import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall
from torchmetrics import MetricCollection
from .registry import MODEL_REGISTRY
from .CAFNet_resnet18 import CrackAwareFusionNet
from .CAFNet_mbnv3l import CAFNet_MBNV3L
from .DTrcNet import CTCNet
from .unet import Unet
from .hrsegnet import HrSegNet
from .segformer import SegFormer
from .deeplabv3plus import DeepLabV3Plus
from .segnet import SegNet
from .dcsnet import DcsNet
from .hacnetv2 import hacnetv2
from utils.metric import DiceBCELoss


def _make_metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        {
            "acc": BinaryAccuracy(),
            "pre": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "iou": BinaryJaccardIndex(),
        },
        prefix=f"{prefix}_",
    )


class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, max_samples=256):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples

    def forward(self, feat_clean, feat_pert, mask):
        if feat_clean.shape[2:] != mask.shape[2:]:
            mask = F.interpolate(mask, size=feat_clean.shape[2:], mode='nearest')

        feat_clean = F.normalize(feat_clean, p=2, dim=1)
        feat_pert = F.normalize(feat_pert, p=2, dim=1)

        crack_mask = mask.squeeze(1) == 1.0
        bg_mask = mask.squeeze(1) == 0.0

        if crack_mask.sum() == 0 or bg_mask.sum() == 0:
            return 0.0 * feat_clean.sum()

        z_clean_crack = feat_clean.permute(0, 2, 3, 1)[crack_mask]
        z_pert_crack = feat_pert.permute(0, 2, 3, 1)[crack_mask]
        z_clean_bg = feat_clean.permute(0, 2, 3, 1)[bg_mask]

        num_cracks = z_clean_crack.size(0)
        num_bgs = z_clean_bg.size(0)

        if num_cracks > self.max_samples:
            idx_crack = torch.randperm(num_cracks, device=feat_clean.device)[:self.max_samples]
            z_clean_crack = z_clean_crack[idx_crack]
            z_pert_crack = z_pert_crack[idx_crack]

        max_bg_samples = self.max_samples * 2
        if num_bgs > max_bg_samples:
            idx_bg = torch.randperm(num_bgs, device=feat_clean.device)[:max_bg_samples]
            z_clean_bg = z_clean_bg[idx_bg]

        sim_pos = (z_clean_crack * z_pert_crack).sum(dim=-1)
        sim_neg = torch.matmul(z_clean_crack, z_clean_bg.transpose(0, 1))

        exp_pos = torch.exp(sim_pos / self.temperature)
        exp_neg = torch.exp(sim_neg / self.temperature).sum(dim=-1)

        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8)).mean()
        return loss


class CrackModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, lr=1e-4, weight_decay=1e-5,
                 lambda_c=0, temperature=0.1, contra_layer="both"):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name, **model_hparams)
        self.lr = lr
        self.weight_decay = weight_decay

        self.lambda_c = lambda_c
        self.contra_layer = str(contra_layer).lower()

        self.loss_module = DiceBCELoss()
        if self.lambda_c > 0:
            self.contra_loss = PixelContrastiveLoss(temperature=temperature)

        self.train_metrics = _make_metrics("train")
        self.val_metrics   = _make_metrics("val")
        self.test_metrics  = _make_metrics("test")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _forward_pass(self, batch):
        imgs, masks, filenames = batch
        masks = (masks > 0).float()

        logits = self(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        loss  = self.loss_module(logits, masks)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        if self.lambda_c > 0:
            return self._training_step_contrastive(batch)
        else:
            loss, preds, masks = self._forward_pass(batch)
            self.train_metrics.update(preds, masks.int())
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            return loss

    def _training_step_contrastive(self, batch):
        img_clean, img_pert, masks, filenames = batch
        if len(masks.shape) == 3:
            masks = masks.float().unsqueeze(1)

        bs = img_clean.size(0)
        imgs_combined = torch.cat([img_clean, img_pert], dim=0)
        outputs = self(imgs_combined)

        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            logits_combined = outputs[0]
            features_combined = outputs[1]
        else:
            logits_combined = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            features_combined = None

        logits_clean = logits_combined[:bs]
        logits_pert = logits_combined[bs:]

        loss_task = (self.loss_module(logits_clean, masks) + self.loss_module(logits_pert, masks)) * 0.5

        loss_contra = torch.tensor(0.0, device=self.device)
        if features_combined is not None:
            if self.contra_layer == "3":
                feats = [features_combined[0]]
            elif self.contra_layer == "4":
                feats = [features_combined[1]]
            else:
                feats = [features_combined[0], features_combined[1]]

            total = 0.0
            for feat in feats:
                feat_clean = feat[:bs]
                feat_pert = feat[bs:]
                total += self.contra_loss(feat_clean, feat_pert, masks)
            loss_contra = total / len(feats)

        total_loss = loss_task + self.lambda_c * loss_contra

        probs = torch.sigmoid(logits_clean)
        preds = (probs > 0.5).float()
        self.train_metrics.update(preds, masks.int())

        self.log_dict({
            "train_loss": total_loss,
            "train_bce": loss_task,
            "train_contra": loss_contra,
            **self.train_metrics,
        }, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self._forward_pass(batch)
        self.val_metrics.update(preds, masks.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, preds, masks = self._forward_pass(batch)
        self.test_metrics.update(preds, masks.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, masks, filenames = batch
        logits = self(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.sigmoid(logits)
        return (probs > 0.5).float()


def create_model(model_name: str, **model_params):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f'Unknown model "{model_name}". '
            f'Available models: {list(MODEL_REGISTRY.keys())}'
        )
    return MODEL_REGISTRY[model_name](**model_params)
