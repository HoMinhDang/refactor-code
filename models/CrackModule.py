import torch
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


class CrackModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name, **model_hparams)
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss_module = DiceBCELoss()

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
        masks = (masks > 0).float().unsqueeze(1)

        logits = self(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        loss  = self.loss_module(logits, masks)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self._forward_pass(batch)
        self.train_metrics.update(preds, masks.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

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
