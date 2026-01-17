import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall
from .registry import MODEL_REGISTRY
from .proposed import CrackAwareFusionNet
from utils.metric import DiceBCELoss


class CrackModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = create_model(model_name, **model_hparams)
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.loss_module = DiceBCELoss()
        
        self.acc = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.re = BinaryRecall()
        self.pre = BinaryPrecision()
        self.iou = BinaryJaccardIndex()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def _common_step(self, batch, batch_idx):
        imgs, masks = batch
        masks = masks.float().unsqueeze(1)   # [B, 1, H, W]

        logits = self(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]               # lấy main output

        loss = self.loss_module(logits, masks)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        return loss, preds, masks

    
    def training_step(self, batch, batch_idx):
        loss, preds, masks = self._common_step(batch, batch_idx)
        
        f1 = self.f1(preds, masks)
        iou = self.iou(preds, masks)
        
        self.log_dict({
            "train_loss": loss,
            "train_f1": f1,
            "train_iou": iou 
            },
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self._common_step(batch, batch_idx)
        
        f1 = self.f1(preds, masks)
        iou = self.iou(preds, masks)
        
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log_dict(
            {
                "val_f1": f1,
                "val_iou": iou,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        
    
    def test_step(self, batch, batch_idx):
        loss, preds, masks = self._common_step(batch, batch_idx)

        acc = self.acc(preds, masks)
        pre = self.pre(preds, masks)
        re = self.re(preds, masks)
        f1 = self.f1(preds, masks)
        iou = self.iou(preds, masks)
        
        
        self.log_dict({
            "test_loss": loss,
            "test_acc": acc,
            "test_precision": pre,
            "test_recall": re,
            "test_f1": f1,
            "test_iou": iou
            },
            on_epoch=True,
            prog_bar=True
        )
    
    def predict_step(self, batch, batch_idx):
        imgs = batch
        logits = self(imgs)
        probs = torch.sigmoid(logits)
        preds = (preds > 0.5).float()
        return preds

def create_model(model_name: str, **model_params):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f'Unknown model "{model_name}". '
            f'Available models: {list(MODEL_REGISTRY.keys())}'
        )

    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(**model_params)