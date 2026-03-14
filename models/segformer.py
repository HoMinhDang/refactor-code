import torch.nn as nn
import segmentation_models_pytorch as smp
from .registry import register_model

@register_model("segformer")
class SegFormer(nn.Module):
    def __init__(self, encoder_name="mit_b0", encoder_weights="imagenet", in_channels=3, classes=1):
        super().__init__()
        # SegFormer sử dụng các encoder Mix Vision Transformer (mit_b0 đến mit_b5)
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None # Giữ nguyên để khớp với DiceBCELoss
        )

    def forward(self, x):
        return self.model(x)