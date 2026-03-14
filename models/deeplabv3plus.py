import torch.nn as nn
import segmentation_models_pytorch as smp
from .registry import register_model

@register_model("deeplabv3plus")
class DeepLabV3Plus(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None # Giữ None để dùng với DiceBCELoss trong CrackModule
        )

    def forward(self, x):
        return self.model(x)