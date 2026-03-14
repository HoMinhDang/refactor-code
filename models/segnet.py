import torch.nn as nn
import segmentation_models_pytorch as smp
from .registry import register_model

@register_model("segnet")
class SegNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        super().__init__()
        # Tôi sử dụng Unet++ hoặc DeepLabV3+ vì SegNet nguyên bản khá cũ
        # Ở đây dùng Unet++ để có độ chính xác cao cho vết nứt (cracks)
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)