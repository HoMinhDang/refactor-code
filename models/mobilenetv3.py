from torch import nn
from torchvision import models

class MobileNetV3LargeBackbone(nn.Module):
    """
    Tạo một module "xương sống" (backbone) từ MobileNetV3-Large
    Module này sẽ trả về 5 feature map ở các mức stride /2, /4, /8, /16, /32.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Tải mô hình MobileNetV3-Large gốc
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None
            
        original_model = models.mobilenet_v3_large(weights=weights)
        
        # Lấy 5 đầu ra: /2, /4, /8, /16, /32
        # Tương ứng với các khối trong MobileNetV3-Large
        self.stage0 = nn.Sequential(*original_model.features[0:1])  # out: H/2 (16 channels)
        self.stage1 = nn.Sequential(*original_model.features[1:4])  # out: H/4 (24 channels)
        self.stage2 = nn.Sequential(*original_model.features[4:7])  # out: H/8 (40 channels)
        self.stage3 = nn.Sequential(*original_model.features[7:13]) # out: H/16 (112 channels)
        self.stage4 = nn.Sequential(*original_model.features[13:16])# out: H/32 (160 channels)

    def forward(self, x):
        # Cho đầu vào đi qua từng giai đoạn
        c0 = self.stage0(x)
        c1 = self.stage1(c0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        
        # Trả về 5 feature map ở các tỷ lệ khác nhau
        # [H/2, H/4, H/8, H/16, H/32]
        return c0, c1, c2, c3, c4