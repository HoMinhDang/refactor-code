from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetEncoder(nn.Module):
    def __init__(self, backbone=resnet18(weights=ResNet18_Weights.DEFAULT)):
        super().__init__()
        
        # stem
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )
        
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1    # 64
        self.layer2 = backbone.layer2    # 128
        self.layer3 = backbone.layer3    # 256
        self.layer4 = backbone.layer4    # 512

    def forward(self, x):
        c1 = self.stem(x)        # [B, 64, H/2,  W/2]
        c2 = self.maxpool(c1)
        c2 = self.layer1(c2)    # [B, 64, H/4,  W/4]

        c3 = self.layer2(c2)    # [B, 128, H/8,  W/8]
        c4 = self.layer3(c3)    # [B, 256, H/16, W/16]
        c5 = self.layer4(c4)    # [B, 512, H/32, W/32]

        return [c1, c2, c3, c4, c5]