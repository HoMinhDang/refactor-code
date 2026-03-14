import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

@register_model("hrsegnet")
class HrSegNet(nn.Module):
    def __init__(self, in_channels=3, base=16, num_classes=2):
        super(HrSegNet, self).__init__()
        self.base = base
        self.num_classes = num_classes

        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base // 2),
            nn.ReLU()
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(base // 2, base, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )

        # Segmentation blocks
        self.seg1 = SegBlock(base=base, stage_index=1)
        self.seg2 = SegBlock(base=base, stage_index=2)
        self.seg3 = SegBlock(base=base, stage_index=3)

        # Segmentation heads
        self.aux_head1 = SegHead(inplanes=base, outplanes=num_classes, aux_head=True)
        self.aux_head2 = SegHead(inplanes=base, outplanes=num_classes, aux_head=True)
        self.head = SegHead(inplanes=base, outplanes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]  # Input height and width

        # Forward stages
        stem1_out = self.stage1(x)
        stem2_out = self.stage2(stem1_out)
        seg1_out = self.seg1(stem2_out)
        seg2_out = self.seg2(seg1_out)
        seg3_out = self.seg3(seg2_out)

        # Segmentation head outputs
        logit_list = []
        main_out = self.head(seg3_out)  # Main output
        logit_list.append(main_out)

        if self.training:
            aux_out1 = self.aux_head1(seg1_out)
            aux_out2 = self.aux_head2(seg2_out)
            logit_list.extend([aux_out1, aux_out2])  # Auxiliary outputs

        # Upsample logits to the original resolution
        logit_list = [
            F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=True)
            for logit in logit_list
        ]
        return logit_list


class SegBlock(nn.Module):
    def __init__(self, base, stage_index):
        super(SegBlock, self).__init__()

        self.h_conv1 = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )
        self.h_conv2 = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )

        self.l_conv1 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU()
        )
        self.l_conv2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU()
        )

        self.l2h_conv = nn.Conv2d(base * 2, base, kernel_size=1)

    def forward(self, x):
        h1 = self.h_conv1(x)
        h2 = self.h_conv2(h1)

        l1 = self.l_conv1(x)
        l2 = self.l_conv2(l1)

        l2_upsampled = F.interpolate(l2, size=h2.shape[2:], mode="bilinear", align_corners=True)
        output = h2 + self.l2h_conv(l2_upsampled)
        return output


class SegHead(nn.Module):
    def __init__(self, inplanes, outplanes, aux_head=False):
        super(SegHead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.con_bn_relu = (
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1)
            if aux_head
            else nn.ConvTranspose2d(inplanes, inplanes, kernel_size=2, stride=2)
        )
        self.conv_out = nn.Conv2d(inplanes, outplanes, kernel_size=1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.con_bn_relu(x)
        return self.conv_out(x)