import torch
from torch import nn
from .MiT import MiT
from .ResNet import ResNetEncoder
from .decoder import CrackAwareBiFusionModule, Upsample, Conv
from .registry import register_model

@register_model("cafnet")
class CrackAwareFusionNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=(64, 64, 128, 256),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4, 4, 4, 4),
        reduction_ratios=(8, 4, 2, 1),
        depths=(2, 2, 2, 2),
        crackam=True,
        crackspam=True,
        attn_gate=True
    ):
        super().__init__()
        
        self.mit = MiT(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            reduction_ratios=reduction_ratios,
            depths=depths
        )
        self.cnn = ResNetEncoder()
        
        self.fusion1 = CrackAwareBiFusionModule(cnn_dim=64, trans_dim=64, bi_dim=32, out_dim=32, crackspam=crackspam, crackam=crackam)
        self.fusion2 = CrackAwareBiFusionModule(cnn_dim=64, trans_dim=64, bi_dim=64, out_dim=64, crackspam=crackspam, crackam=crackam)
        self.fusion3 = CrackAwareBiFusionModule(cnn_dim=128, trans_dim=128, bi_dim=128, out_dim=128, crackspam=crackspam, crackam=crackam)
        self.fusion4 = CrackAwareBiFusionModule(cnn_dim=256, trans_dim=256, bi_dim=256, out_dim=256, crackspam=crackspam, crackam=crackam)
        
        self.up5 = Upsample(in_dim=512, skip_dim=256, out_dim=256, attn_gate=attn_gate)
        self.up4 = Upsample(in_dim=256, skip_dim=128, out_dim=128, attn_gate=attn_gate)
        self.up3 = Upsample(in_dim=128, skip_dim=64, out_dim=64, attn_gate=attn_gate)
        self.up2 = Upsample(in_dim=64, skip_dim=32, out_dim=64, attn_gate=attn_gate)
        self.up1 = Upsample(in_dim=64, skip_dim=0, out_dim=64, attn_gate=False)
        
        self.final = nn.Sequential(
            Conv(in_dim=64, out_dim=8, kernel_size=3, bn=True,relu=True),
            Conv(in_dim=8, out_dim=1, kernel_size=1, padding=0, bn=False, relu=False)
        )
    
    def forward(self, x):
        mit_f = self.mit(x)
        cnn_f = self.cnn(x)
        
        fused1 = self.fusion1(cnn_f[0], mit_f[0])
        fused2 = self.fusion2(cnn_f[1], mit_f[1])
        fused3 = self.fusion3(cnn_f[2], mit_f[2])
        fused4 = self.fusion4(cnn_f[3], mit_f[3])
        
        d5 = self.up5(cnn_f[4], fused4)
        d4 = self.up4(d5, fused3)
        d3 = self.up3(d4, fused2)
        d2 = self.up2(d3, fused1)
        d1 = self.up1(d2, x_skip=None)

        out = self.final(d1)
        return out, d1, d2, d3, d4, d5


