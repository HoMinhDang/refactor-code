import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, bn=True, relu=True):
        super().__init__()
        
        
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias= not bn)
        self.bn = nn.BatchNorm2d(out_dim) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv(in_dim, out_dim),
            Conv(out_dim, out_dim, relu=False)
        )
        
        self.skip = Conv(in_dim, out_dim, kernel_size=1, padding=0, relu=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.double_conv(x) + self.skip(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        mid_dim = out_dim // 2
        
        self.block = nn.Sequential(
            Conv(in_dim, mid_dim, 1, stride=1, padding=0, bn=True, relu=True),
            Conv(mid_dim, mid_dim, 3, stride=1, padding=1, bn=True, relu=True),
            Conv(mid_dim, out_dim, 1, stride=1, padding=0, bn=True, relu=False) 
        )

        if in_dim != out_dim:
            self.skip = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim)
            )
        else:
            self.skip = nn.Identity()
            
        self.final_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.block(x) + self.skip(x)
        return self.final_relu(out)

class CrackAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.fc = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: B C H W
        h_pool = torch.max(x, dim=3, keepdim=True)[0] # B C H 1
        v_pool = torch.max(x, dim=2, keepdim=True)[0] # B C 1 W
        
        se = h_pool.mean(2, keepdim=True) + v_pool.mean(3, keepdim=True)
        se = self.fc(se)

        return x * self.sigmoid(se)

class CrackSPAM(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = lambda x: torch.cat((torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]), dim=1)
        self.conv_h = nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.conv_w = nn.Conv2d(2, 1, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        h_attn = self.conv_h(x_compress)
        w_attn = self.conv_w(x_compress)
        attn = self.sigmoid(h_attn + w_attn)
        return x * attn

class CrackAwareBiFusionModule(nn.Module):
    def __init__(self, cnn_dim, trans_dim, bi_dim, out_dim, drop=0., crackam=True, crackspam=True):
        super().__init__()
        
        self.am = CrackAM(trans_dim) if crackam else nn.Identity()
        self.spam = CrackSPAM() if crackspam else nn.Identity()
        
        self.cnn_proj = Conv(cnn_dim, bi_dim, kernel_size=1, padding=0, bn=True, relu=False)
        self.trans_proj = Conv(trans_dim, bi_dim, kernel_size=1, padding=0, bn=True, relu=False)
        self.conv = Conv(bi_dim, bi_dim, kernel_size=3, bn=True, relu=True)
        
        self.refine = ResidualBlock(cnn_dim + trans_dim + bi_dim, out_dim)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        
    def forward(self, cnn, trans):
        C = self.spam(cnn)
        T = self.am(trans)        
        bi = self.conv(self.cnn_proj(cnn) * self.trans_proj(trans))
        
        fuse = torch.cat([C, T, bi], dim=1)
        
        return self.drop(self.refine(fuse))


class AttnGate(nn.Module):
    def __init__(self,  gate_dim, skip_dim, inter_dim):
        super().__init__()
        
        self.conv_gate = Conv(gate_dim, inter_dim, kernel_size=1, padding=0, bn=True, relu=False)
        self.conv_skip = Conv(skip_dim, inter_dim, kernel_size=1, padding=0, bn=True, relu=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_dim, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, skip):
        g1 = self.conv_gate(gate)
        s1 = self.conv_skip(skip)
        
        combine = self.relu(g1 + s1)
        attn = self.psi(combine)
        
        return skip * attn

class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, skip_dim=0, attn_gate= True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if skip_dim > 0:
            self.conv = DoubleConv(in_dim + skip_dim, out_dim)
        else:
            self.conv = DoubleConv(in_dim, out_dim)
            
        if attn_gate:
            self.attn_block = AttnGate(
                gate_dim=in_dim,                          
                skip_dim=skip_dim,                       
                inter_dim=min(in_dim, skip_dim) if skip_dim > 0 else in_dim
            )
        else:
            self.attn_block = None

    def forward(self, x_dec, x_skip=None):
        x_dec = self.up(x_dec)
        if x_skip is not None:
            diff_h = x_skip.size(2) - x_dec.size(2)
            diff_w = x_skip.size(3) - x_dec.size(3)
            x_dec = F.pad(
                x_dec,
                [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2
                ]
            )
            if self.attn_block is not None:
                x_skip = self.attn_block(x_dec, x_skip)
            x_dec = torch.cat([x_skip, x_dec], dim=1)

        return self.conv(x_dec)