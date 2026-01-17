import torch
from torch import nn
from einops import rearrange


# MiT
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class DWConv(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        return self.pw(self.dw(x))

class OverLapPatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(
            in_dim, out_dim,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        
        self.norm = LayerNorm2d(out_dim)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class EfficientMSA(nn.Module):
    def __init__(self, dim, num_heads, reduction_ratio):
        super().__init__()
        
        self.norm = LayerNorm2d(dim)
        
        self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.sr_norm = LayerNorm2d(dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.proj = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        # Q
        q = rearrange(x_norm, "b c h w -> b (h w) c")
        
        # K, V (spatial reduction)
        kv = self.sr(x_norm)
        kv = self.sr_norm(kv)
        kv = rearrange(kv, "b c h w -> b (h w) c")
        
        out, _ = self.attn(q, kv, kv)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        out = self.proj(out)

        return out
    
class MixFFN(nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        hidden_dim = dim * expansion
        
        self.norm = LayerNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DWConv(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
    
    def forward(self, x):
        return self.ffn(self.norm(x))

class MiTBlock(nn.Module):
    def __init__(self, dim, num_heads, expansion, reduction_ratio):
        super().__init__()
        self.attn = EfficientMSA(dim, num_heads, reduction_ratio)
        self.ffn = MixFFN(dim, expansion)
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class MiT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        reduction_ratios=(8, 4, 2, 1),
        depths=(2, 2, 2, 2)
    ):
        super().__init__()
        
        self.stages = nn.ModuleList()
        prev_dim = in_channels
        
        for i in range(len(embed_dims)):
            stage = nn.ModuleList()
            
            stage.append(
                OverLapPatchEmbedding(
                    in_dim=prev_dim,
                    out_dim=embed_dims[i],
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            
            blocks = nn.Sequential(*[
                MiTBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    expansion=mlp_ratios[i],
                    reduction_ratio=reduction_ratios[i]
                )
                for _ in range(depths[i])
            ])

            stage.append(blocks)
            self.stages.append(stage)
            
            prev_dim = embed_dims[i]
    
    def forward(self, x):
        features = []
        for patch_embed, blocks in self.stages:
            x = patch_embed(x)
            x = blocks(x)
            features.append(x)
        return features
