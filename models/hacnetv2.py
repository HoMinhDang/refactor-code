import torch
import torch.nn as nn
from .registry import register_model

class CrackAM(nn.Module):
    def __init__(self, channels, rate=1, add_maxpool=False, **_):
        super(CrackAM, self).__init__()
        self.fc = nn.Conv2d(int(channels), channels, kernel_size=1, padding=0)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        max_pool_h = torch.max(x, dim=3)[0] # (N, C, H, 1)
        max_pool_v = torch.max(x, dim=2)[0] # (N, C, 1, W)
        xtmp = torch.concat((max_pool_h, max_pool_v), dim=2)  # Shape: [batch_size, channels, width+height]
        x_se = xtmp.mean((2), keepdim=True).unsqueeze(-1)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)      

def Conv2dBN(in_channels,out_channels, kernel_size=3,stride=1, rate=1, name=None): 
    if stride>1:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=rate, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:    
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=rate, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    

def Conv2dBNv2(in_channels,out_channels, kernel_size=3, rate=1, name=None):
    mid_channels=int(out_channels/2)
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding="same"),
        #nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),    
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, dilation=rate, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def EConv2dBN(in_channels,out_channels, kernel_size=3, rate=1, name=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=1, dilation=rate, padding="same"),
        #nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), stride=1, dilation=rate, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)        
    ) 

def upsampling_branch_block(in_channels,out_channels, scale_factor=(2,2), name=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=scale_factor)    
    ) 

class hybird_aspa(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,rates=[1,1,1,1],hs_att=True):
        super(hybird_aspa,self).__init__()
        self.Conv2dBN1 = Conv2dBNv2(in_channels,out_channels,kernel_size,rate=rates[0])
        self.Conv2dBN2 = Conv2dBNv2(in_channels,out_channels,kernel_size,rate=rates[1])
        self.Conv2dBN3 = Conv2dBNv2(in_channels,out_channels,kernel_size,rate=rates[2])       
        self.Conv2dBN4 = Conv2dBNv2(in_channels,out_channels,kernel_size,rate=rates[3]) 
        self.att =CrackAM(out_channels)
        self.conv11 = Conv2dBN(out_channels,out_channels,kernel_size=1) 
        self.conv_out = EConv2dBN(out_channels,out_channels,rate=1) 

    def forward(self, x):
        d1 = self.Conv2dBN1(x)
        d2 = self.Conv2dBN2(d1)
        d3 = self.Conv2dBN3(d2)
        d4 = self.Conv2dBN4(d3)        
        o1 = d1+d2+d3+d4
        o1 = self.att(o1)
        o2 = self.conv11(o1)
        o2 = x+o2
        o3 = self.conv_out(o2)        
        return o3

class effective_stem_block(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(effective_stem_block,self).__init__()
        self.conv1 = Conv2dBN(in_channels,int(out_channels/2),kernel_size=(1,5),stride=stride)
        self.conv2 = Conv2dBN(in_channels,int(out_channels/2),kernel_size=(5,1),stride=stride)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        o1 = torch.cat((d1,d2),dim=1) 
        return o1   


@register_model("hacnetv2")
class hacnetv2(nn.Module):
    def __init__(self,channels=32):
        super(hacnetv2,self).__init__()
        self.stemf = effective_stem_block(3,channels,stride=1)        

        self.hybird_aspa_f1=hybird_aspa(channels,channels,3,[1,3,9,27])
        self.hybird_aspa_f2=hybird_aspa(channels,channels,3,[1,3,9,27])        
        self.hybird_aspa_f3=hybird_aspa(channels,channels,3,[1,3,9,27])       
        mid_channels=channels*2
        
        self.stemh = effective_stem_block(channels,mid_channels,stride=2)                   
        self.up_p1 = upsampling_branch_block(mid_channels,channels)
        self.up_p2 = upsampling_branch_block(mid_channels,channels)
        self.up_p3 = upsampling_branch_block(mid_channels,channels)                
    

        self.hybird_aspa_h1=hybird_aspa(mid_channels,mid_channels,3,[1,3,9,27])
        self.hybird_aspa_h2=hybird_aspa(mid_channels,mid_channels,3,[1,3,9,27])        
        self.hybird_aspa_h3=hybird_aspa(mid_channels,mid_channels,3,[1,3,9,27])                
        
        self.conv_out = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding="same")
        self.branch_o1= nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding="same")
        self.branch_o2= nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding="same")
        self.branch_o3= nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding="same")
          
    def forward(self, x):
        f0 = self.stemf(x)
        h0 = self.stemh(f0)

        h1 = self.hybird_aspa_h1(h0)
        hf1 = self.up_p1(h1)
        h2 = self.hybird_aspa_h2(h1)
        hf2 = self.up_p2(h2)        
        h3 = self.hybird_aspa_h3(h2)
        hf3 = self.up_p3(h3)   
        
        f1 = self.hybird_aspa_f1(f0)
        f1 = f1+hf1
        f2 = self.hybird_aspa_f2(f1)
        f2 = f2+hf2        
        f3 = self.hybird_aspa_f3(f2)
        f3 = f3+hf3            

        o1 = self.branch_o1(f1) 
        o2 = self.branch_o2(f2) 
        o3 = self.branch_o2(f3)   
        o4 = self.conv_out(torch.cat([o1, o2, o3], dim=1))  
        outputs= [o1,o2,o3,o4]
        return outputs