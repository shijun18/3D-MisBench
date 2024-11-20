import torch 
import torch.nn as nn
from nnunetv2.mymodel.CAMS.mamba_simple1 import Mamba


D_state = 2
Expand = 1
drop_rate = 0.10

# 翻转操作
def flip_Dim(x,dim): 
    if dim == '12':
        x1=x
    if dim == '14':
        x1 = x.permute(0, 1, 3, 2).contiguous()  
    if dim == '98':
        x1 = torch.flip(x,[2,3])
    if dim == '96':
        x1 = torch.flip(x,[2,3]).permute(0, 1, 3, 2).contiguous()  
    if dim == '78':
        x1 = torch.flip(x,[2])           
    if dim == '74':
        x1 = torch.flip(x,[2]).permute(0, 1, 3, 2).contiguous()  
    if dim == '32':
        x1 = torch.flip(x,[3])          
    if dim == '36':
        x1 = torch.flip(x,[3]).permute(0, 1, 3, 2).contiguous()
    return x1

def flip_Dim_back(x,dim):
    if dim == '12':
        x1=x
    if dim == '14':
        x1 = x.permute(0, 1, 3, 2).contiguous()  
    if dim == '98':
        x1 = torch.flip(x,[2,3])
    if dim == '96':
        x1 = torch.flip(x,[2,3]).permute(0, 1, 3, 2).contiguous()  
    if dim == '78':
        x1 = torch.flip(x,[2])           
    if dim == '74':
        x1 = torch.flip(x,[3]).permute(0, 1, 3, 2).contiguous()  
    if dim == '32':
        x1 = torch.flip(x,[3])          
    if dim == '36':
        x1 = torch.flip(x,[2]).permute(0, 1, 3, 2).contiguous()
    return x1

# 线性层
class Linear_Layer(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Linear_Layer, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.linear1 = nn.Linear(self.n_channels,self.out_channels)
        self.drop = nn.Dropout(p=drop_rate)
        
        self.m = nn.SiLU()
        #self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1).flatten(start_dim=1,end_dim=2) ##  [B,H*W,C]
        x1 = self.linear1(x1)
        #x1 = self.norm(x1)
        x1 = self.m(x1)
        x1 = self.drop(x1)
        x1 = x1.view(b,h,w,self.out_channels).permute(0,3,1,2)
        return x1

class Linear_Layer_Last(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(Linear_Layer_Last, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.linear1 = nn.Linear(self.n_channels,self.out_channels)

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1).flatten(start_dim=1,end_dim=2) ##  [B,H*W,C]
        x1 = self.linear1(x1)
        x1 = x1.view(b,h,w,self.out_channels).permute(0,3,1,2)
        return x1
    
    
class SSM_spa(nn.Module):
    def __init__(self, sp_dim1,sp_dim2):
        super(SSM_spa, self).__init__()
        
        self.sp_dim2 = sp_dim2
        self.ssm2 = Mamba(
          d_model = sp_dim1*sp_dim1,
          out_c = sp_dim2*sp_dim2,
          d_state = D_state,  
          expand=Expand)
        
        self.norm = nn.LayerNorm(normalized_shape=sp_dim2*sp_dim2)

    def forward(self, x1):
        
        b,c,h,w = x1.shape
        x1 = x1.flatten(start_dim=2,end_dim=3) ##  [B,C,H*W]
                
        x1 = self.ssm2(x1)
        x1 = self.norm(x1)          
        x1 = x1.view(b,c,self.sp_dim2,self.sp_dim2) 
        return x1
    
class SSM_cha(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(SSM_cha, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.ssm1 = Mamba(
          d_model = self.n_channels,
          out_c = self.out_channels,
          d_state=D_state,  
          expand=Expand,
      )

        self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1).flatten(start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.ssm1(x1)
        x1 = self.norm(x1) 
         
        x1 = x1.view(b,h,w,self.out_channels).permute(0,3,1,2)
        return x1

class Linear_Layer_SP_Res(nn.Module):
    def __init__(self, sp_dim1,sp_dim2,in_channels,out_channels):
        super(Linear_Layer_SP_Res, self).__init__()
        self.sp_dim1 = sp_dim1
        self.sp_dim2 = sp_dim2
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Linear(sp_dim1*sp_dim1,sp_dim2*sp_dim2)
        self.drop = nn.Dropout(p=drop_rate)
        
        
        
        self.lin_chan = Linear_Layer(self.in_channels,self.out_channels)
        #self.norm = nn.LayerNorm(normalized_shape=sp_dim2*sp_dim2)
        
        self.m = nn.SiLU()

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.flatten(start_dim=2,end_dim=3) # (B,C,H,W) ->(B,C,H*W)
        
        x1 = self.linear1(x1)
        #x1 = self.norm(x1)
        x1 = self.m(x1)
        x1 = self.drop(x1)
         
        x1 = x1.view(b,c,h,w) 
        x1 = self.lin_chan(x1)
        return x1
    
class SSM_cha_Last(nn.Module):
    def __init__(self, n_channels,out_channels):
        super(SSM_cha_Last, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.ssm1 = Mamba(
          d_model = self.n_channels,
          out_c = self.out_channels,
          d_state=D_state,  
          expand=Expand,
      )
        

    def forward(self, x1):
    
        b,c,h,w = x1.shape
        x1 = x1.permute(0,2,3,1).flatten(start_dim=1,end_dim=2) ##  [B,H*W,C]
        
        x1 = self.ssm1(x1)
         
        x1 = x1.view(b,h,w,self.out_channels).permute(0,3,1,2) 
        return x1
    
# 只有一个SSM模块，与论文中不相符
class Branch_3(nn.Module):
    def __init__(self,in_channels, out_channels,sp_dim1,sp_dim2):
        super().__init__()
        self.branch3 = nn.Sequential(
            SSM_spa(sp_dim1,sp_dim2),
            Linear_Layer(in_channels, out_channels),
        )
        
        self.br_r = Linear_Layer_SP_Res(sp_dim1, sp_dim2,in_channels, out_channels)
    def forward(self, x):
        return  (self.branch3(x) + self.br_r(x))
    
# 论文中的LIFM block
class Branch_2(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.branch2 = nn.Sequential(
            SSM_cha(in_channels, in_channels),
            Linear_Layer(in_channels, out_channels),
            SSM_cha(out_channels, out_channels), 
            nn.Dropout(p=drop_rate),
        )
    def forward(self, x):
        return  self.branch2(x)
    
class Branch_2_Last(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.branch2 = nn.Sequential(
            SSM_cha_Last(in_channels, in_channels),
            Linear_Layer_Last(in_channels, out_channels),
            SSM_cha_Last(out_channels, out_channels), 
        )
    def forward(self, x):
        return  self.branch2(x)

# decoder的输出层，无激活函数
class Branch12_Last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear_Layer_Last(in_channels, out_channels)
        self.br2 = Branch_2_Last(in_channels, out_channels)
           
    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.br2(x)
        return (x1+x2)
    
# 线性层+LIFM 论文中的MCA 
# TODO:哪里有weight_sharing?
class Branch12(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear_Layer(in_channels, out_channels)
        self.br2 = Branch_2(in_channels, out_channels)
        self.drop = nn.Dropout(p=drop_rate)
           
    def forward(self, x):
        x1 = self.linear(x) 
        x2 = self.br2(x)
        return self.drop(x1+x2)

# 线性层+LIFM+LIFM 论文中的CS-IF 
class Branch123(nn.Module):
    def __init__(self, in_channels, out_channels, sp_dim1,sp_dim2):
        super().__init__()
        self.linear = Linear_Layer(in_channels, out_channels)
        self.br2 = Branch_2(in_channels, out_channels)
        self.br3 = Branch_3(in_channels, out_channels, sp_dim1,sp_dim2)
        
        self.drop = nn.Dropout(p=drop_rate)
           
    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.br2(x)
        x3 = self.br3(x)
        return self.drop(x1+x2+x3)   #并行的
    
     
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_br12 = nn.Sequential(
            nn.AvgPool2d(2),
            Branch12(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_br12(x)
    
class Down_1(nn.Module):
    def __init__(self, in_channels, out_channels, sp_dim1,sp_dim2):
        super().__init__()
        self.maxpool_br123 = nn.Sequential(
            nn.AvgPool2d(2),
            Branch123(in_channels, out_channels, sp_dim1,sp_dim2)
        )
    def forward(self, x):
        return self.maxpool_br123(x)
    
class Up_1(nn.Module):
    def __init__(self, in_channels, out_channels,sp_dim1,sp_dim2):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.br123 = Branch123(in_channels+in_channels//2, out_channels,sp_dim1,sp_dim2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.br123(x)

import math
def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe



# Base = 64
# image_size =  160
# BN1 = image_size // 16
# BN2 = image_size // 32 

from timm.models.layers import to_2tuple
class PatchEmbed(nn.Module): # [2,1,160,160] -->[2,1600,96]
    def __init__(self, img_size=160, patch_size=2, in_chans=3, embed_dim=64, Apply_Norm=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
            
        self.norm = nn.LayerNorm(embed_dim)
        self.Apply_Norm = Apply_Norm

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.Apply_Norm:
            x = self.norm(x)
        x = x.transpose(1, 2).view(B, self.embed_dim, H//2, W//2) 
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class CustomEncoder(nn.Module):
    def __init__(self, in_channel= 3, base_channels=64, img_size=160):
        super(CustomEncoder, self).__init__()
        self.BN1= img_size // 16
        self.BN2 = img_size // 32

        self.pm = PatchEmbed(img_size=img_size ,embed_dim= base_channels, in_chans=in_channel)
        self.inc = Branch12(base_channels,base_channels)
        self.down1 = Down(base_channels,2*base_channels)
        self.down2 = Down(2*base_channels,4*base_channels)
        
        self.down3 = Down_1(4*base_channels,8*base_channels,self.BN1,self.BN1)
        self.down4 = Down_1(8*base_channels,16*base_channels,self.BN2,self.BN2)
        self.m = nn.SiLU()
        
        self.pos_embed =  positionalencoding2d(base_channels,img_size//2,img_size//2).to(DEVICE)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
          nn.Linear(1024, 1280),
          nn.Hardswish(inplace=True),
          nn.Dropout(p=0.2, inplace=True),
          nn.Linear(1280, 1000),
          )
        
    def forward(self, inp):
        
        # inp = inp.repeat(1,3,1,1) # (B,3,H,W)
        inp = self.pm(inp) #(B,64,H/2,W/2)
        inp = inp  + self.pos_embed  #                       (B,64,H/2,W/2)

        x1_12 = self.inc(inp)   
        x1_98 = flip_Dim(inp,'98')
        x1_98 = self.inc(x1_98) 
        x1_98 = flip_Dim_back(x1_98,'98')
        x1 = x1_12 + x1_98
        
        x1 = self.m(x1) #                                    (B,64,H/2,W/2)
        
        x2_12 = self.down1(x1) #（B,128,H/4,W/4）
        
        x2_98 = flip_Dim(x1,'98')
        x2_98 = self.down1(x2_98) 
        x2_98 = flip_Dim_back(x2_98,'98')
        
        x2 = x2_12 + x2_98
        x2 = self.m(x2) #                                   （B,128,H/4,W/4）
        
        x3_12 = self.down2(x2) #（B,256,H/8,W/8）
        
        x3_98 = flip_Dim(x2,'98')
        x3_98 = self.down2(x3_98) 
        x3_98 = flip_Dim_back(x3_98,'98')
        
        x3 = x3_12 + x3_98
        x3 = self.m(x3) #                                    （B,256,H/8,W/8）
        
        x4_12 = self.down3(x3) #（B,512,H/16,W/16）
        
        x4_98 = flip_Dim(x3,'98')
        x4_98 = self.down3(x4_98) 
        x4_98 = flip_Dim_back(x4_98,'98')
        
        x4 = x4_12 + x4_98
        x4 = self.m(x4) #                                   （B,512,H/16,W/16）
        
        
        x5_12 = self.down4(x4) #（B,1024,H/32,W/32）
        
        x5_98 = flip_Dim(x4,'98')
        x5_98 = self.down4(x5_98) 
        x5_98 = flip_Dim_back(x5_98,'98')
        
        x5 = x5_12 + x5_98
        x5_r = self.m(x5) #                                   （B,1024,H/32,W/32）

        # 分类器 可注释
        # x5 = self.avgpool(x5_r)
        # x5 = torch.flatten(x5, 1)
        # x5 = self.classifier(x5)
        return x5_r,x4,x3,x2,x1,inp
    
  
class Up(nn.Module):
    def __init__(self, in_channels, out_channels,last=None):
        super().__init__()

        self.last = last
        if self.last is None:
            in_channels = in_channels+in_channels//2
        if self.last is not None:
            in_channels = in_channels + in_channels
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.br12 = Branch12(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.last is not None:
            x2 = self.up(x2)    
        x = torch.cat([x2, x1], dim=1) #跳跃连接
        return self.br12(x)

# Num_Classes = 5
class Decoder(nn.Module):
    def __init__(self, base_channels=64, num_classes=5, img_size=160):
        super(Decoder, self).__init__()
        self.Base = base_channels
        self.BN1 = img_size // 16
        self.Num_Classes = num_classes

        self.up1 = Up_1(16*self.Base,8*self.Base,self.BN1,self.BN1)
        #self.up2 = Up_1(8*Base,4*Base,BN3,BN3)
        
        self.up2 = Up(8*self.Base,4*self.Base)
        self.up3 = Up(4*self.Base,2*self.Base)
        self.up4 = Up(2*self.Base,self.Base)
        self.up5 = Up(self.Base,self.Base,last='yes')
        self.outc = Branch12_Last(self.Base, self.Num_Classes)
        
    def forward(self, x5,x4,x3,x2, x1,inp):
        
        x = self.up1(x5, x4)    
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, inp)
        x = self.outc(x)
        return x

class CAMS(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=160, base_channel=64):
        super(CAMS, self).__init__()
        self.encoder = CustomEncoder(in_channel=in_channels,base_channels=base_channel, img_size=img_size)
        self.decoder = Decoder(base_channels=base_channel, num_classes=out_channels, img_size=img_size)

    def forward(self, x):
        x5,x4,x3,x2,x1,inp = self.encoder(x)
        out = self.decoder(x5,x4,x3,x2,x1,inp)
        return out
    
# class CAMS(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, img_size=160, base_channel=64):
#         super(CAMS, self).__init__()
#         self.encoderdecoder = EncoderDecoder(CustomEncoder())
        