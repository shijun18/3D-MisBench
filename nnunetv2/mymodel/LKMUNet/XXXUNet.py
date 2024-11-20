from re import S
from xml.dom import xmlbuilder
import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
import copy

drop_rate = 0.1

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out
    
class PixelMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)


        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)


        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # use_fast_path=False,

        )

      
        # adjust the window size here to fit the feature map
        self.p = p*5
        self.p1 = 5*p
        self.p2 = 5*p
        self.p3 = 5*p
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)

        B, C = x.shape[:2]

        assert C == self.dim
        img_dims = x.shape[2:]

        if ll == 5: #3d
         
            Z,H,W = x.shape[2:]

            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_div = x.reshape(B, C, Z//self.p1, self.p1, H//self.p2, self.p2, W//self.p3, self.p3)  # 原始 (Z, H, W) 空间被划分成若干小块，每个小块的形状为 (self.p1, self.p2, self.p3)，共 (Z//self.p1, H//self.p2, W//self.p3) 个小块。
                # (B, self.p1, self.p2, self.p3, C, Z//self.p1, H//self.p2, W//self.p3) -> (B*self.p1*self.p2*self.p3, C, Z//self.p1, H//self.p2, W//self.p3)
                x_div = x_div.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous().view(B*self.p1*self.p2*self.p3, C, Z//self.p1, H//self.p2, W//self.p3)
            else:
                x_div = x

        elif ll == 4: #2d
            H,W = x.shape[2:]

            if H%self.p==0 and W%self.p==0:                
                x_div = x.reshape(B, C, H//self.p, self.p, W//self.p, self.p).permute(0, 3, 5, 1, 2, 4).contiguous().view(B*self.p*self.p, C, H//self.p, W//self.p)            
            else:
                x_div = x
        

        NB = x_div.shape[0] # N个小块*Batch size
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2) #NB，n_tokens，C
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm) #对应论文中的两个方向输入mamba 功能和vision mamba中的v2一致
        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        # 还原
        if ll == 5:
            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p1, self.p2, self.p3, C, NZ, NH, NW).permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if H%self.p==0 and W%self.p==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p, self.p, C, NH, NW).permute(0, 3, 4, 1, 5, 2).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        out = x_out + x

        return out
    
class WindowMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        

        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # use_fast_path=False,

        )

      

       
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)
     
       

        B, C = x.shape[:2]

        assert C == self.dim
   
        img_dims = x.shape[2:]



        if ll == 5: #3d
            
            Z,H,W = x.shape[2:]

            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool3d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x

        elif ll == 4: #2d

            H,W = x.shape[2:]
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool2d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x
        

      
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NZ, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
                
        out = x_out + x

        return out

class Mamba_Channel_Block(nn.Module):
    def __init__(self, dim, p):
        super().__init__()
        # self.linear = Linear_Layer(in_channels, out_channels)
        self.br1 = PixelMambaLayer(dim, p)
        self.br2 = WindowMambaLayer(dim, p)
        
        self.drop = nn.Dropout(p=drop_rate)
           
    def forward(self, x):
        # x1 = self.linear(x)
        x1 = self.br1(x)
        x2 = self.br2(x)
        return self.drop(x1+x2)   #并行的
    
class Mamba_Spatial_Block(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.drop = nn.Dropout(p=drop_rate)
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]                     # B,C,H,W,D
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens)      # B,C,L where L = H*W*D
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, C, *img_dims)

        return self.drop(out + x)
    
