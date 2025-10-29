"""
Implementation of Saptial and Channel Fusion
Author: xuyunjiang
"""

from turtle import forward
import torch
import torch.nn as nn

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine_simple, get_rotated_roi
from matplotlib import pyplot as plt
from icecream import ic
import torch.nn.functional as F
import numpy as np
import spconv
import spconv.pytorch
from spconv.pytorch import SparseConv2d

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.mu_gen = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1,  bias=False)
        self.logvar_gen = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1,  bias=False)

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1,  bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_feature = self.global_avg_pool(x)
        # avg_feature = self.global_avg_pool(x).view(b, c)
        latent = self.encode(avg_feature)
        mu = self.mu_gen(latent)
        logvar = self.logvar_gen(latent)
        latent_regular = self.reparameterize(mu,logvar)
        out = self.decode(latent_regular)
        out = out.view(b, c, 1, 1)
        return out.expand_as(x) # 1,C,H,W
        # return x * y.expand_as(x)

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(SpatialBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        # 不能加BN，会影响统计特性
        self.mu_gen = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1, bias=False)
   
        self.logvar_gen = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1, bias=False)

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1,  bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def forward(self, x):
        # x,: 1，C，H，W
        latent = self.encode(x)
        mu = self.mu_gen(latent)
        logvar = self.logvar_gen(latent)

        latent_regular = self.reparameterize(mu, logvar) # 1, C/r, H, W

        out = self.decode(latent_regular) # 1, C, H, W
        return out

class SpatialChannelFusion(nn.Module):
    def __init__(self, in_channels) -> None:
        super(SpatialChannelFusion, self).__init__()
        self.se_att = SEBlock(in_channels)

        self.sp_att = SpatialBlock(in_channels)
    def forward(self, x):
        # x: N,C,H,W 一个scenario下的所有agent feature map
        sum_feature = torch.sum(x, dim=0, keepdim=True) # (1, C, H, W)
        ch_att = self.se_att(sum_feature) # 1, C, 1, 1 --> expand as 1, C, H, W
        sp_att = self.sp_att(sum_feature) # 1, C, H, W

        sum_att = ch_att + sp_att

        sum_att = torch.sigmoid(sum_att) # 1, C，H, W

        out = sum_feature * sum_att

        return out.squeeze(0)
