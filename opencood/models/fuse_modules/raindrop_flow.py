# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from turtle import forward
from uu import encode
from sklearn.metrics import zero_one_loss
import torch
import torch.nn as nn
import easydict
import copy
import math
from torch.autograd import Variable
import torch.nn.functional as F
import random
# from dgl.nn.pytorch.factory import KNNGraph
# import dgl
import numpy as np
import os

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine_simple, get_rotated_roi
from opencood.models.sub_modules.convgru import ConvGRU
from icecream import ic
from matplotlib import pyplot as plt
# from opencood.models.sub_modules.SyncLSTM import SyncLSTM
from opencood.models.sub_modules.MotionNet import STPN, MotionPrediction, StateEstimation, FlowUncPrediction
# from opencood.models.sub_modules.dcn_net import DCNNet
from opencood.models.comm_modules.where2comm_multisweep import Communication
from opencood.models.fuse_modules.SyncNet import SyncLSTM
import torch.nn as nn
from torch.nn.modules.utils import _triple
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
# from opencood.models.fuse_modules.SCFusion import SpatialChannelFusion
# from torchsparse.nn import SubMConv2d
# from opencood.models.sub_modules.sparse_resnet import Sparse_resnet_backbone_aspp
# from opencood.models.sub_modules.base_bev_backbone_resnet_onlydown import ResNetBEVBackboneOnlyDownsample


class RouteFuncMLP(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio, kernels, bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernels[0],1,1],
            padding=[kernels[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernels[1],1,1],
            padding=[kernels[1]//2,0,0],
            bias=False
        )
        self.b.skip_init=True
        self.b.weight.data.zero_() # to make sure the initial values 
                                   # for the output is 1.
        
    def forward(self, x):
        g = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(g))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class TAdaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d 
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(TAdaConv2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
            padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)

        if self.cal_dim == "cin":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in//self.groups, kh, kw)
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0,2,1,3,4).unsqueeze(3) * self.weight).reshape(-1, c_in//self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            # in the official implementation of TAda2D, 
            # there is no bias term in the convs
            # hence the performance with bias is not validated
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
            dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)

        return output
        
    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " +\
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"
    
class TAdaFusion(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.tadaconv_rf = RouteFuncMLP(
                        c_in=input_dim,
                        ratio=4,
                        kernels=[3,3]
        )
        self.tadaconv = TAdaConv2d(
                        in_channels=input_dim, 
                        out_channels=input_dim,
                        kernel_size=[1, 3, 3],
                        stride=[1,1,1],
                        padding=[0, 1, 1],
                        bias=False,
                        cal_dim="cin")
    def forward(self, x):
        # x: N, C, H, W

        x_max = torch.max(x, dim=0, keepdim=True)[0] # 1,C,H,W
        x_mean = torch.mean(x, dim=0, keepdim=True)
        out = torch.cat((x_max, x_mean), dim=0).permute(1, 0, 2, 3).unsqueeze(0) # 2, C, H, W -> C, 2, H, W -> 1, C, 2, H, W
        out = self.tadaconv(out, self.tadaconv_rf(out)) # 1, C, 2, H, W -> 1, C, 1, H, W
        out = out.mean(dim=2).squeeze(0) # C, H, W
        return out

class RadixSoftmax_T(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax_T, self).__init__()
        self.radix = radix # 2
        self.cardinality = cardinality # 1

    def forward(self, x):
        # x: (B, 1, 1, 3C)   [B, 1, 1, 256*3]
        batch = x.size(0)
        cav_num = x.size(1)

        if self.radix > 1:
            # x: (B, 1, 3, C) 
            x = x.view(batch,
                       self.cardinality, self.radix, -1) # [B, 1, 2, 256]
            x = F.softmax(x, dim=2) # 分了三组，计算三组每个通道之间的选用谁的概率 [B, 1, 2, 256]
            # B, 3C
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn_maxavg(nn.Module):
    def __init__(self, input_dim): # 128 /256 /512
        super(SplitAttn_maxavg, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_dim, input_dim * 2, bias=False)
        self.rsoftmax = RadixSoftmax_T(2, 1)


    def forward(self, window_list):

        # (2, C, H, W)  max and mean result
        window_list = window_list.permute(0, 2, 3, 1) # (2, H, W, C)
        N = window_list.shape[0]
        x_gap = torch.sum(window_list, dim=0, keepdim=True) # (1, H，W, C)
        x_gap = x_gap.mean((1, 2), keepdim=True) # [1, 1, 1, C]
        x_gap = self.act1(self.bn1(self.fc1(x_gap))) # 经过一个线性层，一个LN，一个ReLu
        x_attn = self.fc2(x_gap) # 通道升维，[1, 1, 1, C*N]
        x_attn = self.rsoftmax(x_attn).view(1, 1, -1) # [1, 1, N*C]
        out = window_list[0] * x_attn[..., 0:self.input_dim] # [1, 1, C]
        for i in range(1,N):
            out +=  window_list[i] * x_attn[..., i*self.input_dim:(i+1)*self.input_dim] # (H, W, C)
        return out.permute(2, 0, 1) # (C, H, W)

class SplitAttn_T(nn.Module):
    def __init__(self, input_dim): # 256
        super(SplitAttn_T, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        # self.fc2 = nn.Linear(input_dim, input_dim * 2, bias=False)
        self.fc2_list = nn.ModuleList()
        for i in range(1, 6):
            self.fc2_list.append(nn.Linear(input_dim, input_dim * i, bias=False))
        
        # self.rsoftmax = RadixSoftmax_T(2, 1)
        self.rsoftmax_list = nn.ModuleList()
        for i in range(1, 6):
            self.rsoftmax_list.append(RadixSoftmax_T(i, 1))


    def forward(self, window_list):

        # (N, C, H, W) N个cav的特征图
        window_list = window_list.permute(0, 2, 3, 1) # (N, H, W, C)
        N = window_list.shape[0]
        x_gap = torch.sum(window_list, dim=0)
        x_gap = x_gap.unsqueeze(0) # (1, H，W, C)
        x_gap = x_gap.mean((1, 2), keepdim=True) # [1, 1, 1, C]
        x_gap = self.act1(self.bn1(self.fc1(x_gap))) # 经过一个线性层，一个LN，一个ReLu
        x_attn = self.fc2_list[N-1](x_gap) # 通道升维，[1, 1, 1, C*N]
        x_attn = self.rsoftmax_list[N-1](x_attn).view(1, 1, -1) # [1, 1, N*C]
        out = window_list[0] * x_attn[..., 0:self.input_dim] # [1, 1, C]
        for i in range(1,N):
            out +=  window_list[i] * x_attn[..., i*self.input_dim:(i+1)*self.input_dim] # (H, W, C)
        return out.permute(2, 0, 1) # (C, H, W)


        # ego_feature, delay_fused_feature = window_list[0], window_list[1] # 两种特征图的融合 每个都是[B, H, W, 256]
        # B = ego_feature.shape[0]

        # # global average pooling, B,  H, W, C
        # x_gap = ego_feature + delay_fused_feature
        # # B, 1, 1, C
        # x_gap = x_gap.mean((1, 2), keepdim=True) # [B, 1, 1, 256]
        # x_gap = self.act1(self.bn1(self.fc1(x_gap))) # 经过一个线性层，一个LN，一个ReLu
        # # B, 1, 1, 3C
        # x_attn = self.fc2(x_gap) # 通道升维到2倍
        # # B 1 1 3C
        # x_attn = self.rsoftmax(x_attn).view(B, 1, 1, -1) # [B, 1, 1, 2*256]

        # out = ego_feature * x_attn[:, :, :, 0:self.input_dim] + \
        #       delay_fused_feature * x_attn[:, :, :, self.input_dim:] # 取两个不同尺度窗口的权重大的通道
        
        # return out # [B,  H， W， 256]


class FineTuneFlow(nn.Module):
    def __init__(self):
        super(FineTuneFlow, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 1)
        return x

class FineTuneFlow_2(nn.Module):
    def __init__(self):
        super(FineTuneFlow_2, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 1)
        return x


class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

def get_warped_feature_mask(flow, updated_grid):
    ''' 
    get the mask of the warped feature map, where the value is 1 means the location is valid, and 0 means the location is invalid.
    ----------
    Args:
        flow: (2, H, W)
        updated_grid: (2, H, W)

    Returns:
        mask: (H, W)
    '''
    def get_large_flow_indices(flow, threshold=0):
        '''
        find the indices of the flow map where the flow is larger than the threshold, which means these locations have been moved.
        ----------
        Args:
            flow: (2, H, W)
            
        '''
        max_values, max_indices = torch.max(torch.abs(flow[:2]), dim=0)
        large_indices = torch.nonzero(max_values > threshold, as_tuple=False)
        return large_indices

    def remove_duplicate_points(points):
        unique_points, inverse_indices = torch.unique(points, sorted=True, return_inverse=True, dim=0)
        return unique_points, inverse_indices
    
    def get_nonzero_idx(flow, idx):
        flow_values = flow[:, idx[:, 0], idx[:, 1]]
        nonzero_idx = torch.nonzero(torch.abs(flow_values).sum(dim=0) == 0, as_tuple=False).squeeze()
        return idx[nonzero_idx]
    
    flow_idx = get_large_flow_indices(flow)

    mask = torch.ones(flow.shape[-2], flow.shape[-1])
    if flow_idx.shape[0] == 0:
        return mask

    # print(flow_idx)
    updated_grid_points_tmp = updated_grid[:, flow_idx[:,0], flow_idx[:,1]].to(torch.int64).T
    # change the order of dim 1
    updated_grid_points = torch.zeros_like(updated_grid_points_tmp)
    updated_grid_points[:, 0] = updated_grid_points_tmp[:, 1]
    updated_grid_points[:, 1] = updated_grid_points_tmp[:, 0]
    # print(updated_grid_points)

    unique_points_idx, _ = remove_duplicate_points(updated_grid_points)
    # print(unique_points_idx)

    nonzero_idx = get_nonzero_idx(flow, unique_points_idx)
    # print(nonzero_idx)
    
    # mask out the outlier idx

    if len(nonzero_idx.shape) > 1:
        mask[nonzero_idx[:, 0], nonzero_idx[:, 1]] = 0
    else: 
        mask[nonzero_idx[0], nonzero_idx[1]] = 0

    return mask

class SpatialFusion(nn.Module):
    def __init__(self):
        super(SpatialFusion, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(2, 1, 3, stride=1, padding=1),
            # nn.BatchNorm3d(1, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )


    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x):
        # x: N, C, H, W

        x_max = torch.max(x, dim=0, keepdim=True)[0]
        x_mean = torch.mean(x, dim=0, keepdim=True)
        out = torch.cat((x_max, x_mean), dim=0).unsqueeze(0) # 1, 2, C, H, W

        out = self.conv3d(out).squeeze(0).squeeze(0) # 1, 1, C, H, W -> C, H, W
        return out

class HybridSpatialFusion(nn.Module):
    def __init__(self, hybrid_mode):
        # hybrid_mode: 0 max+media
        #              1 max+media+avg
        super(HybridSpatialFusion, self).__init__()
        self.hm = hybrid_mode
        print(f"===hybrid mode is {hybrid_mode}===")
        input_dim = 3 if hybrid_mode == 1 else 2
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_dim, 1, 3, stride=1, padding=1),
            nn.BatchNorm3d(1, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )

    def forward(self, x):
        # x: N, C, H, W

        x_max = torch.max(x, dim=0, keepdim=True)[0]
        x_median = torch.median(x, dim=0, keepdim=True)[0]

        if self.hm == 1:
            x_avg = torch.mean(x, dim=0, keepdim=True)
            out = torch.cat((x_max, x_median, x_avg), dim=0).unsqueeze(0) # 1, 3, C, H, W
        else:
            out = torch.cat((x_max, x_median), dim=0).unsqueeze(0) # 1, 2, C, H, W

        out = self.conv3d(out).squeeze(0).squeeze(0) # 1, 1, C, H, W -> C, H, W
        return out

class HybridReductionAttFusion(nn.Module):
    def __init__(self, feature_dims, dropout_rate = 0.1):
        super(HybridReductionAttFusion, self).__init__()
        self.atten = ScaledDotProductAttention(feature_dims)
        self.dropout = nn.Dropout(0.1)

        self.linear1 = nn.Linear(feature_dims, feature_dims * 2)
        self.linear2 = nn.Linear(feature_dims * 2, feature_dims)

        self.norm1 = nn.LayerNorm(feature_dims)
        self.norm2 = nn.LayerNorm(feature_dims)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dims, feature_dims * 4),
            nn.ReLU(),
            nn.Linear(feature_dims * 4, feature_dims)
        )
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (N_cav, C, H, W)
        
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (C, H, W)
        """
        x_max = torch.max(x, dim=0, keepdim=True)[0] # (1, C, H, W)
        x_avg = torch.mean(x, dim=0, keepdim=True)

        x_conbine = torch.cat((x_max, x_avg), dim=0) # (2, C, H, W)
        N, C, H, W = x_conbine.shape
        x_conbine = x_conbine.view(N, C, -1).permute(2, 0, 1) #  (H*W, 2, C), perform self attention on each pixel.
        att_map = self.atten(x_conbine,x_conbine,x_conbine) #  (H*W, 2, C)

        x_conbine = x_conbine + att_map
        x_conbine = x_conbine[:,0,:] #  (H*W, C)
        x_conbine = self.norm1(x_conbine)

        mlp_out = x_conbine + self.dropout(self.mlp(x_conbine)) #  (H*W, C)
        mlp_out = self.norm2(mlp_out)

        out = mlp_out.permute(1,0).view(C,H,W)
        # out = out.permute(1, 2, 0).view(N, C, H, W)[0] # (2, C, H, W)取第一个(C, H, W)

        return out

class Splitfusion(nn.Module):
    def __init__(self, in_dim):
        super(Splitfusion, self).__init__()
        # self.split_attention = SplitAttn_maxavg(in_dim)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim*2, in_dim, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_dim, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
        self.ch_select = SEBlock(in_dim)
    def forward(self, x, x_raw=None):
        # x: N, C, H, W

        x_max = torch.max(x, dim=0, keepdim=True)[0] # 1,C,H,W

        # print("x_max shape is ", x_max.shape)
        # print("x_raw shape is ", x_raw.shape)
        out = torch.cat((x_max, x_raw), dim=1) # 1, 2C, H, W
        out = self.conv1(out) # 1, C, H, W

        out = self.ch_select(out) # 2, C, H, W -> C, H, W
        out = out.squeeze(0)
        # x_max = torch.max(x, dim=0, keepdim=True)[0] # 1,C,H,W
        # x_mean = torch.mean(x, dim=0, keepdim=True)
        # out = torch.cat((x_max, x_mean), dim=0) # 2, C, H, W

        # out = self.split_attention(out) # 2, C, H, W -> C, H, W
        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EasyFusion(nn.Module):
    def __init__(self, in_channels):
        super(EasyFusion, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels // 2, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
    def forward(self, x):
        # x：N，C，H，W
        max_out, _ = torch.max(x, dim=0, keepdim=True)
        avg_out = torch.mean(x, dim=0, keepdim=True) # 1, C, H, W
        
        co_feature = torch.cat([max_out, avg_out], dim=1) # 1, 2C, H, W
        out = self.conv1(co_feature) # 1, C, H, W
        return out.squeeze(0)

# inspired by CBAM: Convolutional Block Attention Module   https://arxiv.org/abs/1807.06521
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # self.conv1 = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        # self.dropout = nn.Dropout2d(0.2)
        self.sigmoid = nn.Sigmoid()
        self.conv_att1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                                    nn.Mish(),
                                    nn.Dropout2d(0.2))
        self.conv_att2 = nn.Sequential(nn.Conv2d(64, 32 , kernel_size=1, bias=False),
                                    nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
                                    nn.Mish())
        self.conv_att3 = nn.Sequential(nn.Conv2d(32, 2 , kernel_size=1, bias=False),
                                    nn.BatchNorm2d(2, eps=1e-3, momentum=0.01))
    def forward(self, x):
        # x：N，C，H，W
        max_out, _ = torch.max(x, dim=0, keepdim=True)
        avg_out = torch.mean(x, dim=0, keepdim=True) # 1, C, H, W

        co_feature = torch.cat([max_out, avg_out], dim=1) # 1, 2C, H, W
        att = self.conv_att1(co_feature) # 1, 64, H, W
        att = self.conv_att2(att) # 1, 32, H, W
        att = self.conv_att3(att) # 1, 2, H, W
        att = self.sigmoid(att) # 1, 2, H, W
        # print("max att is", att[:,0,:,:])
        # print("mean att is", att[:,1,:,:])
        # xxx
        return max_out * att[:,0:1,:,:] + avg_out * att[:,1:2,:,:] # 1, C, H, W
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True) # N, 1, H, W
        attention = torch.cat([avg_out, max_out], dim=1)  # (batch_size, 2, H, W)
        attention = self.conv1(attention)  # (batch_size, 1, H, W)
        attention = self.dropout(attention)
        attention = self.sigmoid(attention)  # (batch_size, 1, H, W)
        return x * attention

class HybridFusion(nn.Module):
    def __init__(self, in_channel) -> None:
        super(HybridFusion, self).__init__()
        # 方案一：splitfusion后接spatial attention 效果不好，跌性能 (弃)
        # 方案二：spatial attention后接splitfuiosn
        # 方案三：只使用spatial attention
        # self.channel_fuiosn = Splitfusion(in_channel)
        self.channel_fuiosn = SEBlock(in_channel)

        self.spatial_fusion = SpatialAttention(in_channel * 2)
        # self.spatial_fusion = EasyFusion(in_channel * 2)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N，C，H, W 表示一个scenario下的feature
        x = self.spatial_fusion(x) # N, C, H, W --> 1, C, H, W

        out = self.channel_fuiosn(x) # 1, C, H, W
        # out = x
        return out.squeeze(0)

class VAEFusion_Simple(nn.Module):
    def __init__(self, in_channel, ms_h, ms_w) -> None:
        super(VAEFusion_Simple, self).__init__()
        
        self.decoder_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=1),
                                    nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))

        self.conv_mu = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=1),
                                    nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
        self.conv_logvar = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=1),
                                    nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
        # self.conv_logvar = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)

    def encode(self, x):
        # x: N, C, H, W

        mu = self.conv_mu(x) # N, C, H, W
        logvar = self.conv_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, features):
        # features : (N, C, H, W)
        mu, logvar = self.encode(features) # (N, C, H, W)
        re_feature = self.reparameterize(mu, logvar) # (N, C, H, W)

        fused_feature = torch.max(re_feature, dim=0, keepdim=True)[0] # (1, C, H, W)
        fused_feature = self.decoder_conv(fused_feature).squeeze(0) # (1, C, H, W)
        fused_feature = fused_feature + torch.max(features, dim=0)[0]
        return fused_feature

class VAEFusion_Simple_V2(VAEFusion_Simple):
    def __init__(self, in_channel, ms_h, ms_w) -> None:
        super().__init__(in_channel, ms_h, ms_w)
    
    def forward(self, features):
        # features : (N, C, H, W)
        mu, logvar = self.encode(features) # (N, C, H, W)
        re_feature = self.reparameterize(mu, logvar) # (N, C, H, W)

        fused_feature = torch.mean(re_feature, dim=0, keepdim=True) # (1, C, H, W)
        fused_feature = self.decoder_conv(fused_feature).squeeze(0) # (1, C, H, W)
        fused_feature = fused_feature + torch.max(features, dim=0)[0]
        return fused_feature

class FastFusion(nn.Module):
    def __init__(self, num_channels, num_inputs):
        super(FastFusion, self).__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(num_channels), requires_grad=True) for _ in range(num_inputs)])
        self.relu = nn.ReLU()

    def forward(self, *inputs):
        assert len(inputs) == len(self.weights), "Number of inputs must match number of weights"
        weights = [self.relu(w) for w in self.weights]
        weights_sum = sum(weights)
        weights = [w / (weights_sum + 1e-4) for w in weights]
        fused = sum(w.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x for w, x in zip(weights, inputs))
        return fused

class VAEFusion(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(VAEFusion, self).__init__()
        self.encoder_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction , kernel_size=1, bias=False),
                                    nn.BatchNorm2d(in_channels // reduction, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
        self.decoder_conv = nn.Sequential(nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
        
        # 重参数化时不能加BN以及ReLU，会影响其统计特性
        self.fc_mu = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1, bias=False) # 取消BN和ReLu 否则影响统计分布
        self.fc_logvar = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1, bias=False)
        
        # 要有这个 不然会多个feature相加会导致数值不稳从而梯度爆炸
        self.post_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels , kernel_size=1, bias=False),
                                    nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
        # self.fast_fusion = FastFusion(in_channels // reduction, 2)
    def encode(self, x):
        h = self.encoder_conv(x) # N, C, H, W
        mu = self.fc_mu(h) # N, C, H, W
        logvar = self.fc_logvar(h) # N, C, H, W
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        h = self.decoder_conv(z)
        return h
    
    def forward(self, features):
        # (N, C, H, W)
        N, C, H, W = features.shape
        ego_feature = features[[0], :, :, :]
        # cav_feature = features[1:, :, :, :]
        # final_features = torch.zeros_like(ego_feature)
        final_features = []
        for i in range(1, N):
            sum_feature = ego_feature + features[[i], :, :, :] # (1, C, H, W)
            mu, logvar = self.encode(sum_feature)
            z = self.reparameterize(mu, logvar) # 1, C//r, H, W
            att = self.decode(z) # (1, C, H, W)
            att = torch.sigmoid(att)
            sum_feature = att * sum_feature
            final_features.append(sum_feature.squeeze(0)) # 存的全部是 C H W
        final_features = torch.stack(final_features)
        final_features = final_features.sum(dim=0, keepdim=True)
        final_features = self.post_conv(final_features)
        return final_features.squeeze(0)
        # sum_feature = torch.sum(features, dim=0, keepdim=True) # (1, C, H, W)
        # mu, logvar = self.encode(sum_feature)
        # z = self.reparameterize(mu, logvar) # 1, C//r, H, W
        # att = self.decode(z) # (1, C, H, W)
        # att = torch.sigmoid(att)
        # sum_feature = att * sum_feature
        # return sum_feature.squeeze(0)

        # mu, logvar = self.encode(features)
        # z = self.reparameterize(mu, logvar) # N, C, H, W
        # z_fused_max = torch.max(z, dim=0, keepdim=True)[0] # (1, C, H, W)
        # z_fused_mean = torch.mean(z, dim=0, keepdim=True)

        # # z_fused = z_fused_max
        # z_fused = self.fast_fusion(z_fused_max, z_fused_mean)

        # fused_feature = self.decode(z_fused) # (1, C, H, W)
        # # fused_feature = fused_feature + torch.max(features, dim=0, keepdim=True)[0] # residual
        # return fused_feature.squeeze(0)

        # N = features.size(0)
        # ego_feature = features[0].unsqueeze(0)
        # cav_features = features[1:]
        
        # mu_ego, logvar_ego = self.encode(ego_feature)
        # mu_cav, logvar_cav = [], []
        
        # for j in range(N - 1):
        #     mu, logvar = self.encode(cav_features[j].unsqueeze(0))
        #     mu_cav.append(mu)
        #     logvar_cav.append(logvar)
        
        # mu_cav = torch.stack(mu_cav)
        # logvar_cav = torch.stack(logvar_cav)
        
        # z_ego = self.reparameterize(mu_ego, logvar_ego)
        # z_cav = [self.reparameterize(mu, logvar) for mu, logvar in zip(mu_cav, logvar_cav)]
        # z_cav = torch.stack(z_cav)
        
        # z_fused = (z_ego + torch.mean(z_cav, dim=0)) / 2
        # fused_feature = self.decode(z_fused)
        # return fused_feature



class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]

class MinFusion(nn.Module):
    def __init__(self):
        super(MinFusion, self).__init__()

    def forward(self, x):
        return torch.min(x, dim=0)[0]

class MedianFusion(nn.Module):
    def __init__(self):
        super(MedianFusion, self).__init__()

    def forward(self, x):
        return torch.median(x, dim=0)[0]

class AvgFusion(nn.Module):
    def __init__(self):
        super(AvgFusion, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=0)

class raindrop_fuse(nn.Module):
    def __init__(self, args, design_mode=0):
        super(raindrop_fuse, self).__init__()
        
        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication']) # TODO 
            if 'round' in args['communication']:
                self.round = args['communication']['round'] # 1
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]  下采样率 
        
        self.agg_mode = args['agg_operator']['mode'] # MAX
        self.multi_scale = args['multi_scale'] # True
        
        # self.raw_extract = Sparse_resnet_backbone_aspp()
        # self.split_attan = SplitAttn_T(128)
        # self.raw_extract = ResNetBEVBackboneOnlyDownsample()
        #################################################
        if self.multi_scale:
            layer_nums = args['layer_nums'] #  [3, 4, 5]
            num_filters = args['num_filters'] # [64, 128, 256]
            self.num_levels = len(layer_nums) # 3
            self.fuse_modules = nn.ModuleList()
            
            for idx in range(self.num_levels): # fuse_network 加入三个MAX融合
                if self.agg_mode == 'MAX':
                    print("===使用MAXFusion===")
                    fuse_network = MaxFusion()
                elif self.agg_mode == 'S-AdapFusion':
                    print("===使用S-AdaFusion===")
                    fuse_network = SpatialFusion()
                elif self.agg_mode == "splitfusion":
                    print("===使用SplitFusion===")
                    fuse_network = Splitfusion(num_filters[idx])
                elif self.agg_mode == "TAdaFusion":
                    print("===使用TAdaFusion===")
                    expansion = 2 ** idx
                    fuse_network = TAdaFusion(expansion * 128)
                elif self.agg_mode == "MAX_MedianFusion":
                    print("===使用MAX_MedianFusion===")
                    fuse_network = MaxFusion()
                    fuse_network_median = MedianFusion()
                elif self.agg_mode == "HybridReductionFusion":
                    print("===使用HybridReductionFusion===")
                    fuse_network = MaxFusion()
                    fuse_network_avg = MedianFusion()
                    fuse_network_median = MedianFusion()
                elif self.agg_mode == "HybridSpatialFusion":
                    print("===使用HybridSpatialFusion===")
                    fuse_network = HybridSpatialFusion(args['agg_operator']['hm'])
                elif self.agg_mode == "HybridAtten":
                    print("===使用HybridAtten===")
                    fuse_network = HybridReductionAttFusion(num_filters[idx])
                elif self.agg_mode == "HybridFusion":
                    print("===使用HybridFusion===")
                    fuse_network = HybridFusion(num_filters[idx])
                    # self.fuse_modules = HybridFusion(384)
                    # break
                elif self.agg_mode == "vaefusion_simple":
                    print("===使用Simple VAEFusion===")
                    fuse_network = VAEFusion_Simple(num_filters[idx], args['agg_operator']['ms_h'][idx], args['agg_operator']['ms_w'][idx])
                elif self.agg_mode == "vaefusion_simplev_2":
                    print("===使用Simple VAEFusion V2===")
                    fuse_network = VAEFusion_Simple_V2(num_filters[idx], args['agg_operator']['ms_h'][idx], args['agg_operator']['ms_w'][idx])
                elif self.agg_mode == "VAEFusion":
                    print("===使用 VAEFusion===")
                    fuse_network = VAEFusion(num_filters[idx])
                elif self.agg_mode == "ATTEN":
                    print("===使用 ATTENFusion===")
                    fuse_network = AttenFusion(num_filters[idx])
                elif self.agg_mode == "SCFusion":
                    print("===使用 SCFusion===")
                    self.fuse_modules = SpatialChannelFusion(384)
                    break
                    # fuse_network = SpatialChannelFusion(num_filters[idx])
                else:
                    # self.agg_mode == 'ATTEN':
                    fuse_network = AttenFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
                if self.agg_mode == "MAX_MedianFusion":
                    self.fuse_network_median.append(fuse_network_median)
                if self.agg_mode == "HybridReductionFusion":
                    self.fuse_network_avg.append(fuse_network_avg)               
                    self.fuse_network_median.append(fuse_network_median)               
        else: 
            if self.agg_mode == 'MAX': # max fusion, debug use
                self.fuse_modules = MaxFusion()

        self.design = design_mode
        if self.design == 1:
            self.fine_conv = FineTuneFlow()
        elif self.design == 2:
            self.stpn = STPN(args['channel_size'])
            self.motion_pred = MotionPrediction(seq_len=1)
            self.state_classify = StateEstimation(motion_category_num=1)
            self.fine_conv = FineTuneFlow()
            self.fine_conv_2 = FineTuneFlow_2()
        elif self.design == 3:
            self.stpn = STPN(args['channel_size'])
            self.motion_pred = MotionPrediction(seq_len=1)
            self.state_classify = StateEstimation(motion_category_num=1)
        elif self.design == 4: # SyncNet
            design4_h = 200; design4_w = 504; design4_k = 2; design4_TM_Flag = False
            self.syncnet = SyncLSTM(channel_size = 64, h = design4_h, w = design4_w, k = design4_k, TM_Flag = design4_TM_Flag)
            print(f"*** SyncNet init finished! channel_size = 64, h = {design4_h}, w = {design4_w}, k = {design4_k}, TM_Flag = {design4_TM_Flag} ***")

    def regroup(self, x, len, k=0):
        """
        split into different batch and time k

        Parameters
        ----------
        x : torch.Tensor
            input data, (B, ...)

        len : list 
            cav num in differnt batch, eg [3, 3]

        k : torch.Tensor
            num of past frames
        
        Returns:
        --------
        split_x : list
            different cav's lidar feature
            for example: k=4, then return [(3x4, C, H, W), (2x4, C, H, W), ...]
        """
        cum_sum_len = torch.cumsum(len*k, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def generateFlow(self, feats, pairwise_t_matrix, record_len, flow_gt=None):
        '''
        1. generate flow from feature sequence
        2. then update the feature
        Note:
            1. ego feature does not need to be updated

        params:
            feats: (sum(B,N,K), C, H, W)
            pairwise_t_matrix: (B, L, K, 2, 3)
            record_len: (B)

        return:
            flow: 
        '''
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K)

        # debug use
        batch_flow_gt = self.regroup(flow_gt, record_len, 2)

        updated_features_list = []
        flow_list = []
        state_class_pred_list = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            # t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
            node_features = batch_node_features[b]
            # neighbor_feature = warp_affine_simple(node_features, t_matrix, (H, W))
            node_features = node_features.view(-1, K, C, H, W) # (N, k, C, H, W)
            bevs = self.stpn(node_features)
            flow = self.motion_pred(bevs) # [N, 2, H, W]
            flow_list.append(flow)

            # TODO: debug use
            flow = batch_flow_gt[b].view(-1, 2, H, W)
            # print('=== gt flow used! ===')
            
            # Motion State Classification head
            state_class_pred = self.state_classify(bevs)
            state_class_pred_list.append(state_class_pred)

            # Given disp shift feature
            x_coord = torch.arange(W).float()   # [0, ..., W]
            y_coord = torch.arange(H).float()   # [0, ..., H]
            y, x = torch.meshgrid(y_coord, x_coord)  # [H, W], [H, W]
            grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(flow.shape[0], -1, -1, -1).to(flow.device)  # N, 2, H, W
            # updated_grid = grid + flow * (state_class_pred.sigmoid() > self.flow_thre)
            updated_grid = grid + flow # TODO: 这个位置为什么是减号
            
            # TODO: debug use, do not use gt flow, keep the original delay feature as the fusion input. 
            # updated_grid = grid
            
            # generate the mask for filtering out the pixels which moved to other locations but not filled by others
            mask_list = []
            for i in range(flow.shape[0]):
                mask_list.append(get_warped_feature_mask(flow[i], updated_grid[i]))
            mask = torch.stack(mask_list, dim=0)
            mask = mask.unsqueeze(1).repeat(1, C, 1, 1).to(flow.device)
            ##################################
            
            updated_grid[:, 0, :, :] = updated_grid[:, 0, :, :] / (W / 2.0) - 1.0
            updated_grid[:, 1, :, :] = updated_grid[:, 1, :, :] / (H / 2.0) - 1.0
            latest_node_features = node_features[:, 0, :, :, :] # (N, C, H, W)
            updated_features = F.grid_sample(latest_node_features, grid=updated_grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)
            # mask the features
            updated_features = mask * updated_features
            # ego feature use the latest feature (no delay)
            updated_features[0, :, :, :] = latest_node_features[0, :, :, :]

            updated_features_list.append(updated_features)
        
        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)

        # TODO: for debug use
        # debug_path = '/remote-home/share/sizhewei/logs/where2comm_flow_debug/viz_flow/'
        # torch.save(feats, debug_path + 'feats.pt')
        # torch.save(updated_features_all, debug_path + 'updated_features_all.pt')

        flow_all = torch.cat(flow_list, dim=0)  # (sum(B,N), 2, H, W)
        state_class_pred_all = torch.cat(state_class_pred_list, dim=0)  # (sum(B,N), 2, H, W)
        
        return updated_features_all, flow_all, state_class_pred_all

    def update_features_boxflow(self, feats, pairwise_t_matrix, record_len, flow_map, reserved_mask, flow_gt):
        """
        Update features with box flow.
        
        Parameters
        ----------
        feats : torch.Tensor
            input data, 
            shape: (sum(n_cav), C, H, W)
        
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        record_len : list
            shape: (B)
            
        flow_map: torch.Tensor
            flow generate by past 2 frames detection boxes
            shape: (sum(N_b), H, W, 2)

        reserved_mask: torch.Tensor
            shape: (sum(N_b), C, H, W)

        gt_flow: torch.Tensor
            shape: (sum(N_b), H, W, 2)
            
        Returns
        -------
        Warped feature.
        """
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K) # 返回一个List [（N1*k, C, H, W）, （N2*k, C, H, W）...]
        batch_flow_map = self.regroup(flow_map, record_len, k=1) # 预测的流图 返回List[(N1, H, W, 2)...]
        batch_reserved_mask = self.regroup(reserved_mask, record_len, k=1) # 返回List[(N1, C, H, W)...]

        updated_features_list = []
        flow_list = []
        state_class_pred_list = []
        gt_flow_map_list = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            node_features = batch_node_features[b]
            node_features = node_features.view(-1, K, C, H, W) # (N, k, C, H, W)
            flow = batch_flow_map[b] # .view(-1, 2, H, W) # [N, H, W, 2]
            # flow = self.fine_conv(flow) # [N, H, W, 2] 
            # flow_list.append(flow)
            
            latest_node_features = node_features[:, 0, :, :, :] # past_0 features, (N, C, H, W) 

            updated_features = F.grid_sample(latest_node_features, grid=flow, mode='bilinear', align_corners=False) # (N, C, H, W) 
            # updated_features = latest_node_features

            updated_features = updated_features*batch_reserved_mask[b] # (N, C, H, W)
            # updated_features[0, ...] = node_features[0, 0, :, :, :] # (N, C, H, W)
            updated_features_list.append(updated_features)

        # flow_pred = torch.cat(flow_list, dim=0) # (sum(N_b), H, W, 2)
        # gt_flow_norm = torch.cat(gt_flow_map_list, dim=0) # (sum(N_b), H, W, 2)
        # # compute the flow map loss:
        # loss = F.smooth_l1_loss(flow_pred, gt_flow_norm)

        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)

        return updated_features_all
    
    def update_features_boxflow_w_raw(self, feats, pairwise_t_matrix, record_len, flow_map, reserved_mask, flow_gt):
        """
        Update features with box flow.
        
        Parameters
        ----------
        feats : torch.Tensor
            input data, 
            shape: (sum(n_cav), C, H, W)
        
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        record_len : list
            shape: (B)
            
        flow_map: torch.Tensor
            flow generate by past 2 frames detection boxes
            shape: (sum(N_b), H, W, 2)

        reserved_mask: torch.Tensor
            shape: (sum(N_b), C, H, W)

        gt_flow: torch.Tensor
            shape: (sum(N_b), H, W, 2)
            
        Returns
        -------
        Warped feature.
        """
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K) # 返回一个List [（N1*k, C, H, W）, （N2*k, C, H, W）...]
        batch_flow_map = self.regroup(flow_map, record_len, k=1) # 预测的流图 返回List[(N1, H, W, 2)...]
        batch_reserved_mask = self.regroup(reserved_mask, record_len, k=1) # 返回List[(N1, C, H, W)...]
        # torch.autograd.set_detect_anomaly(True)
        updated_features_list = []
        raw_feature_list = []
        flow_list = []
        state_class_pred_list = []
        gt_flow_map_list = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            node_features = batch_node_features[b]
            node_features = node_features.view(-1, K, C, H, W) # (N, k, C, H, W)
            flow = batch_flow_map[b] # .view(-1, 2, H, W) # [N, H, W, 2]
            # flow = self.fine_conv(flow) # [N, H, W, 2] 
            # flow_list.append(flow)
            
            latest_node_features = node_features[:, 0, :, :, :] # past_0 features, (N, C, H, W) 

            ego_feature_wo_mask = latest_node_features[0:1, :, :, :] # (1, C, H, W) 这个是ego的特征图，没有被mask

            updated_features = F.grid_sample(latest_node_features, grid=flow, mode='bilinear', align_corners=False) # (N, C, H, W) 
            # updated_features = latest_node_features

            updated_features = updated_features*batch_reserved_mask[b] # (N, C, H, W)
            
            t_matrix = pairwise_t_matrix[b][:N, 0, :, :] # (N, 2, 3)
            neighbor_feature = warp_affine_simple(updated_features.clone(), # (N, C, H, W)
                                                    t_matrix, # (N, 2, 3)
                                                    (H, W)) # project到ego 原本的feature都是在cav view
            raw_feature_sum = neighbor_feature.sum(dim=0, keepdim=True) # (1, C, H, W) 稀疏ROI区域直接相加
            raw_feature_list.append(raw_feature_sum)

            updated_features[0:1] = ego_feature_wo_mask
            # updated_features[0, ...] = node_features[0, 0, :, :, :] # (N, C, H, W)
            updated_features_list.append(updated_features)

        # flow_pred = torch.cat(flow_list, dim=0) # (sum(N_b), H, W, 2)
        # gt_flow_norm = torch.cat(gt_flow_map_list, dim=0) # (sum(N_b), H, W, 2)
        # # compute the flow map loss:
        # loss = F.smooth_l1_loss(flow_pred, gt_flow_norm)

        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)
        raw_feature_sum = torch.cat(raw_feature_list, dim=0)  # (B, C, H, W)

        return updated_features_all, raw_feature_sum
    
    def update_features_boxflow_design_1(self, feats, pairwise_t_matrix, record_len, flow_map, reserved_mask, flow_gt):
        """
        Update features with box flow.
        
        Parameters
        ----------
        feats : torch.Tensor
            input data, 
            shape: (sum(n_cav), C, H, W)
        
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        record_len : list
            shape: (B)
            
        flow_map: torch.Tensor
            flow generate by past 2 frames detection boxes
            shape: (sum(N_b), H, W, 2)

        reserved_mask: torch.Tensor
            shape: (sum(N_b), C, H, W)

        gt_flow: torch.Tensor
            shape: (sum(N_b), H, W, 2)
            
        Returns
        -------
        Warped feature.
        """
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K)

        batch_flow_map = self.regroup(flow_map, record_len, k=1)
        batch_reserved_mask = self.regroup(reserved_mask, record_len, k=1)

        # debug use
        batch_flow_gt = self.regroup(flow_gt, record_len, k=2)

        updated_features_list = []
        flow_list = []
        state_class_pred_list = []
        gt_flow_map_list = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            node_features = batch_node_features[b]
            node_features = node_features.view(-1, K, C, H, W) # (N, k, C, H, W)
            flow = batch_flow_map[b] # .view(-1, 2, H, W) # [N, H, W, 2]
            # flow = self.fine_conv(flow) # [N, H, W, 2] # TODO: conv
            flow_list.append(flow)
            # # motion net flow
            # bevs = self.stpn(node_features)
            # flow_pred = self.motion_pred(bevs) # [N, 2, H, W]
            # # Motion State Classification head
            # state_class_pred = self.state_classify(bevs)
            # state_class_pred_list.append(state_class_pred)
            
            latest_node_features = node_features[:, 0, :, :, :] # past_0 features, (N, C, H, W) 
            updated_features = F.grid_sample(latest_node_features, grid=flow, mode='nearest', align_corners=False)

            updated_features = updated_features*batch_reserved_mask[b] # (N, C, H, W)
            updated_features_list.append(updated_features)

            # normalizing GT flow: gt_flow_map [N, H, W, 2]
            # Given disp shift feature
            x_coord = torch.arange(W).float()   # [0, ..., W]
            y_coord = torch.arange(H).float()   # [0, ..., H]
            y, x = torch.meshgrid(y_coord, x_coord)  # [H, W], [H, W]
            grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(N, -1, -1, -1).to(flow.device)
            gt_flow_delta = batch_flow_gt[b].view(-1, 2, H, W)
            gt_flow_map = grid - gt_flow_delta
            gt_flow_map[:, 0, :, :] = gt_flow_map[:, 0, :, :] / (W / 2.0) - 1.0
            gt_flow_map[:, 1, :, :] = gt_flow_map[:, 1, :, :] / (H / 2.0) - 1.0
            gt_flow_map = gt_flow_map.permute(0, 2, 3, 1) 
            gt_flow_map_list.append(gt_flow_map)

        flow_pred = torch.cat(flow_list, dim=0) # (sum(N_b), H, W, 2)
        gt_flow_norm = torch.cat(gt_flow_map_list, dim=0) # (sum(N_b), H, W, 2)
        # compute the flow map loss:
        loss = F.smooth_l1_loss(flow_pred, gt_flow_norm)

        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)

        return updated_features_all, loss
    
    def update_features_boxflow_design_2(self, feats, pairwise_t_matrix, record_len, flow_map, reserved_mask, flow_gt):
        """
        Update features with box flow.
        
        Parameters
        ----------
        feats : torch.Tensor
            input data, 
            shape: (sum(n_cav), C, H, W)
        
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        record_len : list
            shape: (B)
            
        flow_map: torch.Tensor
            flow generate by past 2 frames detection boxes
            shape: (sum(N_b), H, W, 2)

        reserved_mask: torch.Tensor
            shape: (sum(N_b), C, H, W)

        gt_flow: torch.Tensor
            shape: (sum(N_b), H, W, 2)
            
        Returns
        -------
        Warped feature.
        """
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K)

        batch_flow_map = self.regroup(flow_map, record_len, k=1) # 过去两帧的预测结果显示的flow
        batch_reserved_mask = self.regroup(reserved_mask, record_len, k=1)

        # debug use
        batch_flow_gt = self.regroup(flow_gt, record_len, k=2)

        updated_features_list = []
        flow_list = []
        state_class_pred_list = []
        gt_flow_map_list = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            node_features = batch_node_features[b]
            node_features = node_features.view(-1, K, C, H, W) # (N, k, C, H, W) 一个scene下的所有车的k帧特征图
            flow_box = batch_flow_map[b] # .view(-1, 2, H, W) # [N, H, W, 2]
            # flow_box = self.fine_conv(flow_box) # [N, H, W, 2]
            
            # motion net flow
            bevs = self.stpn(node_features) # 这里发现一些问题，paper中提及的运动预测似乎不是直接用特征图去生成的flow的
            flow_delta_pred = self.motion_pred(bevs) # [N, 2, H, W]
            # Motion State Classification head
            state_class_pred = self.state_classify(bevs)
            state_class_pred_list.append(state_class_pred)
            # Given disp shift feature
            x_coord = torch.arange(W).float()   # [0, ..., W]
            y_coord = torch.arange(H).float()   # [0, ..., H]
            y, x = torch.meshgrid(y_coord, x_coord)  # [H, W], [H, W]
            grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(N, -1, -1, -1).to(flow_box.device)
            # flow_pred = grid - flow_delta_pred
            # # normalizing flow: flow_pred [N, H, W, 2]
            # flow_pred[:, 0, :, :] = flow_pred[:, 0, :, :] / (W / 2.0) - 1.0
            # flow_pred[:, 1, :, :] = flow_pred[:, 1, :, :] / (H / 2.0) - 1.0
            # flow_pred = flow_pred.permute(0, 2, 3, 1) # [N, H, W, 2]

            # normalizing flow_delta_pred 
            flow_delta_pred[:, 0, :, :] = flow_delta_pred[:, 0, :, :] / (W / 2.0) - 1.0
            flow_delta_pred[:, 1, :, :] = flow_delta_pred[:, 1, :, :] / (H / 2.0) - 1.0
            flow_delta_pred = flow_delta_pred.permute(0, 2, 3, 1) # [N, H, W, 2]
            flow = self.fine_conv(flow_delta_pred + flow_box) # [N, H, W, 2]

            # flow = self.fine_conv_2(flow_box, flow_pred) # [N, H, W, 2]
            flow_list.append(flow)
            
            latest_node_features = node_features[:, 0, :, :, :] # past_0 features, (N, C, H, W) 
            updated_features = F.grid_sample(latest_node_features, grid=flow, mode='nearest', align_corners=False)

            updated_features = updated_features*batch_reserved_mask[b] # (N, C, H, W)
            updated_features_list.append(updated_features)

            # normalizing GT flow: gt_flow_map [N, H, W, 2]
            gt_flow_delta = batch_flow_gt[b].view(-1, 2, H, W)
            gt_flow_map = grid - gt_flow_delta
            gt_flow_map[:, 0, :, :] = gt_flow_map[:, 0, :, :] / (W / 2.0) - 1.0
            gt_flow_map[:, 1, :, :] = gt_flow_map[:, 1, :, :] / (H / 2.0) - 1.0
            gt_flow_map = gt_flow_map.permute(0, 2, 3, 1) 
            gt_flow_map_list.append(gt_flow_map)

        flow_all = torch.cat(flow_list, dim=0) # (sum(N_b), H, W, 2)
        state_class_pred_all = torch.cat(state_class_pred_list, dim=0)  # (sum(B,N), 2, H, W)
        gt_flow_norm = torch.cat(gt_flow_map_list, dim=0) # (sum(N_b), H, W, 2)
        # compute the flow map loss:
        loss = F.smooth_l1_loss(flow_all, gt_flow_norm)

        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)

        return updated_features_all, loss, flow_all, state_class_pred_all

    # syncnet
    def generate_estimated_feats(self, feats, pairwise_t_matrix, record_len):
        '''
        update the feature by SyncNet
        Note:
            1. ego feature does not need to be updated

        params:
            feats: (sum(B,N,K), C, H, W)
            pairwise_t_matrix: (B, L, K, 2, 3)
            record_len: (B)

        return:
            updated feature: [N, C, H, W]
        '''
        _, C, H, W = feats.shape
        B, L, K = pairwise_t_matrix.shape[:3]
        batch_node_features = self.regroup(feats, record_len, k=K)

        updated_features_list = []
        flow_list = []
        # all_recon_loss = 0
        for b in range(B):
            all_feats = batch_node_features[b]
            non_ego_feats = all_feats[K:].clone()
            non_ego_feats = non_ego_feats.reshape(-1, K, C, H, W)
            non_ego_feats = torch.flip(non_ego_feats, [1])
            estimated_non_ego_feats = self.syncnet(non_ego_feats, [1]) # [N-1, 1, C, H, W]
            # estimated_non_ego_feats = non_ego_feats[:, 1:2] # TODO:
            updated_features_list.append(torch.cat([all_feats[0].unsqueeze(0), estimated_non_ego_feats[:,0]], dim=0))

        updated_features_all = torch.cat(updated_features_list, dim=0)  # (sum(B,N), C, H, W)
        
        return updated_features_all

    def forward_backup_1(self, x, rm, record_len, pairwise_t_matrix, time_diffs, backbone=None, heads=None, flow_gt=None, box_flow=None, reserved_mask=None, viz_bbx_flag=False, noise_pairwise_t_matrix=None, num_roi_thres=-1):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
        
        rm : torch.Tensor
            confidence map, (sum(n_cav), 2, H, W)

        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        time_diffs : torch.Tensor
            time interval, (sum(n_cav), )
        
        flow_gt: ground truth flow, generate by object center and id

        box_flow: flow generate by past 2 frames detection boxes
            
        Returns
        -------
        Fused feature
        flow_map loss
        """
        _, C, H, W = x.shape
        B, L, K = pairwise_t_matrix.shape[:3]

        # (B,L,k,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]]  # 从（B，L， K， 4， 4）中取出2D变换部分
        # [B, L, L, 2, 3], 只要x,y这两个维度就可以(z只有一层)，所以提取[0,1,3]作为仿射变换矩阵, 大小(2, 3)
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        if noise_pairwise_t_matrix is not None:
            noise_pairwise_t_matrix = noise_pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] 
            noise_pairwise_t_matrix[...,0,1] = noise_pairwise_t_matrix[...,0,1] * H / W
            noise_pairwise_t_matrix[...,1,0] = noise_pairwise_t_matrix[...,1,0] * W / H
            noise_pairwise_t_matrix[...,0,2] = noise_pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
            noise_pairwise_t_matrix[...,1,2] = noise_pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        # single frame TODO: 
        # batch_confidence_maps = self.regroup(psm_single, record_len, 1) # [[2*3, 2, 100/2, 252/2], [3*3, 2, 100/2, 252/2], ...]
        self.num_roi_thres = num_roi_thres # roi的数量限制
        if self.num_roi_thres > 0:
            self.thre = 0.01
            _, _, H, W = rm.shape
            num_valid_thres = self.num_roi_thres * 10
            communication_maps = rm.sigmoid().max(dim=1)[0].unsqueeze(1) # 构建置信度图 (sum(n_cav), 2, H, W)变为(sum(n_cav), 1, H, W)存最大值
            communication_masks_list = []
            curr_batch = communication_maps.shape[0] # num_cav * k
            communication_mask = (communication_maps>self.thre).to(torch.int) # 置信度图大于0.01的标记1 形状 (sum(n_cav), 1, H, W)
            valid_nums = torch.sum(communication_mask, dim=(-1, -2, -3)) # (sum(n_cav), ) 每个cav每个帧中的有效像素数量

            final_comm_mask = torch.zeros_like(communication_mask) # (sum(n_cav), 1, H, W)
            for i in range(curr_batch):# 遍历具体某个车的某个帧
                if valid_nums[i] > num_valid_thres : # 有效像素数量超出限制
                    # 选择 comm_maps[i] 里面 top num_valid 的位置
                    tmp = communication_maps[i].reshape((1, -1)) # (1, H*W*sum(n_cav))
                    _, idx = torch.topk(tmp, num_valid_thres) # 选出最高置信度的有效像素个数
                    # print(idx)
                    curr_mask = torch.zeros_like(tmp)
                    curr_mask[0, idx[0]] = 1
                    final_comm_mask[i] = curr_mask.reshape((1, H, W)) # 形成最终的有效区域掩码
                else:
                    final_comm_mask[i] = communication_mask[i] # 保证前k个是全1

            #     communication_mask = final_comm_mask
            #     communication_masks_list.append(communication_mask)
            # # _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            # communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
            # print(final_comm_mask.shape)
            # final_comm_mask scale x 2 [B, 1, H, W] -> [B, 1, 2H, 2W]

            # 将张量插值到目标大小 (B, 1, 2H, 2W)
            interpolated_tensor = F.interpolate(final_comm_mask.to(torch.float32).to(x.device), scale_factor=2, mode='nearest')
            interpolated_tensor[:K, ...] = 1 # 保证前k个是全1，TODO 这是为什么？猜测是为了保证ego的所有特征要能保留，所以ego不能掩码，而其他agent要参与传输，为了节省带宽，因此需要掩码

            x = x*interpolated_tensor
        
        # 2. feature compensation with flow
        # 2.1 generate flow, 在同一个坐标系内，计算每个cav的flow
        # 2.2 compensation
        # x: (BxNxK, C, H, W) -> (BxN, C, H, W)
        if self.design == 0 or self.design == 5:
            updated_features = self.update_features_boxflow(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt) # (BxN, C, H, W) warp后的结果
        elif self.design == 1:
            updated_features, flow_recon_loss = self.update_features_boxflow_design_1(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 2:
            updated_features, flow_recon_loss, flow, state_preds = self.update_features_boxflow_design_2(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 3:
            updated_features, flow, state_preds = self.generateFlow(x, pairwise_t_matrix, record_len, flow_gt) # (BxN, C, H, W)
        elif self.design == 4: # SyncNet
            updated_features = self.generate_estimated_feats(x, pairwise_t_matrix, record_len) # (BxN, C, H, W)
        
        # 3. feature fusion
        if self.multi_scale:
            ups = []
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(updated_features)  # tuple((B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4))  得到三个结果，也即多尺寸结果，包含natch中每个车的特征图
            
            for i in range(self.num_levels): # 3
                x = feats[i] if with_resnet else backbone.blocks[i](x)  # (BxN, C', H, W)

                ############ 1. Communication (Mask the features) #########
                if i==0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100/2, 252/2], [3*3, 2, 100/2, 252/2], ...] 返回List [(N1*k, 2, H, W), (N2*k, 2, H, W)...] 置信度图，划分到每一辆车
                        batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...] 返回List [(N1*k), (N2*k)...] 距离cur的时间间隔 划分到每一辆车
                        _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
                        # x = x * communication_masks_tensor
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                else:
                    if self.communication:
                        communication_masks_tensor = F.max_pool2d(communication_masks_tensor, kernel_size=2)
                        # TODO: scale = 1, 2 不加 mask
                        # x = x * communication_masks_tensor
                
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C*2^i, H/2^i, W/2^i), (L2, C*2^i, H/2^i, W/2^i), ...]
                # for example, i=1, past_k=3, b_1 has 2 cav, b_2 has 3 cav ... :
                # [[2*3, 256*2, 100/2, 252/2], [3*3, 256*2, 100/2, 252/2], ...]
                batch_node_features = self.regroup(x, record_len, 1) # List [(N1, C, H, W)， （N2， C， H， W）] 长度是B

                ############ 3. Fusion ####################################
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b] # 一个场景下的cav数量
                    # t_matrix[i, j]-> from i to j
                    if noise_pairwise_t_matrix is not None:
                        t_matrix = noise_pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3)
                    else:
                        t_matrix = pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3) 本来形状是（B,L,K,2,3） 取出past0 cav 到cur ego的转换矩阵
                    node_features = batch_node_features[b] # (N, C, H, W) 这里的N已经是一个场景下的车数了
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, # (N, C, H, W)
                                                    t_matrix, # (N, 2, 3)
                                                    (H, W)) # project到ego 原本的feature都是在cav view
                    if len(backbone.deblocks) > 0:
                        neighbor_feature = backbone.deblocks[i](neighbor_feature) # 输出（N， C， H， W）
                    
                    x_fuse.append(self.split_attan(neighbor_feature))  

                #     record_frames = np.ones((N))*K
                #     # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                    
                #     if self.agg_mode == 'RAIN':
                #         # for sensor embedding
                #         sensor_dist = -1# (B, H, W)
                #         x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                #         # # TODO for scale debug
                #         # if i==self.num_levels-1:
                #         #     x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                #         # else:
                #         #     x_fuse.append(neighbor_feature[0])
                #     else: # ATTEN, MAX, Transformer
                #         x_fuse.append(self.fuse_modules[i](neighbor_feature)) # 输入形状为（N， C，H，W）MAX Fusion 取第0维最大值，返回（C，H， W）
                #         # # TODO for scale debug
                #         # if i==self.num_levels-1:
                #         #     x_fuse.append(self.fuse_modules[i](neighbor_feature))
                #         # else:
                #         #     x_fuse.append(neighbor_feature[0])

                x_fuse = torch.stack(x_fuse) # （B， C， H， W） B在这里就是batch数了，
                ups.append(x_fuse)
                # ############ 4. Deconv #################################### 上采样
                # if len(backbone.deblocks) > 0:
                #     ups.append(backbone.deblocks[i](x_fuse))
                # else:
                #     ups.append(x_fuse)
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1) #（B， 128*3， H/2, W/2）
            elif len(ups) == 1: 
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
                
        else:
            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len, K)
            batch_confidence_maps = self.regroup(rm, record_len, K)
            batch_time_intervals = self.regroup(time_diffs, record_len, K)

            ############ 2. Communication (Mask the features) #########
            if self.communication:
                # _, (B, 1, H, W), float
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            else:
                communication_rates = torch.tensor(0).to(x.device)
            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix,
                                                (H, W))
                record_frames = np.ones((N))*K
                # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                if self.agg_mode == "RAIN":
                    # for sensor embedding
                    sensor_dist = -1# (B, H, W)
                    x_fuse.append(self.fuse_modules(neighbor_feature, sensor_dist, batch_time_intervals[b], communication_masks[b]))
                else: # ATTEN, MAX, Transformer
                    x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)

            # self.fuse_modsules(x_fuse, record_len)
        if self.design == 0 or self.design == 5:
            if viz_bbx_flag:
                return  x_fuse, communication_rates, {}, updated_features
            return x_fuse, communication_rates, {}
        elif self.design == 1:
            return x_fuse, communication_rates, {}, flow_recon_loss
        elif self.design == 2:
            return x_fuse, communication_rates, {}, flow_recon_loss #, flow, state_preds
        elif self.design == 3:
            return x_fuse, communication_rates, {}, flow, state_preds
        elif self.design == 4:
            return x_fuse, communication_rates, {}
    
    def forward(self, x, rm, record_len, pairwise_t_matrix, time_diffs, backbone=None, heads=None, flow_gt=None, box_flow=None, reserved_mask=None, viz_bbx_flag=False, noise_pairwise_t_matrix=None, num_roi_thres=-1):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)  V2XSet: (sum(n_cav), 64, 200, 704)
        
        rm : torch.Tensor
            confidence map, (sum(n_cav), 2, H, W)  V2XSet: (sum(n_cav), 64, 100, 352)

        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        time_diffs : torch.Tensor
            time interval, (sum(n_cav), )
        
        flow_gt: ground truth flow, generate by object center and id

        box_flow: flow generate by past 2 frames detection boxes
            
        Returns
        -------
        Fused feature
        flow_map loss
        """
        _, C, H, W = x.shape # (sum(n_cav), 64, 200, 704)
        B, L, K = pairwise_t_matrix.shape[:3]

        # print("x shape is ", x.shape)
        # print("pairwise_t_matrix shape is ", pairwise_t_matrix.shape)
        # print("rm shape is ", rm.shape)
        # print("record_len shape is ", record_len.shape)
        # print("time_diffs shape is ", time_diffs.shape)
        # print("time_diffs  is ", time_diffs)
        # exit1
        # (B,L,k,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]]  # 从（B，L， K， 4， 4）中取出2D变换部分（B，L， K， 2， 3）
        # [B, L, L, 2, 3], 只要x,y这两个维度就可以(z只有一层)，所以提取[0,1,3]作为仿射变换矩阵, 大小(2, 3)
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W # 因为这个权重要和y坐标相乘，它表示是旋转的比例，但是他是基于x坐标的，也就是W（704），但是现在要作用在H（200）上，所以要转变一下
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2 # 乘以2是将（-1/2， 1/2）调整到（-1， 1）
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        if noise_pairwise_t_matrix is not None:
            noise_pairwise_t_matrix = noise_pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] 
            noise_pairwise_t_matrix[...,0,1] = noise_pairwise_t_matrix[...,0,1] * H / W
            noise_pairwise_t_matrix[...,1,0] = noise_pairwise_t_matrix[...,1,0] * W / H
            noise_pairwise_t_matrix[...,0,2] = noise_pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
            noise_pairwise_t_matrix[...,1,2] = noise_pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        # single frame TODO: 
        # batch_confidence_maps = self.regroup(psm_single, record_len, 1) # [[2*3, 2, 100/2, 252/2], [3*3, 2, 100/2, 252/2], ...]
        self.num_roi_thres = num_roi_thres # roi的数量限制 
        if self.num_roi_thres > 0:
            self.thre = 0.01
            _, _, H, W = rm.shape
            num_valid_thres = self.num_roi_thres * 10 # 这里是规定了所有roi总像素个数限制
            communication_maps = rm.sigmoid().max(dim=1)[0].unsqueeze(1) # 构建置信度图 (sum(n_cav), 2, H, W)变为(sum(n_cav), 1, H, W)存最大值
            communication_masks_list = []
            curr_batch = communication_maps.shape[0] # num_cav * k
            communication_mask = (communication_maps>self.thre).to(torch.int) # 置信度图大于0.01的标记1 形状 (sum(n_cav), 1, H, W)
            valid_nums = torch.sum(communication_mask, dim=(-1, -2, -3)) # (sum(n_cav), ) 每个cav每个帧中的有效像素数量

            final_comm_mask = torch.zeros_like(communication_mask) # (sum(n_cav), 1, H, W)
            for i in range(curr_batch):# 遍历具体某个车的某个帧
                if valid_nums[i] > num_valid_thres : # 有效像素数量超出限制
                    # 选择 comm_maps[i] 里面 top num_valid 的位置
                    tmp = communication_maps[i].reshape((1, -1)) # (1, H*W*sum(n_cav))
                    _, idx = torch.topk(tmp, num_valid_thres) # 选出最高置信度的有效像素个数
                    # print(idx)
                    curr_mask = torch.zeros_like(tmp)
                    curr_mask[0, idx[0]] = 1
                    final_comm_mask[i] = curr_mask.reshape((1, H, W)) # 形成最终的有效区域掩码
                else:
                    final_comm_mask[i] = communication_mask[i] # 保证前k个是全1

            #     communication_mask = final_comm_mask
            #     communication_masks_list.append(communication_mask)
            # # _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            # communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
            # print(final_comm_mask.shape)
            # final_comm_mask scale x 2 [B, 1, H, W] -> [B, 1, 2H, 2W]

            # 将张量插值到目标大小 (B, 1, 2H, 2W)
            interpolated_tensor = F.interpolate(final_comm_mask.to(torch.float32).to(x.device), scale_factor=2, mode='nearest') # 这里的 'nearest' 插值模式意味着在放大过程中，将直接使用最近的像素值来填充新的像素位置，这通常用于分类标签或其他不需要平滑过渡的场景，因为它不会引入新的像素值，只是简单地复制现有的像素值。
            interpolated_tensor[:K, ...] = 1 # 保证前k个是全1，TODO 这是为什么？猜测是为了保证ego的所有特征要能保留，所以ego不能掩码，而其他agent要参与传输，为了节省带宽，因此需要掩码

            x = x*interpolated_tensor
        
        # 2. feature compensation with flow
        # 2.1 generate flow, 在同一个坐标系内，计算每个cav的flow
        # 2.2 compensation
        # x: (BxNxK, C, H, W) -> (BxN, C, H, W)
        if self.design == 0 or self.design == 5:
            updated_features = self.update_features_boxflow(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt) # (BxN, C, H, W)
        elif self.design == 1:
            updated_features, flow_recon_loss = self.update_features_boxflow_design_1(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 2:
            updated_features, flow_recon_loss, flow, state_preds = self.update_features_boxflow_design_2(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 3:
            updated_features, flow, state_preds = self.generateFlow(x, pairwise_t_matrix, record_len, flow_gt) # (BxN, C, H, W)
        elif self.design == 4: # SyncNet
            updated_features = self.generate_estimated_feats(x, pairwise_t_matrix, record_len) # (BxN, C, H, W)
        
        # debug 0 延迟时的结果 默认注释
        # updated_features = x

        # 3. feature fusion
        if self.multi_scale:
            ups = []

            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(updated_features)  # tuple((B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4))  得到三个结果，也即多尺寸结果，包含natch中每个车的特征图 其实正确的形状为tuple((B, C, H/2, W/2), (B, 2C, H/4, W/4), (B, 4C, H/8, W/8))

            for i in range(self.num_levels): # 3
                x = feats[i] if with_resnet else backbone.blocks[i](x)  # (BxN, C', H, W) 这里的BxN应该就是所有车数了，因为所有cav都已经经过延迟补偿

                ############ 1. Communication (Mask the features) #########
                if i==0:
                    if self.communication: # 这进去后会设置置信度图和掩码
                        batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100, 352], [3*3, 2, 100, 352], ...] 返回List [(N1*k, 2, H, W), (N2*k, 2, H, W)...] 置信度图，划分到每一辆车
                        batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...] 返回List [(N1*k), (N2*k)...] 距离cur的时间间隔 划分到每一辆车
                        _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        communication_masks_tensor = torch.concat(communication_masks_list, dim=0) # (sum(n_cav), 1, H, W)

                        # delay_0ms_flag = False # 使用where2comm的mask机制
                        # if delay_0ms_flag:
                        #     H, W =  communication_masks_tensor.shape[2:]
                        #     communication_masks_tensor = communication_masks_tensor.reshape(-1, K, 1, H, W) # (BxN,K, 1, H, W)
                        #     communication_masks_tensor = communication_masks_tensor[:,0,:,:,:] # (BxN, 1, H, W)
                        #     x = x * communication_masks_tensor

                        # print('before rm shape is ', rm.shape) # torch.Size([24, 2, 100, 352])
                        # print('before x shape is ', x.shape) # torch.Size([8, 64, 100, 352])
                        # print('before communication_masks_tensor shape is ', communication_masks_tensor.shape) # torch.Size([24, 1, 100, 352])
                        
                        # x = x * communication_masks_tensor
                        # print('after x shape is ', x.shape)

                        # print('success!')
                        # exi12
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                else:
                    if self.communication:
                        communication_masks_tensor = F.max_pool2d(communication_masks_tensor, kernel_size=2) # 缩小两倍以兼容不同尺寸的feature map
                        # TODO: scale = 1, 2 不加 mask
                        # x = x * communication_masks_tensor
                
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C*2^i, H/2^i, W/2^i), (L2, C*2^i, H/2^i, W/2^i), ...]
                # for example, i=1, past_k=3, b_1 has 2 cav, b_2 has 3 cav ... :
                # [[2*3, 256*2, 100/2, 252/2], [3*3, 256*2, 100/2, 252/2], ...]
                batch_node_features = self.regroup(x, record_len, 1) # List [(N1, C, H, W)， （N2， C， H， W）] 长度是B

                ############ 3. Fusion ####################################
                x_fuse = []

                for b in range(B):
                    # number of valid agent
                    N = record_len[b] # 一个场景下的cav数量
                    # t_matrix[i, j]-> from i to j
                    if noise_pairwise_t_matrix is not None:
                        t_matrix = noise_pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3)
                    else:
                        t_matrix = pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3) 本来形状是（B,L,K,2,3） 取出past0 cav 到cur ego的转换矩阵
                    node_features = batch_node_features[b] # (N, C, H, W) 这里的N已经是一个场景下的车数了
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, # (N, C, H, W)
                                                    t_matrix, # (N, 2, 3)
                                                    (H, W)) # project到ego 原本的feature都是在cav view
                    record_frames = np.ones((N))*K
                    # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                    
                    if self.agg_mode == 'RAIN':
                        # for sensor embedding
                        sensor_dist = -1# (B, H, W)
                        x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                        # # TODO for scale debug
                        # if i==self.num_levels-1:
                        #     x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                        # else:
                        #     x_fuse.append(neighbor_feature[0])
                    else: # ATTEN, MAX, Transformer
                        x_fuse.append(self.fuse_modules[i](neighbor_feature)) # 输入形状为（N， C，H，W）MAX Fusion 取第0维最大值，返回（C，H， W）

                        # # TODO for scale debug
                        # if i==self.num_levels-1:
                        #     x_fuse.append(self.fuse_modules[i](neighbor_feature))
                        # else:
                        #     x_fuse.append(neighbor_feature[0])

                x_fuse = torch.stack(x_fuse) # （B， C， H， W） B在这里就是batch数了，

                ############ 4. Deconv #################################### 上采样
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                    # ups.append(F.dropout2d(backbone.deblocks[i](x_fuse), p=0.1, training=self.training))
                else:
                    ups.append(x_fuse)
                
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1) #（B， 128*3， H/2, W/2）
                    
            elif len(ups) == 1: 
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
                
        else:
            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len, K)
            batch_confidence_maps = self.regroup(rm, record_len, K)
            batch_time_intervals = self.regroup(time_diffs, record_len, K)

            ############ 2. Communication (Mask the features) #########
            if self.communication:
                # _, (B, 1, H, W), float
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            else:
                communication_rates = torch.tensor(0).to(x.device)
            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix,
                                                (H, W))
                record_frames = np.ones((N))*K
                # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                if self.agg_mode == "RAIN":
                    # for sensor embedding
                    sensor_dist = -1# (B, H, W)
                    x_fuse.append(self.fuse_modules(neighbor_feature, sensor_dist, batch_time_intervals[b], communication_masks[b]))
                else: # ATTEN, MAX, Transformer
                    x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)

            # self.fuse_modsules(x_fuse, record_len)
        if self.design == 0 or self.design == 5:
            if viz_bbx_flag:
                return  x_fuse, communication_rates, {}, updated_features
            return x_fuse, communication_rates, {}
        elif self.design == 1:
            return x_fuse, communication_rates, {}, flow_recon_loss
        elif self.design == 2:
            return x_fuse, communication_rates, {}, flow_recon_loss #, flow, state_preds
        elif self.design == 3:
            return x_fuse, communication_rates, {}, flow, state_preds
        elif self.design == 4:
            return x_fuse, communication_rates, {}
        
    def forward_backup_w_raw(self, x, rm, record_len, pairwise_t_matrix, time_diffs, backbone=None, heads=None, flow_gt=None, box_flow=None, reserved_mask=None, viz_bbx_flag=False, noise_pairwise_t_matrix=None, num_roi_thres=-1):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)  V2XSet: (sum(n_cav), 64, 200, 704)
        
        rm : torch.Tensor
            confidence map, (sum(n_cav), 2, H, W)  V2XSet: (sum(n_cav), 64, 100, 352)

        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        time_diffs : torch.Tensor
            time interval, (sum(n_cav), )
        
        flow_gt: ground truth flow, generate by object center and id

        box_flow: flow generate by past 2 frames detection boxes
            
        Returns
        -------
        Fused feature
        flow_map loss
        """
        _, C, H, W = x.shape # (sum(n_cav), 64, 200, 704)
        B, L, K = pairwise_t_matrix.shape[:3]

        # print("x shape is ", x.shape)
        # print("pairwise_t_matrix shape is ", pairwise_t_matrix.shape)
        # print("rm shape is ", rm.shape)
        # print("record_len shape is ", record_len.shape)
        # print("time_diffs shape is ", time_diffs.shape)
        # print("time_diffs  is ", time_diffs)
        # exit1
        # (B,L,k,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]]  # 从（B，L， K， 4， 4）中取出2D变换部分（B，L， K， 2， 3）
        # [B, L, L, 2, 3], 只要x,y这两个维度就可以(z只有一层)，所以提取[0,1,3]作为仿射变换矩阵, 大小(2, 3)
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W # 因为这个权重要和y坐标相乘，它表示是旋转的比例，但是他是基于x坐标的，也就是W（704），但是现在要作用在H（200）上，所以要转变一下
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2 # 乘以2是将（-1/2， 1/2）调整到（-1， 1）
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        if noise_pairwise_t_matrix is not None:
            noise_pairwise_t_matrix = noise_pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] 
            noise_pairwise_t_matrix[...,0,1] = noise_pairwise_t_matrix[...,0,1] * H / W
            noise_pairwise_t_matrix[...,1,0] = noise_pairwise_t_matrix[...,1,0] * W / H
            noise_pairwise_t_matrix[...,0,2] = noise_pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
            noise_pairwise_t_matrix[...,1,2] = noise_pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        # single frame TODO: 
        # batch_confidence_maps = self.regroup(psm_single, record_len, 1) # [[2*3, 2, 100/2, 252/2], [3*3, 2, 100/2, 252/2], ...]
        self.num_roi_thres = num_roi_thres # roi的数量限制 
        if self.num_roi_thres > 0:
            self.thre = 0.01
            _, _, H, W = rm.shape
            num_valid_thres = self.num_roi_thres * 10 # 这里是规定了所有roi总像素个数限制
            communication_maps = rm.sigmoid().max(dim=1)[0].unsqueeze(1) # 构建置信度图 (sum(n_cav), 2, H, W)变为(sum(n_cav), 1, H, W)存最大值
            communication_masks_list = []
            curr_batch = communication_maps.shape[0] # num_cav * k
            communication_mask = (communication_maps>self.thre).to(torch.int) # 置信度图大于0.01的标记1 形状 (sum(n_cav), 1, H, W)
            valid_nums = torch.sum(communication_mask, dim=(-1, -2, -3)) # (sum(n_cav), ) 每个cav每个帧中的有效像素数量

            final_comm_mask = torch.zeros_like(communication_mask) # (sum(n_cav), 1, H, W)
            for i in range(curr_batch):# 遍历具体某个车的某个帧
                if valid_nums[i] > num_valid_thres : # 有效像素数量超出限制
                    # 选择 comm_maps[i] 里面 top num_valid 的位置
                    tmp = communication_maps[i].reshape((1, -1)) # (1, H*W*sum(n_cav))
                    _, idx = torch.topk(tmp, num_valid_thres) # 选出最高置信度的有效像素个数
                    # print(idx)
                    curr_mask = torch.zeros_like(tmp)
                    curr_mask[0, idx[0]] = 1
                    final_comm_mask[i] = curr_mask.reshape((1, H, W)) # 形成最终的有效区域掩码
                else:
                    final_comm_mask[i] = communication_mask[i] # 保证前k个是全1

            #     communication_mask = final_comm_mask
            #     communication_masks_list.append(communication_mask)
            # # _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            # communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
            # print(final_comm_mask.shape)
            # final_comm_mask scale x 2 [B, 1, H, W] -> [B, 1, 2H, 2W]

            # 将张量插值到目标大小 (B, 1, 2H, 2W)
            interpolated_tensor = F.interpolate(final_comm_mask.to(torch.float32).to(x.device), scale_factor=2, mode='nearest') # 这里的 'nearest' 插值模式意味着在放大过程中，将直接使用最近的像素值来填充新的像素位置，这通常用于分类标签或其他不需要平滑过渡的场景，因为它不会引入新的像素值，只是简单地复制现有的像素值。
            interpolated_tensor[:K, ...] = 1 # 保证前k个是全1，TODO 这是为什么？猜测是为了保证ego的所有特征要能保留，所以ego不能掩码，而其他agent要参与传输，为了节省带宽，因此需要掩码

            x = x*interpolated_tensor
        
        # 2. feature compensation with flow
        # 2.1 generate flow, 在同一个坐标系内，计算每个cav的flow
        # 2.2 compensation
        # x: (BxNxK, C, H, W) -> (BxN, C, H, W)
        if self.design == 0 or self.design == 5:
            updated_features, raw_feature_sum = self.update_features_boxflow(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt) # (BxN, C, H, W)
        elif self.design == 1:
            updated_features, flow_recon_loss = self.update_features_boxflow_design_1(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 2:
            updated_features, flow_recon_loss, flow, state_preds = self.update_features_boxflow_design_2(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 3:
            updated_features, flow, state_preds = self.generateFlow(x, pairwise_t_matrix, record_len, flow_gt) # (BxN, C, H, W)
        elif self.design == 4: # SyncNet
            updated_features = self.generate_estimated_feats(x, pairwise_t_matrix, record_len) # (BxN, C, H, W)
        
        # debug 0 延迟时的结果 默认注释
        # updated_features = x

        # 3. feature fusion
        if self.multi_scale:
            ups = []

            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            if with_resnet:
                feats = backbone.resnet(updated_features)  # tuple((B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4))  得到三个结果，也即多尺寸结果，包含natch中每个车的特征图 其实正确的形状为tuple((B, C, H/2, W/2), (B, 2C, H/4, W/4), (B, 4C, H/8, W/8))
                feats_raw = self.raw_extract.resnet(raw_feature_sum) # shape同上

            for i in range(self.num_levels): # 3
                x = feats[i] if with_resnet else backbone.blocks[i](x)  # (BxN, C', H, W) 这里的BxN应该就是所有车数了，因为所有cav都已经经过延迟补偿
                x_raw = feats_raw[i]
                ############ 1. Communication (Mask the features) #########
                if i==0:
                    if self.communication: # 这进去后会设置置信度图和掩码
                        batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100, 352], [3*3, 2, 100, 352], ...] 返回List [(N1*k, 2, H, W), (N2*k, 2, H, W)...] 置信度图，划分到每一辆车
                        batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...] 返回List [(N1*k), (N2*k)...] 距离cur的时间间隔 划分到每一辆车
                        _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                        communication_masks_tensor = torch.concat(communication_masks_list, dim=0) # (sum(n_cav), 1, H, W)

                        # delay_0ms_flag = False # 使用where2comm的mask机制
                        # if delay_0ms_flag:
                        #     H, W =  communication_masks_tensor.shape[2:]
                        #     communication_masks_tensor = communication_masks_tensor.reshape(-1, K, 1, H, W) # (BxN,K, 1, H, W)
                        #     communication_masks_tensor = communication_masks_tensor[:,0,:,:,:] # (BxN, 1, H, W)
                        #     x = x * communication_masks_tensor

                        # print('before rm shape is ', rm.shape) # torch.Size([24, 2, 100, 352])
                        # print('before x shape is ', x.shape) # torch.Size([8, 64, 100, 352])
                        # print('before communication_masks_tensor shape is ', communication_masks_tensor.shape) # torch.Size([24, 1, 100, 352])
                        
                        # x = x * communication_masks_tensor
                        # print('after x shape is ', x.shape)

                        # print('success!')
                        # exi12
                    else:
                        communication_rates = torch.tensor(0).to(x.device)
                else:
                    if self.communication:
                        communication_masks_tensor = F.max_pool2d(communication_masks_tensor, kernel_size=2) # 缩小两倍以兼容不同尺寸的feature map
                        # TODO: scale = 1, 2 不加 mask
                        # x = x * communication_masks_tensor
                
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C*2^i, H/2^i, W/2^i), (L2, C*2^i, H/2^i, W/2^i), ...]
                # for example, i=1, past_k=3, b_1 has 2 cav, b_2 has 3 cav ... :
                # [[2*3, 256*2, 100/2, 252/2], [3*3, 256*2, 100/2, 252/2], ...]
                batch_node_features = self.regroup(x, record_len, 1) # List [(N1, C, H, W)， （N2， C， H， W）] 长度是B

                ############ 3. Fusion ####################################
                x_fuse = []

                for b in range(B):
                    # number of valid agent
                    N = record_len[b] # 一个场景下的cav数量
                    # t_matrix[i, j]-> from i to j
                    if noise_pairwise_t_matrix is not None:
                        t_matrix = noise_pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3)
                    else:
                        t_matrix = pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3) 本来形状是（B,L,K,2,3） 取出past0 cav 到cur ego的转换矩阵
                    node_features = batch_node_features[b] # (N, C, H, W) 这里的N已经是一个场景下的车数了
                    node_raw = x_raw[[b], :,:,:] # (1, C, H, W)
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, # (N, C, H, W)
                                                    t_matrix, # (N, 2, 3)
                                                    (H, W)) # project到ego 原本的feature都是在cav view
                    record_frames = np.ones((N))*K
                    # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                    
                    if self.agg_mode == 'RAIN':
                        # for sensor embedding
                        sensor_dist = -1# (B, H, W)
                        x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                        # # TODO for scale debug
                        # if i==self.num_levels-1:
                        #     x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                        # else:
                        #     x_fuse.append(neighbor_feature[0])
                    else: # ATTEN, MAX, Transformer
                        x_fuse.append(self.fuse_modules[i](neighbor_feature, node_raw)) # 输入形状为（N， C，H，W）MAX Fusion 取第0维最大值，返回（C，H， W）

                        # # TODO for scale debug
                        # if i==self.num_levels-1:
                        #     x_fuse.append(self.fuse_modules[i](neighbor_feature))
                        # else:
                        #     x_fuse.append(neighbor_feature[0])

                x_fuse = torch.stack(x_fuse) # （B， C， H， W） B在这里就是batch数了，

                ############ 4. Deconv #################################### 上采样
                if len(backbone.deblocks) > 0:
                    # ups.append(backbone.deblocks[i](x_fuse))
                    ups.append(F.dropout2d(backbone.deblocks[i](x_fuse), p=0.1, training=self.training))
                else:
                    ups.append(x_fuse)
                
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1) #（B， 128*3， H/2, W/2）
                    
            elif len(ups) == 1: 
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
                
        else:
            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len, K)
            batch_confidence_maps = self.regroup(rm, record_len, K)
            batch_time_intervals = self.regroup(time_diffs, record_len, K)

            ############ 2. Communication (Mask the features) #########
            if self.communication:
                # _, (B, 1, H, W), float
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            else:
                communication_rates = torch.tensor(0).to(x.device)
            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix,
                                                (H, W))
                record_frames = np.ones((N))*K
                # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                if self.agg_mode == "RAIN":
                    # for sensor embedding
                    sensor_dist = -1# (B, H, W)
                    x_fuse.append(self.fuse_modules(neighbor_feature, sensor_dist, batch_time_intervals[b], communication_masks[b]))
                else: # ATTEN, MAX, Transformer
                    x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)

            # self.fuse_modsules(x_fuse, record_len)
        if self.design == 0 or self.design == 5:
            if viz_bbx_flag:
                return  x_fuse, communication_rates, {}, updated_features
            return x_fuse, communication_rates, {}
        elif self.design == 1:
            return x_fuse, communication_rates, {}, flow_recon_loss
        elif self.design == 2:
            return x_fuse, communication_rates, {}, flow_recon_loss #, flow, state_preds
        elif self.design == 3:
            return x_fuse, communication_rates, {}, flow, state_preds
        elif self.design == 4:
            return x_fuse, communication_rates, {}
                
    def forward_backup_2(self, x, rm, record_len, pairwise_t_matrix, time_diffs, backbone=None, heads=None, flow_gt=None, box_flow=None, reserved_mask=None, viz_bbx_flag=False, noise_pairwise_t_matrix=None, num_roi_thres=-1):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)  V2XSet: (sum(n_cav), 64, 200, 704)
        
        rm : torch.Tensor
            confidence map, (sum(n_cav), 2, H, W)  V2XSet: (sum(n_cav), 64, 100, 352)

        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, K, 2, 3) 

        time_diffs : torch.Tensor
            time interval, (sum(n_cav), )
        
        flow_gt: ground truth flow, generate by object center and id

        box_flow: flow generate by past 2 frames detection boxes
            
        Returns
        -------
        Fused feature
        flow_map loss
        """
        _, C, H, W = x.shape # (sum(n_cav), 64, 200, 704)
        B, L, K = pairwise_t_matrix.shape[:3]

        # print("x shape is ", x.shape)
        # print("pairwise_t_matrix shape is ", pairwise_t_matrix.shape)
        # print("rm shape is ", rm.shape)
        # print("record_len shape is ", record_len.shape)
        # print("time_diffs shape is ", time_diffs.shape)
        # print("time_diffs  is ", time_diffs)
        # exit1
        # (B,L,k,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]]  # 从（B，L， K， 4， 4）中取出2D变换部分（B，L， K， 2， 3）
        # [B, L, L, 2, 3], 只要x,y这两个维度就可以(z只有一层)，所以提取[0,1,3]作为仿射变换矩阵, 大小(2, 3)
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W # 因为这个权重要和y坐标相乘，它表示是旋转的比例，但是他是基于x坐标的，也就是W（704），但是现在要作用在H（200）上，所以要转变一下
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2 # 乘以2是将（-1/2， 1/2）调整到（-1， 1）
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        if noise_pairwise_t_matrix is not None:
            noise_pairwise_t_matrix = noise_pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] 
            noise_pairwise_t_matrix[...,0,1] = noise_pairwise_t_matrix[...,0,1] * H / W
            noise_pairwise_t_matrix[...,1,0] = noise_pairwise_t_matrix[...,1,0] * W / H
            noise_pairwise_t_matrix[...,0,2] = noise_pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
            noise_pairwise_t_matrix[...,1,2] = noise_pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        # single frame TODO: 
        # batch_confidence_maps = self.regroup(psm_single, record_len, 1) # [[2*3, 2, 100/2, 252/2], [3*3, 2, 100/2, 252/2], ...]
        self.num_roi_thres = num_roi_thres # roi的数量限制 
        if self.num_roi_thres > 0:
            self.thre = 0.01
            _, _, H, W = rm.shape
            num_valid_thres = self.num_roi_thres * 10 # 这里是规定了所有roi总像素个数限制
            communication_maps = rm.sigmoid().max(dim=1)[0].unsqueeze(1) # 构建置信度图 (sum(n_cav), 2, H, W)变为(sum(n_cav), 1, H, W)存最大值
            communication_masks_list = []
            curr_batch = communication_maps.shape[0] # num_cav * k
            communication_mask = (communication_maps>self.thre).to(torch.int) # 置信度图大于0.01的标记1 形状 (sum(n_cav), 1, H, W)
            valid_nums = torch.sum(communication_mask, dim=(-1, -2, -3)) # (sum(n_cav), ) 每个cav每个帧中的有效像素数量

            final_comm_mask = torch.zeros_like(communication_mask) # (sum(n_cav), 1, H, W)
            for i in range(curr_batch):# 遍历具体某个车的某个帧
                if valid_nums[i] > num_valid_thres : # 有效像素数量超出限制
                    # 选择 comm_maps[i] 里面 top num_valid 的位置
                    tmp = communication_maps[i].reshape((1, -1)) # (1, H*W*sum(n_cav))
                    _, idx = torch.topk(tmp, num_valid_thres) # 选出最高置信度的有效像素个数
                    # print(idx)
                    curr_mask = torch.zeros_like(tmp)
                    curr_mask[0, idx[0]] = 1
                    final_comm_mask[i] = curr_mask.reshape((1, H, W)) # 形成最终的有效区域掩码
                else:
                    final_comm_mask[i] = communication_mask[i] # 保证前k个是全1

            #     communication_mask = final_comm_mask
            #     communication_masks_list.append(communication_mask)
            # # _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            # communication_masks_tensor = torch.concat(communication_masks_list, dim=0) 
            # print(final_comm_mask.shape)
            # final_comm_mask scale x 2 [B, 1, H, W] -> [B, 1, 2H, 2W]

            # 将张量插值到目标大小 (B, 1, 2H, 2W)
            interpolated_tensor = F.interpolate(final_comm_mask.to(torch.float32).to(x.device), scale_factor=2, mode='nearest') # 这里的 'nearest' 插值模式意味着在放大过程中，将直接使用最近的像素值来填充新的像素位置，这通常用于分类标签或其他不需要平滑过渡的场景，因为它不会引入新的像素值，只是简单地复制现有的像素值。
            interpolated_tensor[:K, ...] = 1 # 保证前k个是全1，TODO 这是为什么？猜测是为了保证ego的所有特征要能保留，所以ego不能掩码，而其他agent要参与传输，为了节省带宽，因此需要掩码

            x = x*interpolated_tensor
        
        # 2. feature compensation with flow
        # 2.1 generate flow, 在同一个坐标系内，计算每个cav的flow
        # 2.2 compensation
        # x: (BxNxK, C, H, W) -> (BxN, C, H, W)
        if self.design == 0 or self.design == 5:
            updated_features = self.update_features_boxflow(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt) # (BxN, C, H, W)
        elif self.design == 1:
            updated_features, flow_recon_loss = self.update_features_boxflow_design_1(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 2:
            updated_features, flow_recon_loss, flow, state_preds = self.update_features_boxflow_design_2(x, pairwise_t_matrix, record_len, box_flow, reserved_mask, flow_gt)
        elif self.design == 3:
            updated_features, flow, state_preds = self.generateFlow(x, pairwise_t_matrix, record_len, flow_gt) # (BxN, C, H, W)
        elif self.design == 4: # SyncNet
            updated_features = self.generate_estimated_feats(x, pairwise_t_matrix, record_len) # (BxN, C, H, W)
        
        # debug 0 延迟时的结果 默认注释
        # updated_features = x

        # 3. feature fusion
        if self.multi_scale:
            whole_fpn = True
            if whole_fpn:
                data_dict = {}
                x_fuse = []

                if self.communication: # 这进去后会设置置信度图和掩码
                    batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100, 352], [3*3, 2, 100, 352], ...] 返回List [(N1*k, 2, H, W), (N2*k, 2, H, W)...] 置信度图，划分到每一辆车
                    batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...] 返回List [(N1*k), (N2*k)...] 距离cur的时间间隔 划分到每一辆车
                    _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                    communication_masks_tensor = torch.concat(communication_masks_list, dim=0) # (sum(n_cav), 1, H, W)

                    delay_0ms_flag = False # 使用where2comm的mask机制
                    if delay_0ms_flag:
                        H, W =  communication_masks_tensor.shape[2:]
                        communication_masks_tensor = communication_masks_tensor.reshape(-1, K, 1, H, W) # (BxN,K, 1, H, W)
                        communication_masks_tensor = communication_masks_tensor[:,0,:,:,:] # (BxN, 1, H, W)
                        x = x * communication_masks_tensor

                else:
                    communication_rates = torch.tensor(0).to(x.device)

                data_dict['spatial_features'] = updated_features
                res_dict = backbone(data_dict) # N,384, H, W
                batch_node_features = self.regroup(res_dict['spatial_features_2d'], record_len, 1) # List [(N1, C, H, W)， （N2， C， H， W）] 长度是B
                for b in range(B):
                    # number of valid agent
                    N = record_len[b] # 一个场景下的cav数量
                    # t_matrix[i, j]-> from i to j
                    if noise_pairwise_t_matrix is not None:
                        t_matrix = noise_pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3)
                    else:
                        t_matrix = pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3) 本来形状是（B,L,K,2,3） 取出past0 cav 到cur ego的转换矩阵
                    node_features = batch_node_features[b] # (N, C, H, W) 这里的N已经是一个场景下的车数了
                    C, H, W = node_features.shape[1:]
                    neighbor_feature = warp_affine_simple(node_features, # (N, C, H, W)
                                                    t_matrix, # (N, 2, 3)
                                                    (H, W)) # project到ego 原本的feature都是在cav view
                    x_fuse.append(self.fuse_modules(neighbor_feature)) # N,.C, H, W --> C, H, W
                x_fuse = torch.stack(x_fuse) # （B， C， H， W） B在这里就是batch数了
                '''
                data_dict = {}
                x_fuse = []

                if self.communication: # 这进去后会设置置信度图和掩码
                    batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100, 352], [3*3, 2, 100, 352], ...] 返回List [(N1*k, 2, H, W), (N2*k, 2, H, W)...] 置信度图，划分到每一辆车
                    batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...] 返回List [(N1*k), (N2*k)...] 距离cur的时间间隔 划分到每一辆车
                    _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                    communication_masks_tensor = torch.concat(communication_masks_list, dim=0) # (sum(n_cav), 1, H, W)

                    delay_0ms_flag = False # 使用where2comm的mask机制
                    if delay_0ms_flag:
                        H, W =  communication_masks_tensor.shape[2:]
                        communication_masks_tensor = communication_masks_tensor.reshape(-1, K, 1, H, W) # (BxN,K, 1, H, W)
                        communication_masks_tensor = communication_masks_tensor[:,0,:,:,:] # (BxN, 1, H, W)
                        x = x * communication_masks_tensor

                else:
                    communication_rates = torch.tensor(0).to(x.device)

                data_dict['spatial_features'] = updated_features
                res_dict = backbone(data_dict) # N,384, H, W
                res = []
                for i in range(self.num_levels):
                    x_fuse = []
                    batch_node_features = res_dict['spatial_features_2d'][:,i,:,:,:]

                    batch_node_features = self.regroup(batch_node_features, record_len, 1) # List [(N1, C, H, W)， （N2， C， H， W）] 长度是B

                    for b in range(B):
                        # number of valid agent
                        N = record_len[b] # 一个场景下的cav数量                        
                        # t_matrix[i, j]-> from i to j
                        if noise_pairwise_t_matrix is not None:
                            t_matrix = noise_pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3)
                        else:
                            t_matrix = pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3) 本来形状是（B,L,K,2,3） 取出past0 cav 到cur ego的转换矩阵
                        
                        node_features = batch_node_features[b]# (N, 3, C, H, W) 这里的N已经是一个场景下的车数了

                        C, H, W = node_features.shape[1:]
                        neighbor_feature = warp_affine_simple(node_features, # (N, C, H, W)
                                                        t_matrix, # (N, 2, 3)
                                                        (H, W)) # project到ego 原本的feature都是在cav view
                        x_fuse.append(self.fuse_modules[i](neighbor_feature)) # N,C, H, W --> C, H, W
                    x_fuse = torch.stack(x_fuse) # （B， C， H， W） B在这里就是batch数了
                    res.append(x_fuse)
                x_fuse = torch.cat(res, dim=1)
                '''
                
            else:
                ups = []
                if self.agg_mode == "MAX_MedianFusion":
                    ups_branch = []
                if self.agg_mode == "HybridReductionFusion":
                    ups_avg = []
                    ups_median = []

                # backbone.__dict__()
                with_resnet = True if hasattr(backbone, 'resnet') else False
                if with_resnet:
                    feats = backbone.resnet(updated_features)  # tuple((B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4))  得到三个结果，也即多尺寸结果，包含natch中每个车的特征图 其实正确的形状为tuple((B, C, H/2, W/2), (B, 2C, H/4, W/4), (B, 4C, H/8, W/8))
                
                # print('updated_features shape is ', updated_features.shape) #  torch.Size([5, 64, 200, 704]) 此时B=2，这个5表示两个场景下一共有五个cav 都已经经过延迟补偿
                # for res in  feats:
                #     print("res shape is ", res.shape)

                # exi1

                for i in range(self.num_levels): # 3
                    x = feats[i] if with_resnet else backbone.blocks[i](x)  # (BxN, C', H, W) 这里的BxN应该就是所有车数了，因为所有cav都已经经过延迟补偿

                    ############ 1. Communication (Mask the features) #########
                    if i==0:
                        if self.communication: # 这进去后会设置置信度图和掩码
                            batch_confidence_maps = self.regroup(rm, record_len, K) # [[2*3, 2, 100, 352], [3*3, 2, 100, 352], ...] 返回List [(N1*k, 2, H, W), (N2*k, 2, H, W)...] 置信度图，划分到每一辆车
                            batch_time_intervals = self.regroup(time_diffs, record_len, K) # [[2*3], [3*3], ...] 返回List [(N1*k), (N2*k)...] 距离cur的时间间隔 划分到每一辆车
                            _, communication_masks_list, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                            communication_masks_tensor = torch.concat(communication_masks_list, dim=0) # (sum(n_cav), 1, H, W)

                            delay_0ms_flag = False # 使用where2comm的mask机制
                            if delay_0ms_flag:
                                H, W =  communication_masks_tensor.shape[2:]
                                communication_masks_tensor = communication_masks_tensor.reshape(-1, K, 1, H, W) # (BxN,K, 1, H, W)
                                communication_masks_tensor = communication_masks_tensor[:,0,:,:,:] # (BxN, 1, H, W)
                                x = x * communication_masks_tensor

                            # print('before rm shape is ', rm.shape) # torch.Size([24, 2, 100, 352])
                            # print('before x shape is ', x.shape) # torch.Size([8, 64, 100, 352])
                            # print('before communication_masks_tensor shape is ', communication_masks_tensor.shape) # torch.Size([24, 1, 100, 352])
                            
                            # x = x * communication_masks_tensor
                            # print('after x shape is ', x.shape)

                            # print('success!')
                            # exi12
                        else:
                            communication_rates = torch.tensor(0).to(x.device)
                    else:
                        if self.communication:
                            communication_masks_tensor = F.max_pool2d(communication_masks_tensor, kernel_size=2) # 缩小两倍以兼容不同尺寸的feature map
                            # TODO: scale = 1, 2 不加 mask
                            # x = x * communication_masks_tensor
                    
                    ############ 2. Split the confidence map #######################
                    # split x:[(L1, C*2^i, H/2^i, W/2^i), (L2, C*2^i, H/2^i, W/2^i), ...]
                    # for example, i=1, past_k=3, b_1 has 2 cav, b_2 has 3 cav ... :
                    # [[2*3, 256*2, 100/2, 252/2], [3*3, 256*2, 100/2, 252/2], ...]
                    batch_node_features = self.regroup(x, record_len, 1) # List [(N1, C, H, W)， （N2， C， H， W）] 长度是B

                    ############ 3. Fusion ####################################
                    x_fuse = []
                    if self.agg_mode == "MAX_MedianFusion":
                        x_fuse_branch = []
                    if self.agg_mode == "HybridReductionFusion":
                        x_fuse_avg = []
                        x_fuse_median = []

                    for b in range(B):
                        # number of valid agent
                        N = record_len[b] # 一个场景下的cav数量
                        # t_matrix[i, j]-> from i to j
                        if noise_pairwise_t_matrix is not None:
                            t_matrix = noise_pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3)
                        else:
                            t_matrix = pairwise_t_matrix[b][:N, 0, :, :] #(N, 2, 3) 本来形状是（B,L,K,2,3） 取出past0 cav 到cur ego的转换矩阵
                        node_features = batch_node_features[b] # (N, C, H, W) 这里的N已经是一个场景下的车数了
                        C, H, W = node_features.shape[1:]
                        neighbor_feature = warp_affine_simple(node_features, # (N, C, H, W)
                                                        t_matrix, # (N, 2, 3)
                                                        (H, W)) # project到ego 原本的feature都是在cav view
                        record_frames = np.ones((N))*K
                        # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                        
                        if self.agg_mode == 'RAIN':
                            # for sensor embedding
                            sensor_dist = -1# (B, H, W)
                            x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                            # # TODO for scale debug
                            # if i==self.num_levels-1:
                            #     x_fuse.append(self.fuse_modules[i](neighbor_feature, sensor_dist, batch_time_intervals[b], self.regroup(communication_masks_tensor, record_len, K)[b]))
                            # else:
                            #     x_fuse.append(neighbor_feature[0])
                        else: # ATTEN, MAX, Transformer
                            x_fuse.append(self.fuse_modules[i](neighbor_feature)) # 输入形状为（N， C，H，W）MAX Fusion 取第0维最大值，返回（C，H， W）
                            if self.agg_mode == "MAX_MedianFusion":
                                x_fuse_branch.append(self.fuse_network_median[i](neighbor_feature))
                            elif self.agg_mode == "HybridReductionFusion":
                                x_fuse_avg.append(self.fuse_network_avg[i](neighbor_feature))
                                x_fuse_median.append(self.fuse_network_median[i](neighbor_feature))

                            # # TODO for scale debug
                            # if i==self.num_levels-1:
                            #     x_fuse.append(self.fuse_modules[i](neighbor_feature))
                            # else:
                            #     x_fuse.append(neighbor_feature[0])

                    x_fuse = torch.stack(x_fuse) # （B， C， H， W） B在这里就是batch数了，
                    if self.agg_mode == "MAX_MedianFusion":
                        x_fuse_branch = torch.stack(x_fuse_branch)
                    if self.agg_mode == "HybridReductionFusion":
                        x_fuse_avg = torch.stack(x_fuse_avg)
                        x_fuse_median = torch.stack(x_fuse_median)


                    ############ 4. Deconv #################################### 上采样
                    if len(backbone.deblocks) > 0:
                        # ups.append(backbone.deblocks[i](x_fuse))
                        ups.append(F.dropout2d(backbone.deblocks[i](x_fuse), p=0.1, training=self.training))
                        if self.agg_mode == "MAX_MedianFusion":
                            ups_branch.append(F.dropout2d(self.deblocks_median[i](x_fuse_branch), p=0.1, training=self.training))
                        elif self.agg_mode == "HybridReductionFusion":
                            ups_avg.append(F.dropout2d(backbone.deblocks[i](x_fuse_avg), p=0.1, training=self.training))
                            ups_median.append(F.dropout2d(backbone.deblocks[i](x_fuse_median), p=0.1, training=self.training))

                    else:
                        ups.append(x_fuse)
                    
                if len(ups) > 1:
                    x_fuse = torch.cat(ups, dim=1) #（B， 128*3， H/2, W/2）
                    if self.agg_mode == "MAX_MedianFusion":
                        x_fuse_branch = torch.cat(ups_branch, dim=1).unsqueeze(1) #（B，1,  128*3， H/2, W/2）
                        x_fuse = x_fuse.unsqueeze(1)
                        concatenated_tensor = torch.cat((x_fuse, x_fuse_branch), dim=1) #（B，2,  128*3， H/2, W/2）
                        x_fuse = self.hybridconv3d(concatenated_tensor)#（B，1,  128*3， H/2, W/2）
                        x_fuse = x_fuse.squeeze(1)
                    elif self.agg_mode == "HybridReductionFusion":
                        x_fuse_avg = torch.cat(ups_avg, dim=1).unsqueeze(1) #（B，1,  128*3， H/2, W/2）
                        x_fuse_median = torch.cat(ups_median, dim=1).unsqueeze(1) #（B，1,  128*3， H/2, W/2）
                        x_fuse = x_fuse.unsqueeze(1)
                        concatenated_tensor = torch.cat((x_fuse, x_fuse_avg, x_fuse_median), dim=1) #（B，3,  128*3， H/2, W/2）
                        x_fuse = self.hybridconv3d(concatenated_tensor)#（B，1,  128*3， H/2, W/2）
                        x_fuse = x_fuse.squeeze(1)
                        
                elif len(ups) == 1: 
                    x_fuse = ups[0]
                
                if len(backbone.deblocks) > self.num_levels:
                    x_fuse = backbone.deblocks[-1](x_fuse)
                
        else:
            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len, K)
            batch_confidence_maps = self.regroup(rm, record_len, K)
            batch_time_intervals = self.regroup(time_diffs, record_len, K)

            ############ 2. Communication (Mask the features) #########
            if self.communication:
                # _, (B, 1, H, W), float
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
            else:
                communication_rates = torch.tensor(0).to(x.device)
            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :, :, :].view(-1, 2, 3) #(Nxk, 2, 3)
                node_features = batch_node_features[b]
                if self.communication:
                    node_features = node_features * communication_masks[b]
                neighbor_feature = warp_affine_simple(node_features,
                                                t_matrix,
                                                (H, W))
                record_frames = np.ones((N))*K
                # def forward(self, feartures, comm_mask, sensor_dist, record_frames, time_diffs):
                if self.agg_mode == "RAIN":
                    # for sensor embedding
                    sensor_dist = -1# (B, H, W)
                    x_fuse.append(self.fuse_modules(neighbor_feature, sensor_dist, batch_time_intervals[b], communication_masks[b]))
                else: # ATTEN, MAX, Transformer
                    x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)

            # self.fuse_modsules(x_fuse, record_len)
        if self.design == 0 or self.design == 5:
            if viz_bbx_flag:
                return  x_fuse, communication_rates, {}, updated_features
            return x_fuse, communication_rates, {}
        elif self.design == 1:
            return x_fuse, communication_rates, {}, flow_recon_loss
        elif self.design == 2:
            return x_fuse, communication_rates, {}, flow_recon_loss #, flow, state_preds
        elif self.design == 3:
            return x_fuse, communication_rates, {}, flow, state_preds
        elif self.design == 4:
            return x_fuse, communication_rates, {}        

class TemporalFusion(nn.Module):
    def __init__(self, args):
        super(TemporalFusion, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        # channel_size = args['channel_size']
        # spatial_size = args['spatial_size']
        # TM_Flag = args['TM_Flag']
        # compressed_size = args['compressed_size']
        # self.compensation_net = SyncLSTM(channel_size, spatial_size, TM_Flag, compressed_size)
        self.stpn = STPN(height_feat_size=args['channel_size'])
        self.flow_thre = args['flow_thre']
        self.motion_pred = MotionPrediction(seq_len=1)
        self.state_classify = StateEstimation(motion_category_num=1)

        self.flow_unc_flag = False
        if 'flow_unc_flag' in args:
            self.flow_unc_flag = True
            self.flow_unc_pred = FlowUncPrediction(seq_len=1)

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        if 'FlowPredictionFix' in args.keys() and args['FlowPredictionFix']:
            self.FlowPredictionFix()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def FlowPredictionFix(self):
        print('------ Flow Prediction Fix, Only train DCN ------')
        for p in self.stpn.parameters():
            p.requires_grad = False
        for p in self.motion_pred.parameters():
            p.requires_grad = False
        for p in self.state_classify.parameters():
            p.requires_grad = False

    def forward(self, x, record_len, pairwise_t_matrix, data_dict=None):
        """
        Fusion forwarding.

        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        record_len : list
            shape: (B)

        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego,
            shape: (B, L, L, 4, 4)

        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        feat_seqs = self.regroup(x, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
                    self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
                    self.downsample_rate * self.discrete_ratio * H) * 2

        # iteratively warp feature to current timestamp
        batch_feat_seqs = []
        for b in range(B):
            # number of valid timestamps
            K = record_len[b]
            t_matrix = pairwise_t_matrix[b][:K, :K, :, :]
            curr_feat_seq = warp_affine_simple(feat_seqs[b],
                                               t_matrix[0, :, :, :],
                                               (H, W))
            batch_feat_seqs.append(curr_feat_seq[None, ...])
        batch_feat_seqs = torch.cat(batch_feat_seqs, dim=0)  # b, K, c, h, w
        batch_hist_feat_seqs = batch_feat_seqs[:, 1:].flip(1)

        # Backbone network
        bevs = self.stpn(batch_hist_feat_seqs)  # b, K, c, h, w

        # Motion Displacement prediction
        flow = self.motion_pred(bevs)
        flow = flow.view(-1, 2, bevs.size(-2), bevs.size(-1))

        flow_unc = None
        if self.flow_unc_flag:
            flow_unc = self.flow_unc_pred(bevs)
            flow_unc = flow_unc.view(-1, 2, bevs.size(-2), bevs.size(-1))

        # flow = data_dict['flow_gt']

        # Motion State Classification head
        state_class_pred = self.state_classify(bevs)

        # Given disp shift feature
        x_coord = torch.arange(bevs.size(-1)).float()
        y_coord = torch.arange(bevs.size(-2)).float()
        y, x = torch.meshgrid(y_coord, x_coord)
        grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(flow.shape[0], -1, -1, -1).to(
            flow.device)
        # updated_grid = grid + flow * (state_class_pred.sigmoid() > self.flow_thre)
        updated_grid = grid - flow
        updated_grid[:, 0, :, :] = updated_grid[:, 0, :, :] / (bevs.size(-1) / 2.0) - 1.0
        updated_grid[:, 1, :, :] = updated_grid[:, 1, :, :] / (bevs.size(-2) / 2.0) - 1.0
        out = F.grid_sample(batch_feat_seqs[:, 1], grid=updated_grid.permute(0, 2, 3, 1), mode='bilinear')
        # out = F.grid_sample(batch_feat_seqs[:,1], grid=updated_grid.permute(0,2,3,1), mode='nearest')
        # out = F.grid_sample(batch_feat_seqs[:,1], grid=grid.permute(0,2,3,1), mode='bilinear')
        # out = batch_feat_seqs[:,1]
        if self.dcn:
            out = self.dcn_net(out)

        flow_dict = {'flow':flow, 'flow_unc': flow_unc}
        return flow_dict, state_class_pred, out

    def forward_debug(self, x, origin_x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding
        Used for debug and visualization


        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        origin_x: torch.Tensor
            pillars (sum(n_cav), C, H * downsample_rate, W * downsample_rate)

        record_len : list
            shape: (B)

        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego,
            shape: (B, L, L, 4, 4)

        Returns
        -------
        Fused feature.
        """
        from matplotlib import pyplot as plt

        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = self.regroup(x, record_len)
        split_origin_x = self.regroup(origin_x, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
                    self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
                    self.downsample_rate * self.discrete_ratio * H) * 2

        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L, 1, H, W)).to(x)
                roi_mask[b, i] = warp_affine_simple(one_tensor, pairwise_t_matrix[b][i, :, :, :], (H, W))

        batch_node_features = split_x
        # iteratively update the features for num_iteration times

        # visualize warped feature map
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            i = 0  # ego
            mask = roi_mask[b, i, :N, ...]
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                  t_matrix[i, :, :, :],
                                                  (H, W))
            for idx in range(N):
                plt.imshow(torch.max(neighbor_feature[idx], 0)[0].detach().cpu().numpy())
                plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/feature_{b}_{idx}")
                plt.clf()
                plt.imshow(mask[idx][0].detach().cpu().numpy())
                plt.savefig(
                    f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/mask_feature_{b}_{idx}")
                plt.clf()

        # visualize origin pillar feature
        origin_node_features = split_origin_x

        for b in range(B):
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            i = 0  # ego
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(origin_node_features[b],
                                                  t_matrix[i, :, :, :],
                                                  (H * self.downsample_rate, W * self.downsample_rate))

            for idx in range(N):
                plt.imshow(torch.max(neighbor_feature[idx], 0)[0].detach().cpu().numpy())
                plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/origin_{b}_{idx}")
                plt.clf()