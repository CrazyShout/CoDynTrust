from turtle import forward
import torch
from torch import nn
import spconv
import spconv.pytorch
from spconv.pytorch import SparseSequential, SparseConv2d

from opencood.models.sub_modules.sparse_conv import SparseConvBlock, SparseBasicBlock
from opencood.models.sub_modules.aspp import ASPPNeck


class SparseResNet(spconv.pytorch.SparseModule):
    def __init__(
            self,
            layer_nums = [3, 4, 5], # [2, 2, 2, 2]
            ds_layer_strides = [2, 2, 2], # [1, 2, 2, 2]
            ds_num_filters = [128, 256, 512], # [64, 128, 256, 256]
            num_input_features = 64, # 64
            kernel_size=[3, 3, 3],
            out_channels=256):

        super(SparseResNet, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features # 64

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                kernel_size[i],
                self._layer_strides[i],
                layer_num)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        # self.mapping = SparseSequential(
        #     SparseConv2d(self._num_filters[-1],
        #                  out_channels, 1, 1, bias=False),
        #     nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        # )

    def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

        layers = []
        layers.append(SparseConvBlock(inplanes, planes,
                      kernel_size=kernel_size, stride=stride, use_subm=False))

        for j in range(num_blocks):
            layers.append(SparseBasicBlock(planes, kernel_size=kernel_size))

        return spconv.pytorch.SparseSequential(*layers)

    def forward(self, pseudo_image, return_interm: bool = True):
        batch_size, channels, height, width = pseudo_image.shape
        nonzero_indices = torch.nonzero(pseudo_image.sum(dim=1), as_tuple=False)  # (num_nonzero_pillars, 3)
        # 为非零柱子构建 indices
        num_nonzero_pillars = nonzero_indices.shape[0]
        batch_indices = nonzero_indices[:, 0].unsqueeze(-1)
        # z_indices = torch.zeros_like(batch_indices)
        y_indices = nonzero_indices[:, 1].unsqueeze(-1)
        x_indices = nonzero_indices[:, 2].unsqueeze(-1)
        indices = torch.cat([batch_indices, y_indices, x_indices], dim=1)  # (num_nonzero_pillars, 4)

        # 获取非零柱子的特征值
        values = pseudo_image[batch_indices, :, y_indices, x_indices].view(num_nonzero_pillars, channels)

        spatial_shape = [height, width]  # 对于 2D 卷积，只需要 height 和 width
        # print("Indices shape:", indices.shape)
        # print("Values shape:", values.shape)
        # print("Spatial shape:", spatial_shape)
        interm_features = []
        x = spconv.pytorch.SparseConvTensor(
            values, indices.int(), spatial_shape, batch_size)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            interm_features.append(x.dense())

        if return_interm:
            return interm_features

        # x = self.mapping(x)
        return x.dense()


# class SparseResNet(spconv.pytorch.SparseModule):
#     def __init__(
#             self,
#             layer_nums = [2, 2, 2, 2], # [2, 2, 2, 2]
#             ds_layer_strides = [1, 2, 2, 2], # [1, 2, 2, 2]
#             ds_num_filters = [64, 128, 256, 256], # [64, 128, 256, 256]
#             num_input_features = 64, # 64
#             kernel_size=[3, 3, 3, 3],
#             out_channels=256):

#         super(SparseResNet, self).__init__()
#         self._layer_strides = ds_layer_strides
#         self._num_filters = ds_num_filters
#         self._layer_nums = layer_nums
#         self._num_input_features = num_input_features # 64

#         assert len(self._layer_strides) == len(self._layer_nums)
#         assert len(self._num_filters) == len(self._layer_nums)

#         in_filters = [self._num_input_features, *self._num_filters[:-1]]
#         blocks = []

#         for i, layer_num in enumerate(self._layer_nums):
#             block = self._make_layer(
#                 in_filters[i],
#                 self._num_filters[i],
#                 kernel_size[i],
#                 self._layer_strides[i],
#                 layer_num)
#             blocks.append(block)

#         self.blocks = nn.ModuleList(blocks)

#         self.mapping = SparseSequential(
#             SparseConv2d(self._num_filters[-1],
#                          out_channels, 1, 1, bias=False),
#             nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
#             nn.ReLU(),
#         )

#     def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

#         layers = []
#         layers.append(SparseConvBlock(inplanes, planes,
#                       kernel_size=kernel_size, stride=stride, use_subm=False))

#         for j in range(num_blocks):
#             layers.append(SparseBasicBlock(planes, kernel_size=kernel_size))

#         return spconv.pytorch.SparseSequential(*layers)

#     def forward(self, pseudo_image):
#         batch_size, channels, height, width = pseudo_image.shape
#         nonzero_indices = torch.nonzero(pseudo_image.sum(dim=1), as_tuple=False)  # (num_nonzero_pillars, 3)
#         # 为非零柱子构建 indices
#         num_nonzero_pillars = nonzero_indices.shape[0]
#         batch_indices = nonzero_indices[:, 0].unsqueeze(-1)
#         # z_indices = torch.zeros_like(batch_indices)
#         y_indices = nonzero_indices[:, 1].unsqueeze(-1)
#         x_indices = nonzero_indices[:, 2].unsqueeze(-1)
#         indices = torch.cat([batch_indices, y_indices, x_indices], dim=1)  # (num_nonzero_pillars, 4)

#         # 获取非零柱子的特征值
#         values = pseudo_image[batch_indices, :, y_indices, x_indices].view(num_nonzero_pillars, channels)

#         spatial_shape = [height, width]  # 对于 2D 卷积，只需要 height 和 width
#         # print("Indices shape:", indices.shape)
#         # print("Values shape:", values.shape)
#         # print("Spatial shape:", spatial_shape)
#         x = spconv.pytorch.SparseConvTensor(
#             values, indices.int(), spatial_shape, batch_size)
#         for i in range(len(self.blocks)):
#             x = self.blocks[i](x)
#         x = self.mapping(x)
#         return x.dense()

    # def forward(self, pillar_features, coors, input_shape):
    #     batch_size = len(torch.unique(coors[:, 0]))
    #     x = spconv.pytorch.SparseConvTensor(
    #         pillar_features, coors, input_shape, batch_size)
    #     for i in range(len(self.blocks)):
    #         x = self.blocks[i](x)
    #     x = self.mapping(x)
    #     return x.dense()

class UpsamplePixelShuffle(nn.Module):
    def __init__(self, in_channels, upscale_factor=4):
        super(UpsamplePixelShuffle, self).__init__()
        # 卷积层将通道数增加 upscale_factor^2 倍
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        # 批归一化层
        self.bn = nn.BatchNorm2d(in_channels * (upscale_factor ** 2))
        # ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)
        # PixelShuffle 操作将通道数重新排列，得到上采样后的特征图
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class UpsampleTransposedConv(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleTransposedConv, self).__init__()
        # 转置卷积层，卷积核大小设为4，步幅设为4，填充设为0
        self.trans_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=4, padding=0)
        # 批量归一化层
        self.bn = nn.BatchNorm2d(in_channels)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Sparse_resnet_backbone_aspp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = SparseResNet()
        # self.neck = ASPPNeck(256)
        # self.deblocks = nn.ModuleList()
        # num_levels = 3
        # upsample_strides = [1, 2, 4]
        # num_filters = [128, 256, 512]
        # num_upsample_filters = [128, 128, 128]
        # for idx in range(num_levels):
        #     if len(upsample_strides) > 0:
        #         stride = upsample_strides[idx]
        #         if stride >= 1:
        #             self.deblocks.append(nn.Sequential(
        #                 nn.ConvTranspose2d(
        #                     num_filters[idx], num_upsample_filters[idx],
        #                     upsample_strides[idx],
        #                     stride=upsample_strides[idx], bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[idx],
        #                                eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))
        #         else:
        #             stride = np.round(1 / stride).astype(np.int)
        #             self.deblocks.append(nn.Sequential(
        #                 nn.Conv2d(
        #                     num_filters[idx], num_upsample_filters[idx],
        #                     stride,
        #                     stride=stride, bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
        #                                momentum=0.01),
        #                 nn.ReLU()
        #             ))
        # self.upsample = UpsampleTransposedConv(384)

    def forward(self, data_dict):

        
        spatial_features = data_dict['spatial_features'] # (batch_size, feature, H, W)

        out = self.resnet(spatial_features)
        # out = self.upsample(out)

        data_dict['spatial_features_2d'] = out

        return data_dict
        # spatial_features = data_dict['spatial_features'] # (batch_size, feature, H, W)

        # out = self.resnet(spatial_features)
        # out = self.neck(out)
        # out = self.upsample(out)

        # data_dict['spatial_features_2d'] = out

        # return data_dict

