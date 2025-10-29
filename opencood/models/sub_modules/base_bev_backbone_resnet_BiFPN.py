"""
Resblock is much strong than normal conv

Provide api for multiscale intermeidate fuion
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride= 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, up_stride):
        super(UpSampleBlock, self).__init__()
        self.up_conv = nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels, out_channels,
                            kernel_size=kernel_size,
                            stride=up_stride, bias=False
                        ),
                        nn.BatchNorm2d(out_channels,
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    )

    def forward(self, x):
        return self.up_conv(x)

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

class BiFPNLayer(nn.Module):
    def __init__(self):
        super(BiFPNLayer, self).__init__()

        self.conv6_up = ConvBlock(128, 128)
        self.conv5_up = ConvBlock(256, 256)

        self.conv4_down = ConvBlock(128, 128)
        self.conv5_down = ConvBlock(128, 128)
        self.conv6_down = ConvBlock(128, 128)

        self.fusion_up_5 = FastFusion(256, 2)
        self.fusion_up_6 = FastFusion(128, 2)
        self.fusion_down_4 = FastFusion(128, 2)
        self.fusion_down_5 = FastFusion(128, 3)
        self.fusion_down_6 = FastFusion(128, 2)


        self.up_conv5 = UpSampleBlock(512, 256, kernel_size=2, up_stride=2)
        self.up_conv6 = UpSampleBlock(256, 128, kernel_size=2, up_stride=2)

        self.lateral_conv5 = UpSampleBlock(256, 128, kernel_size=2, up_stride=2)

        self.skip_conv5 = UpSampleBlock(256, 128, kernel_size=2, up_stride=2)
        self.skip_conv4 = UpSampleBlock(512, 128, kernel_size=4, up_stride=4)
        


    def forward(self, P3, P4, P5):
        # P3: 2C, H/2, W/2
        # P4: 4C, H/4, W/4
        # P5: 8C, H/8, W/8
        P6_0 = P3
        P5_0 = P4
        P4_0 = P5

        # Top-down pathway
        P4_up = P4_0

        P5_up = self.conv5_up(self.fusion_up_5(P5_0, self.up_conv5(P4_up)))
        P6_up = self.conv6_up(self.fusion_up_6(P6_0, self.up_conv6(P5_up)))


        # Bottom-up pathway
        P6_down = self.conv6_down(self.fusion_down_6(P6_up, P6_0))
        P5_down = self.conv5_down(self.fusion_down_5(P6_down, self.lateral_conv5(P5_up), self.skip_conv5(P5_0)))
        P4_down = self.conv4_down(self.fusion_down_4(P5_down, self.skip_conv4(P4_0)))

        return P6_down, P5_down, P4_down # 三个都是 N，128，H，W


class ResNetBEVBackbone_BiFPN(nn.Module):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__()
        print("===使用BiFPN===")
        self.model_cfg = model_cfg

        self.use_dropout = model_cfg.get('use_dropout', False)
        self.enable_dropout = model_cfg.get('dropout_enable', False)
        if self.use_dropout:
            print("===backbone use dropout===")
            if self.enable_dropout:
                print("  --enforce enable dropout with F.Dropout2d")

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums'] # [3, 4, 5]
            layer_strides = self.model_cfg['layer_strides'] # [2, 2, 2]
            num_filters = self.model_cfg['num_filters'] # [128, 256, 512]
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter'] # [128, 128, 128]
            upsample_strides = self.model_cfg['upsample_strides'] # [1, 2, 4]

        else:
            upsample_strides = num_upsample_filters = []

        self.resnet = ResNetModified(BasicBlock, 
                                        layer_nums,
                                        layer_strides,
                                        num_filters,
                                        inplanes = model_cfg.get('inplanes', 64))

        self.bi_fpn = BiFPNLayer() # TODO 需要将yaml中的参数传递进去

        # num_levels = len(layer_nums)
        # self.num_levels = len(layer_nums)
        # self.deblocks = nn.ModuleList()

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

        # c_in = sum(num_upsample_filters)
        # if len(upsample_strides) > num_levels:
        #     self.deblocks.append(nn.Sequential(
        #         nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
        #                            stride=upsample_strides[-1], bias=False),
        #         nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
        #         nn.ReLU(),
        #     ))

        # self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        x = self.resnet(spatial_features)  # tuple of features

        P6_down, P5_down, P4_down = self.bi_fpn(x[0], x[1], x[2])
        out = torch.cat((P6_down, P5_down, P4_down), dim = 1) # N, 384, H, W
        # out_w_scale = torch.stack((P6_down, P5_down, P4_down), dim = 1) # N, 3, 128, H, W

        data_dict['spatial_features_2d'] = out # N, 384, H, W
        return data_dict
    
        ups = []
        
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                if self.use_dropout:
                    if self.enable_dropout:
                        ups.append(F.dropout2d(self.deblocks[i](x[i]), p=0.1, training = True))
                    else:
                        ups.append(F.dropout2d(self.deblocks[i](x[i]), p=0.1, training = self.training))
                else:
                    ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict

    # these two functions are seperated for multiscale intermediate fusion
    def get_multiscale_feature(self, spatial_features):
        """
        before multiscale intermediate fusion
        """
        x = self.resnet(spatial_features)  # tuple of features
        return x

    def decode_multiscale_feature(self, x):
        """
        after multiscale interemediate fusion
        """
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x
        
    def get_layer_i_feature(self, spatial_features, layer_i):
        """
        before multiscale intermediate fusion
        """
        return eval(f"self.resnet.layer{layer_i}")(spatial_features)  # tuple of features
    