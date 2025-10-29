from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.fuse_modules.raindrop_swin import raindrop_swin
from opencood.models.fuse_modules.raindrop_swin_w_single import raindrop_swin_w_single
from opencood.models.fuse_modules.raindrop_flow import raindrop_fuse
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
from opencood.utils.model_utils import weight_init
from opencood.tools.matcher import Matcher
from collections import OrderedDict
import torch
import numpy as np


class PointPillarSingleUncertainty(nn.Module):
    def __init__(self, args):
        super(PointPillarSingleUncertainty, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", True) # default true
        if is_resnet:
            print("===use ResNet as backbone!==")
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            print("===use downsample conv to reduce memory===")
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.uncertainty_dim = args['uncertainty_dim'] # dim=3 means x, y, yaw, dim=2 means x, y

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                kernel_size=1)

        self.unc_head = nn.Conv2d(self.out_channel, self.uncertainty_dim * args['anchor_number'],
                                    kernel_size=1)

        self.re_parameterization = args.get('re_parameterization', False)

        if self.re_parameterization is True:
            print("===re-parameterization trick==")
            self.unc_head_cls = nn.Conv2d(128 * 2, args['anchor_number'],
                                        kernel_size=1)

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                    kernel_size=1) # BIN_NUM = 2， # 384
        else:
            self.use_dir = False

        self.inference_state = False
        self.inference_num = 0
        self.simple_dropout = False
        if 'mc_dropout' in args.keys():
            print("===use dropout to regulate output! dropout rate: %.2f==="% (args['mc_dropout']['dropout_rate']))
            if self.simple_dropout:
                self.feature_dropout = nn.Dropout2d(args['mc_dropout']['dropout_rate'])
            if 'inference_stage' in args['mc_dropout'].keys():
                if args['mc_dropout']['inference_stage'] is True:
                    self.inference_state = True
                    self.inference_num = args['mc_dropout']['inference_num']
                    print("===Use MC Dropout! infer %d times!=="%(self.inference_num))

        self.apply(weight_init)

    def forward(self, data_dict, dataset=None):
        voxel_features = data_dict['processed_lidar']['voxel_features']         #(M, 32, 4) 一个batch下所有cav的体素
        voxel_coords = data_dict['processed_lidar']['voxel_coords']             #(M, 4)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']     #(M, )
        record_len = data_dict['record_len']                                    #(B, )
        record_frames = data_dict['past_k_time_interval']                       #(sum(n_cav) ) batch中所有cav中每一帧到cur的时间间隔
        pairwise_t_matrix = data_dict['pairwise_t_matrix']                      #(B, L, k=1, 4, 4)
        
        # print('voxel_features shape is:', voxel_features.shape)
        # print('voxel_coords shape is:', voxel_coords.shape)
        # print('voxel_num_points shape is:', voxel_num_points.shape)
        # print('record_len shape is:', record_len.shape)
        # print('pairwise_t_matrix shape is:', pairwise_t_matrix.shape)



        # debug = 0
        B, _, k, _, _ = pairwise_t_matrix.shape
        # for i in range(B):
        #     debug += record_len[i]*k

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len} # (B) 每个场景下的车
        # n, 4 -> n, c  ('pillar_features')
        batch_dict = self.pillar_vfe(batch_dict)
        # (n, c) -> (batch_cav_size, C, H, W) put pillars into spatial feature map ('spatial_features')
        # import ipdb; ipdb.set_trace()
        batch_dict = self.scatter(batch_dict) # N_b, 64, H, W
        batch_dict = self.backbone(batch_dict) # 'spatial_features_2d': (batch_cav_size, 128*3, H/2, W/2)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d'] # 特征提取后的结果
        
        # print("spatial_features shape is :", batch_dict['spatial_features'].shape)
        # print("spatial_features_2d shape is :", spatial_features_2d.shape)

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')

        # [B, 256, 50, 176]
        cls_preds = self.cls_head(spatial_features_2d) # (B, 2, H', W')
        reg_preds = self.reg_head(spatial_features_2d) # (B, 14, H', W')

        unc_preds = self.unc_head(spatial_features_2d) # s is log(b) or log(sigma^2)
        # re-parametrization trick, or var of classification logit is very difficulty to learn -- xyj 2024/5/27
        if self.re_parameterization is True and self.training: # 训练阶段是需要添加噪声的，但是验证阶段以及推理阶段则不需要
            unc_cls_log_var = self.unc_head_cls(spatial_features_2d) # (N, 2, H, W)
            unc_cls_log_var = torch.exp(unc_cls_log_var) # 得到方差
            unc_cls_log_var = torch.sqrt(unc_cls_log_var) # 得到标准差
            epsilon = torch.randn_like(unc_cls_log_var).to(unc_cls_log_var.device) # sample
            cls_preds = cls_preds + epsilon * unc_cls_log_var

        output_dict = {'cls_preds': cls_preds,
                       'reg_preds': reg_preds,
                       'unc_preds': unc_preds}
        
        if self.use_dir:
            dir_preds = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dir_preds})

        # MC Dropout    
        if self.inference_state is True:
            # MC Dropout
            B,_,H0,W0 = cls_preds.shape
            cls_preds_ntimes_tensor = torch.zeros_like(cls_preds, dtype=cls_preds.dtype, device=cls_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            reg_preds_ntimes_tensor = torch.zeros_like(reg_preds, dtype=reg_preds.dtype, device=reg_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            unc_preds_ntimes_tensor = torch.zeros_like(unc_preds, dtype=unc_preds.dtype, device=unc_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            cls_preds_ntimes_tensor[0] = cls_preds
            reg_preds_ntimes_tensor[0] = reg_preds
            unc_preds_ntimes_tensor[0] = torch.exp(unc_preds) # 变回方差
            if self.re_parameterization is True:
                cls_noise_ntimes_tensor = torch.zeros_like(unc_cls_log_var, dtype=unc_cls_log_var.dtype, device=unc_cls_log_var.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
                cls_noise_ntimes_tensor[0] = torch.exp(unc_cls_log_var) # 这是已经被处理过的标准差 要变回到方差 需要平方
                cls_noise_ntimes_tensor[0] = torch.square(cls_noise_ntimes_tensor[0]) # 这是已经被处理过的标准差 要变回到方差 需要平方

            debug_flag = False
            
            if self.use_dir:
                dir_preds_ntimes_tensor = torch.zeros_like(dir_preds, dtype=dir_preds.dtype, device=dir_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
                dir_preds_ntimes_tensor[0] = dir_preds

            if debug_flag:
                print(f"cls_preds_ntimes_tensor shape is {cls_preds_ntimes_tensor.shape}") # torch.Size([10, 1, 2, 100, 176])
                print(f"reg_preds_ntimes_tensor shape is {reg_preds_ntimes_tensor.shape}") # torch.Size([10, 1, 14, 100, 176])
                print(f"dir_preds_ntimes_tensor shape is {dir_preds_ntimes_tensor.shape}") # torch.Size([10, 1, 4, 100, 176])
                print(f"unc_preds_ntimes_tensor shape is {unc_preds_ntimes_tensor.shape}") # torch.Size([10, 1, 6, 100, 176])
                
            for i in range(1, self.inference_num):
                # spatial_features_2d_temp = self.feature_dropout(batch_dict['spatial_features_2d'])
                # cls_preds_ntimes_tensor[i] = self.cls_head(spatial_features_2d_temp)
                # reg_preds_ntimes_tensor[i] = self.reg_head(spatial_features_2d_temp)
                # unc_preds_ntimes_tensor[i] = self.unc_head(spatial_features_2d_temp)
                # if self.use_dir:
                #     dir_preds_ntimes_tensor[i] = self.dir_head(spatial_features_2d_temp)
                # # print(f"reg_preds_ntimes_tensor[{i}] is {reg_preds_ntimes_tensor[i][0,:,0,0]}") # torch.Size([1, 2, 100, 176])
                # del spatial_features_2d_temp

                batch_dict = self.backbone(batch_dict)

                spatial_features_2d = batch_dict['spatial_features_2d']

                # downsample feature to reduce memory
                if self.shrink_flag:
                    spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')
                cls_preds_ntimes_tensor[i] = self.cls_head(spatial_features_2d)
                reg_preds_ntimes_tensor[i] = self.reg_head(spatial_features_2d)
                
                if self.use_dir:
                    dir_preds_ntimes_tensor[i] = self.dir_head(spatial_features_2d)
                del spatial_features_2d

                unc_preds_ntimes_tensor[i] = self.unc_head(spatial_features_2d) # s is log(b) or log(sigma^2)
                unc_preds_ntimes_tensor[i] = torch.exp(unc_preds_ntimes_tensor[i]) # 方差作为回归噪声
                if self.re_parameterization is True:
                    cls_noise_ntimes_tensor[i] = self.unc_head_cls(spatial_features_2d)
                    cls_noise_ntimes_tensor[i] = torch.exp(cls_noise_ntimes_tensor[i]) # 方差作为分类噪声
                    cls_noise_ntimes_tensor[i] = torch.sqrt(cls_noise_ntimes_tensor[i]) # 标准差

            cls_preds_mean = torch.mean(cls_preds_ntimes_tensor, dim=0)
            reg_preds_mean = torch.mean(reg_preds_ntimes_tensor, dim=0) # (1, 14, H, W)
            unc_preds_mean = torch.mean(unc_preds_ntimes_tensor, dim=0) # (1, 14, H, W)
            if self.re_parameterization is True:
                cls_noise_mean = torch.mean(cls_noise_ntimes_tensor, dim=0)

            if self.use_dir:
                dir_preds_mean = torch.mean(dir_preds_ntimes_tensor, dim=0) 
            reg_preds_var = torch.var(reg_preds_ntimes_tensor, dim=0) # (1, 14, H, W)

            if debug_flag:
                print("===============mean value===============")
                # print(f"reg_preds_mean is {reg_preds_mean[0,:,0,0]}")
                print(f"cls_preds_mean shape is {cls_preds_mean.shape}") # torch.Size([1, 2, 100, 176])
                print(f"reg_preds_mean shape is {reg_preds_mean.shape}") # torch.Size([1, 14, 100, 176])
                print(f"unc_preds_mean shape is {unc_preds_mean.shape}") # torch.Size([1, 6, 100, 176])
                print(f"dir_preds_mean shape is {dir_preds_mean.shape}") # torch.Size([1, 4, 100, 176])
                print(f"reg_preds_var shape is {reg_preds_var.shape}") # torch.Size([1, 14, 100, 176])
                # print(f"reg_preds_var  is {reg_preds_var}") # torch.Size([1, 14, 100, 176])
                
            # 模型分类不确定性  衡量每个anchor
            cls_score = torch.sigmoid(cls_preds_mean)

            if self.re_parameterization is True:
                cls_noise = calc_deviation_ratio(cls_noise_mean, cls_score, tp_cls_mean = 0.0295, tp_cls_std = 0.0175, tp_score_mean = 0.6348, tp_score_std = 0.1810) # 计算分类偏差比

            # 计算分类分数的对数
            log_cls_score = torch.log(cls_score)
            log_1_cls_score = torch.log(1 - cls_score)

            # 计算熵 除以 log(2) 以将结果从 nats 转换为 bits
            unc_epi_cls = -(cls_score * log_cls_score + (1 - cls_score) * log_1_cls_score) / torch.log(torch.tensor(2.0, device=cls_preds.device))           
            # 将熵的张量转换为 cls_preds 的设备
            unc_epi_cls = unc_epi_cls.to(cls_score.device)
            # unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score) # 计算分类偏差比
            # unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score, tp_cls_mean = 0.8201, tp_cls_std = 0.1749, tp_score_mean = 0.6484, tp_score_std = 0.1872) # 计算分类偏差比
            unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score, tp_cls_mean = 0.8417, tp_cls_std = 0.1547, tp_score_mean = 0.6345, tp_score_std = 0.1808) # 计算分类偏差比

            # 模型回归不确定性
            d_a_square = 1.6**2 + 3.9**2 # anchor的长宽平方和
            reg_preds_var = reg_preds_var.permute(0,2,3,1).reshape(-1,7) # (2HW, 7) 
            reg_preds_var[:,:2] *= d_a_square
            unc_epi_reg = torch.sqrt(reg_preds_var)
            unc_epi_reg = unc_epi_reg[:,0] + unc_epi_reg[:,1] + unc_epi_reg[:,6]
            unc_epi_reg = unc_epi_reg.reshape(B, -1, H0, W0)

            if debug_flag:
                print("===============uncertainty epistemic value===============")
                print(f"unc_epi_cls shape is {unc_epi_cls.shape}") # torch.Size([1, 2, 100, 176])
                print(f"unc_epi_reg shape is {unc_epi_reg.shape}") # torch.Size([1, 2, 100, 176])
                # print(f"unc_epi_cls  is {unc_epi_cls}") # torch.Size([1, 14, 100, 176])
                # print(f"unc_epi_reg  is {unc_epi_reg}") # torch.Size([1, 14, 100, 176])
                                
            output_dict.update({'cls_preds': cls_preds_mean,
                        'reg_preds': reg_preds_mean,
                        'unc_preds': unc_preds_mean,
                        'cls_noise': cls_noise,
                        'dir_preds': dir_preds_mean,
                        'unc_epi_cls': unc_epi_cls,
                        'unc_epi_reg': unc_epi_reg})
        return output_dict
