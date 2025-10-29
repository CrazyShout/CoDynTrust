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
from opencood.tools.matcher import Matcher
from collections import OrderedDict
import torch
import numpy as np

def generate_noise(pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use gaussian distribution to generate noise.

    Args:

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree

    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.normal(pos_mean, pos_std, size=(2))
    yaw = np.random.normal(rot_mean, rot_std, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])

    return pose_noise

def get_past_k_pairwise_transformation2ego(past_k_lidar_pose, noise_level, k=3, max_cav=5):
    """
    Get transformation matrixes accross different agents to curr ego at all past timestamps.

    Parameters
    ----------
    base_data_dict : dict
        Key : cav id, item: transformation matrix to ego, lidar points.
    
    ego_pose : list
        ego pose

    max_cav : int
        The maximum number of cav, default 5

    Return
    ------
    pairwise_t_matrix : np.array
        The transformation matrix each cav to curr ego at past k frames.
        shape: (L, k, 4, 4), L is the max cav number in a scene, k is the num of past frames
        pairwise_t_matrix[i, j] is T i_to_ego at past_j frame
    """
    pos_std = noise_level['pos_std']
    rot_std = noise_level['rot_std']
    pos_mean = 0 
    rot_mean = 0
    
    pairwise_t_matrix = np.tile(np.eye(4), (max_cav, k, 1, 1)) # (L, k, 4, 4)

    ego_pose = past_k_lidar_pose[0, 0]

    t_list = []

    # save all transformation matrix in a list in order first.
    for cav_id in range(past_k_lidar_pose.shape[0]):
        past_k_poses = []
        for time_id in range(k):
            loc_noise = generate_noise(pos_std, rot_std)
            past_k_poses.append(x_to_world(past_k_lidar_pose[cav_id, time_id].cpu().numpy()+loc_noise))
        t_list.append(past_k_poses) # Twx
    
    ego_pose = x_to_world(ego_pose.cpu().numpy())
    for i in range(len(t_list)): # different cav
        if i!=0 :
            for j in range(len(t_list[i])): # different time
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                t_matrix = np.linalg.solve(t_list[i][j], ego_pose)  # Tjw*Twi = Tji
                pairwise_t_matrix[i, j] = t_matrix
    pairwise_t_matrix = torch.tensor(pairwise_t_matrix).to(past_k_lidar_pose.device)
    return pairwise_t_matrix


class PointPillarSingleXyj(nn.Module):
    def __init__(self, args):
        super(PointPillarSingleXyj, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = MaxResNetBEVBackbone(args['base_bev_backbone'], 64)
        if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        # if args['compression'] > 0:
        #     self.compression = True
        #     self.naive_compressor = NaiveCompressor(256, args['compression'])

 
        if self.shrink_flag:
            dim = args['shrink_header']['dim'][0]
            self.cls_head = nn.Conv2d(int(dim), args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(int(dim), 7 * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                    kernel_size=1)

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                    kernel_size=1) # BIN_NUM = 2， # 384
            # self.fused_dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'], # num_bins 设置成多种桶
            #                         kernel_size=1)
        else:
            self.use_dir = False


        self.backbone_fix_flag = False
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix_flag = True
            self.backbone_fix()
            print('=== backbone fixed ===')

        self.only_tune_header_flag = False
        if 'only_tune_header' in args.keys() and args['only_tune_header']:
            self.only_tune_header_flag = True
            self.only_tune_header()
            print('=== only tune header ===')

        self.viz_bbx_flag = False
        if 'viz_bbx_flag' in args.keys() and args['viz_bbx_flag']:
            self.viz_bbx_flag = True
        
        assert self.backbone_fix_flag == False or self.only_tune_header_flag == False, 'backbone_fix and only_tune_header cannot be True at the same time'
    
    def only_tune_header(self):
        """
        Fix the parameters of backbone during finetune on timedelay
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.rain_fusion.parameters():
            p.requires_grad = False

        for p in self.matcher.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    

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
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)


        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d) # (B, 2, H', W')
        rm_single = self.reg_head(spatial_features_2d) # (B, 14, H', W')

        output_dict = {'psm_single': psm_single,
                       'rm_single': rm_single}

        # print("spatial_features_2d shape is :", spatial_features_2d.shape)
        # print("psm_single shape is :", psm_single.shape)
        # print("rm_single shape is :", rm_single.shape)
        # print("self.use_dir is :", self.use_dir)
        # print("self.design_mode is :", self.design_mode)
        
        if self.use_dir:
            dm_single = self.dir_head(spatial_features_2d)
            output_dict.update({'dm_single': dm_single})

        return output_dict
