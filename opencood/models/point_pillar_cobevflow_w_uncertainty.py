from re import match
from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
# from opencood.models.sub_modules.base_bev_backbone_resnet_BiFPN import ResNetBEVBackbone_BiFPN
# from opencood.models.sub_modules.sparse_resnet import Sparse_resnet_backbone_aspp
from opencood.models.sub_modules.downsample_conv import DownsampleConv, AttentionDownsampleConv
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

def calc_dist_uncert(x, x_mean, x_std):
    
    # return torch.clamp((x - x_mean - x_std), min=0, max=1)
    return torch.relu(x - x_mean - x_std)

def calc_dist_score(x, x_mean, x_std):
    
    # return torch.abs(x - min(1, x_mean + x_std))
    return torch.relu(min(1, x_mean + x_std) - x)

def calc_deviation_ratio(test_cls, test_score, tp_cls_mean = 0.8355, tp_cls_std = 0.1638, tp_score_mean = 0.6425, tp_score_std = 0.1794):

    dr_uncert = tp_cls_mean / (calc_dist_uncert(test_cls, tp_cls_mean, tp_cls_std) + tp_cls_mean)
    dr_score = tp_score_mean / (calc_dist_score(test_score, tp_score_mean, tp_score_std) + tp_score_mean)
    dr = dr_uncert * dr_score

    return dr

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


class PointPillarCobevflowWUncertainty(nn.Module):
    def __init__(self, args):
        super(PointPillarCobevflowWUncertainty, self).__init__()
        print("===train cobevflow with uncertainty!===")
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = MaxResNetBEVBackbone(args['base_bev_backbone'], 64)
        if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
            print("===use ResNet as backbone!==")
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
            self.fused_backbone = ResNetBEVBackbone(args['fuse_bev_backbone'], 64)           
            # self.fused_backbone = Sparse_resnet_backbone_aspp()            
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_conv = None
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            print("===use downsample conv to reduce memory===")
            self.shrink_flag = True
            if args['shrink_header']['use_atten']:
                print("  ===AttentionDownsampleConv===")
                self.shrink_conv = AttentionDownsampleConv(args['shrink_header'])
                self.fused_shrink_conv = AttentionDownsampleConv(args['shrink_header'])
            else:
                self.shrink_conv = DownsampleConv(args['shrink_header'])
                self.fused_shrink_conv = DownsampleConv(args['shrink_header'])

            self.out_channel = args['shrink_header']['dim'][-1]
        self.compression = False

        self.uncertainty_dim = args['uncertainty_dim'] # dim=3 means x, y, yaw, dim=2 means x, y

        # if args['compression'] > 0:
        #     self.compression = True
        #     self.naive_compressor = NaiveCompressor(256, args['compression'])

        if 'num_sweep_frames' in args:    # number of frames we use in LSTM
            self.k = args['num_sweep_frames']
        else:
            self.k = 0

        if 'time_delay' in args:          # number of time delay
            self.tau = args['time_delay'] 
        else:
            self.tau = 0

        self.dcn = False
        # if 'dcn' in args:
        #     self.dcn = True
        #     self.dcn_net = DCNNet(args['dcn'])

        self.design_mode = 0
        if 'design_mode' in args.keys():
            self.design_mode = args['design_mode']
            print(f'=== design mode : {self.design_mode} ===')

        self.noise_flag = 0
        if 'noise' in args.keys():
            self.noise_flag = True
            self.noise_level = {'pos_std': args['noise']['pos_std'], 'rot_std': args['noise']['rot_std'], 'pos_mean': args['noise']['pos_mean'], 'rot_mean': args['noise']['rot_mean']}
            print(f'=== noise level : {self.noise_level} ===')

        self.num_roi_thres = -1
        if 'num_roi_thres' in args.keys():
            self.num_roi_thres = args['num_roi_thres']
            print(f'=== num_roi_thres : {self.num_roi_thres} ===')

        self.single_supervise = False
        if 'with_compensation' in args and args['with_compensation']: # 如果已经有补偿
            self.compensation = True
            if 'with_single_supervise' in args and args['with_single_supervise']: # 单车监督，ROI生成？
                self.rain_fusion = raindrop_swin_w_single(args['rain_model'])
                self.single_supervise = True
            else: # 最终的融合检测器
                self.rain_fusion = raindrop_swin(args['rain_model'])
        else: 
            self.compensation = False
            self.rain_fusion = raindrop_fuse(args['rain_model'], self.design_mode) # 训练流

        self.multi_scale = args['rain_model']['multi_scale']
        

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                kernel_size=1)
        self.fused_cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                kernel_size=1)
        self.fused_reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                kernel_size=1)


        self.unc_head = nn.Conv2d(self.out_channel, self.uncertainty_dim * args['anchor_number'],
                                    kernel_size=1) 
        # 重参数化 trick
        self.re_parameterization = args.get('re_parameterization', False)

        if self.re_parameterization is True:
            print("===re-parameterization trick==")
            self.unc_head_cls = nn.Conv2d(self.out_channel, args['anchor_number'],
                                        kernel_size=1) 
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                    kernel_size=1) # BIN_NUM = 2， # 384
            self.fused_dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'], # num_bins 设置成多种桶
                                    kernel_size=1)
        else:
            self.use_dir = False

        self.inference_state = False
        self.inference_num = 0
        self.simple_dropout = False # 仅仅对输出特征头进行dropout
        if 'mc_dropout' in args.keys():
            print("===use dropout to regulate output! dropout rate: %.2f==="% (args['mc_dropout']['dropout_rate']))
            if self.simple_dropout:
                self.feature_dropout = nn.Dropout2d(args['mc_dropout']['dropout_rate'])
            if 'inference_stage' in args['mc_dropout'].keys():
                if args['mc_dropout']['inference_stage'] is True:
                    self.inference_state = True
                    self.inference_num = args['mc_dropout']['inference_num']

                    self.tp_score_mean = args['mc_dropout']['tp_score_mean']
                    self.tp_score_std = args['mc_dropout']['tp_score_std']
                    self.tp_data_ucls_mean = args['mc_dropout']['tp_data_ucls_mean']
                    self.tp_data_ucls_std = args['mc_dropout']['tp_data_ucls_std']
                    self.tp_model_ucls_mean = args['mc_dropout']['tp_model_ucls_mean']
                    self.tp_model_ucls_std = args['mc_dropout']['tp_model_ucls_std']
                    self.data_ureg_mean = args['mc_dropout']['data_ureg_mean']
                    self.data_ureg_std = args['mc_dropout']['data_ureg_std']
                    self.model_ureg_mean = args['mc_dropout']['model_ureg_mean']
                    self.model_ureg_std = args['mc_dropout']['model_ureg_std']
                    
                    self.dairv2x = args['mc_dropout'].get("dairv2x", False)
                    if self.dairv2x:
                        self.anchor_l = 4.5
                        self.anchor_w = 2
                    else:
                        self.anchor_l = 3.9
                        self.anchor_w = 1.6
                    print(f"===回归不确定性使用的缩放因子: l={self.anchor_l} w={self.anchor_w}===")
                    print("===Use MC Dropout! infer %d times!=="%(self.inference_num))
                    
        if self.design_mode == 0:
            if 'matcher' in args.keys():
                self.matcher = Matcher('flow_uncertainty', args=args['matcher'])
            else:
                self.matcher = Matcher('flow_uncertainty')
        else:
            self.matcher = Matcher('linear')

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
            print("===补偿结果可视化===")
            self.viz_bbx_flag = True
        
        assert self.backbone_fix_flag == False or self.only_tune_header_flag == False, 'backbone_fix and only_tune_header cannot be True at the same time'

        # self._initialize_weights() # 初始化head的参数

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fused_cls_head.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.fused_reg_head.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.fused_dir_head.weight, mode='fan_out', nonlinearity='linear')
        if self.fused_cls_head.bias is not None:
            nn.init.constant_(self.fused_cls_head.bias, 0)
        if self.fused_reg_head.bias is not None:
            nn.init.constant_(self.fused_reg_head.bias, 0)
        if self.fused_dir_head.bias is not None:
            nn.init.constant_(self.fused_dir_head.bias, 0)

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

        for p in self.fused_backbone.parameters():
            p.requires_grad = False

        for p in self.rain_fusion.parameters():
            p.requires_grad = False

        # for p in self.matcher.parameters(): # 
        #     p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
            for p in self.fused_shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
        # for p in self.fused_cls_head.parameters():
        #     p.requires_grad = False
        # for p in self.fused_reg_head.parameters():
        #     p.requires_grad = False
        if self.use_dir:
            for p in self.dir_head.parameters():
                p.requires_grad = False
            # for p in self.fused_dir_head.parameters():
            #     p.requires_grad = False

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
        for p in self.unc_head.parameters():
            p.requires_grad = False
        if self.re_parameterization is True:
            for p in self.unc_head_cls.parameters():
                p.requires_grad = False

        if self.use_dir:
            for p in self.dir_head.parameters():
                p.requires_grad = False
    
    def regroup(self, x, record_len, k=1):
        '''
        x的形状为(B, C, H, W)
        record_len的形状为(B) 记录每一个样本场景下的agent个数
        k为保存的帧数

        '''
        cum_sum_len = torch.cumsum(record_len*k, dim=0) # 求累计和
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu()) # 分割批数据，返回的是List
        return split_x # List[p1()]

    def bandwidth_filter(self, input, num_box):
        topk_idx = torch.argsort(input['scores'], descending=True)[:num_box]

        output = {}
        output['scores'] = input['scores'][topk_idx]
        output['pred_box_3dcorner_tensor']  = input['pred_box_3dcorner_tensor'][topk_idx]
        output['pred_box_center_tensor'] = input['pred_box_center_tensor'][topk_idx]

        return output

    def generate_box_flow(self, data_dict, pred_dict, dataset, shape_list, device, pairwise_t_matrix): 
        """
        data_dict : 
        pred_dict: 单车检测结果
        dataset: 数据集对象
        pairwise_t_matrix: (B, L, k, 4, 4) 每一帧到cur ego的变换矩阵
        pred_dict : 
            {
                psm_single_list: len = B, each element's shape is (N_b x k, 2, H, W) 
                rm_single_list: len = B, each element's shape is (N_b x k, 14, H, W) 
            }
        """
        # for b in range(B):
        # 1. get box results
        lidar_pose_batch = self.regroup(data_dict['past_lidar_pose'], data_dict['record_len'], k=1) # lidar pose一般是6dof，猜测形状为(B, k, 6) 通过record_len划分为List 每个元素为(N_b, k, 6)
        past_k_time_diff = self.regroup(data_dict['past_k_time_interval'], data_dict['record_len'], k=2) # k帧的时间间隔 长度为 (B) 返回List 每个元素为(N_b x k)  其实这里就是(N_b x 3)
        anchor_box = data_dict['anchor_box'] # (H, W, 2, 7) 这是预设置的锚框
        psm_single_list = pred_dict['psm_single_list'] # (N_b x k, 2, H, W)  列表每个元素都是前面这个形状，列表的长度为B
        rm_single_list = pred_dict['rm_single_list'] # (N_b x k, 2*7, H, W)
        dm_single_list = pred_dict['dm_single_list'] # 方向预测 (N_b x k, 2*2, H, W)

        # 包含两个 predict_unc_cls ：(N_b x k, 2, H, W)  predict_unc_reg ： (N_b x k, 2, H, W)
        # predict_unc_dict = pred_dict['predict_unc'] # 预测不确定性，分为了分类不确定性和回归不确定性
        predict_unc_cls_list = pred_dict['predict_unc']['predict_unc_cls']
        predict_unc_reg_list = pred_dict['predict_unc']['predict_unc_reg']

        # H, W = psm_single_list[0].shape[-2:]
        # shape_list = torch.tensor([64, H, W]).to(device)
        
        trans_mat_pastk_2_past0_batch = []
        B = len(lidar_pose_batch)
        box_flow_map_list = []
        reserved_mask_list = []

        if self.viz_bbx_flag:
            ori_reserved_mask_list = []
            single_box_results = None
        
        # for all batches
        for b in range(B):
            box_results = OrderedDict()
            psm_single = psm_single_list[b].reshape(-1, self.k, 2, psm_single_list[b].shape[-2], psm_single_list[b].shape[-1]) # (N_b, k, 2, H, W)
            rm_single = rm_single_list[b].reshape(-1, self.k, 14, rm_single_list[b].shape[-2], rm_single_list[b].shape[-1]) # (N_b, k, 14, H, W)
            if self.use_dir:
                dm_single = dm_single_list[b].reshape(-1, self.k, 4, dm_single_list[b].shape[-2], dm_single_list[b].shape[-1]) # (N_b, k, 4, H, W)

            predict_unc_cls = predict_unc_cls_list[b].reshape(-1, self.k, 4, predict_unc_cls_list[b].shape[-2], predict_unc_cls_list[b].shape[-1]) # (N_b, k, 2, H, W)
            predict_unc_reg = predict_unc_reg_list[b].reshape(-1, self.k, 4, predict_unc_reg_list[b].shape[-2], predict_unc_reg_list[b].shape[-1])

            cav_past_k_time_diff = past_k_time_diff[b] # (N_b x k)  从列表中选择出一个样本场景，这表示了每一帧到第0帧的时间间隔  而且注意上面在regroup时，传递的k一个是1一个是3，那是因为一个的形状为(B, k, 6)，而data_dict['past_k_time_interval']的形状为B，所以他需要额外按照k的大小来继续划分，这里就看出来k在代码中默认是3，也就是每个车存储了3帧
            cav_trans_mat_pastk_2_past0 = []
            pairwise_t_matrix_batch = pairwise_t_matrix[b, ...] # 5, k, 4, 4
            # for all cavs
            '''
            box_result : dict for each cav at each time
            {
                cav_idx : {
                    'past_k_time_diff' : 时间差
                    'matrix_past0_2_cur' : 当前cav past0到ego的空间变换矩阵
                    [0] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. (n, 8, 3)
                        pred_box_center_tensor : (n, 7)
                        scores: (n, )
                        u_cls: (n, )
                        u_reg: (n, )
                    },
                    ...
                    [k-1] : { ... }
                }
            }
            '''
            comm_volum = []
            for cav_idx in range(data_dict['record_len'][b]):# 遍历一个场景下的所有车 循环次数是该场景下的cav数
                
                pairwise_t_matrix_past0_2_cur = pairwise_t_matrix_batch[:data_dict['record_len'][b], 0, :, :] # n, 4 , 4 past0到ego的变换矩阵

                # generate one cav's trans_mat_pastk_2_past0
                pastk_trans_mat_pastk_2_past0 = []
                for i in range(self.k): # 遍历k帧
                    unit_mat = x1_to_x2(lidar_pose_batch[b][cav_idx, i, :].cpu().numpy(), lidar_pose_batch[b][cav_idx, 0].cpu()) # (4, 4) 一个agent的第i帧位置到第0帧的变换矩阵
                    pastk_trans_mat_pastk_2_past0.append(unit_mat)# 存放每一帧到第0帧的变换矩阵， 所以一共有k个元素
                pastk_trans_mat_pastk_2_past0 = torch.from_numpy(np.stack(pastk_trans_mat_pastk_2_past0, axis=0)).to(device) # (k, 4, 4)
                
                m_single = {}
                m_single['psm_single'] = psm_single[cav_idx] # (k, 2, H, W) 每一个agent的
                m_single['rm_single'] = rm_single[cav_idx] # (k, 14, H, W)
                if self.use_dir:
                    m_single['dm_single'] = dm_single[cav_idx] #(k, 4, H, W)
                m_single['predict_unc_cls'] = predict_unc_cls[cav_idx] # (k, 2, H, W)
                m_single['predict_unc_reg'] = predict_unc_reg[cav_idx]
                # 1. generate one cav's box results 接下来这一步本质上是过滤，选出合适的bbx作为预测结果 输入k帧的检测结果
                box_results[cav_idx] = dataset.generate_pred_bbx_frames_w_uncertainty(m_single, pastk_trans_mat_pastk_2_past0, cav_past_k_time_diff[cav_idx*self.k:cav_idx*self.k+self.k], anchor_box, pairwise_t_matrix_past0_2_cur[cav_idx])
            
            comm_rois_nums = box_results[0][0]['scores'].shape[0] # 只计算ego向外发送的
            # import math as mathmatics
            
            # comm_volum.append(mathmatics.log2(comm_rois_nums * 40))
            comm_volum.append(comm_rois_nums * 40)
            cav_trans_mat_pastk_2_past0.append(pastk_trans_mat_pastk_2_past0)
            
            # 2. generate box flow in one batch
            if self.viz_bbx_flag: # 这个应该是判断是否需要可视化单车检测的bbx流数据 额外返回了 ori_mask：（N_b， C， H，W）标记了object的位置，置1 matched_idx_list：列表 其中元素有 (N_obj, 3)  也有 (N_obj, 2) compensated_results_list: 列表 (N_obj, 4, 2)
                box_flow_map, mask, ori_mask, matched_idx_list, compensated_results_list = self.matcher(box_results, shape_list=shape_list, viz_flag=self.viz_bbx_flag)
                ori_reserved_mask_list.append(ori_mask)
            else:
                box_flow_map, mask = self.matcher(box_results, shape_list=shape_list, viz_flag=self.viz_bbx_flag) # [N_b, H, W, 2] [N_b, C, H, W]
            box_flow_map_list.append(box_flow_map)
            reserved_mask_list.append(mask)

            if self.viz_bbx_flag:
                single_box_results = box_results
        
        final_flow_map = torch.concat(box_flow_map_list, dim=0) # [N, H, W, 2] 一个batch中的合在一起
        final_reserved_mask = torch.concat(reserved_mask_list, dim=0)# [N, C, H, W] 一个batch中的合在一起
        comm_volum = sum(comm_volum) / B

        if self.viz_bbx_flag:
            ori_reserved_mask = torch.concat(ori_reserved_mask_list, dim=0) 
            return final_flow_map, final_reserved_mask, ori_reserved_mask, single_box_results, matched_idx_list, compensated_results_list, comm_volum

        return final_flow_map, final_reserved_mask, comm_volum

    def forward(self, data_dict, dataset=None):
        voxel_features = data_dict['processed_lidar']['voxel_features']         #(M, 32, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords']             #(M, 4)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']     #(M, )
        record_len = data_dict['record_len']                                    #(B, )
        record_frames = data_dict['past_k_time_interval']                       #(sum(n_cav) ) batch中所有cav中每一帧到cur的时间间隔
        pairwise_t_matrix = data_dict['pairwise_t_matrix']                      #(B, L, k, 4, 4) 每一帧到cur ego
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
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict) # 'spatial_features_2d': (batch_cav_size, 128*3, H/2, W/2)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d'] # 特征提取后的结果

        shape_list = torch.tensor(batch_dict['spatial_features'].shape[-3:]).to(pairwise_t_matrix.device) # [64, 200, 704]

        noise_pairwise_t_matrix = None
        if self.noise_flag and B==1:
            # noise_level = {'pos_std': 0.5, 'rot_std': 0, 'pos_mean': 0, 'rot_mean': 0}
            noise_pairwise_t_matrix = get_past_k_pairwise_transformation2ego(data_dict['past_lidar_pose'], self.noise_level, k=self.k, max_cav=5)[:record_len[0]].unsqueeze(0)

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d).detach() # (B, 2, H', W')
        rm_single = self.reg_head(spatial_features_2d).detach() # (B, 14, H', W')
        unc_preds = self.unc_head(spatial_features_2d).detach() # s is log(b) or log(sigma^2)  移动到这里是因为发现下采样会对不确定性量化造成很严重的数值不稳
        # re-parametrization trick, or var of classification logit is very difficulty to learn -- xyj 2024/5/27
        # if self.re_parameterization is True and self.training:
        #     unc_cls_log_var = self.unc_head_cls(spatial_features_2d) # (N, 2, H, W)
        #     unc_cls_log_var = torch.exp(unc_cls_log_var) # 得到方差
        #     unc_cls_log_var = torch.sqrt(unc_cls_log_var) # 得到标准差
        #     epsilon = torch.randn_like(unc_cls_log_var).to(unc_cls_log_var.device)
        #     psm_single = psm_single + epsilon * unc_cls_log_var
        if self.re_parameterization is True: # 协同训练的时候不需要加入噪声 
            unc_cls_log_var = self.unc_head_cls(spatial_features_2d).detach() # (N, 2, H, W)
        
        if self.use_dir:
            dm_single = self.dir_head(spatial_features_2d).detach()

        # single MC Dropout start
        if self.inference_state is True:
            # MC Dropout
            B0,_,H0,W0 = psm_single.shape # (B, 2, 100, 352)
            cls_preds_ntimes_tensor = torch.zeros_like(psm_single, dtype=psm_single.dtype, device=psm_single.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            reg_preds_ntimes_tensor = torch.zeros_like(rm_single, dtype=rm_single.dtype, device=rm_single.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            unc_preds_ntimes_tensor = torch.zeros_like(unc_preds, dtype=unc_preds.dtype, device=unc_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            cls_preds_ntimes_tensor[0] = psm_single
            reg_preds_ntimes_tensor[0] = rm_single
            unc_preds_ntimes_tensor[0] = torch.exp(unc_preds) # 本身预测的是log var 所以现在要变回方差
            if self.re_parameterization is True:
                cls_noise_ntimes_tensor = torch.zeros_like(unc_cls_log_var, dtype=unc_cls_log_var.dtype, device=unc_cls_log_var.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
                cls_noise_ntimes_tensor[0] = torch.exp(unc_cls_log_var) # 这是已经被处理过的标准差 要变回到方差 需要平方
                cls_noise_ntimes_tensor[0] = torch.sqrt(cls_noise_ntimes_tensor[0])

            if self.use_dir:
                dir_preds_ntimes_tensor = torch.zeros_like(dm_single, dtype=dm_single.dtype, device=dm_single.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
                dir_preds_ntimes_tensor[0] = dm_single

            for i in range(1, self.inference_num): # infer
                batch_dict = self.backbone(batch_dict)

                spatial_features_2d = batch_dict['spatial_features_2d']

                if self.shrink_flag:
                    spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')

                cls_preds_ntimes_tensor[i] = self.cls_head(spatial_features_2d)
                reg_preds_ntimes_tensor[i] = self.reg_head(spatial_features_2d)
                if self.use_dir:
                    dir_preds_ntimes_tensor[i] = self.dir_head(spatial_features_2d)

                unc_preds_ntimes_tensor[i] = self.unc_head(spatial_features_2d) # s is log(b) or log(sigma^2)
                unc_preds_ntimes_tensor[i] = torch.exp(unc_preds_ntimes_tensor[i]) # 恢复成方差
                if self.re_parameterization is True:
                    cls_noise_ntimes_tensor[i] = self.unc_head_cls(spatial_features_2d)
                    cls_noise_ntimes_tensor[i] = torch.exp(cls_noise_ntimes_tensor[i]) # 方差作为分类噪声
                    cls_noise_ntimes_tensor[i] = torch.sqrt(cls_noise_ntimes_tensor[i]) # 标准差

                del spatial_features_2d

            cls_preds_mean = torch.mean(cls_preds_ntimes_tensor, dim=0)
            reg_preds_mean = torch.mean(reg_preds_ntimes_tensor, dim=0) # (1, 14, H, W)
            unc_preds_mean = torch.mean(unc_preds_ntimes_tensor, dim=0) # (1, 2*3, H, W) 直接建模回归不确定性
            d_a_square = self.anchor_l**2 + self.anchor_w**2
            unc_preds_mean = unc_preds_mean.permute(0,2,3,1) # (N, H, W, 2*3) 这会改变内存布局，因此后面必须逆操作
            unc_preds_mean = unc_preds_mean.reshape(2*H0*W0*B0, -1) # (2HW, 3)
            # print(unc_preds_mean.shape)
            assert unc_preds_mean.shape[1] == 3
            unc_preds_mean[:, :2] *= d_a_square
            unc_preds_mean = torch.sqrt(unc_preds_mean)
            unc_preds_mean = unc_preds_mean.sum(dim=-1, keepdim = True) # (2HW, 1)
            unc_preds_mean = (unc_preds_mean - self.data_ureg_mean) / self.data_ureg_std # 标准化
            unc_preds_mean = unc_preds_mean.reshape(B0, H0, W0, -1).permute(0, 3, 1, 2) # （N, H, W, 2）

            if self.re_parameterization is True:
                cls_noise_mean = torch.mean(cls_noise_ntimes_tensor, dim=0)
            if self.use_dir:
                dir_preds_mean = torch.mean(dir_preds_ntimes_tensor, dim=0) 
            reg_preds_var = torch.var(reg_preds_ntimes_tensor, dim=0) # (1, 14, H, W)

            # 模型分类不确定性  衡量每个anchor
            cls_score = torch.sigmoid(cls_preds_mean)
            if self.re_parameterization is True:
                # cls_noise = cls_noise_mean
                cls_noise = calc_deviation_ratio(cls_noise_mean, cls_score, tp_cls_mean = self.tp_data_ucls_mean, tp_cls_std = self.tp_data_ucls_std, tp_score_mean = self.tp_score_mean, tp_score_std = self.tp_score_std) # 计算分类偏差比
            else:
                print("==close re-parameterzation==")
                cls_noise = torch.zeros_like(cls_score)
            # print('unc_preds_mean shape is ', unc_preds_mean.shape)
            # print('cls_noise shape is ', cls_noise.shape)
            # print('unc_preds_mean is ', unc_preds_mean)
            # print('cls_noise is ', cls_noise)
            # xxx
            # 计算分类分数的对数
            log_cls_score = torch.log(cls_score)
            log_1_cls_score = torch.log(1 - cls_score)

            # 计算熵 除以 log(2) 以将结果从 nats 转换为 bits
            unc_epi_cls = -(cls_score * log_cls_score + (1 - cls_score) * log_1_cls_score) / torch.log(torch.tensor(2.0, device=psm_single.device))           
            # 将熵的张量转换为 psm_single 的设备
            # unc_epi_cls = unc_epi_cls.to(cls_score.device)
            # unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score) # 计算分类偏差比
            # unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score, tp_cls_mean = 0.8201, tp_cls_std = 0.1749, tp_score_mean = 0.6484, tp_score_std = 0.1872) # 计算分类偏差比
            # temp_model_cls = unc_epi_cls.clone()
            unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score, tp_cls_mean = self.tp_model_ucls_mean, tp_cls_std = self.tp_model_ucls_std, tp_score_mean = self.tp_score_mean, tp_score_std = self.tp_score_std) # 计算分类偏差比
            # equal_elements = (cls_noise == unc_epi_cls)
            # num_equal_elements = equal_elements.sum().item()
            # print('num_equal_elements is ', num_equal_elements)
            # print('equal_elements shape is ', equal_elements.shape)
            # print("same data cls unc is ", cls_noise[equal_elements])
            # print("same model cls unc is ", unc_epi_cls[equal_elements])
            # aaa

            # 模型回归不确定性
            # d_a_square = 1.6**2 + 3.9**2 # anchor的长宽平方和
            reg_preds_var = reg_preds_var.permute(0,2,3,1).reshape(-1,7) # (2BHW, 7) 
            reg_preds_var[:,:2] *= d_a_square
            unc_epi_reg = torch.sqrt(reg_preds_var)
            unc_epi_reg = unc_epi_reg[:,0] + unc_epi_reg[:,1] + unc_epi_reg[:,6] # (2BHW,)
            unc_epi_reg = (unc_epi_reg - self.model_ureg_mean) / self.model_ureg_std # 标准化
            unc_epi_reg = unc_epi_reg.reshape(B0, H0, W0, -1).permute(0, 3, 1, 2) # (B, 2, H, W)

            psm_single = cls_preds_mean.detach() # (B, 2, H', W')
            rm_single = reg_preds_mean.detach() # (B, 14, H', W')
            
            if self.use_dir:
                dm_single = dir_preds_mean.detach()

            # 数据不确定性+ 模型不确定性 = 预测不确定性
                
            reg_noise = unc_preds_mean
            # predict_unc_cls = cls_noise # 数据不确定性求的是标准差,模型不确定性求的是熵,但二者都通过分类偏差值使得尺度统一
            # predict_unc_reg = reg_noise # 数据不确定性求的是方差,模型不确定性求的也是方差
            # predict_unc_cls = cls_noise + unc_epi_cls # 数据不确定性求的是标准差,模型不确定性求的是熵,但二者都通过分类偏差值使得尺度统一
            # predict_unc_reg = reg_noise + unc_epi_reg # 数据不确定性求的是方差,模型不确定性求的也是方差

            predict_unc_cls = torch.cat((cls_noise, unc_epi_cls), dim=1) # 数据不确定性求的是标准差,模型不确定性求的是熵,但二者都通过分类偏差值使得尺度统一
            predict_unc_reg = torch.cat((reg_noise, unc_epi_reg), dim=1) # 数据不确定性求的是方差,模型不确定性求的也是方差
            # predict_unc_reg = torch.cat((cls_noise_mean, temp_model_cls), dim=1) # 数据不确定性求的是方差,模型不确定性求的也是方差

            predict_unc_cls = self.regroup(predict_unc_cls, record_len, k) # 原本是（B，2，H，W) 变成了list，每个元素为一个scenario，为(Nxk, 2, H, W)
            predict_unc_reg = self.regroup(predict_unc_reg, record_len, k)

            predict_unc = {'predict_unc_cls': predict_unc_cls,
                           'predict_unc_reg': predict_unc_reg}
        else:
            predict_unc = {'predict_unc_cls': self.regroup(torch.zeros_like(psm_single), record_len, k),
                           'predict_unc_reg': self.regroup(torch.zeros_like(psm_single), record_len, k)}


        single_detection_bbx = None

        # if self.design_mode != 4 and not self.only_tune_header_flag:
        if self.design_mode != 4: 
            # generate box flow
            # [B, 256, 50, 176]
            single_output = {}
            single_output.update({'psm_single_list': self.regroup(psm_single, record_len, k), 
            'rm_single_list': self.regroup(rm_single, record_len, k),
            'predict_unc': predict_unc})
            if self.use_dir:
                single_output.update({
                    'dm_single_list': self.regroup(dm_single, record_len, k)
                })
            if self.viz_bbx_flag: # 可视化bbx
                box_flow_map, reserved_mask, ori_reserved_mask, single_detection_bbx, matched_idx_list, compensated_results_list, comm_volum = self.generate_box_flow(data_dict, single_output, dataset, shape_list, psm_single.device, pairwise_t_matrix)
            else:
                box_flow_map, reserved_mask, comm_volum = self.generate_box_flow(data_dict, single_output, dataset, shape_list, psm_single.device, pairwise_t_matrix) # 这是根据方向还有距离匹配past0和past1 再计算平均速度乘上时间得到的预估flow


        # print("box_flow_map shape is :", box_flow_map.shape)
        # print("reserved_mask shape is :", reserved_mask.shape) # torch.Size([2, 64, 200, 704])
        # xxx
        # print("batch_dict['spatial_features'] shape is :", batch_dict['spatial_features'].shape)
        


        if 'flow_gt' in data_dict['label_dict']:
            flow_gt = data_dict['label_dict']['flow_gt']
            mask_gt = data_dict['label_dict']['warp_mask']
        else:
            flow_gt = None
    
        # debug 在不使用运动预测的时候 默认先用GT流，GT流的生成为直接用 past0 与 cur下的object做
        # box_flow_map = flow_gt
        # reserved_mask = mask_gt
        
        # if self.only_tune_header_flag:
        #     box_flow_map = flow_gt
        #     reserved_mask = mask_gt

        # rain attention:
        if self.multi_scale:
            if self.design_mode == 0 or self.design_mode==5:
                if self.viz_bbx_flag: # 会额外返回 (BxN, C, H, W) 这是补偿之后的特征
                    fused_feature, communication_rates, result_dict, single_updated_feature = self.rain_fusion(batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix, 
                    record_frames,
                    self.fused_backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head],
                    box_flow=box_flow_map, reserved_mask=reserved_mask,
                    flow_gt=flow_gt, viz_bbx_flag=self.viz_bbx_flag)
                else:
                    fused_feature, communication_rates, result_dict = self.rain_fusion(batch_dict['spatial_features'], # (sum(n_cav), C, H, W) 所有帧的
                        psm_single, # (B, 2, H', W')
                        record_len, # (B, )
                        pairwise_t_matrix, # (B, L, k, 4, 4)
                        record_frames,# (B, )
                        self.fused_backbone, # ResNetBEVBackbone(args['base_bev_backbone'], 64)
                        [self.shrink_conv, self.cls_head, self.reg_head],
                        box_flow=box_flow_map, reserved_mask=reserved_mask,
                        flow_gt=flow_gt, viz_bbx_flag=self.viz_bbx_flag, noise_pairwise_t_matrix=noise_pairwise_t_matrix)
            elif self.design_mode == 4:
                fused_feature, communication_rates, result_dict = self.rain_fusion(batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix, 
                    record_frames,
                    self.backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head])
            else: 
                fused_feature, communication_rates, result_dict, flow_recon_loss = self.rain_fusion(batch_dict['spatial_features'],
                    psm_single,
                    record_len,
                    pairwise_t_matrix, 
                    record_frames,
                    self.backbone,
                    [self.shrink_conv, self.cls_head, self.reg_head],
                    box_flow=box_flow_map, reserved_mask=reserved_mask,
                    flow_gt=flow_gt)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.fused_shrink_conv(fused_feature) # (B, 256, H/2, W/2)
                if self.single_supervise:
                    single_feature = self.shrink_conv(single_feature)

        else:
            fused_feature, communication_rates, result_dict = self.rain_fusion(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix,
                                            record_frames)
            if self.compensation:
                if self.single_supervise:
                    fused_feature, single_feature, communication_rates, all_recon_loss, result_dict = self.rain_fusion(spatial_features_2d,
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix, 
                                                record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])
                else:
                    fused_feature,fused_feature_curr,fused_feature_latency, communication_rates, all_recon_loss, all_latency_recon_loss, result_dict = self.rain_fusion(spatial_features_2d,
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix, 
                                                record_frames,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])            

        
        # print('fused_feature: ', fused_feature.shape)
        # exit9

        if self.only_tune_header_flag:
            psm = self.fused_cls_head(fused_feature)
            rm = self.fused_reg_head(fused_feature)
        else: 
            # psm = self.cls_head(fused_feature) # 默认是开启下方这个，这里先注释 2024年04月20日
            # rm = self.reg_head(fused_feature)
            psm = self.fused_cls_head(fused_feature)
            rm = self.fused_reg_head(fused_feature)
        
        output_dict = {'psm': psm,
                       'rm': rm}

        if self.use_dir:
            # dm = self.dir_head(fused_feature)
            dm = self.fused_dir_head(fused_feature)
            output_dict.update({'dm': dm})

        if self.compensation:
            if self.single_supervise:
                psm_nonego_single = self.cls_head(single_feature)
                rm_nonego_single = self.reg_head(single_feature)
                output_dict.update({
                    'psm_nonego_single': psm_nonego_single,
                    'rm_nonego_single': rm_nonego_single
                })
            
            output_dict.update({
                'recon_loss': all_recon_loss, 
                'record_len': record_len
            })

        output_dict.update({'psm_single': psm_single,
                       'rm_single': rm_single,
                       'comm_rate': comm_volum
                       })

        if self.viz_bbx_flag:
            output_dict.update({
                'single_detection_bbx': single_detection_bbx, # Dict 包含了三帧的detect结果
                'matched_idx_list': matched_idx_list,  # List [(N_obj_cav1, 2/3), ...]
                'compensated_results_list': compensated_results_list # List [(N_obj_cav1, 4, 2), ...]
            })
            _, C, H, W = batch_dict['spatial_features'].shape
            output_dict.update({
                'single_updated_feature': single_updated_feature, # 补偿后的特征 (BxN, C, H, W)
                'single_original_feature': batch_dict['spatial_features'].reshape(-1, self.k, C, H, W)[:, 0, :, :, :],  # 没有特征提取前的特征（BxN, 0, C, H, W） past0 其中
                'single_flow_map': box_flow_map,  # 生成的流  [N, H, W, 2] 
                'single_reserved_mask': reserved_mask,  # 流掩码  [N, C, H, W]
                'single_original_reserved_mask': ori_reserved_mask # 和上面的流掩码形状一样，但是是将原始past0中的object的区域都置1
            })

        
        output_dict.update({'reserved_mask': reserved_mask[:, 0, :, :]}) # (N, H, W)
        if self.design_mode == 1:
            output_dict.update({'flow_recon_loss': flow_recon_loss})
        
        output_dict.update(result_dict) 
        
        return output_dict
