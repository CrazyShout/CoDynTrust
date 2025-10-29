# -*- coding: utf-8 -*-
# Sizhe Wei <sizhewei@sjtu.edu.cn>

import argparse
import os
import time
from typing import OrderedDict
import sys
sys.path.append(os.getcwd())
import shutil
import random

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils, box_utils
from opencood.visualization import vis_utils, my_vis, simple_vis

from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange


def worker_init_fn(worker_id):
    # 获取主进程的种子
    seed = torch.initial_seed() % 2**32
    # 为每个 worker 设定不同的随机种子
    test_deterministic(seed + worker_id)

def test_deterministic(myseed=303):
    # 设置Python原生随机库的种子
    random.seed(myseed)

    # 设置NumPy的随机种子
    np.random.seed(myseed)

    # 设置PyTorch的随机种子
    torch.manual_seed(myseed)

    # 如果你使用了GPU，还需要设置cuda的随机种子
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)  # 如果你使用多张GPU

    # 确保PyTorch在使用GPU时的可重复性, 这两项的设置是防止CuDNN动态选择最优算法，为此，可能损失计算速度
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=1,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--pretrained_path', default='',
                        help='The path of the model need to be fine tuned.')
    parser.add_argument('--note', default="ir_thre_0_d_20", type=str, help='save folder name')
    parser.add_argument('--p', default=None, type=float, help='binomial probability')
    parser.add_argument('--ir_range', default=None, type=float, help='binomial probability')
    parser.add_argument('--two_stage', help='whether to use two stage training', default=0, type=int)
    parser.add_argument('--config_suffix', default='', type=str, help='config suffix')
    parser.add_argument('--dataset', default='o', type=str, choices=['o', 'd'], help='which dataset will be used, o is for OPV2V/IRV2V, d is for DAIR-V2X.')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    test_deterministic()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty'] 

    hypes = yaml_utils.load_yaml(None, opt, config_suffix=opt.config_suffix)
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # update binomial prob
    if 'binomial_p' in hypes or opt.p is not None:
        hypes['binomial_p'] = opt.p
    if 'binomial_p' not in hypes and opt.p is None:
        hypes['binomial_p'] = 0.0
        print(f"!!! No binomial probability is set, using default value: {hypes['binomial_p']} !!!")

    if 'binomial_n' not in hypes:
        hypes['binomial_n'] = 1
        print(f"!!! No binomial n is set, using default value: {hypes['binomial_n']} !!!")

    hypes['time_delay'] = int(hypes['binomial_p'] * hypes['binomial_n'])
    
    if 'ir_range' in hypes and opt.ir_range is not None:
        hypes['ir_range'] = opt.ir_range
    if 'ir_range' not in hypes and opt.ir_range is None:
        hypes['ir_range'] = 0
        print(f"!!! No ir range is set, using default value: {hypes['ir_range']} !!!")
    
    viz_bbx_flag = False
    if 'viz_bbx_flag' in hypes:
        print("viz_bbx_flag is True!")
        viz_bbx_flag = hypes['viz_bbx_flag']

    # This is used in visualization
    # left hand: IRV2V, OPV2V, V2XSET
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if "my_opv2v" in hypes['test_dir'] or "my_v2xset" in hypes['test_dir'] else False
    # left_hand = True
    
    # control comm volume
    num_roi_thres = -1
    if 'num_roi_thres' in hypes:
        num_roi_thres = hypes['num_roi_thres']

    # add pose noise:
    noise = {'pos_std':0, 'rot_std':0, 'pos_mean':0, 'rot_mean':0}
    if 'noise' in hypes:
        noise = hypes['noise']
    noise_note = f"noise_{noise['pos_std']}_{noise['rot_std']}_{noise['pos_mean']}_{noise['rot_mean']}"

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']
        
    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model_diff(saved_path, model)
    
    if opt.pretrained_path: # load traj pred model
        saved_path = opt.pretrained_path
        pretrained_model_dict = torch.load(saved_path, map_location='cpu')
        diff_keys = {k:v for k, v in pretrained_model_dict.items() if k not in model.state_dict()}
        modified_pretrained_model_dict = OrderedDict()
        for k, v in pretrained_model_dict['model_dict'].items():
            modified_pretrained_model_dict.update({'matcher.compensate_motion.'+k: v})
        diff_keys = {k:v for k, v in modified_pretrained_model_dict.items() if k not in model.state_dict()}
        if diff_keys:
            print(f"!!! PreTrained model has keys: {diff_keys.keys()}, \
                which are not in the model you have created!!!")
        model.load_state_dict(modified_pretrained_model_dict, strict=False)
    
    model.eval()

    # setting noise
    np.random.seed(303)
    noise_setting = OrderedDict()
    noise_setting['add_noise'] = False
    # noise_setting['args'] = OrderedDict()
    # noise_setting['args']['pos_std'] = 0.4
    # noise_setting['args']['rot_std'] = 0.4
    # noise_setting['args']['pos_mean'] = 0
    # noise_setting['args']['rot_mean'] = 0

    # build dataset for each noise setting
    print('Dataset Building')
    print(f"No Noise Added.")
    hypes.update({"noise_setting": noise_setting})
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            worker_init_fn=worker_init_fn,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, # fp [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] tp [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] score 22
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    
    total_comm_rates = []
    noise_level = "no_noise"
    if hasattr(opencood_dataset, "set_cur_epoch"):
        print("===记录当前epoch===")
        opencood_dataset.set_cur_epoch(0)
    start_time = time.time()
    # infer_info = opt.fusion_method + f"{opt.cavnum}agent" + opt.note
    i = -1
    avg_time_delay = 0.0
    avg_sample_interval = 0.0
    avg_time_var = 0.0
    if opt.dataset == 'd':
        avg_cp_rate = 0.0
    for i, batch_data in tenumerate(data_loader):
        if batch_data is None:
            continue
        with torch.no_grad():
            if opt.fusion_method == 'late':
                if opt.dataset == 'o':
                    unit_time_delay = []
                    unit_sample_interval = []
                    # for cav_id, cav_content in batch_data.items():
                    #     unit_time_delay.append(cav_content['debug']['time_diff'])
                    #     unit_sample_interval.append(hypes['time_delay']*100)
                    #     # unit_sample_interval.append(cav_content['debug']['sample_interval'])
                    # avg_time_delay += (sum(unit_time_delay[1:])/len(unit_time_delay[1:]))
                    # avg_sample_interval += (float(sum(unit_sample_interval[1:]))/len(unit_sample_interval[1:]))
            if opt.fusion_method == 'intermediate':
                if opt.dataset == 'o':
                    try:
                        avg_time_delay += batch_data['ego']['avg_time_delay']
                        avg_sample_interval += batch_data['ego']['avg_sample_interval']
                    except KeyError:
                        avg_sample_interval += 0
                        avg_time_delay += 0
                    try: 
                        avg_time_var += batch_data['ego']['avg_time_var']
                    except:
                        avg_time_var += -1
                elif opt.dataset == 'd':
                    try:
                        avg_time_delay += (-batch_data['ego']['avg_time_delay'])
                    except KeyError:
                        avg_time_delay += 0
                    try:
                        avg_sample_interval += batch_data['ego']['avg_time_delay']
                    except KeyError:
                        avg_sample_interval += 0
                    try:
                        avg_cp_rate += (batch_data['ego']['cp_rate'])
                    except KeyError:
                        avg_cp_rate += 1
                        
            batch_data = train_utils.to_device(batch_data, device)
            uncertainty_tensor = None
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                if opt.two_stage == 1:
                    if viz_bbx_flag:
                        pred_box_tensor, pred_score, gt_box_tensor, single_detection_bbx, matched_idx_list, compensated_results_list, single_updated_feature, single_original_feature, single_flow_map, single_reserved_mask, single_original_reserved_mask = \
                            inference_utils.inference_intermediate_fusion_flow_module(batch_data,
                                                                        model,
                                                                        opencood_dataset, True)
                        # pred_box_tensor, pred_score, gt_box_tensor = \
                        #     inference_utils.inference_intermediate_fusion_flow_module(batch_data,
                        #                                                 model,
                        #                                                 opencood_dataset, True)

                    else:
                        # pred_box_tensor, pred_score, gt_box_tensor, reserved_mask = \
                        pred_box_tensor, pred_score, gt_box_tensor, comm_rates = \
                            inference_utils.inference_intermediate_fusion_flow_module(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                        total_comm_rates.append(comm_rates)
                else:
                    pred_box_tensor, pred_score, gt_box_tensor, comm_rates = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset) # (N, 8, 3)
                    total_comm_rates.append(comm_rates)
                    
            elif opt.fusion_method == 'no':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
                    inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            else:
                raise NotImplementedError('Only no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')
            
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego']['origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'complement_vis_{opt.note}_%.2f_{noise_note}_roi_{num_roi_thres}'%(hypes['binomial_p']))
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                try:
                    debug_path = batch_data['ego']['debug']['scene_name'] + '_' + batch_data['ego']['debug']['cav_id'] + '_' + batch_data['ego']['debug']['timestamp']
                    # print(debug_path)
                except:
                    debug_path = 'path'
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d_%s.png' % (i, debug_path))
                # vis_save_path_3d = os.path.join(vis_save_path_root, '3d_%05d_%s.png' % (i, debug_path))

                # # 保存ego前摄图片
                # if 'camera0_files' in  batch_data['ego']:
                #     camera0_files = batch_data['ego']['camera0_files']
                #     camera0_save_path = os.path.join(vis_save_path_root, 'camera_%05d_%s.png' % (i, debug_path))
                #     shutil.copy(camera0_files, camera0_save_path)


                # reserved_mask = single_reserved_mask[:, 0, :, :].cpu().numpy() # (N, H, W)
                # import matplotlib.pyplot as plt
                # for pic_id in range(reserved_mask.shape[0]):
                #     # if pic_id != 1:
                #     #     continue
                #     plt.figure(figsize=(12, 6))
                #     # 将掩码为0的部分设为透明，只显示有值的部分
                #     masked_data = np.ma.masked_where(reserved_mask[pic_id] == 0, reserved_mask[pic_id])

                #     plt.imshow(np.ones_like(reserved_mask[pic_id]), cmap='gray', vmin=0.5, vmax=1)  # 深灰背景  # 设置白色背景

                #     im = plt.imshow(masked_data, cmap='viridis', alpha=0.8)
                #     plt.colorbar(mappable=im, orientation='horizontal', pad=0.1)
                #     # 设置黑色背景
                #     # plt.gca().set_facecolor('white')
                #     # 移除坐标轴标签
                #     plt.xticks([])
                #     plt.yticks([])
                #     vis_mask_path = os.path.join(vis_save_path_root, f'mask_image_{i}_{pic_id}.png')
                #     plt.savefig(vis_mask_path, bbox_inches='tight', dpi=400)
                #     plt.close()

                # mean_updated_feature = torch.mean(single_updated_feature, dim=1) # (N, 64, 704, 200) warp后的特征图  (N, 704, 200)
                # # 自定义颜色映射，背景为深紫色，其他值用热力图
                # import matplotlib.colors as mcolors
                # from matplotlib.colors import LinearSegmentedColormap

                # # 自定义金黄色到黄色的颜色映射
                # # colors = [(1, 1, 1), (1, 0.75, 0)]  # 白色到黄色
                # colors = [(0, 1, 0), (1, 0.84, 0)]  # 绿色到金黄色
                # n_bins = 100  # 颜色分段数
                # cmap = LinearSegmentedColormap.from_list('green_gold', colors, N=n_bins)
                # # cmap = plt.get_cmap('inferno')  # 热力图的配色
                # bg_color = mcolors.to_rgba('#2E0854')  # 深紫色（Ubuntu终端的背景色)

                # # 设置特征图的颜色映射，背景部分为深紫色，其他部分通过热力图显示
                # # 归一化特征图数据，准备进行颜色映射
                # min_val = torch.min(mean_updated_feature)
                # max_val = torch.max(mean_updated_feature)
                # normed_feature_map = (mean_updated_feature - min_val) / (max_val - min_val) # (N, 704, 200)
                # # 将特征图从 tensor 转为 numpy 数组，以便使用 matplotlib 的 colormap 进行颜色映射
                # normed_feature_map_np = normed_feature_map.cpu().numpy()
                # colored_map = cmap(normed_feature_map_np)  # 将数据映射为颜色
                # colored_map[normed_feature_map_np == 0] = bg_color  # 将掩码为0的部分设为背景色

                # mean_original_feature = torch.mean(single_original_feature, dim=1) # 没有warp后的特征（N，C，H，W）--> (N, 704, 200)
                # original_mask = single_original_reserved_mask[:,0,:,:]
                # mean_original_feature = mean_original_feature * original_mask
                # min_val = torch.min(mean_original_feature)
                # max_val = torch.max(mean_original_feature)
                # normed_ori_feature_map = (mean_original_feature - min_val) / (max_val - min_val) # (N, 704, 200)
                # normed_ori_feature_map_np = normed_ori_feature_map.cpu().numpy()
                # colored_map_ori = cmap(normed_ori_feature_map_np)
               
                # colored_map_ori[normed_ori_feature_map_np == 0] = bg_color

                # # single_original_reserved_mask # 原始mask （N，C，H，W）

                # for cav_id in range(normed_feature_map_np.shape[0]):
                #     cav_updated_feature = colored_map[cav_id]
                    
                #     vis_updated_feature_path = os.path.join(vis_save_path_root, f'updated_feature_{i}_{cav_id}.png')
                #     plt.imshow(cav_updated_feature)
                #     plt.savefig(vis_updated_feature_path, bbox_inches='tight', dpi=400)
                #     plt.axis('off')  # 不显示坐标轴
                #     plt.close()
                # for cav_id in range(normed_ori_feature_map_np.shape[0]):
                #     cav_original_feature = colored_map_ori[cav_id]
                    
                #     vis_original_feature_path = os.path.join(vis_save_path_root, f'original_feature_{i}_{cav_id}.png')
                #     plt.imshow(cav_original_feature)
                #     plt.savefig(vis_original_feature_path, bbox_inches='tight', dpi=400)
                #     plt.axis('off')  # 不显示坐标轴
                #     plt.close()

                simple_vis.visualize(pred_box_tensor,
                                    gt_box_tensor,
                                    batch_data['ego']['origin_lidar'][0],# 一个场景下的所有cav的cur点云信息，且全部投影到cur ego
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand,
                                    uncertainty=uncertainty_tensor,
                                    pcd_agent_split = batch_data['ego']['origin_lidar_splitnum'][0])
                # simple_vis.visualize(pred_box_tensor,
                #                     gt_box_tensor,
                #                     batch_data['ego']['origin_lidar'][0],# 一个场景下的所有cav的cur点云信息，且全部投影到cur ego
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path_3d,
                #                     method='3d',
                #                     left_hand=left_hand,
                #                     uncertainty=uncertainty_tensor,
                #                     pcd_agent_split = batch_data['ego']['origin_lidar_splitnum'][0])

                # if viz_bbx_flag:
                #     '''
                #     box_dict : {
                #         'single_detection_bbx': # dict, [0, 1, ... , N-1], 
                #             [i]: dict:{
                #                 [0]/[1]/[2]: past 3 frames detection results:{  past 3 frames detection results:{ # 都是在[0]的各自坐标系下的检测结果。
                #                     pred_box_3dcorner_tensor
                #                     pred_box_center_tensor
                #                     scores
                #                 }
                #             }
                #         }
                #         'lidar_pose_0': 过去第一帧 所有车的pose [N, 6]
                #         'lidar_pose_current': 当前帧 所有车的pose [N, 6]
                #         'matched_idx_list': matched_idx_list, # len=N-1, 表示每个non-ego的过去两帧的匹配id, each of which is [N_obj, 2], 比如 ['matched_idx_list'][0] shape(22,2) 表示过去第一帧的22个框与第二帧的22个框的索引的匹配情况
                #         'compensated_results_list': # len=N-1, 每个non-ego补偿出来的box, each of which is [N_obj, 4, 3], 注意这里用的是4个点, 最后一个维度上是 xyz, z是多余的
                #         'single_gt_box_tensor': # list, len=N, 表示ego与non-ego每辆车在current时刻的检测框结果, 例如box_dict['single_gt_box_tensor'][1]大小为[N_obj, 8, 3] 表示第二辆车在current时刻的检测框结果
                #         'single_lidar': # list, len=N, 表示ego与non-ego每辆车在current时刻的lidar np
                #         'gt_range': # [-140.8, -40, -3, 140.8, 40, 1], 表示lidar的范围
                #         'single_updated_feature': tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的更新后的feature
                #         'single_original_feature': tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的原始feature
                #         'single_flow_map': tensor, [N, H, W, 2], 表示ego与non-ego每辆车在past-0时刻的flow map
                #         'single_reserved_mask': tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的reserved mask
                #         'single_original_reserved_mask': tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的原始reserved mask
                #     }
                #     '''
                #     # box_dict = {}
                #     # box_dict.update({
                #     #     'single_lidar': batch_data['ego']['single_lidar_list'], # len = N, (n_lidar, 3)
                #     #     'single_past_lidar': batch_data['ego']['single_past_lidar_list'], 
                #     #     'gt_range': hypes['postprocess']['gt_range'],
                #     #     'lidar_pose_current': batch_data['ego']['curr_lidar_pose'], #[N, 6] 
                #     #     'lidar_pose_0': batch_data['ego']['past_lidar_pose'][:, 0, :], # (N, 6)
                #     #     'pred_box_tensor': pred_box_tensor, # (n ,8, 3)
                #     #     'gt_box_tensor': gt_box_tensor
                #     # })
                #     # '''
                #     # box_dict : {
                #     #     'lidar_pose_0': 过去第一帧 所有车的pose [N, 6]
                #     #     'lidar_pose_current': 当前帧 所有车的pose [N, 6]
                #     #     'single_lidar': # list, len=N, 表示ego与non-ego每辆车在current时刻的lidar np, 每一个都是 (n_lidar, 3)
                #     #     'single_past_lidar': # list, len=N, 表示ego与non-ego每辆车在past-0时刻的lidar np, 每一个都是 (n_lidar, 3)
                #     #     'pred_box_tensor': # tensor, [N, 8, 3], 表示ego与non-ego每辆车在current时刻的检测框结果
                #     #     'gt_box_tensor': # tensor, [N, 8, 3], 表示ego与non-ego每辆车在current时刻的检测框结果
                #     #     'gt_range': # [-140.8, -40, -3, 140.8, 40, 1], 表示lidar的范围
                #     # }
                #     # '''
                #     # box_save_folder = os.path.join(vis_save_path_root, 'bbx_folder')
                #     # if not os.path.exists(box_save_folder):
                #     #     os.mkdir(box_save_folder)
                #     # box_save_path = os.path.join(box_save_folder, 'bbx_%05d_%s.pt' % (i, debug_path))
                #     # torch.save(box_dict, box_save_path)

                #     box_dict = {}
                #     single_gt_box_tensor = []
                #     for cav in range(len(batch_data['ego']['single_object_bbx_center'])):
                #         object_bbx_center = batch_data['ego']['single_object_bbx_center'][cav]
                #         object_bbx_corner = box_utils.boxes_to_corners_3d(object_bbx_center, 'hwl')
                #         single_gt_box_tensor.append(object_bbx_corner)
                #     box_dict.update({
                #         'single_detection_bbx': single_detection_bbx, # dict, [0, 1, 2] 
                #         'compensated_results_list': compensated_results_list, # len=N-1, each of which is [N_obj, 8, 3]
                #         'matched_idx_list': matched_idx_list, # len=N-1, each of which is [N_obj, 2]
                #         'single_gt_box_tensor': single_gt_box_tensor, # tensor, [N_obj, 8, 3]
                #         'single_lidar': batch_data['ego']['single_lidar_list'], # len = N, (n_lidar, 3)
                #         'single_past_lidar': batch_data['ego']['single_past_lidar_list'], 
                #         'gt_range': hypes['postprocess']['gt_range'],
                #         'lidar_pose_current': batch_data['ego']['curr_lidar_pose'], #[N, 6] 
                #         'lidar_pose_0': batch_data['ego']['past_lidar_pose'][:, 0, :], # (N, 6)
                #         'single_updated_feature': single_updated_feature, # tensor, [N, C, H, W], 表示ego与non-ego每辆车在current时刻的更新后的feature
                #         'single_original_feature': single_original_feature, # tensor, [N, C, H, W], 表示ego与non-ego每辆车在current时刻的原始feature
                #         'single_flow_map': single_flow_map, # tensor, [N, H, W, 2], 表示ego与non-ego每辆车在past-0时刻的flow map
                #         'single_reserved_mask': single_reserved_mask, # tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的reserved mask
                #         'single_original_reserved_mask': single_original_reserved_mask, # tensor, [N, C, H, W], 表示ego与non-ego每辆车在past-0时刻的原始reserved mask
                #     })
                #     box_save_folder = os.path.join(vis_save_path_root, 'bbx_folder')
                #     if not os.path.exists(box_save_folder):
                #         os.mkdir(box_save_folder)
                #     box_save_path = os.path.join(box_save_folder, 'bbx_%05d_%s.pt' % (i, debug_path))
                #     torch.save(box_dict, box_save_path)
        torch.cuda.empty_cache()
    end_time = time.time()
    print("Time Consumed: %.2f minutes" % ((end_time - start_time)/60))
        
    if opt.dataset == 'o':
        avg_time_delay = (avg_time_delay/i) * 50 # unit is ms
        avg_sample_interval /= i
        avg_time_var /= i
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    opt.model_dir, noise_level, avg_time_delay, avg_sample_interval, avg_time_var, opt.note+'_'+str("%.2f"%hypes['binomial_p'])+f'_{noise_note}_roi_{num_roi_thres}')
    elif opt.dataset == 'd':
        avg_sample_interval /= i
        avg_cp_rate /= i
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    opt.model_dir, noise_level=noise_level, avg_time_delay=avg_cp_rate, avg_sample_interval=avg_sample_interval, note=opt.note+'_'+str("%.2f"%hypes['binomial_p'])+f'_{noise_note}_roi_{num_roi_thres}', dataset=opt.dataset)
    print("Module with sample interval expection: {}".format(hypes['binomial_n']*hypes['binomial_p']))
    if opt.dataset == 'o':
        print(f"IR sample range is {hypes['ir_range']}")

    if len(total_comm_rates) > 0:
        comm_rates = (sum(total_comm_rates)/len(total_comm_rates))
    else:
        comm_rates = 0
    print("comm_rate is {:.04f}".format(comm_rates))

if __name__ == '__main__':
    main()
