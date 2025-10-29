# -*- coding: utf-8 -*-
# Modified: Sizhe Wei <sizhewei@sjtu.edu.cn>

"""
Dataset class for intermediate fusion with time delay k
"""
import random
import math
from collections import OrderedDict

import os
import os.path as osp
import numpy as np
from pytest import param
import torch
from torch.utils.data import DataLoader
import json
import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from scipy import stats

from opencood.data_utils.datasets import intermediate_fusion_dataset_opv2v_irregular_flow_new
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import x_to_world
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.flow_utils import generate_flow_map, generate_flow_map_szwei



def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def build_idx_to_info(data):
    idx2info = {}
    for elem in data:
        if elem["pointcloud_path"] == "":
            continue
        idx = elem["pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info

def build_idx_to_co_info(data):
    idx2info = {}
    for elem in data:
        if elem["vehicle_pointcloud_path"] == "":
            continue
        idx = elem["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info

def build_inf_fid_to_veh_fid(data):
    inf_fid2veh_fid = {}
    for elem in data:
        veh_fid = elem["vehicle_pointcloud_path"].split("/")[-1].rstrip('.pcd')
        inf_fid = elem["infrastructure_pointcloud_path"].split("/")[-1].rstrip('.pcd')
        inf_fid2veh_fid[inf_fid] = veh_fid
    return inf_fid2veh_fid

def id_to_str(id, digits=6):
    result = ""
    for i in range(digits):
        result = str(id % 10) + result
        id //= 10
    return result

class IntermediateFusionDatasetDAIRIrregularFlow(intermediate_fusion_dataset_opv2v_irregular_flow_new.IntermediateFusionDatasetIrregularFlowNew):
    """
    Written by sizhewei @ 2022/09/28
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        #注意yaml文件应该有sensor_type：lidar/camera
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        self.max_cav = 2
        
        if 'num_sweep_frames' in params:    # number of frames we use in LSTM
            self.k = params['num_sweep_frames']
        else:
            self.k = 1

        if 'binomial_n' in params:
            self.binomial_n = params['binomial_n']
        else:
            self.binomial_n = 10

        if 'binomial_p' in params:
            self.binomial_p = params['binomial_p']
        else:
            self.binomial_p = 0

        if 'with_history_frames' in params and params['with_history_frames'] is True:
            print("===需要历史帧===")
            self.w_history = True
        else:
            print("===不需要历史帧===")
            self.w_history = False

        self.strict_data = False
        if 'strict_data' in params and params['strict_data'] is True:
            print("===验证集使用严格策略, 即必须10*k历史帧严格存在===")
            self.strict_data = True

        self.only_async = False
        if 'only_Async' in params and params['only_Async'] is True:
            print("===仅Async而非Irregular===")
            self.only_async = True

        self.expectation_delay = self.binomial_n * self.binomial_p
        print(f"===expectation_delay is {self.expectation_delay}===")

        self.num_roi_thres = -1
        if 'num_roi_thres' in params:
            self.num_roi_thres = params['num_roi_thres']
            print("限制ROI个数以测量带宽性能权衡: ", self.num_roi_thres)

        # 控制是否需要生成GT flow
        self.is_generate_gt_flow = False
        if 'is_generate_gt_flow' in params and params['is_generate_gt_flow']:
            self.is_generate_gt_flow = True

        self.viz_bbx_flag = False
        
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False


        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
            
        self.generate_uncertainty = False
        if 'use_uncertainty_guide' in params['model']['args']:
            if params['model']['args']['use_uncertainty_guide']:
                print("===use uncertainty to guide roi===")
                self.generate_uncertainty = True

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        #这里root_dir是一个json文件！--> 代表一个split
        if self.train:
            split_dir = params['root_dir'] # "my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/train.json"
        else:
            split_dir = params['validate_dir'] # "my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/val.json"

        self.root_dir = params['data_dir'] # "my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"

        self.inf_idx2info = build_idx_to_info( # 读取路端标签 形成路端id对应其信息字典的形式
            load_json(osp.join(self.root_dir, "infrastructure-side/data_info.json"))
        )
        self.co_idx2info = build_idx_to_co_info( # 读取协同标签，形成车端id对应该项协同场景的所有信息的形式
            load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
        ) # 依旧读取协同标签，形成路端id对应车端id的形式，也就是形成了一一对应
        self.inf_fid2veh_fid = build_inf_fid_to_veh_fid(load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
        )

        self.data_split = load_json(split_dir)
        self.data = []
        for veh_idx in self.data_split: # 读取数据集划分文件，这里分为训练集和验证集两种，存储车端id
            if self.is_valid_id(veh_idx): # 检查车端id是否合格：1、往前后10*k帧，2、这其中每一帧都存在车路对应
                self.data.append(veh_idx)
        
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead. 
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        self.anchor_box = self.post_processor.generate_anchor_box()

        self.cur_epoch = 0
        
        print("Irregular async dataset with past %d frames and expectation time delay = %d initialized! %d samples totally!" % (self.k, int(self.binomial_n*self.binomial_p), len(self.data)))

    def get_vehicle_trans(self, veh_frame_id):

        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))

        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)
        trans = tfm_to_pose(transformation_matrix)

        return trans
    
    def get_inf_trans(self, inf_frame_id, system_error_offset):
        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        trans = tfm_to_pose(transformation_matrix1)

        return trans

    def is_valid_id(self, veh_frame_id):
        """
        Written by sizhewei @ 2022/10/05
        Given veh_frame_id, determine whether there is a corresponding inf_frame that meets the k delay requirement.

        Parameters
        ----------
        veh_frame_id : 05d
            Vehicle frame id

        Returns
        -------
        bool valud
            True means there is a corresponding road-side frame.
        """
        # print('veh_frame_id: ',veh_frame_id,'\n')
        if self.strict_data is not True:
            frame_info = {}
            
            frame_info = self.co_idx2info[veh_frame_id] # 取出协同信息
            inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "") # 当前路端的帧id
            cur_inf_info = self.inf_idx2info[inf_frame_id] # 取出路端信息
            delay_id = id_to_str(int(inf_frame_id) - 0) 
            if delay_id not in self.inf_fid2veh_fid.keys(): # 必须有车路对应
                return False
            # if (int(inf_frame_id) - self.binomial_n*self.k < int(cur_inf_info["batch_start_id"])): # 当前帧作为cur，往前倒推10*2帧，检查是否合法
            #     return False
            # for i in range(self.binomial_n * self.k): # 循环10 * 2 次 也就是从cur id往前倒推，必须每一帧都存在
            #     delay_id = id_to_str(int(inf_frame_id) - i) 
            #     if delay_id not in self.inf_fid2veh_fid.keys(): # 必须有车路对应
            #         return False

            return True

        frame_info = {}
        
        frame_info = self.co_idx2info[veh_frame_id] # 取出协同信息
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "") # 当前路端的帧id
        cur_inf_info = self.inf_idx2info[inf_frame_id] # 取出路端信息
        if (int(inf_frame_id) - self.binomial_n*self.k < int(cur_inf_info["batch_start_id"])): # 当前帧作为cur，往前倒推10*2帧，检查是否合法
            return False
        for i in range(self.binomial_n * self.k): # 循环10 * 2 次 也就是从cur id往前倒推，必须每一帧都存在
            delay_id = id_to_str(int(inf_frame_id) - i) 
            if delay_id not in self.inf_fid2veh_fid.keys(): # 必须有车路对应
                return False

        return True

    def retrieve_base_data(self, idx):
        """
        Modified by sizhewei @ 2022/09/28
        Given the index, return the corresponding async data (time delay expection = sum(B(n, p)) ).

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            {
                [0]  / [1]: {
                    'ego' : True / False,
                    'params' : {
                        'vehicles':
                        'lidar_pose':
                    },
                    'lidar_np' : 
                    'veh_frame_id' :
                    'avg_time_delay' : ([1] only)
                }
            }
        """
        final_data = OrderedDict()
        veh_frame_id = self.data[idx] # 获取到车id
        
        # print('veh_frame_id: ',veh_frame_id,'\n')
        frame_info = {}
        system_error_offset = {}
        
        frame_info = self.co_idx2info[veh_frame_id] # 拿到协同信息
        bernoulliDist = stats.bernoulli(self.binomial_p) 

        '''
        final_data : {
            [0] / [1] : {
                'ego' : True / False,
                'curr' : {
                    'frame_id' :
                }
                'past_k' : {
                    [0] / [1] / ... / [k-1] : {
                        'frame_id'
                    }
                }
            }

        }
        '''

        final_data = OrderedDict()
        curr_veh_frame_id = self.data[idx] # 车端id
        frame_info = self.co_idx2info[veh_frame_id] # 帧信息
        system_error_offset = frame_info['system_error_offset']
        curr_inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "") # 路端id
        for i in range(2):
            final_data[i] = OrderedDict()
            final_data[i]['ego'] = True if i == 0 else False # first one is veh, second is inf
            final_data[i]['curr'] = {}
            final_data[i]['curr']['frame_id'] = curr_veh_frame_id if i == 0 else curr_inf_frame_id
            final_data[i]['curr']['timestamp'] = curr_veh_frame_id if i == 0 else curr_inf_frame_id
            final_data[i]['curr']['time_diff'] = 0
            final_data[i]['curr']['sample_interval'] = 0
            # 用于可视化前摄
            final_data[i]['curr']['camera0_files'] = os.path.join(self.root_dir, frame_info["vehicle_image_path"])

            final_data[i]['curr']['lidar_np'] = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))[0] if i==0 else \
                pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))[0]
            final_data[i]['curr']['params'] = OrderedDict()
            final_data[i]['curr']['params']['vehicles'] = \
                load_json(osp.join(self.root_dir, frame_info['cooperative_label_path'])) if i == 0 else [] # 这里面存放的是世界标签
            final_data[i]['curr']['params']['vehicles_single'] = \
                load_json(os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(curr_veh_frame_id))) if i==0 else \
                load_json(os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(curr_inf_frame_id))) # 这里面存放的是single标签
            
            final_data[i]['curr']['params']['lidar_pose'] = self.get_vehicle_trans(curr_veh_frame_id) if i==0 else \
                self.get_inf_trans(curr_inf_frame_id, system_error_offset)
                          
            final_data[i]['past_k'] = OrderedDict()
            if i==0:
                for j in range(self.k):
                    final_data[i]['past_k'][j] = final_data[0]['curr']
            else:
                latest_frame_id = curr_inf_frame_id # 当前路端帧id
                for j in range(self.k):
                    if self.only_async:
                        sample_interval = int(self.expectation_delay)
                        if j != 0:
                            sample_interval = 1
                    else:
                        # B(n, p)
                        trails = bernoulliDist.rvs(self.binomial_n)
                        sample_interval = sum(trails) # 向前倒推多少帧
                    # sample_interval = 5
                    # if j == 0:
                    #     sample_interval = 3
                    # elif j == 1:
                    #     sample_interval = 1
                    # else:
                    #     sample_interval = 1
                    if self.strict_data is not True:
                        cur_inf_info = self.inf_idx2info[latest_frame_id] # 取出路端信息 要判断两个：1、延迟减去后的帧是否存在，否为无延迟，2、延迟减去后的帧是否有车路帧，如果没有则倒退
                        if (int(latest_frame_id) - sample_interval < int(cur_inf_info["batch_start_id"])): # 如果往前已经没有帧，则默认使用当前帧 ，设置延迟为0
                            sample_interval = 0
                        if sample_interval > 0:
                            delay_id = id_to_str(int(latest_frame_id) - sample_interval)
                            for _ in range(sample_interval): # 判断延迟帧是否有车路帧，无则回溯
                                if delay_id not in self.inf_fid2veh_fid.keys(): # 必须有车路对应
                                    sample_interval -= 1
                                else:
                                    break

                    latest_frame_id = id_to_str(int(latest_frame_id) - int(sample_interval)) # 找到延迟前的帧id


                    try:
                        veh_frame_id_of_inf = self.inf_fid2veh_fid[latest_frame_id] # 根据这个id找到延迟车帧id
                    except KeyError:
                        print("=====Error====")
                    frame_info = self.co_idx2info[veh_frame_id_of_inf] # 延迟车帧id取出协同信息

                    system_offset = self.co_idx2info[veh_frame_id_of_inf]['system_error_offset'] # 误差偏移
                    final_data[i]['past_k'][j] = {}
                    final_data[i]['past_k'][j]['frame_id'] = latest_frame_id
                    final_data[i]['past_k'][j]['timestamp'] = latest_frame_id
                    final_data[i]['past_k'][j]['time_diff'] = int(latest_frame_id) - int(curr_inf_frame_id) # 这里被我反了一下，这是为了得到和其他数据集一样的正负关系
                    final_data[i]['past_k'][j]['sample_interval'] = - sample_interval
                    final_data[i]['past_k'][j]['lidar_np'] = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))[0]
                    final_data[i]['past_k'][j]['params'] = OrderedDict()
                    final_data[i]['past_k'][j]['params']['vehicles'] = []
                    final_data[i]['past_k'][j]['params']['lidar_pose'] = self.get_inf_trans(latest_frame_id, system_offset)
                    # 2024年7月22日 xuyujiang 增加路端的单车信息，因为在where2comm监督单车的训练中需要用到
                    # final_data[i]['past_k'][j]['params']['vehicles_single'] = \
                    #     load_json(os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(latest_frame_id))) # 这里面存放的是single标签
        
        """
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            {
                [0] / [1]: {
                    'ego' : True / False,
                    'curr': {
                        'frame_id' :
                        'params': vehicle & pose
                    }
                    'past_k' : {
                        [0] / [1] / ... / [k-1] : {
                            'frame_id' :
                            'params': vehicle & pose
                        }
                    }
                    'avg_time_delay' : ([1] only)
                }
            }
        """



        return final_data



    def get_item_single_car(self, selected_cav_base, ego_pose, idx):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information, 
            structure: {
                'ego' : true,
                'curr' : {
                    'params': (yaml),
                    'lidar_np': (numpy),
                    'timestamp': string
                },
                'past_k' : {		           # (k) totally
                    [0]:{
                        'params': (yaml),
                        'lidar_np': (numpy),
                        'timestamp': string,
                        'time_diff': float,
                        'sample_interval': int
                    },
                    [1] : {},
                    ...,		
                    [k-1] : {}
                },
                'debug' : {                     # debug use
                    scene_name : string         
                    cav_id : string
                }
            }

        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.

        idx: int,
            debug use.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
            {
                'projected_lidar':      # lidar in ego space, 用于viz
                'single_label_dict':	# single view label. 没有经过坐标变换,                      cav view + curr 的label
                "single_object_bbx_center": single_object_bbx_center,       # 用于viz single view
                "single_object_ids": single_object_ids,                # 用于viz single view
                'flow_gt':              # single view flow
                'curr_feature':         # current feature, lidar预处理得到的feature
                'object_bbx_center':	# ego view label. np.ndarray. Shape is (max_num, 7).    ego view + curr 的label
                'object_ids':			# ego view label index. list. length is (max_num, 7).   ego view + curr 的label
                'curr_pose':			# current pose, list, len = 6
                'past_k_poses': 		    # list of past k frames' poses
                'past_k_features': 		    # list of past k frames' lidar
                'past_k_time_diffs': 	    # list of past k frames' time diff with current frame
                'past_k_tr_mats': 		    # list of past k frames' transformation matrix to current ego coordinate
                'pastk_2_past0_tr_mats':    # list of past k frames' transformation matrix to past 0 frame
                'past_k_sample_interval':   # list of past k frames' sample interval with later frame
                'avg_past_k_time_diffs':    # avg_past_k_time_diffs,
                'avg_past_k_sample_interval': # avg_past_k_sample_interval,
                'if_no_point':              # bool, 用于判断是否合法
            }
        """
        selected_cav_processed = {}

        # curr lidar feature
        lidar_np = selected_cav_base['curr']['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np) # remove points that hit itself

        if self.visualize:
            # trans matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['curr']['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray
            projected_lidar = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
            selected_cav_processed.update({'projected_lidar': projected_lidar})

        lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
        if self.viz_bbx_flag:
            selected_cav_processed.update({'single_lidar': lidar_np[:, :3]})
        curr_feature = self.pre_processor.preprocess(lidar_np) # 转变成体素
        
        # past k transfomation matrix
        past_k_tr_mats = []
        # past_k to past_0 tansfomation matrix
        pastk_2_past0_tr_mats = []
        # past k lidars
        past_k_features = []
        # past k poses
        past_k_poses = []
        # past k timestamps
        past_k_time_diffs = []
        # past k sample intervals
        past_k_sample_interval = []

        # for debug use
        # avg_past_k_time_diff = 0
        # avg_past_k_sample_interval = 0
        
        # past k label 
        # past_k_label_dicts = [] # todo 这个部分可以删掉

        # 判断点的数量是否合法
        if_no_point = False

        # past k frames [trans matrix], [lidar feature], [pose], [time interval]
        for i in range(self.k):
            # 1. trans matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['past_k'][i]['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray
            past_k_tr_mats.append(transformation_matrix)
            # past_k trans past_0 matrix
            pastk_2_past0 = \
                x1_to_x2(selected_cav_base['past_k'][i]['params']['lidar_pose'], selected_cav_base['past_k'][0]['params']['lidar_pose'])
            pastk_2_past0_tr_mats.append(pastk_2_past0)
            
            # 2. lidar feature
            lidar_np = selected_cav_base['past_k'][i]['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_ego_points(lidar_np) # remove points that hit itself
            lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
            processed_features = self.pre_processor.preprocess(lidar_np)
            past_k_features.append(processed_features)

            if lidar_np.shape[0] == 0: # 没有点留下
                if_no_point = True

            # 3. pose
            past_k_poses.append(selected_cav_base['past_k'][i]['params']['lidar_pose'])

            # 4. time interval and sample interval
            past_k_time_diffs.append(selected_cav_base['past_k'][i]['time_diff'])
            # print("selected_cav_base['past_k'][i]['time_diff'] is ", selected_cav_base['past_k'][i]['time_diff'])
            past_k_sample_interval.append(selected_cav_base['past_k'][i]['sample_interval'])

            ################################################################
            # sizhewei
            # for past k frames' single view label
            ################################################################
            # # 5. single view label
            # # past_i label at past_i single view
            # # opencood/data_utils/post_processor/base_postprocessor.py
            # object_bbx_center, object_bbx_mask, object_ids = \
            #     self.generate_object_center([selected_cav_base['past_k'][i]], selected_cav_base['past_k'][i]['params']['lidar_pose'])  
            # # generate the anchor boxes
            # # opencood/data_utils/post_processor/voxel_postprocessor.py
            # anchor_box = self.anchor_box
            # single_view_label_dict = self.post_processor.generate_label(
            #         gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask
            #     )
            # past_k_label_dicts.append(single_view_label_dict)
        
        

        '''
        # past k merge
        past_k_label_dicts = self.post_processor.merge_label_to_dict(past_k_label_dicts)
        '''

        past_k_tr_mats = np.stack(past_k_tr_mats, axis=0) # (k, 4, 4) 这是每一帧到cur的变换矩阵
        pastk_2_past0_tr_mats = np.stack(pastk_2_past0_tr_mats, axis=0) # (k, 4, 4) 这是每一帧到past0的变换矩阵

        # avg_past_k_time_diffs = float(sum(past_k_time_diffs) / len(past_k_time_diffs))
        # avg_past_k_sample_interval = float(sum(past_k_sample_interval) / len(past_k_sample_interval))

        # curr label at single view
        # opencood/data_utils/post_processor/base_postprocessor.py
        # xuyunjiang at 2024/7/22 修正，这里不能使用curr的信息，因为训练不会使用单车的cur点云，而是使用的past信息，所以label也要用past0的
        # 但是仍有问题，因为这个变量既会用于监督where2comm的置信度图损失，又会用来做显式，但是后者应该用curr的bbx，以可视化运动预测与实际位置的差距，@TODO 这里需要权衡修改
        # single_object_bbx_center, single_object_bbx_mask, single_object_ids = \
        #     self.generate_object_center_dair_single([selected_cav_base['past_k'][0]], selected_cav_base['past_k'][0]['params']['lidar_pose'])
        single_object_bbx_center, single_object_bbx_mask, single_object_ids = \
            self.generate_object_center_dair_single([selected_cav_base['curr']], selected_cav_base['curr']['params']['lidar_pose'])  
        # generate the anchor boxes
        # opencood/data_utils/post_processor/voxel_postprocessor.py
        anchor_box = self.anchor_box
        label_dict = self.post_processor.generate_label(
                gt_box_center=single_object_bbx_center, anchors=anchor_box, mask=single_object_bbx_mask
            )
        
        # curr label at ego view
        object_bbx_center, object_bbx_mask, object_ids = \
            self.generate_object_center([selected_cav_base['curr']], ego_pose)
            
        selected_cav_processed.update({
            "single_label_dict": label_dict,
            "single_object_bbx_center": single_object_bbx_center[single_object_bbx_mask == 1],
            "single_object_ids": single_object_ids,
            "curr_feature": curr_feature,
            'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
            'object_ids': object_ids,
            'curr_pose': selected_cav_base['curr']['params']['lidar_pose'],
            'past_k_tr_mats': past_k_tr_mats,
            'pastk_2_past0_tr_mats': pastk_2_past0_tr_mats,
            'past_k_poses': past_k_poses,
            'past_k_features': past_k_features,
            'past_k_time_diffs': past_k_time_diffs,
            'past_k_sample_interval': past_k_sample_interval,
        #  'avg_past_k_time_diffs': avg_past_k_time_diffs,
        #  'avg_past_k_sample_interval': avg_past_k_sample_interval,
        #  'past_k_label_dicts': past_k_label_dicts,
            'if_no_point': if_no_point
            })

        if self.is_generate_gt_flow:
            # generate flow, from past_0 and curr
            prev_object_id_stack = {}
            prev_object_stack = {}
            for t_i in range(2):
                split_part = selected_cav_base['past_k'][0] if t_i == 0 else selected_cav_base['curr'] # TODO: 这里面的 prev 和 curr 可能反了
                object_bbx_center, object_bbx_mask, object_ids = \
                    self.generate_object_center([split_part], selected_cav_base['past_k'][0]['params']['lidar_pose'])
                prev_object_id_stack[t_i] = object_ids
                prev_object_stack[t_i] = object_bbx_center
            
            for t_i in range(2):
                unique_object_ids = list(set(prev_object_id_stack[t_i]))
                unique_indices = \
                    [prev_object_id_stack[t_i].index(x) for x in unique_object_ids]
                prev_object_stack[t_i] = np.vstack(prev_object_stack[t_i])
                prev_object_stack[t_i] = prev_object_stack[t_i][unique_indices]
                prev_object_id_stack[t_i] = unique_object_ids
            
            # TODO: generate_flow_map: yhu, generate_flow_map_szwei: szwei
            flow_map, warp_mask = generate_flow_map_szwei(prev_object_stack,
                                            prev_object_id_stack,
                                            self.params['preprocess']['cav_lidar_range'],
                                            self.params['preprocess']['args']['voxel_size'],
                                            past_k=1)

            selected_cav_processed.update({'flow_gt': flow_map})
            selected_cav_processed.update({'warp_mask': warp_mask})

        return selected_cav_processed

    def set_cur_epoch(self, epoch):
        self.cur_epoch = epoch

    def __getitem__(self, idx):
        # seed = idx + self.cur_epoch * len(self.data)
        # np.random.seed(seed)
        # 首先要判断一下是不是要用历史帧，因为我们需要同样数目的数据集来公平比较
        # 构造函数在初始化的时候已经用两帧来限制数据集，这就保证了都在两帧筛选过的数据集上训练和测试，
        # 但有的model不需要历史帧，因此在这里才开始改变
        if self.w_history is not True:
            self.k = 1
        base_data_dict = self.retrieve_base_data(idx)
        ''' base_data_dict structure:
        {
            cav_id_1 : {
                'ego' : true,
                curr : {
                    'params': (yaml),
                    'lidar_np': (numpy),
                    'timestamp': string
                },
                past_k : {		                # (k) totally
                    [0]:{
                        'params': (yaml),
                        'lidar_np': (numpy),
                        'timestamp': string,
                        'time_diff': float,
                        'sample_interval': int
                    },
                    [1] : {},	(id-1)
                    ...,		
                    [k-1] : {} (id-(k-1))
                },
                'debug' : {                     # debug use
                    scene_name : string         
                    cav_id : string
                }
            },
            cav_id_2 : { ... }
        }
        '''

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        # first find the ego vehicle's lidar pose
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['curr']['params']['lidar_pose']
                camera0_files = cav_content['curr']['camera0_files']

                break	
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        too_far = []
        curr_lidar_pose_list = []
        cav_id_list = []
        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            # for non-ego cav, we use the latest frame's pose
            distance = math.sqrt( \
                (selected_cav_base['past_k'][0]['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + \
                    (selected_cav_base['past_k'][0]['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue
            curr_lidar_pose_list.append(selected_cav_base['curr']['params']['lidar_pose']) # 6dof pose
            cav_id_list.append(cav_id)  

        for cav_id in too_far: # filter those out of communicate range
            base_data_dict.pop(cav_id)

        single_label_dict_stack = []
        object_stack = []
        object_id_stack = []
        curr_pose_stack = []
        curr_feature_stack = []
        past_k_pose_stack = []
        past_k_features_stack = [] 
        past_k_tr_mats = []
        pastk_2_past0_tr_mats = []
        past_k_label_dicts_stack = []
        past_k_sample_interval_stack = []
        past_k_time_diffs_stack = []
        if self.is_generate_gt_flow:
            flow_gt = []
            warp_mask = []
        # avg_past_k_time_diffs = 0.0
        # avg_past_k_sample_interval = 0.0
        illegal_cav = []
        if self.visualize:
            projected_lidar_stack = []

        if self.viz_bbx_flag:
            single_lidar_stack = []
            single_object_stack = []
            single_object_id_stack = []
            single_mask_stack = []
        
        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id] # cav_id只有两个 0、 1
            ''' selected_cav_base:
            {
                'ego' : true,
                curr : {
                    'params': (yaml),
                    'lidar_np': (numpy),
                    'timestamp': string
                },
                past_k : {		                # (k) totally
                    [0]:{
                        'params': (yaml),
                        'lidar_np': (numpy),
                        'timestamp': string,
                        'time_diff': float,
                        'sample_interval': int
                    },
                    [1] : {},	(id-1)
                    ...,		
                    [k-1] : {} (id-(k-1))
                },
                'debug' : {                     # debug use
                    scene_name : string         
                    cav_id : string
                }
            }
            '''
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose,
                idx
            )
            ''' selected_cav_processed : dict
            The dictionary contains the cav's processed information.
            {
                'projected_lidar':      # curr lidar in ego space, 用于viz
                'single_label_dict':	# single view label. 没有经过坐标变换, 用于单体监督            cav view + curr 的label
                'single_object_bbx_center'
                'single_object_ids'
                'curr_feature':         # current feature, lidar预处理得到的feature                 cav view + curr feature
                'object_bbx_center':	# ego view label. np.ndarray. Shape is (max_num, 7).    ego view + curr 的label
                'object_ids':			# ego view label index. list. length is (max_num, 7).   ego view + curr 的label
                'curr_pose':			# current pose, list, len = 6
                'past_k_poses': 		    # list of past k frames' poses
                'past_k_features': 		    # list of past k frames' lidar
                'past_k_time_diffs': 	    # list of past k frames' time diff with current frame
                'past_k_tr_mats': 		    # list of past k frames' transformation matrix to current ego coordinate
                'past_k_sample_interval':   # list of past k frames' sample interval with later frame
                # 'avg_past_k_time_diffs':    # avg_past_k_time_diffs,
                # 'avg_past_k_sample_interval': # avg_past_k_sample_interval,
                'if_no_point':              # bool, 用于判断是否合法
                'flow_gt':               # [2, H, W] flow ground truth
            }
            '''

            if selected_cav_processed['if_no_point']: # 把点的数量不合法的车排除
                illegal_cav.append(cav_id)
                # 把出现不合法sample的 场景、车辆、时刻 记录下来:
                illegal_path = os.path.join(base_data_dict[cav_id]['debug']['scene'], cav_id, base_data_dict[cav_id]['past_k'][0]['timestamp']+'.npy')
                # illegal_path_list.add(illegal_path)
                # print(illegal_path)
                continue

            
            if self.visualize:
                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
            if self.viz_bbx_flag:
                single_lidar_stack.append(selected_cav_processed['single_lidar'])
                single_object_stack.append(selected_cav_processed['single_object_bbx_center'])
                single_object_id_stack.append(selected_cav_processed['single_object_ids'])
                # mask = np.zeros(self.params['postprocess']['max_num'])
                # mask[:single_object_stack.shape[0]] = 1
                # single_mask_stack.append(mask)
            
            # single view feature
            curr_feature_stack.append(selected_cav_processed['curr_feature'])
            # single view label
            single_label_dict_stack.append(selected_cav_processed['single_label_dict'])

            # curr ego view label
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

            # current pose: N, 6
            curr_pose_stack.append(selected_cav_processed['curr_pose']) 
            # features: N, k, 
            past_k_features_stack.append(selected_cav_processed['past_k_features'])
            # poses: N, k, 6
            past_k_pose_stack.append(selected_cav_processed['past_k_poses'])
            # time differences: N, k
            past_k_time_diffs_stack += selected_cav_processed['past_k_time_diffs']
            # sample intervals: N, k
            past_k_sample_interval_stack += selected_cav_processed['past_k_sample_interval']
            # past k frames to ego pose trans matrix, list of len=N, past_k_tr_mats[i]: ndarray(k, 4, 4)
            past_k_tr_mats.append(selected_cav_processed['past_k_tr_mats'])
            # past k frames to cav past 0 frame trans matrix, list of len=N, pastk_2_past0_tr_mats[i]: ndarray(k, 4, 4)
            pastk_2_past0_tr_mats.append(selected_cav_processed['pastk_2_past0_tr_mats'])
            # past k label dict: N, k, object_num, 7
            # past_k_label_dicts_stack.append(selected_cav_processed['past_k_label_dicts'])
        
            if self.is_generate_gt_flow:
                # for flow
                flow_gt.append(selected_cav_processed['flow_gt'])
                warp_mask.append(selected_cav_processed['warp_mask'])
        
        pastk_2_past0_tr_mats = np.stack(pastk_2_past0_tr_mats, axis=0) # N, k, 4, 4

        # {pos: array[num_cav, k, 100, 252, 2], neg: array[num_cav, k, 100, 252, 2], target: array[num_cav, k, 100, 252, 2]}
        # past_k_label_dicts = self.post_processor.merge_label_to_dict(past_k_label_dicts_stack)
        # self.times.append(time.time())

        # filter those cav who has no points left
        # then we can calculate get_pairwise_transformation
        for cav_id in illegal_cav:
            base_data_dict.pop(cav_id)
            cav_id_list.remove(cav_id)

        merged_curr_feature_dict = self.merge_features_to_dict(curr_feature_stack)  # current 在各自view 下 feature
        
        single_label_dict = self.post_processor.collate_batch(single_label_dict_stack) # current 在各自view 下 label

        past_k_time_diffs_stack = np.array(past_k_time_diffs_stack)
        past_k_sample_interval_stack = np.array(past_k_sample_interval_stack)
        
        pairwise_t_matrix = \
            self.get_past_k_pairwise_transformation2ego(base_data_dict, 
            ego_lidar_pose, self.max_cav) # np.tile(np.eye(4), (max_cav, self.k, 1, 1)) (L, k, 4, 4) TODO: 这里面没有搞懂为什么不用 past_k_tr_mats

        curr_lidar_poses = np.array(curr_pose_stack).reshape(-1, 6)  # (N_cav, 6)
        past_k_lidar_poses = np.array(past_k_pose_stack).reshape(-1, self.k, 6)  # (N, k, 6)

        # exclude all repetitive objects    
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        try:
            object_stack = np.vstack(object_stack)
        except ValueError:
            # print("!!! vstack ValueError !!!")
            return None
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # self.times.append(time.time())

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(cav_id_list)
        # past_k_features_stack: list, len is num_cav. [i] is list, len is k. [cav_id][time_id] is Orderdict, {'voxel_features': array, ...}
        merged_feature_dict = self.merge_past_k_features_to_dict(past_k_features_stack)

        # generate the anchor boxes
        anchor_box = self.anchor_box

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        # self.times.append(time.time())

        # self.times = (np.array(self.times[1:]) - np.array(self.times[:-1]))

        # self.times = np.hstack((self.times, time4data))

        processed_data_dict['ego'].update(
            {'single_object_dict_stack': single_label_dict,
             'curr_processed_lidar': merged_curr_feature_dict,
             'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'curr_lidar_poses': curr_lidar_poses,
             'past_k_lidar_poses': past_k_lidar_poses,
             'past_k_time_diffs': past_k_time_diffs_stack,
             'past_k_sample_interval': past_k_sample_interval_stack, 
             'camera0_files': camera0_files, # 可视化前摄用的
             'pastk_2_past0_tr_mats': pastk_2_past0_tr_mats})
            #  'times': self.times})

        if self.is_generate_gt_flow:
            flow_gt = np.vstack(flow_gt) # (N, H, W, 2)
            processed_data_dict['ego'].update({'flow_gt': flow_gt})
            warp_mask = np.vstack(warp_mask) # (N, C, H, W)
            processed_data_dict['ego'].update({'warp_mask': warp_mask})
        
        processed_data_dict['ego'].update({'sample_idx': idx,
                                            'cav_id_list': cav_id_list})
        try:
            tmp = past_k_time_diffs_stack[self.k:].reshape(-1, self.k) # (N, k)
            tmp_past_k_time_diffs = np.concatenate((tmp[:, :1] , (tmp[:, 1:] - tmp[:, :-1])), axis=1) # (N, k)
            avg_time_diff = sum(tmp_past_k_time_diffs.reshape(-1)) / tmp_past_k_time_diffs.reshape(-1).shape[0]
            processed_data_dict['ego'].update({'avg_sample_interval':\
                sum(past_k_sample_interval_stack[self.k:]) / len(past_k_sample_interval_stack[self.k:])})
            processed_data_dict['ego'].update({'avg_time_delay':\
                avg_time_diff})
        except ZeroDivisionError:
            # print("!!! ZeroDivisionError !!!")
            return None

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(projected_lidar_stack)})
            split_num = []
            for pd in projected_lidar_stack:
                split_num.append(pd.shape[0])
            processed_data_dict['ego'].update({'origin_lidar_splitnum':
                split_num}) # [n1, n2, ...]

        if self.viz_bbx_flag:
            for id, cav in enumerate(cav_id_list):
                processed_data_dict[id] = {}
                processed_data_dict[id].update({
                    'single_lidar': single_lidar_stack[id],
                    'single_object_bbx_center': single_object_stack[id],
                    # 'single_object_bbx_mask': single_mask_stack[i],
                    'single_object_ids': single_object_id_stack[id]
                })

        return processed_data_dict

    def __len__(self):
        # 符合条件的 frame 的数量
        return len(self.data)

    ### rewrite generate_object_center ###
    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
    
    ### Add new func for single side
    def generate_object_center_dair_single(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        veh or inf 's coordinate. 

        reference_lidar_pose is of no use.
        """
        suffix = "_single"
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)
    
    def generate_pred_bbx_frames_w_uncertainty(self, m_single, trans_mat_pastk_2_past0, past_time_diff, anchor_box, pairwise_t_matrix_past0_2_cur):
        '''
        m_single : {
            psm_single, rm_single, (dm_single)
        }
        box_result : {
            'past_k_time_diff' : 
            [0] : {
                pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. (n, 8, 3)
                pred_box_center_tensor : (n, 7)
                scores: (n, )
            },
            ...
            [k-1] : { ... }
        }
        pairwise_t_matrix_past0_2_cur: (4, 4) past0到ego的变换矩阵
        '''
        if self.generate_uncertainty:
            box_results = self.post_processor.single_post_process_w_uncertainty(m_single, trans_mat_pastk_2_past0, past_time_diff, anchor_box, self.k, self.num_roi_thres, pairwise_t_matrix_past0_2_cur)
        else:
            box_results = self.post_processor.single_post_process(m_single, trans_mat_pastk_2_past0, past_time_diff, anchor_box, self.k, self.num_roi_thres)
        return box_results

