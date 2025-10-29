# -*- coding: utf-8 -*-
# Author: sizhewei @ 2023/1/27
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for Late fusion with past k frames on irregular OPV2V
TODO: train 部分因为暂时没有用到 可能存在bug 使用前需要检查
"""

from collections import OrderedDict
import os
import numpy as np
import torch
import math
import random
import copy
import sys
import time
import json
from scipy import stats
import opencood.data_utils.post_processor as post_processor
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.keypoint_utils import bev_sample, get_keypoints
from opencood.data_utils.datasets import basedataset, intermediate_fusion_dataset_opv2v_irregular
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
from opencood.utils.pose_utils import add_noise_data_dict, remove_z_axis
from opencood.utils.common_utils import read_json
from opencood.utils import box_utils
from opencood.data_utils.datasets import late_fusion_dataset

# from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np

# from opencood.data_utils.datasets import build_dataset 

class LateFusionDatasetIrregular(late_fusion_dataset.LateFusionDataset):
    """
    This class is for late fusion where each vehicle transmit the
    detection outputs to ego.
    """
    # def __init__(self, params, visualize, train=True):
    #     super(LateFusionDatasetIrregular, self).__init__(params, visualize, train)

    #     # print("=== OPV2V-Irregular Multi-sweep dataset for late fusion with non-ego cavs' past {} frames collected initialized! Expectation of sample interval is {}. ### {} ###  samples totally! ===".format(self.k, self.binomial_n * self.binomial_p, self.len_record[-1]))

    def __init__(self, params, visualize, train=True):

        self.times = []

        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if 'num_sweep_frames' in params:    # number of frames we use in LSTM
            self.k = params['num_sweep_frames']
        else:
            self.k = 0

        if 'time_delay' in params:          # number of time delay
            self.tau = params['time_delay'] 
        else:
            self.tau = 0

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

        # 控制 past-0 处是否有扰动
        self.is_no_shift = False
        if 'is_no_shift' in params and params['is_no_shift']:
            self.is_no_shift = True

        # 控制每个场景内不同车的采样间隔是否一致
        self.is_same_sample_interval = False
        if 'is_same_sample_interval' in params and params['is_same_sample_interval']:
            self.is_same_sample_interval = True

        # 控制是否采用完全 regular 的数据 （整数timestamp）
        self.is_ab_regular = False
        if 'is_ab_regular' in params and params['is_ab_regular']:
            self.is_ab_regular = True
            print(f'======= is_ab_regular? {self.is_ab_regular} =======')
        
        self.sample_interval_exp = int(self.binomial_n * self.binomial_p)

        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        
        print("Dataset dir:", root_dir)

        if 'train_params' not in params or\
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        scenario_folders_name = sorted([x
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        '''
        scenario_database Structure: 
        {
            scenario_id : {
                cav_1 : {
                    'ego' : true / false , 
                    timestamp1 : {
                        yaml: path,
                        lidar: path, 
                        cameras: list of path
                    },
                    ...
                },
                ...
            }
        }
        '''
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})


            # at least 1 cav should show up
            cav_list = sorted([x 
                            for x in os.listdir(scenario_folder) if 
                            os.path.isdir(os.path.join(scenario_folder, x))], key=lambda y:int(y))
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]] # 路端放到最后

            # use the frame number as key, the full path as the values, store all json or yaml files in this scenario
            yaml_files = sorted([x
                        for x in os.listdir(os.path.join(scenario_folder, cav_list[0])) if # 遍历第一辆车目录下的所有json文件
                        x.endswith(".json")], 
                        key=lambda y:float((y.split('/')[-1]).split('.json')[0]))
            if len(yaml_files)==0:
                yaml_files = sorted([x 
                            for x in os.listdir(os.path.join(scenario_folder, cav_list[0])) if # 遍历第一辆车目录下的所有yaml文件
                            x.endswith('.yaml')], key=lambda y:float((y.split('/')[-1]).split('.yaml')[0]))
                
            regular_timestamps = self.extract_timestamps(yaml_files) # 返回的时间戳的列表，如[00068,000070,....] 其中都是字符串

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                cav_path = os.path.join(scenario_folder, cav_id)

                timestamps = regular_timestamps # 场景下的所有时间戳条目

                for timestamp in timestamps:
                    timestamp = "%06d" % int(timestamp) # 字符串格式化，保证6位
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    # camera_files = self.load_camera_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    # self.scenario_database[i][cav_id][timestamp]['camera0'] = \
                        # camera_files
                
                # regular的timestamps 用于做 curr 真实时刻的ground truth
                self.scenario_database[i][cav_id]['regular'] = self.scenario_database[i][cav_id]

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:  # ego 
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    # num_ego_timestamps = len(timestamps) - (self.tau + self.k - 1)		# 从第 tau+k 个往后, store 0 时刻的 time stamp
                    num_ego_timestamps = len(timestamps) - self.binomial_n * self.k
                    if not self.len_record:
                        self.len_record.append(num_ego_timestamps)
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + num_ego_timestamps)
                else:
                    self.scenario_database[i][cav_id]['ego'] = False
                    

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead. 
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        self.anchor_box = self.post_processor.generate_anchor_box()

        print("=== OPV2V-Irregular Multi-sweep dataset with non-ego cavs' past {} frames collected initialized! Expectation of sample interval is {}. ### {} ###  samples totally! ===".format(self.k, self.binomial_n * self.binomial_p, self.len_record[-1]))

    @staticmethod
    def dist_time(ts1, ts2, i = -1):
        """caculate the time interval between two timestamps

        Args:
            ts1 (string): time stamp at some time
            ts2 (string): current time stamp
            i (int, optional): past frame id, for debug use. Defaults to -1.
        
        Returns:
            time_diff (float): time interval (ts1 - ts2)
        """
        if not i==-1:
            return -i
        else:
            return (float(ts1) - float(ts2))

    def extract_timestamps(self, yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]
            if res.endswith('.yaml'):
                timestamp = res.replace('.yaml', '')
            elif res.endswith('.json'):
                timestamp = res.replace('.json', '')
            else:
                print("Woops! There is no processing method for file {}".format(res))
                sys.exit(1)
            timestamps.append(timestamp)

        return timestamps

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            Structure: 
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
                cav_id_2 : {		                # (k) totally
                    'ego': false, 
                    curr : 	{
                            'params': (yaml),
                            'lidar_np': (numpy),
                            'timestamp': string
                    },
                    past_k: {
                        [0] : {
                            'params': (yaml),
                            'lidar_np': (numpy),
                            'timestamp': string,
                            'time_diff': float,
                            'sample_interval': int
                        }			
                        ..., 	
                        [k-1]:{}  (id-\tau-(k-1))
                    },
                    'debug' : {                     # debug use
                        scene_name : string         
                        cav_id : string
                    }
                }, 
                ...
            }
        """
        # we loop the accumulated length list to get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record): # 遍历一个递增的list [场景1时间戳数，场景1+场景2时间戳数, ...]
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index] # 找到对应场景
        
        # 生成冻结分布函数
        bernoulliDist = stats.bernoulli(self.binomial_p) 

        data = OrderedDict()
        # 找到 current 时刻的 timestamp_index 这对于每辆车来讲都一样
        curr_timestamp_idx = idx if scenario_index == 0 else \
                        idx - self.len_record[scenario_index - 1] # 找到这是场景下的第几个数据 也就是对应时间戳的索引
        curr_timestamp_idx = curr_timestamp_idx + self.binomial_n * self.k # * 3 # TODO: 往后加了10*3=30帧作为cur帧 其实是因为十次伯努利实验防止都成功 那三帧刚好就要往前跳三十帧

        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items(): # 遍历一个场景下的所有cav
            '''
            cav_content 
            {
                timestamp_1 : {
                    yaml: path,
                    lidar: path, 
                    cameras: list of path
                },
                ...
                timestamp_n : {
                    yaml: path,
                    lidar: path, 
                    cameras: list of path
                },
                'regular' : {
                    timestamp_1 : {},
                    ...
                    timestamp_n : {}
                },
                'ego' : true / false , 
            },
            '''
            data[cav_id] = OrderedDict()
            
            # 1. set 'ego' flag
            data[cav_id]['ego'] = cav_content['ego']                

            # 2. current frame, for co-perception lable use
            data[cav_id]['curr'] = {}

            timestamp_key = list(cav_content['regular'].items())[curr_timestamp_idx][0] # 这是找到当前时间戳，字符串  每个列表项是一个键值对，如("000021", OrderedDict())

            if data[cav_id]['ego']:

                # 2.1 load curr params
                # json is faster than yaml 这里是为了加速想使用json 但是目前还没有做json文件
                json_file = cav_content['regular'][timestamp_key]['yaml'].replace("yaml", "json")
                json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")

                # TODO: debug use, uncomment: to use regular version yaml GT
                # time_new = str( '%06d' % int(float(timestamp_key)))
                # json_file = json_file.replace('OPV2V_irregular_npy','OPV2V_w_npy')
                # json_file = json_file.replace(timestamp_key, time_new)

                # json_file = cav_content['regular'][timestamp_key]['yaml'].replace("OPV2V_irregular_npy", "OPV2V_Irregular_V2/dataset_irregular_v2")
                # data[cav_id]['curr']['params'] = \
                #                 load_yaml(json_file)

                # print(json_file)
                # store the scene name, which is convinient for debug
                scene_name = json_file.split('/')[-3]
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data[cav_id]['curr']['params'] = json.load(f)
                else:
                    data[cav_id]['params'] = \
                                load_yaml(cav_content['regular'][timestamp_key]['yaml'])
                # 没有 lidar pose
                if not ('lidar_pose' in data[cav_id]['params']):
                    tmp_ego_pose = np.array(data[cav_id]['params']['true_ego_pos'])
                    tmp_ego_pose += np.array([-0.5, 0, 1.9, 0, 0, 0])
                    data[cav_id]['params']['lidar_pose'] = list(tmp_ego_pose)

                # 2.2 load curr lidar file
                # npy is faster than pcd
                npy_file = cav_content['regular'][timestamp_key]['lidar'].replace("pcd", "npy")
                
                # TODO: debug use, uncomment: to use regular version lidar input
                # time_new = str( '%06d' % int(float(timestamp_key))) 
                # npy_file = npy_file.replace('OPV2V_irregular_npy','OPV2V_w_npy')
                # npy_file = npy_file.replace(timestamp_key, time_new)

                # npy_file = cav_content['regular'][timestamp_key]['lidar'].replace("OPV2V_irregular_npy", "OPV2V_Irregular_V2")
                # data[cav_id]['curr']['lidar_np'] = \
                #     pcd_utils.pcd_to_np(npy_file)
                # print(npy_file)

                if os.path.exists(npy_file): 
                    data[cav_id]['lidar_np'] = np.load(npy_file)
                else:
                    data[cav_id]['lidar_np'] = \
                            pcd_utils.pcd_to_np(cav_content['regular'][timestamp_key]['lidar']) # （n, 4）

                # 2.3 store curr timestamp and time_diff
                data[cav_id]['timestamp'] = timestamp_key
                data[cav_id]['time_diff'] = 0.0
                data[cav_id]['sample_interval'] = 0
            else:
                # B(n, p)
                trails = bernoulliDist.rvs(self.binomial_n)  # 做10次伯努利实验
                sample_interval = sum(trails) # 统计成功次数，作为采样间隔

                delay_sample_stamp_idx = curr_timestamp_idx - sample_interval
                delay_timestamp_key = list(cav_content.items())[delay_sample_stamp_idx][0] # 获取时间戳

                # load the corresponding data into the dictionary
                # load param file: json is faster than yaml
                json_file = cav_content[delay_timestamp_key]['yaml'].replace("yaml", "json")
                json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")

                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data[cav_id]['params'] = json.load(f)
                else:
                    data[cav_id]['params'] = \
                        load_yaml(cav_content[delay_timestamp_key]['yaml']) # 获取yaml文件内容
                # 没有 lidar pose
                if not ('lidar_pose' in data[cav_id]['params']):
                    tmp_ego_pose = np.array(data[cav_id]['params']['true_ego_pos'])
                    tmp_ego_pose += np.array([-0.5, 0, 1.9, 0, 0, 0])
                    data[cav_id]['params']['lidar_pose'] = list(tmp_ego_pose)

                # load lidar file: npy is faster than pcd
                npy_file = cav_content[delay_timestamp_key]['lidar'].replace("pcd", "npy")
                if os.path.exists(npy_file):
                    data[cav_id]['lidar_np'] = np.load(npy_file)
                else:
                    data[cav_id]['lidar_np'] = \
                            pcd_utils.pcd_to_np(cav_content[delay_timestamp_key]['lidar'])

                data[cav_id]['timestamp'] = delay_timestamp_key
                data[cav_id]['sample_interval'] = sample_interval

                # label要替换回 curr 的内容
                curr_params = \
                        load_yaml(cav_content[timestamp_key]['yaml']) # 获取yaml文件内容
                data[cav_id]['params']['vehicles'] = curr_params['vehicles'] # 用curr的来取代，这是因为在late fusion的时候要用来考虑延迟

            # # 3. past frames, for model input
            # data[cav_id]['past_k'] = OrderedDict()
            # latest_sample_stamp_idx = curr_timestamp_idx
            # # past k frames, pose | lidar | label(for single view confidence map generator use)
            # for i in range(self.k): # 遍历所有帧 以下做出了修改 将ego也作为none-ego的cav处理从而去训练数据
            #     # sample_interval
            #     if data[cav_id]['ego']:             # ego sample_interval = E(B(n, p))
            #         if i == 0: # ego-past-0 与 ego-curr 是一样的
            #             data[cav_id]['past_k'][i] = data[cav_id]['curr']
            #             continue
            #         sample_interval = self.sample_interval_exp # 同样的采样间隔
            #         if sample_interval == 0:
            #             sample_interval = 1
            #     else:                               # non-ego sample_interval ~ B(n, p)
            #         # delay_debug = 6
            #         # if i == 0:
            #         #     sample_interval = 3 #delay_debug
            #         # elif i ==1:
            #         #     sample_interval = 3
            #         #     # sample_set = [2,3]
            #         #     # import random
            #         #     # sample_interval = random.sample(sample_set, 1)[0] #3 #10 - delay_debug
            #         #     # trails = bernoulliDist.rvs(self.binomial_n)
            #         #     # sample_interval = sum(trails)
            #         # else:
            #         #     sample_interval = 3
            #         #     # trails = bernoulliDist.rvs(self.binomial_n)
            #         #     # sample_interval = sum(trails)
            #         if self.sample_interval_exp==0 \
            #             and self.is_no_shift \
            #                 and i == 0:
            #             data[cav_id]['past_k'][i] = data[cav_id]['curr']
            #             continue
            #         if self.is_same_sample_interval: # 相同时间间隔采样
            #             sample_interval = self.sample_interval_exp
            #         else:
            #             # B(n, p)
            #             trails = bernoulliDist.rvs(self.binomial_n)  # 做10次伯努利实验
            #             sample_interval = sum(trails) # 统计成功次数，作为采样间隔
            #         if sample_interval==0: # 如果采样间隔为0
            #             if i==0: # 检查past 0 的实际时间是否在curr 的后面
            #                 tmp_time_key = list(cav_content.items())[latest_sample_stamp_idx][0] # 取出时间戳的字符串
            #                 if self.dist_time(tmp_time_key, data[cav_id]['curr']['timestamp'])>0: # 检查past0的时间戳是>= cur时间戳，那么最起码要间隔一帧
            #                     sample_interval = 1
            #             if i>0: # 过去的几帧不要重复
            #                 sample_interval = 1                

            #     # check the timestamp index
            #     data[cav_id]['past_k'][i] = {}
            #     latest_sample_stamp_idx -= sample_interval
            #     timestamp_key = list(cav_content.items())[latest_sample_stamp_idx][0] # 获取时间戳
            #     # load the corresponding data into the dictionary
            #     # load param file: json is faster than yaml
            #     json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            #     json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")

            #     if os.path.exists(json_file):
            #         with open(json_file, "r") as f:
            #             data[cav_id]['past_k'][i]['params'] = json.load(f)
            #     else:
            #         data[cav_id]['past_k'][i]['params'] = \
            #             load_yaml(cav_content[timestamp_key]['yaml']) # 获取yaml文件内容
            #     # 没有 lidar pose
            #     if not ('lidar_pose' in data[cav_id]['past_k'][i]['params']):
            #         tmp_ego_pose = np.array(data[cav_id]['past_k'][i]['params']['true_ego_pos'])
            #         tmp_ego_pose += np.array([-0.5, 0, 1.9, 0, 0, 0])
            #         data[cav_id]['past_k'][i]['params']['lidar_pose'] = list(tmp_ego_pose)

            #     # load lidar file: npy is faster than pcd
            #     npy_file = cav_content[timestamp_key]['lidar'].replace("pcd", "npy")
            #     if os.path.exists(npy_file):
            #         data[cav_id]['past_k'][i]['lidar_np'] = np.load(npy_file)
            #     else:
            #         data[cav_id]['past_k'][i]['lidar_np'] = \
            #                 pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

            #     data[cav_id]['past_k'][i]['timestamp'] = timestamp_key
            #     data[cav_id]['past_k'][i]['sample_interval'] = sample_interval
            #     data[cav_id]['past_k'][i]['time_diff'] = \
            #         self.dist_time(timestamp_key, data[cav_id]['curr']['timestamp']) # 延迟后的时间戳-cur时间戳
            
            data[cav_id]['debug'] = {}
            data[cav_id]['debug']['scene'] = scene_name
            data[cav_id]['debug']['cav_id'] = cav_id

        return data


    # def __getitem__(self, idx):
    #     '''
    #     Returns:
    #     ------ 
    #     processed_data_dict : dict consist of all processed info, whose structure is:
    #     {
    #         'single_object_dict_stack': single_label_dict_stack,
    #         'object_bbx_center': object_bbx_center,
    #         'object_bbx_mask': mask,
    #         'object_ids': [object_id_stack[i] for i in unique_indices],
    #         'anchor_box': anchor_box,
    #         'processed_lidar': merged_feature_dict,
    #         'label_dict': label_dict,
    #         'cav_num': cav_num,
    #         'pairwise_t_matrix': pairwise_t_matrix,
    #         'curr_lidar_poses': curr_lidar_poses,
    #         'past_k_lidar_poses': past_k_lidar_poses,
    #         'sample_idx': idx,
    #         'cav_id_list': cav_id_list,
    #         'past_k_time_diffs': past_k_time_diffs_stack, np.array of len(\sum_i^num_cav k_i), k_i represents the num of past frames of cav_i
    #         'avg_sample_interval': float,
    #         'avg_time_delay': float
    #     }
    #     '''
    #     # TODO: debug use
    #     global illegal_path_list

    #     # 首先要判断一下是不是要用历史帧，因为我们需要同样数目的数据集来公平比较
    #     # 构造函数在初始化的时候已经用两帧来限制数据集，这就保证了都在两帧筛选过的数据集上训练和测试，
    #     # 但有的model不需要历史帧，因此在这里才开始改变
    #     if self.w_history is not True:
    #         self.k = 1

    #     # np.random.seed(idx) # 固定随机种子，目的是让每一次训练or推理的结果可以被复现
    #     base_data_dict = self.retrieve_base_data(idx) # 获取了一个场景下的信息
    #     ''' base_data_dict structure:
    #     {
    #         cav_id_1 : {
    #             'ego' : true,
    #             curr : {
    #                 'params': (yaml),
    #                 'lidar_np': (numpy),
    #                 'timestamp': string
    #             },
    #             past_k : {		                # (k) totally
    #                 [0]:{
    #                     'params': (yaml),
    #                     'lidar_np': (numpy),
    #                     'timestamp': string,
    #                     'time_diff': float,
    #                     'sample_interval': int
    #                 },
    #                 [1] : {},	(id-1)
    #                 ...,		
    #                 [k-1] : {} (id-(k-1))
    #             },
    #             'debug' : {                     # debug use
    #                 scene_name : string         
    #                 cav_id : string
    #             }
    #         },
    #         cav_id_2 : { ... }
    #     }
    #     '''

    #     processed_data_dict = OrderedDict()
    #     processed_data_dict['ego'] = {}

    #     # first find the ego vehicle's lidar pose
    #     ego_id = -1
    #     ego_lidar_pose = []
    #     for cav_id, cav_content in base_data_dict.items():
    #         if cav_content['ego']:
    #             ego_id = cav_id
    #             ego_lidar_pose = cav_content['curr']['params']['lidar_pose']
    #             break	
    #     assert cav_id == list(base_data_dict.keys())[
    #         0], "The first element in the OrderedDict must be ego"
    #     assert ego_id != -1
    #     assert len(ego_lidar_pose) > 0

    #     too_far = []
    #     curr_lidar_pose_list = []
    #     cav_id_list = []

    #     # loop over all CAVs to process information
    #     for cav_id, selected_cav_base in base_data_dict.items(): # 遍历场景下的每一辆车
    #         # check if the cav is within the communication range with ego
    #         # for non-ego cav, we use the latest frame's pose
    #         distance = math.sqrt( \
    #             (selected_cav_base['past_k'][0]['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + \
    #                 (selected_cav_base['past_k'][0]['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
    #         # if distance is too far, we will just skip this agent
    #         if distance > self.params['comm_range']:
    #             too_far.append(cav_id)
    #             continue
    #         curr_lidar_pose_list.append(selected_cav_base['curr']['params']['lidar_pose']) # 6dof pose  保存 pose 信息
    #         cav_id_list.append(cav_id)  # 放入cav id
    #     for cav_id in too_far: # filter those out of communicate range
    #         base_data_dict.pop(cav_id)
        
    #     single_label_dict_stack = []
    #     object_stack = []
    #     object_id_stack = []
    #     curr_pose_stack = []
    #     curr_feature_stack = []
    #     past_k_pose_stack = []
    #     past_k_features_stack = [] 
    #     past_k_tr_mats = []
    #     pastk_2_past0_tr_mats = []
    #     past_k_label_dicts_stack = []
    #     past_k_sample_interval_stack = []
    #     past_k_time_diffs_stack = []

    #     # past_k_object_bbx_stack = []
    #     # past_k_cav_object_num = []
    #     # cur_cav_object_bbx_debug= []
    #     # past_k_common_id_index_stack = []

    #     # avg_past_k_time_diffs = 0.0
    #     # avg_past_k_sample_interval = 0.0
    #     illegal_cav = []
    #     if self.visualize:
    #         projected_lidar_stack = []
        
    #     for cav_id in cav_id_list: # 一个场景下遍历每一辆车
    #         selected_cav_base = base_data_dict[cav_id]
    #         ''' selected_cav_base:
    #         {
    #             'ego' : true,
    #             curr : {
    #                 'params': (yaml),
    #                 'lidar_np': (numpy),
    #                 'timestamp': string
    #             },
    #             past_k : {		                # (k) totally
    #                 [0]:{
    #                     'params': (yaml),
    #                     'lidar_np': (numpy),
    #                     'timestamp': string,
    #                     'time_diff': float,
    #                     'sample_interval': int
    #                 },
    #                 [1] : {},	(id-1)
    #                 ...,		
    #                 [k-1] : {} (id-(k-1))
    #             },
    #             'debug' : {                     # debug use
    #                 scene_name : string         
    #                 cav_id : string
    #             }
    #         }
    #         '''
    #         selected_cav_processed = self.get_item_single_car(
    #             selected_cav_base,
    #             ego_lidar_pose,
    #             idx
    #         )
    #         ''' selected_cav_processed : dict
    #         The dictionary contains the cav's processed information.
    #         {
    #             'projected_lidar':      # curr lidar in ego space, 用于viz
    #             'single_label_dict':	# single view label. 没有经过坐标变换, 用于单体监督            cav view + curr 的label
    #             'single_object_bbx_center'
    #             'single_object_ids'
    #             'curr_feature':         # current feature, lidar预处理得到的feature                 cav view + curr feature
    #             'object_bbx_center':	# ego view label. np.ndarray. Shape is (max_num, 7).    ego view + curr 的label
    #             'object_ids':			# ego view label index. list. length is (max_num, 7).   ego view + curr 的label
    #             'curr_pose':			# current pose, list, len = 6
    #             'past_k_poses': 		    # list of past k frames' poses
    #             'past_k_features': 		    # list of past k frames' lidar
    #             'past_k_time_diffs': 	    # list of past k frames' time diff with current frame
    #             'past_k_tr_mats': 		    # list of past k frames' transformation matrix to current ego coordinate
    #             'past_k_sample_interval':   # list of past k frames' sample interval with later frame
    #             # 'avg_past_k_time_diffs':    # avg_past_k_time_diffs,
    #             # 'avg_past_k_sample_interval': # avg_past_k_sample_interval,
    #             'if_no_point':              # bool, 用于判断是否合法
    #             'flow_gt':               # [2, H, W] flow ground truth
    #         }
    #         '''

    #         # if selected_cav_processed is None: # 如果进入到这里，说明flow的三帧匹配失败，即三帧中甚至找不到一个共有的object
    #         #     illegal_cav.append(cav_id) # 加入到非法cav id里，这是因为这个cav在这个时候已经没有必要继续处理下去了
    #         #     continue
    #         if selected_cav_processed['if_no_point']: # 把点的数量不合法的车排除
    #             illegal_cav.append(cav_id)
    #             # 把出现不合法sample的 场景、车辆、时刻 记录下来:
    #             illegal_path = os.path.join(base_data_dict[cav_id]['debug']['scene'], cav_id, base_data_dict[cav_id]['past_k'][0]['timestamp']+'.npy')
    #             illegal_path_list.add(illegal_path)
    #             # print(illegal_path)
    #             continue

            
    #         if self.visualize:
    #             projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
            
    #         # single view feature
    #         curr_feature_stack.append(selected_cav_processed['curr_feature']) #  List 每个元素为字典，记录体素化信息
    #         # single view label
    #         single_label_dict_stack.append(selected_cav_processed['single_label_dict'])

    #         # curr ego view label
    #         object_stack.append(selected_cav_processed['object_bbx_center']) # (n_cur, 7) 已经project到curr ego view
    #         object_id_stack += selected_cav_processed['object_ids'] # 所有的车id全部组合进入一个list

    #         # current pose: N, 6
    #         curr_pose_stack.append(selected_cav_processed['curr_pose']) 
    #         # features: N, k, 
    #         past_k_features_stack.append(selected_cav_processed['past_k_features']) # List 每一个元素也是一个List 包含三帧体素化信息
    #         # poses: N, k, 6
    #         past_k_pose_stack.append(selected_cav_processed['past_k_poses'])
    #         # time differences: N, k
    #         past_k_time_diffs_stack += selected_cav_processed['past_k_time_diffs']
    #         # sample intervals: N, k
    #         past_k_sample_interval_stack += selected_cav_processed['past_k_sample_interval']
    #         # past k frames to ego pose trans matrix, list of len=N, past_k_tr_mats[i]: ndarray(k, 4, 4)
    #         past_k_tr_mats.append(selected_cav_processed['past_k_tr_mats'])
    #         # past k frames to cav past 0 frame trans matrix, list of len=N, pastk_2_past0_tr_mats[i]: ndarray(k, 4, 4)
    #         pastk_2_past0_tr_mats.append(selected_cav_processed['pastk_2_past0_tr_mats'])
    #         # past k label dict: N, k, object_num, 7
    #         # past_k_label_dicts_stack.append(selected_cav_processed['past_k_label_dicts'])
        
    #         # past_k_object_bbx_stack.append(selected_cav_processed['past_k_common_bbx']) # List 长度为agent个数，每个元素为（M，k，7）表示这个cav的k帧中匹配成功的M个object bbx 形如[(M1, k, 7), (M2, k, 7), (M3, k, 7)...]
    #         # if len(selected_cav_processed['past_k_common_bbx'].shape) != 3 or selected_cav_processed['past_k_common_bbx'].shape[0] == 0: # TODO 如果进入了这里，说明有一个cav的三帧中没有连续出现三次的object，即匹配不成功
    #         #     print("Encounter a scenario without any sample to train flow!")
    #         #     return None
    #         # past_k_cav_object_num.append(selected_cav_processed['past_k_common_bbx'].shape[0]) # 记录每个cav的object数 形如[30, 20...] 长度为cav的个数

    #         # cur_cav_object_bbx_debug.append(selected_cav_processed['debug_gt_bbx']) # List [(M1, 7), (M2, 7)...]

        
    #     pastk_2_past0_tr_mats = np.stack(pastk_2_past0_tr_mats, axis=0) # N, k, 4, 4
    #     # past_k_object_bbx_stack = np.vstack(past_k_object_bbx_stack) # 堆叠起来 （M1+M2+...，k, 7） 这就是一个scenario中的所有匹配好的三帧 检索第一维度就是每一个object，后两维则是代表其在三帧内的运动变化
    #     # if past_k_object_bbx_stack.shape[0] == 0 or len(past_k_object_bbx_stack.shape) != 3:
    #     #     print("Encounter a scenario without any sample to train flow!")
    #     #     return None
    #     # cur_cav_object_bbx_debug = np.vstack(cur_cav_object_bbx_debug) # (M1+M2+...., 7)

    #     # {pos: array[num_cav, k, 100, 252, 2], neg: array[num_cav, k, 100, 252, 2], target: array[num_cav, k, 100, 252, 2]}
    #     # past_k_label_dicts = self.post_processor.merge_label_to_dict(past_k_label_dicts_stack)
    #     # self.times.append(time.time())

    #     # filter those cav who has no points left
    #     # then we can calculate get_pairwise_transformation
    #     for cav_id in illegal_cav:
    #         base_data_dict.pop(cav_id)
    #         cav_id_list.remove(cav_id)

    #     merged_curr_feature_dict = self.merge_features_to_dict(curr_feature_stack)  # current 在各自view 下 feature
        
    #     single_label_dict = self.post_processor.collate_batch(single_label_dict_stack) # current 在各自view 下 label 

    #     past_k_time_diffs_stack = np.array(past_k_time_diffs_stack)
    #     past_k_sample_interval_stack = np.array(past_k_sample_interval_stack)
        
    #     pairwise_t_matrix = \
    #         self.get_past_k_pairwise_transformation2ego(base_data_dict, 
    #         ego_lidar_pose, self.max_cav) # np.tile(np.eye(4), (max_cav, self.k, 1, 1)) (L, k, 4, 4) TODO: 这里面没有搞懂为什么不用 past_k_tr_mats 

    #     curr_lidar_poses = np.array(curr_pose_stack).reshape(-1, 6)  # (N_cav, 6)
    #     past_k_lidar_poses = np.array(past_k_pose_stack).reshape(-1, self.k, 6)  # (N, k, 6)

    #     # exclude all repetitive objects    
    #     unique_indices = \
    #         [object_id_stack.index(x) for x in set(object_id_stack)]
    #     try:
    #         object_stack = np.vstack(object_stack) # 元素堆叠，原本每个元素为(cur下object数，7) 表示一个agent的cur下object信息，堆叠后为一个scenario下curr时所有object信息 （N_all， 7）
    #     except ValueError:
    #         print("!!! vstack ValueError !!!")
    #         return None
    #     object_stack = object_stack[unique_indices] # 去重

    #     # make sure bounding boxes across all frames have the same number
    #     object_bbx_center = \
    #         np.zeros((self.params['postprocess']['max_num'], 7)) # （100， 7）
    #     mask = np.zeros(self.params['postprocess']['max_num']) # （100）
    #     object_bbx_center[:object_stack.shape[0], :] = object_stack # （100， 7） 将cur下，一个scenario的所有agent的object bbx整合到一起
    #     mask[:object_stack.shape[0]] = 1 # 最多支持100object 设置掩码 标记有效object

    #     # self.times.append(time.time())

    #     # merge preprocessed features from different cavs into the same dict
    #     cav_num = len(cav_id_list) # 场景中有几辆车记录下
    #     # past_k_features_stack: list, len is num_cav. [i] is list, len is k. [cav_id][time_id] is Orderdict, {'voxel_features': array, ...}
    #     merged_feature_dict = self.merge_past_k_features_to_dict(past_k_features_stack)

    #     # generate the anchor boxes
    #     anchor_box = self.anchor_box

    #     # generate targets label
    #     label_dict = \
    #         self.post_processor.generate_label(
    #             gt_box_center=object_bbx_center,
    #             anchors=anchor_box,
    #             mask=mask)

    #     # self.times.append(time.time())

    #     # self.times = (np.array(self.times[1:]) - np.array(self.times[:-1]))

    #     # self.times = np.hstack((self.times, time4data))

    #     processed_data_dict['ego'].update(
    #         {'single_object_dict_stack': single_label_dict,
    #          'curr_processed_lidar': merged_curr_feature_dict,
    #          'object_bbx_center': object_bbx_center,
    #          'object_bbx_mask': mask,
    #          'object_ids': [object_id_stack[i] for i in unique_indices],
    #          'anchor_box': anchor_box,
    #          'processed_lidar': merged_feature_dict,
    #          'label_dict': label_dict,
    #          'cav_num': cav_num,
    #          'pairwise_t_matrix': pairwise_t_matrix, # （L， K， 4， 4）：每一个agent，每一帧的lidar pose到cur ego的lidar pose的变换矩阵
    #          'curr_lidar_poses': curr_lidar_poses, # （L， 6）：每一个agent的在cur下的lidar  pose
    #          'past_k_lidar_poses': past_k_lidar_poses, # （L，k， 6）：每一个agent的每一帧的lidar pose
    #          'past_k_time_diffs': past_k_time_diffs_stack, # （L*k）：每一个agent每一帧到cur的时间间隔
    #          'past_k_sample_interval': past_k_sample_interval_stack,  # （L*k）：每一个agent下每一帧的采样间隔
    #         #  'past_k_object_bbx': past_k_object_bbx_stack, # (M_all, k, 7) 一个scenario中的所有匹配好的三帧 
    #         #  'past_k_cav_object_num': past_k_cav_object_num, # List (M1， M2...) 一个scenario中每个cav三帧匹配好的object个数 注意M1+M2+... M_n= M_all 其中n是场景中所有的cav数
    #         #  'cur_cav_object_bbx_debug': cur_cav_object_bbx_debug, # (M1+M2+..., 7)
    #          'pastk_2_past0_tr_mats': pastk_2_past0_tr_mats})
    #         #  'times': self.times})
        
    #     processed_data_dict['ego'].update({'sample_idx': idx,
    #                                         'cav_id_list': cav_id_list})
    #     try:
    #         tmp = past_k_time_diffs_stack[self.k:].reshape(-1, self.k) # (N, k)
            
    #         # tmp_past_k_time_diffs = np.concatenate((tmp[:, :1] , (tmp[:, 1:] - tmp[:, :-1])), axis=1) # (N, k) # TODO: 不同的方法这个计算方式不一样
    #         tmp_past_k_time_diffs = tmp[:, :1] # (N, 1) irregular setting 下，只用最近的一个时间间隔
            
    #         avg_time_diff = sum(tmp_past_k_time_diffs.reshape(-1)) / tmp_past_k_time_diffs.reshape(-1).shape[0]
    #         processed_data_dict['ego'].update({'avg_time_delay':\
    #             avg_time_diff})

    #         tmp = past_k_sample_interval_stack[self.k:].reshape(-1, self.k) # (N, k)
    #         # avg_sample_interval = sum(tmp.reshape(-1)) / len(tmp.reshape(-1)) # TODO: 不同的方法这个计算方式不一样
    #         avg_sample_interval = sum(tmp[:, :1].reshape(-1)) / len(tmp[:, :1].reshape(-1)) # (N, 1) irregular setting 下，只用最近的一个时间间隔
    #         processed_data_dict['ego'].update({'avg_sample_interval':\
    #             avg_sample_interval})

    #         tmp_var = np.var(tmp, axis=1) # (N, )
    #         avg_var = np.mean(tmp_var)
    #         processed_data_dict['ego'].update({'avg_var':\
    #             avg_var})
            
    #     except ZeroDivisionError:
    #         # print("!!! ZeroDivisionError !!!")
    #         # print("past_k_time_diffs_stack shape  is", past_k_time_diffs_stack.shape)
    #         # print("illegal_cav is", illegal_cav)
    #         # print("cav_num  is", cav_num)
    #         # 应该是距离导致的超过距离的cav会被踢掉就剩一个ego
    #         return None

    #     if self.visualize:
    #         processed_data_dict['ego'].update({'origin_lidar':
    #             np.vstack(projected_lidar_stack)})

    #     return processed_data_dict

    # def get_item_train(self, base_data_dict):
    #     processed_data_dict = OrderedDict()
    #     # base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])
    #     # during training, we return a random cav's data
    #     # only one vehicle is in processed_data_dict
    #     if not self.visualize:
    #         selected_cav_id, selected_cav_base = \
    #             random.choice(list(base_data_dict.items()))
    #     else:
    #         selected_cav_id, selected_cav_base = \
    #             list(base_data_dict.items())[0]

    #     selected_cav_processed = self.get_item_single_car(selected_cav_base)
    #     processed_data_dict.update({'ego': selected_cav_processed})

    #     return processed_data_dict

    # def get_item_test(self, base_data_dict):
    #     ''' 
    #     Fetch useful info from base_data_dict, filter out too far cav.
    #     Return a dict match the point_pillar model.
        
    #     Params:
    #     ------
    #     base_data_dict : dict
    #         The dictionary contains loaded yaml params and lidar data for
    #         each cav.
    #         Structure: 
    #         {
    #             cav_id_1 : {
    #                 'ego' : true,
    #                 curr : {		(id)			#      |       | label
    #                     'params': (yaml),
    #                     'lidar_np': (numpy),
    #                     'timestamp': string
    #                 },
    #                 past_k : {		# (k) totally
    #                     [0]:{		(id)			# pose | lidar | label
    #                         'params': (yaml),
    #                         'lidar_np': (numpy),
    #                         'timestamp': string,
    #                         'time_diff': float,
    #                         'sample_interval': int
    #                     },
    #                     [1] : {},	(id-1)			# pose | lidar | label
    #                     ...,						# pose | lidar | label
    #                     [k-1] : {} (id-(k-1))		# pose | lidar | label
    #                 },
    #                 'debug' : {                     # debug use
    #                     scene_name : string         
    #                     cav_id : string
    #                 }
                    
    #             }, 
    #             cav_id_2 : {
    #                 'ego': false, 
    #                 curr : 	{		(id)			#      |       | label
    #                         'params': (yaml),
    #                         'lidar_np': (numpy),
    #                         'timestamp': string
    #                 },
    #                 past_k: {		# (k) totally
    #                     [0] : {		(id - \tau)		# pose | lidar | label
    #                         'params': (yaml),
    #                         'lidar_np': (numpy),
    #                         'timestamp': string,
    #                         'time_diff': float,
    #                         'sample_interval': int
    #                     }			
    #                     ..., 						# pose | lidar | label
    #                     [k-1]:{}  (id-\tau-(k-1))	# pose | lidar | label
    #                 },
    #             }, 
    #             ...
    #         }

    #     Returns:
    #     ------ 
    #     {
    #         'ego' : {
    #             'transformation_matrix_curr':
    #             'transformation_matrix_past': 
    #             'if_no_point' : True ,
    #             'debug' : {
    #                 'scene_name' : string,
    #                 'cav_id' : string,
    #                 'time_diff': float (0.0 if 'ego'),
    #                 'sample_interval': int (0 if 'ego')
    #             },
    #             'past_k' : {
    #                 'origin_lidar' : 
    #                 'processed_lidar' : 
    #                 'anchor_box' : 
    #                 'object_bbx_center' : 
    #                 'object_bbx_mask' : 
    #                 'object_ids' : 
    #                 'label_dict' : 
    #             },
    #             'curr' : { ... }
    #         },
    #         cav_id: { ... }     
    #     }
    #     '''
    #     processed_data_dict = OrderedDict()

    #     # first find the ego vehicle's lidar pose
    #     ego_id = -1
    #     ego_lidar_pose = []
    #     for cav_id, cav_content in base_data_dict.items():
    #         if cav_content['ego']:
    #             ego_id = cav_id
    #             ego_lidar_pose = cav_content['curr']['params']['lidar_pose']
    #             break	
    #     assert cav_id == list(base_data_dict.keys())[
    #         0], "The first element in the OrderedDict must be ego"
    #     assert ego_id != -1
    #     assert len(ego_lidar_pose) > 0

    #     # scene_name = base_data_dict[ego_id]['debug']['scene_name']
        
    #     too_far = []
    #     cav_id_list = []

    #     # loop over all CAVs to process information
    #     for cav_id, selected_cav_base in base_data_dict.items():
    #         # check if the cav is within the communication range with ego
    #         # for non-ego cav, we use the latest frame's pose
    #         distance = \
    #             math.sqrt( \
    #                 (selected_cav_base['past_k'][0]['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + \
    #                 (selected_cav_base['past_k'][0]['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
    #         # if distance is too far, we will just skip this agent
    #         if distance > self.params['comm_range']:
    #             too_far.append(cav_id)
    #             continue
    #         cav_id_list.append(cav_id)  
    #     # filter those out of communicate range
    #     for cav_id in too_far:
    #         base_data_dict.pop(cav_id)

    #     if self.visualize:
    #         projected_lidar_stack = []

    #     illegal_cav = []
    #     for cav_id in cav_id_list:
    #         selected_cav_base = base_data_dict[cav_id]

    #         selected_cav_processed = self.get_item_single_car_test(
    #             selected_cav_base
    #         )
    #         '''
    #         selected_cav_processed : {
    #             'if_no_point' : False / True ,
    #             'debug' : {
    #                 'scene_name' : string,
    #                 'cav_id' : string,
    #                 'time_diff': float (0.0 if 'ego'),
    #                 'sample_interval': int (0 if 'ego')
    #             },
    #             'past_k' : {
    #                 'origin_lidar' : 
    #                 'processed_lidar' : 
    #                 'anchor_box' : 
    #                 'object_bbx_center' : 
    #                 'object_bbx_mask' : 
    #                 'object_ids' : 
    #                 'label_dict' : 
    #             },
    #             'curr' : {}
    #         }
    #         '''

    #         if selected_cav_processed['if_no_point']: # 把点的数量不合法的车排除
    #             illegal_cav.append(cav_id)
    #             # # 把出现不合法sample的 场景、车辆、时刻 记录下来:
    #             # illegal_path = os.path.join(base_data_dict[cav_id]['debug']['scene'], cav_id, base_data_dict[cav_id]['past_k'][0]['timestamp']+'.npy')
    #             # illegal_path_list.add(illegal_path)
    #             # print(illegal_path)
    #             continue
            
    #         cav_lidar_pose_past = selected_cav_base['past_k'][0]['params']['lidar_pose']
    #         transformation_matrix_past = x1_to_x2(cav_lidar_pose_past, ego_lidar_pose)
    #         selected_cav_processed.update({'transformation_matrix_past': transformation_matrix_past})

    #         cav_lidar_pose_curr = selected_cav_base['curr']['params']['lidar_pose']
    #         transformation_matrix_curr = x1_to_x2(cav_lidar_pose_curr, ego_lidar_pose)
    #         selected_cav_processed.update({'transformation_matrix_curr': transformation_matrix_curr})
            
    #         update_cav = "ego" if cav_id == ego_id else cav_id
    #         processed_data_dict.update({update_cav: selected_cav_processed})
        
    #     return processed_data_dict

    # def get_item_single_car_test(self, selected_cav_base):
    #     """
    #     Process a single CAV's information for the train/test pipeline.

    #     Parameters
    #     ----------
    #     selected_cav_base : dict
    #         The dictionary contains a single CAV's raw information.
    #         Structure : {
    #             'ego' : true,
    #             curr : {		(id)			#      |       | label
    #                 'params': (yaml),
    #                 'lidar_np': (numpy),
    #                 'timestamp': string
    #             },
    #             past_k : {		# (k) totally
    #                 [0]:{		(id)			# pose | lidar | label
    #                     'params': (yaml),
    #                     'lidar_np': (numpy),
    #                     'timestamp': string,
    #                     'time_diff': float,
    #                     'sample_interval': int
    #                 },
    #                 [1] : {},	(id-1)			# pose | lidar | label
    #                 ...,						# pose | lidar | label
    #                 [k-1] : {} (id-(k-1))		# pose | lidar | label
    #             },
    #             'debug' : {                     # debug use
    #                 scene_name : string         
    #                 cav_id : string
    #             }    
    #         }

    #     Returns
    #     -------
    #     selected_cav_processed : dict
    #         The dictionary contains the cav's processed information.
    #         Structure : {
    #             'if_no_point' : False / True ,
    #             'debug' : {
    #                 'scene_name' : string,
    #                 'cav_id' : string,
    #                 'time_diff': float (0.0 if 'ego'),
    #                 'sample_interval': int (0 if 'ego'),
    #                 'timestamp' : string ('past_k'-0-timestamp)
    #             },
    #             'past_k' : {
    #                 'origin_lidar' : 
    #                 'processed_lidar' : 
    #                 'anchor_box' : 
    #                 'object_bbx_center' : 
    #                 'object_bbx_mask' : 
    #                 'object_ids' : 
    #                 'label_dict' : 
    #             },
    #             'curr' : {
    #                 'origin_lidar' : 
    #                 'processed_lidar' : 
    #                 'anchor_box' : 
    #                 'object_bbx_center' : 
    #                 'object_bbx_mask' : 
    #                 'object_ids' : 
    #                 'label_dict' : 
    #             }
    #         }
    #     """
    #     selected_cav_processed = {}

    #     if_no_point = False
    #     # for past: lidar | feature | pose | label
    #     for part in ['past_k', 'curr']:
    #         processed_part = {}
    #         if part=='past_k':
    #             processing_base = selected_cav_base['past_k'][0]
    #         else:
    #             processing_base = selected_cav_base['curr']

    #         # filter lidar
    #         lidar_np = processing_base['lidar_np']
    #         lidar_np = shuffle_points(lidar_np)
    #         lidar_np = mask_points_by_range(lidar_np,
    #                                         self.params['preprocess'][
    #                                             'cav_lidar_range'])
    #         # remove points that hit ego vehicle
    #         lidar_np = mask_ego_points(lidar_np)
            
    #         # tag illegal situation
    #         if lidar_np.shape[0] == 0: # 没有点留下
    #             selected_cav_processed.update({'if_no_point': True})
    #             return selected_cav_processed

    #         # generate the bounding box(n, 7) under the cav's space
    #         object_bbx_center, object_bbx_mask, object_ids = \
    #             self.generate_object_center([processing_base], processing_base['params']['lidar_pose'])  

    #         # data augmentation
    #         lidar_np, object_bbx_center, object_bbx_mask = \
    #             self.augment(lidar_np, object_bbx_center, object_bbx_mask) # TODO: check

    #         if self.visualize:
    #             processed_part.update({'origin_lidar': lidar_np})

    #         # pre-process the lidar to voxel/bev/downsampled lidar
    #         lidar_dict = self.pre_processor.preprocess(lidar_np)
    #         processed_part.update({'processed_lidar': lidar_dict})

    #         # generate the anchor boxes
    #         anchor_box = self.post_processor.generate_anchor_box()
    #         processed_part.update({'anchor_box': anchor_box})

    #         processed_part.update({'object_bbx_center': object_bbx_center,
    #                                     'object_bbx_mask': object_bbx_mask,
    #                                     'object_ids': object_ids})

    #         # generate targets label
    #         label_dict = \
    #             self.post_processor.generate_label(
    #                 gt_box_center=object_bbx_center,
    #                 anchors=anchor_box,
    #                 mask=object_bbx_mask)
            
    #         processed_part.update({'label_dict': label_dict})

    #         selected_cav_processed.update({part : processed_part})
 
    #     debug_part = {}
    #     # print(selected_cav_base['debug'])
    #     debug_part.update({'time_diff': selected_cav_base['past_k'][0]['time_diff'], 
    #                         'sample_interval': selected_cav_base['past_k'][0]['sample_interval'],
    #                         'scene_name': selected_cav_base['debug']['scene'],
    #                         'cav_id': selected_cav_base['debug']['cav_id'],
    #                         'timestamp': selected_cav_base['past_k'][0]['timestamp']})
    #     selected_cav_processed.update({'if_no_point': if_no_point,
    #                                     'debug': debug_part})

    #     return selected_cav_processed

    # def get_item_single_car(self, selected_cav_base, ego_pose, idx):
    #     """
    #     Project the lidar and bbx to ego space first, and then do clipping.

    #     Parameters
    #     ----------
    #     selected_cav_base : dict
    #         The dictionary contains a single CAV's raw information, 
    #         structure: {
    #             'ego' : true,
    #             'curr' : {
    #                 'params': (yaml),
    #                 'lidar_np': (numpy),
    #                 'timestamp': string
    #             },
    #             'past_k' : {		           # (k) totally
    #                 [0]:{
    #                     'params': (yaml),
    #                     'lidar_np': (numpy),
    #                     'timestamp': string,
    #                     'time_diff': float,
    #                     'sample_interval': int
    #                 },
    #                 [1] : {},
    #                 ...,		
    #                 [k-1] : {}
    #             },
    #             'debug' : {                     # debug use
    #                 scene_name : string         
    #                 cav_id : string
    #             }
    #         }

    #     ego_pose : list, length 6
    #         The ego vehicle lidar pose under world coordinate.

    #     idx: int,
    #         debug use.

    #     Returns
    #     -------
    #     selected_cav_processed : dict
    #         The dictionary contains the cav's processed information.
    #         {
    #             'projected_lidar':      # lidar in ego space, 用于viz
    #             'single_label_dict':	# single view label. 没有经过坐标变换,                      cav view + curr 的label
    #             "single_object_bbx_center": single_object_bbx_center,       # 用于viz single view
    #             "single_object_ids": single_object_ids,                # 用于viz single view
    #             'flow_gt':              # single view flow
    #             'curr_feature':         # current feature, lidar预处理得到的feature
    #             'object_bbx_center':	# ego view label. np.ndarray. Shape is (max_num, 7).    ego view + curr 的label
    #             'object_ids':			# ego view label index. list. length is (max_num, 7).   ego view + curr 的label
    #             'curr_pose':			# current pose, list, len = 6
    #             'past_k_poses': 		    # list of past k frames' poses
    #             'past_k_features': 		    # list of past k frames' lidar
    #             'past_k_time_diffs': 	    # list of past k frames' time diff with current frame
    #             'past_k_tr_mats': 		    # list of past k frames' transformation matrix to current ego coordinate
    #             'pastk_2_past0_tr_mats':    # list of past k frames' transformation matrix to past 0 frame
    #             'past_k_sample_interval':   # list of past k frames' sample interval with later frame
    #             'avg_past_k_time_diffs':    # avg_past_k_time_diffs,
    #             'avg_past_k_sample_interval': # avg_past_k_sample_interval,
    #             'if_no_point':              # bool, 用于判断是否合法
    #         }
    #     """
    #     selected_cav_processed = {}

    #     # curr lidar feature
    #     lidar_np = selected_cav_base['curr']['lidar_np'] # 当前的点云信息
    #     lidar_np = shuffle_points(lidar_np)
    #     lidar_np = mask_ego_points(lidar_np) # remove points that hit itself

    #     if self.visualize:
    #         # trans matrix
    #         transformation_matrix = \
    #             x1_to_x2(selected_cav_base['curr']['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray  cav到ego的转换矩阵  当前的cav到当前的ego
    #         projected_lidar = \
    #             box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix) # 点云投影过去
    #         selected_cav_processed.update({'projected_lidar': projected_lidar})

    #     lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range']) # 剔除雷达范围以外

    #     curr_feature = self.pre_processor.preprocess(lidar_np) # 体素化 返回字典，有三个元素 体素信息(N, 32, 4) 体素坐标：(N, 3) 每个体素中点的数量：（N）
        
    #     # past k transfomation matrix
    #     past_k_tr_mats = []
    #     # past_k to past_0 tansfomation matrix
    #     pastk_2_past0_tr_mats = []
    #     # past k lidars
    #     past_k_features = []
    #     # past k poses
    #     past_k_poses = []
    #     # past k timestamps
    #     past_k_time_diffs = []
    #     # past k sample intervals
    #     past_k_sample_interval = []

    #     # for debug use
    #     # avg_past_k_time_diff = 0
    #     # avg_past_k_sample_interval = 0
        
    #     # past k label 
    #     # past_k_label_dicts = [] # todo 这个部分可以删掉

    #     # past_k_object_bbx = []
    #     # past_k_object_ids = []
    #     # 判断点的数量是否合法
    #     if_no_point = False

    #     # past k frames [trans matrix], [lidar feature], [pose], [time interval]
    #     for i in range(self.k): # 遍历k帧
    #         # 1. trans matrix
    #         transformation_matrix = \
    #             x1_to_x2(selected_cav_base['past_k'][i]['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray 求得这一帧的cav pose 到ego的转换矩阵 (4, 4)
    #         past_k_tr_mats.append(transformation_matrix)
    #         # past_k trans past_0 matrix
    #         pastk_2_past0 = \
    #             x1_to_x2(selected_cav_base['past_k'][i]['params']['lidar_pose'], selected_cav_base['past_k'][0]['params']['lidar_pose']) # 求每一帧到第0帧的转换矩阵
    #         pastk_2_past0_tr_mats.append(pastk_2_past0)
            
    #         # 2. lidar feature
    #         lidar_np = selected_cav_base['past_k'][i]['lidar_np']
    #         lidar_np = shuffle_points(lidar_np)
    #         lidar_np = mask_ego_points(lidar_np) # remove points that hit itself
    #         lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])

    #         processed_features = self.pre_processor.preprocess(lidar_np) # 预处理点云 也即体素化 
    #         past_k_features.append(processed_features)

    #         if lidar_np.shape[0] == 0: # 没有点留下
    #             if_no_point = True

    #         # 3. pose
    #         past_k_poses.append(selected_cav_base['past_k'][i]['params']['lidar_pose'])

    #         # 4. time interval and sample interval
    #         past_k_time_diffs.append(selected_cav_base['past_k'][i]['time_diff'])
    #         past_k_sample_interval.append(selected_cav_base['past_k'][i]['sample_interval'])

    #         ################################################################
    #         # sizhewei
    #         # for past k frames' single view label
    #         ################################################################
    #         # # 5. single view label
    #         # # past_i label at past_i single view
    #         # # opencood/data_utils/post_processor/base_postprocessor.py
    #         # object_bbx_center, object_bbx_mask, object_ids = \
    #         #     self.generate_object_center([selected_cav_base['past_k'][i]], selected_cav_base['past_k'][0]['params']['lidar_pose'])  # 这是将这一帧对应的object表示取出 TODO 这里我已经将参考pose全部统一成pastk0的lidar pose
    #         #     # self.generate_object_center([selected_cav_base['past_k'][i]], selected_cav_base['past_k'][0]['params']['lidar_pose'])  # 这是将这一帧对应的object表示取出 TODO 这里我已经将参考pose全部统一成pastk0的lidar pose
    #         # past_k_object_bbx.append(object_bbx_center[object_bbx_mask == 1])  # List 每一帧的object bbx 每一个元素的形状为（有效bbx个数，7）
    #         # past_k_object_ids.append(object_ids) # 

    #         # # generate the anchor boxes
    #         # # opencood/data_utils/post_processor/voxel_postprocessor.py
    #         # anchor_box = self.anchor_box
    #         # single_view_label_dict = self.post_processor.generate_label(
    #         #         gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask
    #         #     )
    #         # past_k_label_dicts.append(single_view_label_dict)
        
        

    #     '''
    #     # past k merge
    #     past_k_label_dicts = self.post_processor.merge_label_to_dict(past_k_label_dicts)
    #     '''
    #     # # 取出三帧共同索引id 
    #     # # print(len(past_k_object_ids))
    #     # past_k_common_id_index = self.find_common_id(past_k_object_ids) # 三个元素  每个元素对应各自的bbx索引
    #     # # print(past_k_common_id_index[0])
    #     # # print(past_k_common_id_index[1])
    #     # # print(past_k_common_id_index[2])
    #     # # for i in range(3):
    #     # #     past = torch.from_numpy(np.array(past_k_object_ids[i]))
    #     # #     print(past[past_k_common_id_index[i]])
    #     # # print("+++++++++++++++++++++")

    #     # past_k_common_bbx_list = []
    #     # for i in range(len(past_k_common_id_index)): # 通过共有bbx 索引取出其bbx [[M.7], [M,7], [M,7]]表示从past0到past2一共三帧的匹配bbx  较低概率出现每个元素是形状(7, )
    #     #     past = past_k_object_bbx[i][past_k_common_id_index[i]] # 取出共同的object
    #     #     if len(past.shape) == 1: # 只有一个object的时候，past会变成（7，）会导致下面转置出错 TODO 还有一种情况，即匹配失败，那past会是（0， 7） 这个时候样本就可以丢弃
    #     #         past = past.reshape(1, 7)
    #     #     past_k_common_bbx_list.append(past)
    #     #     # print(past_k_common_bbx_list[i].shape)
    #     # past_k_common_bbx = np.stack(past_k_common_bbx_list, axis=0) # （k，M， 7 ） 一个cav下所有的匹配上的object数为M
    #     # # print(past_k_common_bbx.shape)
            
    #     # past_k_common_bbx = np.transpose(past_k_common_bbx, (1, 0, 2)) # (M, k, 7)
    #     # if past_k_common_bbx.shape[0] == 0: # 也就是三帧没有匹配成功
    #     #     return None

    #     past_k_tr_mats = np.stack(past_k_tr_mats, axis=0) # (k, 4, 4)
    #     pastk_2_past0_tr_mats = np.stack(pastk_2_past0_tr_mats, axis=0) # (k, 4, 4)

    #     # avg_past_k_time_diffs = float(sum(past_k_time_diffs) / len(past_k_time_diffs))
    #     # avg_past_k_sample_interval = float(sum(past_k_sample_interval) / len(past_k_sample_interval))

    #     # curr label at single view
    #     # opencood/data_utils/post_processor/base_postprocessor.py
    #     single_object_bbx_center, single_object_bbx_mask, single_object_ids = \
    #         self.generate_object_center([selected_cav_base['curr']], selected_cav_base['curr']['params']['lidar_pose'])  # 生成bbx，投影到current cav位置
    #     # generate the anchor boxes
    #     # opencood/data_utils/post_processor/voxel_postprocessor.py
    #     anchor_box = self.anchor_box
    #     label_dict = self.post_processor.generate_label(
    #             gt_box_center=single_object_bbx_center, anchors=anchor_box, mask=single_object_bbx_mask
    #         ) # 这是用来train 检测器的  cur下的object转化为target学习偏移
        
    #     # curr label at ego view
    #     object_bbx_center, object_bbx_mask, object_ids = \
    #         self.generate_object_center([selected_cav_base['curr']], ego_pose)
            
    #     selected_cav_processed.update({
    #         "single_label_dict": label_dict, # Dict：用来train的数据，表示预置anchor与object的偏移量
    #         "single_object_bbx_center": single_object_bbx_center[single_object_bbx_mask == 1], # （n_cur, 7）当前时间戳的cav周围的object 并且project到相应的cav view
    #         "single_object_ids": single_object_ids, # List: (n_cur) cur下cav的感知范围内的车id
    #         "curr_feature": curr_feature, # Dict：三个元素 表示cur下的体素化信息
    #         'object_bbx_center': object_bbx_center[object_bbx_mask == 1], # （n_cur, 7）时间为cur下，n为场景中实际的object的数量，与上面的single_object_bbx_center区别在于这个是project到ego的view上
    #         'object_ids': object_ids, # List: (n_cur) cur下cav的感知范围内的车id
    #         'curr_pose': selected_cav_base['curr']['params']['lidar_pose'], # cur下的pose
    #         'past_k_tr_mats': past_k_tr_mats, # （K，4, 4）：历史k帧到ego cur的变换矩阵
    #         'pastk_2_past0_tr_mats': pastk_2_past0_tr_mats, # （K，4, 4）：历史k帧到past0帧的变换矩阵
    #         'past_k_poses': past_k_poses, # List：k帧每一帧的pose，每一个元素shape为（6）
    #         'past_k_features': past_k_features, # List：k帧每一帧的体素化信息，k个元素，每一个都是Dict
    #         'past_k_time_diffs': past_k_time_diffs, # List：k帧每一帧距离cur 的时间间隔，每一个元素为float
    #         'past_k_sample_interval': past_k_sample_interval, # List：k帧每一帧的样本间隔，每一个元素为int
    #         # 'past_k_common_bbx': past_k_common_bbx, # （M，k，7）这个cav的k帧中匹配成功的M个object bbx
    #     #  'avg_past_k_time_diffs': avg_past_k_time_diffs,
    #     #  'avg_past_k_sample_interval': avg_past_k_sample_interval,
    #     #  'past_k_label_dicts': past_k_label_dicts,
    #         'if_no_point': if_no_point
    #         })

    #     return selected_cav_processed

    # def merge_past_k_features_to_dict(self, processed_feature_list):
    #     """
    #     Merge the preprocessed features from different cavs to the same
    #     dictionary.

    #     Parameters
    #     ----------
    #     processed_feature_list : list
    #         A list of dictionary containing all processed features from
    #         different cavs.

    #     Returns
    #     -------
    #     merged_feature_dict: dict
    #         key: feature names, value: list of features.
    #     """

    #     merged_feature_dict = OrderedDict()

    #     for cav_id in range(len(processed_feature_list)):
    #         for time_id in range(self.k):
    #             for feature_name, feature in processed_feature_list[cav_id][time_id].items():
    #                 if feature_name not in merged_feature_dict:
    #                     merged_feature_dict[feature_name] = []
    #                 if isinstance(feature, list):
    #                     merged_feature_dict[feature_name] += feature
    #                 else:
    #                     merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
    #     return merged_feature_dict

    # @staticmethod
    # def merge_features_to_dict(processed_feature_list):
    #     """
    #     Merge the preprocessed features from different cavs to the same
    #     dictionary.

    #     Parameters
    #     ----------
    #     processed_feature_list : list
    #         A list of dictionary containing all processed features from
    #         different cavs.

    #     Returns
    #     -------
    #     merged_feature_dict: dict
    #         key: feature names, value: list of features.
    #     """

    #     merged_feature_dict = OrderedDict()

    #     for i in range(len(processed_feature_list)):
    #         for feature_name, feature in processed_feature_list[i].items():
    #             if feature_name not in merged_feature_dict:
    #                 merged_feature_dict[feature_name] = []
    #             if isinstance(feature, list):
    #                 merged_feature_dict[feature_name] += feature
    #             else:
    #                 merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
    #     return merged_feature_dict

    # def collate_batch_train(self, batch):
    #     '''
    #     Parameters:
    #     ----------
    #     batch[i]['ego'] structure:
    #     {   
    #         'single_object_dict_stack': single_label_dict_stack,
    #         'curr_processed_lidar': merged_curr_feature_dict,
    #         'object_bbx_center': object_bbx_center,
    #         'object_bbx_mask': mask,
    #         'object_ids': [object_id_stack[i] for i in unique_indices],
    #         'anchor_box': anchor_box,
    #         'processed_lidar': merged_feature_dict,
    #         'label_dict': label_dict,
    #         'cav_num': cav_num,
    #         'pairwise_t_matrix': pairwise_t_matrix,
    #         'curr_lidar_poses': curr_lidar_poses,
    #         'past_k_lidar_poses': past_k_lidar_poses,
    #         'sample_idx': idx,
    #         'cav_id_list': cav_id_list,
    #         'past_k_time_diffs': past_k_time_diffs_stack, 
    #             list of len(\sum_i^num_cav k_i), k_i represents the num of past frames of cav_i
    #         'flow_gt': [N, C, H, W]
    #     }
    #     '''
    #     for i in range(len(batch)):
    #         if batch[i] is None:
    #             return None
    #     # Intermediate fusion is different the other two
    #     output_dict = {'ego': {}}

    #     single_object_label = []

    #     pos_equal_one_single = []
    #     neg_equal_one_single = []
    #     targets_single = []

    #     object_bbx_center = []
    #     object_bbx_mask = []
    #     object_ids = []
    #     curr_processed_lidar_list = []
    #     processed_lidar_list = []
    #     # used to record different scenario
    #     record_len = []
    #     label_dict_list = []
    #     curr_lidar_pose_list = []
    #     past_k_lidar_pose_list = []
    #     past_k_label_list = []
    #     # store the time interval of each feature map
    #     past_k_time_diff = []
    #     past_k_sample_interval = []
    #     past_k_avg_time_delay = []
    #     past_k_avg_sample_interval = []
    #     past_k_avg_time_var = []
    #     pastk_2_past0_tr_mats = []
    #     # pairwise transformation matrix
    #     pairwise_t_matrix_list = []
    #     # past_k_object_bbx_list = []
    #     # cur_object_bbx_debug_list = []
    #     # past_k_object_cav_num_list = []

    #     # for debug use:
    #     sum_time_diff = 0.0
    #     sum_sample_interval = 0.0
    #     # time_consume = np.zeros_like(batch[0]['ego']['times'])

    #     if self.visualize:
    #         origin_lidar = []
        
    #     for i in range(len(batch)): # 遍历每一个样本，一个样本就是一个scenario
    #         ego_dict = batch[i]['ego']
    #         single_object_label.append(ego_dict['single_object_dict_stack']) # TODO：下面又是分开处理，这里这个有什么用？

    #         pos_equal_one_single.append(ego_dict['single_object_dict_stack']['pos_equal_one']) # 添加进入列表，每个元素都是一维Tensor：长度为N，scenario中的agent个数，每个元素（H， W， 2） 0/1张量表示iou合适的anchor标记起来
    #         neg_equal_one_single.append(ego_dict['single_object_dict_stack']['neg_equal_one'])
    #         targets_single.append(ego_dict['single_object_dict_stack']['targets'])

    #         curr_processed_lidar_list.append(ego_dict['curr_processed_lidar'])
    #         object_bbx_center.append(ego_dict['object_bbx_center'])
    #         object_bbx_mask.append(ego_dict['object_bbx_mask'])
    #         object_ids.append(ego_dict['object_ids'])
    #         curr_lidar_pose_list.append(ego_dict['curr_lidar_poses']) # ego_dict['curr_lidar_pose'] is np.ndarray [N,6]
    #         past_k_lidar_pose_list.append(ego_dict['past_k_lidar_poses']) # ego_dict['past_k_lidar_pose'] is np.ndarray [N,k,6]
    #         past_k_time_diff.append(ego_dict['past_k_time_diffs']) # ego_dict['past_k_time_diffs'] is np.array(), len=nxk
    #         past_k_sample_interval.append(ego_dict['past_k_sample_interval']) # ego_dict['past_k_sample_interval'] is np.array(), len=nxk
    #         past_k_avg_sample_interval.append(ego_dict['avg_sample_interval']) # ego_dict['avg_sample_interval'] is float
    #         past_k_avg_time_delay.append(ego_dict['avg_time_delay']) # ego_dict['avg_sample_interval'] is float
    #         try:
    #             past_k_avg_time_var.append(ego_dict['avg_var'])
    #         except KeyError:
    #             past_k_avg_time_var.append(-1.0)
    #         # avg_time_delay += ego_dict['avg_time_delay']
    #         # avg_sample_interval += ego_dict['avg_sample_interval']
    #         processed_lidar_list.append(ego_dict['processed_lidar']) # different cav_num, ego_dict['processed_lidar'] is list.
    #         record_len.append(ego_dict['cav_num'])
    #         label_dict_list.append(ego_dict['label_dict']) # 每个元素就是一个场景下的协同label
    #         pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
    #         pastk_2_past0_tr_mats.append(ego_dict['pastk_2_past0_tr_mats'])
    #         # past_k_label_list.append(ego_dict['past_k_label_dicts'])

    #         # past_k_object_bbx_list.append(ego_dict['past_k_object_bbx']) # List 每个元素为（一个scenario下所有object数， k， 7）
    #         # past_k_object_cav_num_list += ego_dict['past_k_cav_object_num'] # List 原本ego_dict['past_k_cav_object_num']为列表，表示一个scenario下各个cav的匹配object数，长度为对应场景下的cav数，这里记为n_下标 则形如[n1, n2] batchsize=2
    #         # cur_object_bbx_debug_list.append(ego_dict['cur_cav_object_bbx_debug']) # List 每个元素为（一个scenario下所有object数，  7）
    #         # # print("一个scenario下cav的匹配object数,  ego_dict['past_k_cav_object_num']:  " , ego_dict['past_k_cav_object_num'])


    #         # time_consume += ego_dict['times']
    #         if self.visualize:
    #             origin_lidar.append(ego_dict['origin_lidar'])
        
    #     # past_k_object_bbx = torch.from_numpy(np.vstack(past_k_object_bbx_list)) # 一个batch中的所有object的三帧变化，（M_b， k, 7）
    #     # past_k_object_cav_num = torch.from_numpy(np.array(past_k_object_cav_num_list)) # 记录一个batch中的所有cav的每一个的object数量 其长度和batch中的所有cav个数保持一致 
    #     # cur_object_bbx_debug = torch.from_numpy(np.vstack(cur_object_bbx_debug_list)) # 一个batch中的所有object的cur，（M_b, 7）

    #     # single_object_label = self.post_processor.collate_batch(single_object_label)
    #     single_object_label = { "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
    #                             "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
    #                              "targets": torch.cat(targets_single, dim=0)}
                                 
    #     # collate past k single view label from different batch [B, cav_num, k, 100, 252, 2]...
    #     past_k_single_label_torch_dict = self.post_processor.collate_batch(past_k_label_list)
        
    #     # collate past k time interval from different batch, (B, )
    #     past_k_time_diff = np.hstack(past_k_time_diff) # 本来是一个列表 长度为B，每一个元素长度为L*K，L不是固定长度，表示场景下的cav个数, 第二个轴上堆叠，
    #     past_k_time_diff= torch.from_numpy(past_k_time_diff)

    #     # collate past k sample interval from different batch, (B, )
    #     past_k_sample_interval = np.hstack(past_k_sample_interval)
    #     past_k_sample_interval = torch.from_numpy(past_k_sample_interval)

    #     past_k_avg_sample_interval = np.array(past_k_avg_sample_interval)
    #     avg_sample_interval = float(sum(past_k_avg_sample_interval) / len(past_k_avg_sample_interval))

    #     past_k_avg_time_delay = np.array(past_k_avg_time_delay)
    #     avg_time_delay = float(sum(past_k_avg_time_delay) / len(past_k_avg_time_delay))

    #     past_k_avg_time_var = np.array(past_k_avg_time_var)
    #     avg_time_var = float(sum(past_k_avg_time_var) / len(past_k_avg_time_var))

    #     # convert to numpy, (B, max_num, 7)
    #     object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
    #     # （B, max_num)
    #     object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        
    #     curr_merged_feature_dict = self.merge_features_to_dict(curr_processed_lidar_list)
    #     curr_processed_lidar_torch_dict = \
    #         self.pre_processor.collate_batch(curr_merged_feature_dict)
    #     # processed_lidar_list: list, len is 6. [batch_i] is OrderedDict, 3 keys: {'voxel_features': , ...}
    #     # example: {'voxel_features':[np.array([1,2,3]]),
    #     # np.array([3,5,6]), ...]}
    #     merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
    #     # [sum(record_len), C, H, W]
    #     processed_lidar_torch_dict = \
    #         self.pre_processor.collate_batch(merged_feature_dict)
    #     # [2, 3, 4, ..., M], M <= max_cav
    #     record_len = torch.from_numpy(np.array(record_len, dtype=int))
    #     # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
    #     curr_lidar_pose = torch.from_numpy(np.concatenate(curr_lidar_pose_list, axis=0))
    #     # [[N1, k, 6], [N2, k, 6]...] -> [(N1+N2+...), k, 6]
    #     past_k_lidar_pose = torch.from_numpy(np.concatenate(past_k_lidar_pose_list, axis=0))
    #     label_torch_dict = \
    #         self.post_processor.collate_batch(label_dict_list)

    #     # (B, max_cav, k, 4, 4)
    #     pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

    #     pastk_2_past0_tr_mats = torch.from_numpy(np.vstack(pastk_2_past0_tr_mats))

    #     # add pairwise_t_matrix to label dict
    #     label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
    #     label_torch_dict['record_len'] = record_len

    #     # for debug use: 
    #     # time_consume = torch.from_numpy(time_consume)

    #     # object id is only used during inference, where batch size is 1.
    #     # so here we only get the first element.
    #     output_dict['ego'].update({'single_object_label': single_object_label,
    #                                'curr_processed_lidar': curr_processed_lidar_torch_dict,
    #                                'object_bbx_center': object_bbx_center,
    #                                'object_bbx_mask': object_bbx_mask,
    #                                'processed_lidar': processed_lidar_torch_dict,
    #                                'record_len': record_len,
    #                                'label_dict': label_torch_dict,
    #                                'single_past_dict': past_k_single_label_torch_dict,
    #                                'object_ids': object_ids[0],
    #                                'pairwise_t_matrix': pairwise_t_matrix,
    #                                'pastk_2_past0_tr_mats': pastk_2_past0_tr_mats,
    #                                'curr_lidar_pose': curr_lidar_pose,
    #                                'past_lidar_pose': past_k_lidar_pose,
    #                                'past_k_time_interval': past_k_time_diff, # (所有帧的数量)
    #                                'past_k_sample_interval': past_k_sample_interval,
    #                             #    'past_k_object_bbx': past_k_object_bbx, # 一个batch中的所有object的三帧变化，（M_b， k, 7）
    #                             #    'past_k_object_cav_num': past_k_object_cav_num, # 一维张量 形如（M1，M2....）其中存储每个cav的object数量 用于后续恢复形状对应到相应cav
    #                             #    'cur_object_bbx_debug': cur_object_bbx_debug, 
    #                                'avg_sample_interval': avg_sample_interval,
    #                                'avg_time_delay': avg_time_delay,
    #                                'avg_time_var': avg_time_var})
    #                             #    'times': time_consume})
    #     output_dict['ego'].update({'anchor_box':
    #             torch.from_numpy(np.array(self.anchor_box))})
        
    #     if self.visualize:
    #         origin_lidar = \
    #             np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
    #         origin_lidar = torch.from_numpy(origin_lidar)
    #         output_dict['ego'].update({'origin_lidar': origin_lidar})
            
    #     # if self.params['preprocess']['core_method'] == 'SpVoxelPreprocessor' and \
    #     #     (output_dict['ego']['processed_lidar']['voxel_coords'][:, 0].max().int().item() + 1) != record_len.sum().int().item():
    #     #     return None

    #     return output_dict

    # def collate_batch_test(self, batch):
    #     """
    #     Parameters:
    #     -----------
    #     batch: list, len = batch_size
    #     [0] : {
    #         'ego' : {
    #             'transformation_matrix_curr':               cav['curr'] to ego['curr']
    #             'transformation_matrix_past':               cav['past'][0] to ego['curr']
    #             'if_no_point' : True ,
    #             'debug' : {
    #                 'scene_name' : string,
    #                 'cav_id' : string,
    #                 'time_diff': float (0.0 if 'ego'),
    #                 'sample_interval': int (0 if 'ego')
    #             },
    #             'past_k' : {
    #                 'origin_lidar' : 
    #                 'processed_lidar' : 
    #                 'anchor_box' : 
    #                 'object_bbx_center' : 
    #                 'object_bbx_mask' : 
    #                 'object_ids' : 
    #                 'label_dict' : 
    #             },
    #             'curr' : { ... }
    #         },
    #         cav_id: { ... }        
    #     } 

    #     Returns:
    #     ------
    #     Structure: {
    #         'ego' / cav_id : {
    #             'anchor_box' : ,
    #             'object_bbx_center':                    curr,
    #             'object_bbx_mask':                      curr,
    #             'processed_lidar':                      past_k 0 ,
    #             'label_dict':                           curr,
    #             'object_ids':                           curr,
    #             'transformation_matrix':                cav-past to ego-curr,
    #             'transformation_matrix_clean':          cav-curr to ego-curr,
    #             'debug' : {
    #                 'scene_name' : string,
    #                 'cav_id' : string,
    #                 'time_diff': float (0.0 if 'ego'),
    #                 'sample_interval': int (0 if 'ego')
    #             },
    #             'origin_lidar' :                        cav-curr in ego view
    #         },
    #         cav_id : { ... }
    #     }
    #     """
    #     if batch[0] is None:
    #         return None
    #     # currently, we only support batch size of 1 during testing
    #     assert len(batch) <= 1, "Batch size 1 is required during testing!"
    #     batch = batch[0]

    #     output_dict = {}

    #     # for late fusion, we also need to stack the lidar for better
    #     # visualization
    #     if self.visualize:
    #         projected_lidar_list = []
    #         origin_lidar = []

    #     for cav_id, cav_content in batch.items():
    #         output_dict.update({cav_id: {}})
    #         # shape: (1, max_num, 7)
    #         object_bbx_center = \
    #             torch.from_numpy(np.array([cav_content['curr']['object_bbx_center']]))
    #         object_bbx_mask = \
    #             torch.from_numpy(np.array([cav_content['curr']['object_bbx_mask']]))
    #         object_ids = cav_content['curr']['object_ids']

    #         # the anchor box is the same for all bounding boxes usually, thus
    #         # we don't need the batch dimension.
    #         if cav_content['past_k']['anchor_box'] is not None:
    #             output_dict[cav_id].update({'anchor_box':
    #                 torch.from_numpy(np.array(
    #                     cav_content['past_k']['anchor_box']))})
    #         if self.visualize:
    #             transformation_matrix = cav_content['transformation_matrix_curr']
    #             origin_lidar = [cav_content['curr']['origin_lidar']] # TODO: check

    #             if (self.params['only_vis_ego'] is False) or (cav_id=='ego'):
    #                 # print(cav_id)
    #                 import copy
    #                 projected_lidar = copy.deepcopy(cav_content['curr']['origin_lidar']) # TODO: check
    #                 projected_lidar[:, :3] = \
    #                     box_utils.project_points_by_matrix_torch(
    #                         projected_lidar[:, :3],
    #                         transformation_matrix)
    #                 projected_lidar_list.append(projected_lidar)

    #         # processed lidar dictionary
    #         processed_lidar_torch_dict = \
    #             self.pre_processor.collate_batch(
    #                 [cav_content['past_k']['processed_lidar']])
    #         # label dictionary
    #         label_torch_dict = \
    #             self.post_processor.collate_batch([cav_content['curr']['label_dict']]) # TODO: check

    #         # save the transformation matrix (4, 4) to ego vehicle
    #         transformation_matrix_torch = \
    #             torch.from_numpy(
    #                 np.array(cav_content['transformation_matrix_past'])).float()
            
    #         # late fusion training, no noise
    #         transformation_matrix_clean_torch = \
    #             torch.from_numpy(
    #                 np.array(cav_content['transformation_matrix_curr'])).float()

    #         output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
    #                                     'object_bbx_mask': object_bbx_mask,
    #                                     'processed_lidar': processed_lidar_torch_dict,
    #                                     'label_dict': label_torch_dict,
    #                                     'object_ids': object_ids,
    #                                     'transformation_matrix': transformation_matrix_torch,
    #                                     'transformation_matrix_clean': transformation_matrix_clean_torch})

    #         if self.visualize:
    #             origin_lidar = \
    #                 np.array(
    #                     downsample_lidar_minimum(pcd_np_list=origin_lidar))
    #             origin_lidar = torch.from_numpy(origin_lidar)
    #             output_dict[cav_id].update({'origin_lidar': origin_lidar})

    #         output_dict[cav_id].update({'debug' : cav_content['debug']})

    #     if self.visualize:
    #         projected_lidar_stack = [torch.from_numpy(
    #             np.vstack(projected_lidar_list))]
    #         output_dict['ego'].update({'origin_lidar': projected_lidar_stack})
    #         # output_dict['ego'].update({'projected_lidar_list': projected_lidar_list})

    #     return output_dict

    def post_process_no_fusion(self, data_dict, output_dict_ego, return_uncertainty=False):
        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        if return_uncertainty:
            pred_box_tensor, pred_score, uncertainty = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
            return pred_box_tensor, pred_score, gt_box_tensor, uncertainty
        else:
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego)
            return pred_box_tensor, pred_score, gt_box_tensor
            
    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    # def get_pairwise_transformation(self, base_data_dict, max_cav):
    #     """
    #     Get pair-wise transformation matrix accross different agents.

    #     Parameters
    #     ----------
    #     base_data_dict : dict
    #         Key : cav id, item: transformation matrix to ego, lidar points.

    #     max_cav : int
    #         The maximum number of cav, default 5

    #     Return
    #     ------
    #     pairwise_t_matrix : np.array
    #         The pairwise transformation matrix across each cav.
    #         shape: (L, L, 4, 4), L is the max cav number in a scene
    #         pairwise_t_matrix[i, j] is Tji, i_to_j
    #     """
    #     pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

    #     if self.proj_first:
    #         # if lidar projected to ego first, then the pairwise matrix
    #         # becomes identity
    #         # no need to warp again in fusion time.

    #         # pairwise_t_matrix[:, :] = np.identity(4)
    #         return pairwise_t_matrix
    #     else:
    #         t_list = []

    #         # save all transformation matrix in a list in order first.
    #         for cav_id, cav_content in base_data_dict.items():
    #             lidar_pose = cav_content['curr']['params']['lidar_pose']
    #             t_list.append(x_to_world(lidar_pose))  # Twx

    #         for i in range(len(t_list)):
    #             for j in range(len(t_list)):
    #                 # identity matrix to self
    #                 if i != j:
    #                     # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
    #                     # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
    #                     t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
    #                     pairwise_t_matrix[i, j] = t_matrix

    #     return pairwise_t_matrix
    
    # def get_past_k_pairwise_transformation2ego(self, base_data_dict, ego_pose, max_cav):
    #     """
    #     Get transformation matrixes accross different agents to curr ego at all past timestamps.

    #     Parameters
    #     ----------
    #     base_data_dict : dict
    #         Key : cav id, item: transformation matrix to ego, lidar points.
        
    #     ego_pose : list
    #         ego pose

    #     max_cav : int
    #         The maximum number of cav, default 5

    #     Return
    #     ------
    #     pairwise_t_matrix : np.array
    #         The transformation matrix each cav to curr ego at past k frames.
    #         shape: (L, k, 4, 4), L is the max cav number in a scene, k is the num of past frames
    #         pairwise_t_matrix[i, j] is T i_to_ego at past_j frame
    #     """
    #     pairwise_t_matrix = np.tile(np.eye(4), (max_cav, self.k, 1, 1)) # (L, k, 4, 4)

    #     if self.proj_first:
    #         # if lidar projected to ego first, then the pairwise matrix
    #         # becomes identity
    #         # no need to warp again in fusion time.

    #         # pairwise_t_matrix[:, :] = np.identity(4)
    #         return pairwise_t_matrix
    #     else:
    #         t_list = []

    #         # save all transformation matrix in a list in order first.
    #         for cav_id, cav_content in base_data_dict.items():
    #             past_k_poses = []
    #             for time_id in range(self.k):
    #                 past_k_poses.append(x_to_world(cav_content['past_k'][time_id]['params']['lidar_pose']))
    #             t_list.append(past_k_poses) # Twx
            
    #         ego_pose = x_to_world(ego_pose)
    #         for i in range(len(t_list)): # different cav
    #             if i!=0 :
    #                 for j in range(len(t_list[i])): # different time
    #                     # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
    #                     # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
    #                     t_matrix = np.linalg.solve(t_list[i][j], ego_pose)  # Tjw*Twi = Tji
    #                     pairwise_t_matrix[i, j] = t_matrix

    #     return pairwise_t_matrix

if __name__ == '__main__':   
    
    def train_parser():
        parser = argparse.ArgumentParser(description="synthetic data generation")
        parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                            help='data generation yaml file needed ')
        parser.add_argument('--model_dir', default='',
                            help='Continued training path')
        parser.add_argument('--fusion_method', '-f', default="intermediate",
                            help='passed to inference.')
        opt = parser.parse_args()
        return opt
    
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('### Dataset Building ... ###')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)