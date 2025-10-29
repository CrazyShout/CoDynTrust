# -*- coding: utf-8 -*-
# Author: sizhewei @ 2023/4/15
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
two stage flow update framework

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
from opencood.data_utils.datasets import basedataset
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
from opencood.utils.flow_utils import generate_flow_map, generate_flow_map_szwei

from opencood.utils.box_utils import boxes_to_corners2d # for debug use
# from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np

# global, for debug use
illegal_path_list = set()

class IntermediateFusionDatasetV2XSETFlowXYJ(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        print("start create a IntermediateFusionDatasetV2XSETFlowXYJ object!")
        
        self.times = []

        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if 'num_sweep_frames' in params:    # number of frames we use in LSTM
            self.k = params['num_sweep_frames'] # 3
        else:
            self.k = 0

        if 'time_delay' in params:          # number of time delay
            self.tau = params['time_delay'] 
        else:
            self.tau = 0

        if 'binomial_n' in params:
            self.binomial_n = params['binomial_n']
        else:
            self.binomial_n = 0

        if 'binomial_p' in params:
            self.binomial_p = params['binomial_p']
        else:
            self.binomial_p = 1

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
            print("!!! Absolutely Regular !!!")

        # 控制是否需要生成GT flow
        self.is_generate_gt_flow = False
        if 'is_generate_gt_flow' in params and params['is_generate_gt_flow']:
            self.is_generate_gt_flow = True
        
        # 只有在绘制每个sample的匹配框时用到 sizhewei
        self.viz_bbx_flag = False
        if 'viz_bbx_flag' in params and params['viz_bbx_flag']:
            self.viz_bbx_flag = True

        self.num_roi_thres = -1
        if 'num_roi_thres' in params:
            self.num_roi_thres = params['num_roi_thres']
        
        self.sample_interval_exp = int(self.binomial_n * self.binomial_p) # 理论上十次伯努利实验的结果，1次

        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']: # 默认false
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
                                   os.path.isdir(os.path.join(root_dir, x))]) # 场景文件夹路径名
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

            # # copy timestamps npy file, store all irregular timestamps of each non-ego vehicle in this scenario
            # timestamps_file = os.path.join(scenario_folder, 'timestamps.npy')
            # time_annotations = np.load(timestamps_file)

            # at least 1 cav should show up
            cav_list = sorted([x 
                            for x in os.listdir(scenario_folder) if 
                            os.path.isdir(os.path.join(scenario_folder, x))], key=lambda y:int(y)) # 列出场景下的车id对应的文件路径
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

            # start_timestamp = int(float(self.extract_timestamps(yaml_files)[0]))
            # while(1):
            #     time_id_json = ("%.3f" % float(start_timestamp)) + ".json"
            #     time_id_yaml = ("%.3f" % float(start_timestamp)) + ".yaml"
            #     if not (time_id_json in yaml_files or time_id_yaml in yaml_files):
            #         start_timestamp += 1
            #     else:
            #         break

            # end_timestamp = int(float(self.extract_timestamps(yaml_files)[-1]))
            # if start_timestamp%2 == 0:
            #     # even
            #     end_timestamp = end_timestamp-1 if end_timestamp%2==1 else end_timestamp
            # else:
            #     end_timestamp = end_timestamp-1 if end_timestamp%2==0 else end_timestamp
            # num_timestamps = int((end_timestamp - start_timestamp)/2 + 1)
            # regular_timestamps = [start_timestamp+2*i for i in range(num_timestamps)]

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1: # 最多不能超过五辆车
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                cav_path = os.path.join(scenario_folder, cav_id)
                # yaml_files = \
                #     sorted([os.path.join(cav_path, x)
                #             for x in os.listdir(cav_path) if
                #             x.endswith('.yaml') and 'additional' not in x])
                # timestamps = self.extract_timestamps(yaml_files) # 返回的时间戳的列表，如[00068,000070,....]
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
                if j == 0:  # ego  每一个场景的ego会进来依次
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    # num_ego_timestamps = len(timestamps) - (self.tau + self.k - 1)		# 从第 tau+k 个往后, store 0 时刻的 time stamp
                    num_ego_timestamps = len(timestamps) - self.binomial_n * self.k # * 3 # TODO:  减去30个？ 这是因为后面会向后移动30帧作为cur帧，这里防止后面越界
                    if not self.len_record:
                        self.len_record.append(num_ego_timestamps)
                    else:# 如果self.len_record不为空，说明这不是第一个场景，那就将当前场景ego车辆的时间戳个数累加上去，最终self.len_record类似[场景1的ego车时间戳个数， 场景1+场景2*， 场景1+场景2+场景3*, ...]
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

        self.anchor_box = self.post_processor.generate_anchor_box() # 返回预设置的锚框

        print("=== V2XSET-Irregular Multi-sweep dataset with non-ego cavs' past {} frames collected initialized! Expectation of sample interval is {}. ### {} ###  samples totally! ===".format(self.k, self.binomial_n * self.binomial_p, self.len_record[-1]))

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
            res = file.split('/')[-1] # 将路径的前缀全部剔除 只留下如'000070.yaml'这样的字符串
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
                data[cav_id]['curr']['params'] = \
                            load_yaml(cav_content['regular'][timestamp_key]['yaml'])
            # 没有 lidar pose
            if not ('lidar_pose' in data[cav_id]['curr']['params']):
                tmp_ego_pose = np.array(data[cav_id]['curr']['params']['true_ego_pos'])
                tmp_ego_pose += np.array([-0.5, 0, 1.9, 0, 0, 0])
                data[cav_id]['curr']['params']['lidar_pose'] = list(tmp_ego_pose)

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
                data[cav_id]['curr']['lidar_np'] = np.load(npy_file)
            else:
                data[cav_id]['curr']['lidar_np'] = \
                        pcd_utils.pcd_to_np(cav_content['regular'][timestamp_key]['lidar']) # （n, 4）

            # 2.3 store curr timestamp and time_diff
            data[cav_id]['curr']['timestamp'] = timestamp_key
            data[cav_id]['curr']['time_diff'] = 0.0
            data[cav_id]['curr']['sample_interval'] = 0

            # 3. past frames, for model input
            data[cav_id]['past_k'] = OrderedDict()
            latest_sample_stamp_idx = curr_timestamp_idx
            # past k frames, pose | lidar | label(for single view confidence map generator use)
            for i in range(self.k): # 遍历所有帧 以下做出了修改 将ego也作为none-ego的cav处理从而去训练数据
                # sample_interval
                # if data[cav_id]['ego']:             # ego sample_interval = E(B(n, p))
                #     if i == 0: # ego-past-0 与 ego-curr 是一样的
                #         data[cav_id]['past_k'][i] = data[cav_id]['curr']
                #         continue
                #     sample_interval = self.sample_interval_exp
                #     if sample_interval == 0:
                #         sample_interval = 1
                # else:                               # non-ego sample_interval ~ B(n, p)
                    # delay_debug = 6
                    # if i == 0:
                    #     sample_interval = 3 #delay_debug
                    # elif i ==1:
                    #     sample_interval = 3
                    #     # sample_set = [2,3]
                    #     # import random
                    #     # sample_interval = random.sample(sample_set, 1)[0] #3 #10 - delay_debug
                    #     # trails = bernoulliDist.rvs(self.binomial_n)
                    #     # sample_interval = sum(trails)
                    # else:
                    #     sample_interval = 3
                    #     # trails = bernoulliDist.rvs(self.binomial_n)
                    #     # sample_interval = sum(trails)
                if self.sample_interval_exp==0 \
                    and self.is_no_shift \
                        and i == 0:
                    data[cav_id]['past_k'][i] = data[cav_id]['curr']
                    continue
                if self.is_same_sample_interval: # 相同时间间隔采样
                    sample_interval = self.sample_interval_exp
                else:
                    # B(n, p)
                    trails = bernoulliDist.rvs(self.binomial_n)  # 做10次伯努利实验
                    sample_interval = sum(trails) # 统计成功次数，作为采样间隔
                if sample_interval==0: # 如果采样间隔为0
                    if i==0: # 检查past 0 的实际时间是否在curr 的后面
                        tmp_time_key = list(cav_content.items())[latest_sample_stamp_idx][0] # 取出时间戳的字符串
                        if self.dist_time(tmp_time_key, data[cav_id]['curr']['timestamp'])>0: # 检查past0的时间戳是>= cur时间戳，那么最起码要间隔一帧
                            sample_interval = 1
                    if i>0: # 过去的几帧不要重复
                        sample_interval = 1                

                # check the timestamp index
                data[cav_id]['past_k'][i] = {}
                latest_sample_stamp_idx -= sample_interval
                timestamp_key = list(cav_content.items())[latest_sample_stamp_idx][0] # 获取时间戳
                # load the corresponding data into the dictionary
                # load param file: json is faster than yaml
                json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
                json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")

                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data[cav_id]['past_k'][i]['params'] = json.load(f)
                else:
                    data[cav_id]['past_k'][i]['params'] = \
                        load_yaml(cav_content[timestamp_key]['yaml']) # 获取yaml文件内容
                # 没有 lidar pose
                if not ('lidar_pose' in data[cav_id]['past_k'][i]['params']):
                    tmp_ego_pose = np.array(data[cav_id]['past_k'][i]['params']['true_ego_pos'])
                    tmp_ego_pose += np.array([-0.5, 0, 1.9, 0, 0, 0])
                    data[cav_id]['past_k'][i]['params']['lidar_pose'] = list(tmp_ego_pose)

                # load lidar file: npy is faster than pcd
                npy_file = cav_content[timestamp_key]['lidar'].replace("pcd", "npy")
                if os.path.exists(npy_file):
                    data[cav_id]['past_k'][i]['lidar_np'] = np.load(npy_file)
                else:
                    data[cav_id]['past_k'][i]['lidar_np'] = \
                            pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

                data[cav_id]['past_k'][i]['timestamp'] = timestamp_key
                data[cav_id]['past_k'][i]['sample_interval'] = sample_interval
                data[cav_id]['past_k'][i]['time_diff'] = \
                    self.dist_time(timestamp_key, data[cav_id]['curr']['timestamp'])
            
            data[cav_id]['debug'] = {}
            data[cav_id]['debug']['scene'] = scene_name
            data[cav_id]['debug']['cav_id'] = cav_id

        return data

    def __getitem__(self, idx):
        '''
        Returns:
        ------ 
        processed_data_dict : dict consist of all processed info, whose structure is:
        {
            'single_object_dict_stack': single_label_dict_stack,
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
            'sample_idx': idx,
            'cav_id_list': cav_id_list,
            'past_k_time_diffs': past_k_time_diffs_stack, np.array of len(\sum_i^num_cav k_i), k_i represents the num of past frames of cav_i
            'avg_sample_interval': float,
            'avg_time_delay': float
        }
        '''
        # TODO: debug use
        global illegal_path_list

        base_data_dict = self.retrieve_base_data(idx) # 获取了一个场景下的信息
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
                break	
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        too_far = []
        curr_lidar_pose_list = []
        cav_id_list = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items(): # 遍历场景下的每一辆车
            # check if the cav is within the communication range with ego
            # for non-ego cav, we use the latest frame's pose
            distance = math.sqrt( \
                (selected_cav_base['past_k'][0]['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + \
                    (selected_cav_base['past_k'][0]['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue
            curr_lidar_pose_list.append(selected_cav_base['curr']['params']['lidar_pose']) # 6dof pose  保存 pose 信息
            cav_id_list.append(cav_id)  # 放入cav id
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

        past_k_object_bbx_stack = []
        past_k_cav_object_num = []
        cur_cav_object_bbx_debug= []
        # past_k_common_id_index_stack = []

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
            single_past_lidar_stack = []
        
        for cav_id in cav_id_list: # 一个场景下遍历每一辆车 注意 在训练流的时候，可能出现一辆车都不满足要求，因为要四帧都匹配上才行
            selected_cav_base = base_data_dict[cav_id]
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

            if selected_cav_processed is None: # 如果进入到这里，说明flow的三帧匹配失败，即三帧中甚至找不到一个共有的object
                illegal_cav.append(cav_id) # 加入到非法cav id里，这是因为这个cav在这个时候已经没有必要继续处理下去了
                continue
            if selected_cav_processed['if_no_point']: # 把点的数量不合法的车排除
                illegal_cav.append(cav_id)
                # 把出现不合法sample的 场景、车辆、时刻 记录下来:
                illegal_path = os.path.join(base_data_dict[cav_id]['debug']['scene'], cav_id, base_data_dict[cav_id]['past_k'][0]['timestamp']+'.npy')
                illegal_path_list.add(illegal_path)
                # print(illegal_path)
                continue

            
            if self.visualize:
                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
            if self.viz_bbx_flag:
                single_lidar_stack.append(selected_cav_processed['single_lidar'])
                single_object_stack.append(selected_cav_processed['single_object_bbx_center'])
                single_object_id_stack.append(selected_cav_processed['single_object_ids'])
                single_past_lidar_stack.append(selected_cav_processed['single_past_lidar'])
                # mask = np.zeros(self.params['postprocess']['max_num'])
                # mask[:single_object_stack.shape[0]] = 1
                # single_mask_stack.append(mask)
            
            # single view feature
            curr_feature_stack.append(selected_cav_processed['curr_feature']) #  List 每个元素为字典，记录体素化信息
            # single view label
            single_label_dict_stack.append(selected_cav_processed['single_label_dict'])

            # curr ego view label
            object_stack.append(selected_cav_processed['object_bbx_center']) # (n_cur, 7) 已经proiect到ego view
            object_id_stack += selected_cav_processed['object_ids'] # 所有的车id全部组合进入一个list

            # current pose: N, 6
            curr_pose_stack.append(selected_cav_processed['curr_pose']) 
            # features: N, k, 
            past_k_features_stack.append(selected_cav_processed['past_k_features']) # List 每一个元素也是一个List 包含三帧体素化信息
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
        
            past_k_object_bbx_stack.append(selected_cav_processed['past_k_common_bbx']) # List 长度为agent个数，每个元素为（M，k，7）表示这个cav的k帧中匹配成功的M个object bbx 形如[(M1, k, 7), (M2, k, 7), (M3, k, 7)...]
            if len(selected_cav_processed['past_k_common_bbx'].shape) != 3 or selected_cav_processed['past_k_common_bbx'].shape[0] == 0: # TODO 如果进入了这里，说明有一个cav的三帧中没有连续出现三次的object，即匹配不成功
                print("Encounter a scenario without any sample to train flow!")
                return None
            past_k_cav_object_num.append(selected_cav_processed['past_k_common_bbx'].shape[0]) # 记录每个cav的object数 形如[30, 20...] 长度为cav的个数

            cur_cav_object_bbx_debug.append(selected_cav_processed['debug_gt_bbx']) # List [(M1, 7), (M2, 7)...]

            if self.is_generate_gt_flow:
                # for flow
                flow_gt.append(selected_cav_processed['flow_gt']) # List 每个元素(1, H, W, 2)
                warp_mask.append(selected_cav_processed['warp_mask']) # （1， C， H，W）
        
        for cav_id in illegal_cav:
            base_data_dict.pop(cav_id)
            cav_id_list.remove(cav_id)
            
        if len(cav_id_list) == 0:
            print("遭遇一次scenario下的样本连续4帧匹配灾难!")
            return self.__getitem__(idx+1)
        pastk_2_past0_tr_mats = np.stack(pastk_2_past0_tr_mats, axis=0) # N, k, 4, 4
        past_k_object_bbx_stack = np.vstack(past_k_object_bbx_stack) # 堆叠起来 （M1+M2+...，k, 7） 这就是一个scenario中的所有匹配好的三帧 检索第一维度就是每一个object，后两维则是代表其在三帧内的运动变化
        if past_k_object_bbx_stack.shape[0] == 0 or len(past_k_object_bbx_stack.shape) != 3:
            print("Encounter a scenario without any sample to train flow!")
            return None
        cur_cav_object_bbx_debug = np.vstack(cur_cav_object_bbx_debug) # (M1+M2+...., 7)
        # {pos: array[num_cav, k, 100, 252, 2], neg: array[num_cav, k, 100, 252, 2], target: array[num_cav, k, 100, 252, 2]}
        # past_k_label_dicts = self.post_processor.merge_label_to_dict(past_k_label_dicts_stack)
        # self.times.append(time.time())

        # filter those cav who has no points left
        # then we can calculate get_pairwise_transformation
        # for cav_id in illegal_cav:
        #     base_data_dict.pop(cav_id)
        #     cav_id_list.remove(cav_id)

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
            object_stack = np.vstack(object_stack) # 元素堆叠，原本每个元素为(cur下object数，7) 表示一个agent的cur下object信息，堆叠后为一个scenario下所有object信息 （N_all， 7）
        except ValueError:
            print("!!! vstack ValueError !!!")
            return None
        object_stack = object_stack[unique_indices] # 去重

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7)) # （100， 7）
        mask = np.zeros(self.params['postprocess']['max_num']) # （100）
        object_bbx_center[:object_stack.shape[0], :] = object_stack # （100， 7） 将cur下，一个scenario的所有agent的object bbx整合到一起
        mask[:object_stack.shape[0]] = 1 # 最多支持100object 设置掩码 标记有效object

        # self.times.append(time.time())

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(cav_id_list) # 场景中有几辆车记录下
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
             'pairwise_t_matrix': pairwise_t_matrix, # （L， K， 4， 4）：每一个agent，每一帧的lidar pose到cur ego的lidar pose的变换矩阵
             'curr_lidar_poses': curr_lidar_poses, # （L， 6）：每一个agent的在cur下的lidar  pose
             'past_k_lidar_poses': past_k_lidar_poses, # （L，k， 6）：每一个agent的每一帧的lidar pose
             'past_k_time_diffs': past_k_time_diffs_stack, # （L*k）：每一个agent每一帧到cur的时间间隔
             'past_k_sample_interval': past_k_sample_interval_stack,  # （L*k）：每一个agent下每一帧的采样间隔
             'past_k_object_bbx': past_k_object_bbx_stack, # (M_all, k, 7) 一个scenario中的所有匹配好的三帧 
             'past_k_cav_object_num': past_k_cav_object_num, # List (M1， M2...) 一个scenario中每个cav三帧匹配好的object个数 注意M1+M2+... M_n= M_all 其中n是场景中所有的cav数
             'cur_cav_object_bbx_debug': cur_cav_object_bbx_debug, # (M1+M2+..., 7)
             'pastk_2_past0_tr_mats': pastk_2_past0_tr_mats})
            #  'times': self.times})

        if self.is_generate_gt_flow:
            flow_gt = np.vstack(flow_gt) # (L, H, W, 2) 场景下N 辆车的past0-cur的变换坐标网格
            processed_data_dict['ego'].update({'flow_gt': flow_gt})
            warp_mask = np.vstack(warp_mask) # (L, C, H, W)
            processed_data_dict['ego'].update({'warp_mask': warp_mask})
        
        processed_data_dict['ego'].update({'sample_idx': idx,
                                            'cav_id_list': cav_id_list})
        try:
            tmp = past_k_time_diffs_stack[self.k:].reshape(-1, self.k) # (N, k)
            
            # tmp_past_k_time_diffs = np.concatenate((tmp[:, :1] , (tmp[:, 1:] - tmp[:, :-1])), axis=1) # (N, k) # TODO: 不同的方法这个计算方式不一样
            tmp_past_k_time_diffs = tmp[:, :1] # (N, 1) irregular setting 下，只用最近的一个时间间隔
            
            avg_time_diff = sum(tmp_past_k_time_diffs.reshape(-1)) / tmp_past_k_time_diffs.reshape(-1).shape[0]
            processed_data_dict['ego'].update({'avg_time_delay':\
                avg_time_diff})

            tmp = past_k_sample_interval_stack[self.k:].reshape(-1, self.k) # (N, k)
            # avg_sample_interval = sum(tmp.reshape(-1)) / len(tmp.reshape(-1)) # TODO: 不同的方法这个计算方式不一样
            avg_sample_interval = sum(tmp[:, :1].reshape(-1)) / len(tmp[:, :1].reshape(-1)) # (N, 1) irregular setting 下，只用最近的一个时间间隔
            processed_data_dict['ego'].update({'avg_sample_interval':\
                avg_sample_interval})

            tmp_var = np.var(tmp, axis=1) # (N, )
            avg_var = np.mean(tmp_var)
            processed_data_dict['ego'].update({'avg_var':\
                avg_var})
            
        except ZeroDivisionError:
            # print("!!! ZeroDivisionError !!!")
            return None

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(projected_lidar_stack)})

        if self.viz_bbx_flag:
            for id, cav in enumerate(cav_id_list):
                processed_data_dict[id] = {}
                processed_data_dict[id].update({
                    'single_lidar': single_lidar_stack[id],
                    'single_object_bbx_center': single_object_stack[id],
                    # 'single_object_bbx_mask': single_mask_stack[i],
                    'single_object_ids': single_object_id_stack[id],
                    'single_past_lidar': single_past_lidar_stack[id]
                })

        return processed_data_dict


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
        lidar_np = selected_cav_base['curr']['lidar_np'] # 当前的点云信息
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np) # remove points that hit itself

        if self.visualize:
            # trans matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['curr']['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray  cav到ego的转换矩阵
            projected_lidar = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
            selected_cav_processed.update({'projected_lidar': projected_lidar})

        lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range']) # 剔除雷达范围以外
        if self.viz_bbx_flag:
            selected_cav_processed.update({'single_lidar': lidar_np[:, :3]})
        curr_feature = self.pre_processor.preprocess(lidar_np) # 体素化 返回字典，有三个元素 体素信息(N, 32, 4) 体素坐标：(N, 3) 每个体素中点的数量：（N）
        
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

        past_k_object_bbx = []
        past_k_object_ids = []
        # 判断点的数量是否合法
        if_no_point = False

        # past k frames [trans matrix], [lidar feature], [pose], [time interval]
        for i in range(self.k): # 遍历k帧
            # 1. trans matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['past_k'][i]['params']['lidar_pose'], ego_pose) # T_ego_cav, np.ndarray 求得这一帧的cav pose 到ego的转换矩阵 (4, 4)
            past_k_tr_mats.append(transformation_matrix)
            # past_k trans past_0 matrix
            pastk_2_past0 = \
                x1_to_x2(selected_cav_base['past_k'][i]['params']['lidar_pose'], selected_cav_base['past_k'][0]['params']['lidar_pose']) # 求每一帧到第0帧的转换矩阵
            pastk_2_past0_tr_mats.append(pastk_2_past0)
            
            # 2. lidar feature
            lidar_np = selected_cav_base['past_k'][i]['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_ego_points(lidar_np) # remove points that hit itself
            lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
            if self.viz_bbx_flag:
                selected_cav_processed.update({'single_past_lidar': lidar_np[:, :3]})
            processed_features = self.pre_processor.preprocess(lidar_np) # 预处理点云 也即体素化 
            past_k_features.append(processed_features)

            if lidar_np.shape[0] == 0: # 没有点留下
                if_no_point = True

            # 3. pose
            past_k_poses.append(selected_cav_base['past_k'][i]['params']['lidar_pose'])

            # 4. time interval and sample interval
            past_k_time_diffs.append(selected_cav_base['past_k'][i]['time_diff'])
            past_k_sample_interval.append(selected_cav_base['past_k'][i]['sample_interval'])

            ################################################################
            # sizhewei
            # for past k frames' single view label
            ################################################################
            # # 5. single view label
            # # past_i label at past_i single view
            # # opencood/data_utils/post_processor/base_postprocessor.py
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base['past_k'][i]], selected_cav_base['past_k'][0]['params']['lidar_pose'])  # 这是将这一帧对应的object表示取出 TODO 这里我已经将参考pose全部统一成pastk0的lidar pose
                # self.generate_object_center([selected_cav_base['past_k'][i]], selected_cav_base['past_k'][0]['params']['lidar_pose'])  # 这是将这一帧对应的object表示取出 TODO 这里我已经将参考pose全部统一成pastk0的lidar pose
            past_k_object_bbx.append(object_bbx_center[object_bbx_mask == 1])  # List 每一帧的object bbx 每一个元素的形状为（有效bbx个数，7）
            past_k_object_ids.append(object_ids) # 

            # # generate the anchor boxes
            # # opencood/data_utils/post_processor/voxel_postprocessor.py
            # anchor_box = self.anchor_box
            # single_view_label_dict = self.post_processor.generate_label(
            #         gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask
            #     )
            # past_k_label_dicts.append(single_view_label_dict)
        
        object_bbx_center_cur, object_bbx_mask_cur, object_ids_cur = self.generate_object_center([selected_cav_base['curr']], selected_cav_base['past_k'][0]['params']['lidar_pose']) # 形成obj bbx

        '''
        # past k merge
        past_k_label_dicts = self.post_processor.merge_label_to_dict(past_k_label_dicts)
        '''
        # 取出三帧共同索引id 
        # print(len(past_k_object_ids))
        # past_k_common_id_index = self.find_common_id(past_k_object_ids) # 三个元素  每个元素对应各自的bbx索引
        past_k_common_id_index = self.find_common_id(past_k_object_ids, object_ids_cur) # 四个元素  每个元素对应各自的bbx索引
        object_bbx_center_cur = object_bbx_center_cur[object_bbx_mask_cur == 1]

        # print(past_k_common_id_index[0])
        # print(past_k_common_id_index[1])
        # print(past_k_common_id_index[2])
        # print(past_k_common_id_index[3])
        # print("++++++++++++++++++++++++++++++++++++++")
        # for i in range(3):
        #     past = torch.from_numpy(np.array(past_k_object_ids[i]))
        #     print(past[past_k_common_id_index[i]])

        # past = torch.from_numpy(np.array(object_ids_cur))
        # print(past[past_k_common_id_index[-1]])

        # print("+++++++++++++++++++++")

        past_k_common_bbx_list = []
        for i in range(len(past_k_common_id_index)): # 通过共有bbx 索引取出其bbx [[M.7], [M,7], [M,7]]表示从past0到past2一共三帧的匹配bbx  较低概率出现每个元素是形状(7, )
            if i == len(past_k_common_id_index) - 1: # cur
                object_bbx_center_cur = object_bbx_center_cur[past_k_common_id_index[-1]]
                if len(object_bbx_center_cur.shape) == 1:
                    object_bbx_center_cur.reshape(1, 7)
                continue
            past = past_k_object_bbx[i][past_k_common_id_index[i]] # 取出共同的object
            if len(past.shape) == 1: # 只有一个object的时候，past会变成（7，）会导致下面转置出错 TODO 还有一种情况，即匹配失败，那past会是（0， 7） 这个时候样本就可以丢弃
                past = past.reshape(1, 7)
            past_k_common_bbx_list.append(past)
            # print(past_k_common_bbx_list[i].shape)
        past_k_common_bbx = np.stack(past_k_common_bbx_list, axis=0) # （k，M，7） 一个cav下所有的匹配上的object数为M
        # print(past_k_common_bbx.shape)
            
        past_k_common_bbx = np.transpose(past_k_common_bbx, (1, 0, 2)) # (M, k, 7)
        if past_k_common_bbx.shape[0] == 0: # 也就是三帧没有匹配成功
            return None
        # print("+++++++++++++++++++++")
        # print(past_k_common_bbx.shape)
        # print("+++++++++++++++++++++")

        past_k_tr_mats = np.stack(past_k_tr_mats, axis=0) # (k, 4, 4)
        pastk_2_past0_tr_mats = np.stack(pastk_2_past0_tr_mats, axis=0) # (k, 4, 4)

        # avg_past_k_time_diffs = float(sum(past_k_time_diffs) / len(past_k_time_diffs))
        # avg_past_k_sample_interval = float(sum(past_k_sample_interval) / len(past_k_sample_interval))

        # curr label at single view
        # opencood/data_utils/post_processor/base_postprocessor.py
        single_object_bbx_center, single_object_bbx_mask, single_object_ids = \
            self.generate_object_center([selected_cav_base['curr']], selected_cav_base['curr']['params']['lidar_pose'])  # 生成bbx，投影到current cav位置
        # generate the anchor boxes
        # opencood/data_utils/post_processor/voxel_postprocessor.py
        anchor_box = self.anchor_box
        label_dict = self.post_processor.generate_label(
                gt_box_center=single_object_bbx_center, anchors=anchor_box, mask=single_object_bbx_mask
            ) # 这是用来train 检测器的  cur下的object转化为target学习偏移
        
        # curr label at ego view
        object_bbx_center, object_bbx_mask, object_ids = \
            self.generate_object_center([selected_cav_base['curr']], ego_pose)
            
        selected_cav_processed.update({
            "single_label_dict": label_dict, # Dict：用来train的数据，表示预置anchor与object的偏移量
            "single_object_bbx_center": single_object_bbx_center[single_object_bbx_mask == 1], # （n_cur, 7）当前时间戳的cav周围的object 并且project到相应的cav view
            "single_object_ids": single_object_ids, # List: (n_cur) cur下cav的感知范围内的车id
            "curr_feature": curr_feature, # Dict：三个元素 表示cur下的体素化信息
            'object_bbx_center': object_bbx_center[object_bbx_mask == 1], # （n_cur, 7）时间为cur下，n为场景中实际的object的数量，与上面的single_object_bbx_center区别在于这个是project到ego的view上
            'object_ids': object_ids, # List: (n_cur) cur下cav的感知范围内的车id
            'curr_pose': selected_cav_base['curr']['params']['lidar_pose'], # cur下的pose
            'past_k_tr_mats': past_k_tr_mats, # （K，4, 4）：历史k帧到ego cur的变换矩阵
            'pastk_2_past0_tr_mats': pastk_2_past0_tr_mats, # （K，4, 4）：历史k帧到past0帧的变换矩阵
            'past_k_poses': past_k_poses, # List：k帧每一帧的pose，每一个元素shape为（6）
            'past_k_features': past_k_features, # List：k帧每一帧的体素化信息，k个元素，每一个都是Dict
            'past_k_time_diffs': past_k_time_diffs, # List：k帧每一帧距离cur 的时间间隔，每一个元素为float
            'past_k_sample_interval': past_k_sample_interval, # List：k帧每一帧的样本间隔，每一个元素为int
            'past_k_common_bbx': past_k_common_bbx, # （M，k，7）这个cav的k帧中匹配成功的M个object bbx
        #  'avg_past_k_time_diffs': avg_past_k_time_diffs,
        #  'avg_past_k_sample_interval': avg_past_k_sample_interval,
        #  'past_k_label_dicts': past_k_label_dicts,
            'if_no_point': if_no_point
            })
        # for debug xyj 2024年03月06日
        # 这里注释掉，因为在前面将其与三帧一起做了find common 2024年03月22日

        # past_0 = torch.from_numpy(np.array(past_k_object_ids[2]))
        # common_ids = past_0[past_k_common_id_index[2]].tolist()
        # object_bbx_center_cur, object_bbx_mask_cur, object_ids_cur = self.generate_object_center([selected_cav_base['curr']], selected_cav_base['past_k'][0]['params']['lidar_pose']) # 形成obj bbx
        # dict_pastk = dict(zip(common_ids, range(len(common_ids))))
        # dict_cur = dict(zip(object_ids_cur, range(len(object_ids_cur))))
        # common_id = set(common_ids) & set(object_ids_cur)
        # index_pastk = torch.tensor([dict_pastk[id] for id in common_id], dtype=torch.long)
        # index_cur = torch.tensor([dict_cur[id] for id in common_id], dtype=torch.long)
        # object_bbx_center_cur = object_bbx_center_cur[object_bbx_mask_cur == 1]
        # object_bbx_center_cur = object_bbx_center_cur[index_cur] # (n_common, 7)

        # 以上，2024年03月22日 注释

        # print('common_ids', common_ids)
        # print('object_ids_cur', object_ids_cur)
        # print('common_id', common_id)
        # print('set(common_ids)', set(common_ids))
        # print('set(object_ids_cur)', set(object_ids_cur))
        selected_cav_processed.update({'debug_gt_bbx': object_bbx_center_cur})

        if self.is_generate_gt_flow:
            # generate flow, from past_0 and curr
            prev_object_id_stack = {}
            prev_object_stack = {}
            for t_i in range(2): # 遍历两次 过去一帧和当前帧 第一次为过去一帧， 第二次为当前帧
                split_part = selected_cav_base['past_k'][0] if t_i == 0 else selected_cav_base['curr'] # TODO: 这里面的 prev 和 curr 可能反了
                object_bbx_center, object_bbx_mask, object_ids = \
                    self.generate_object_center([split_part], selected_cav_base['past_k'][0]['params']['lidar_pose']) # 形成obj bbx
                prev_object_id_stack[t_i] = object_ids
                prev_object_stack[t_i] = object_bbx_center # （max_num, 7） 第一个放past0 第二个放cur
            
            for t_i in range(2):
                unique_object_ids = list(set(prev_object_id_stack[t_i])) # 去重
                unique_indices = \
                    [prev_object_id_stack[t_i].index(x) for x in unique_object_ids] # 独立的cav id 的索引
                prev_object_stack[t_i] = np.vstack(prev_object_stack[t_i]) # 竖直堆叠 (n_common, 7)
                prev_object_stack[t_i] = prev_object_stack[t_i][unique_indices] # 到这步其实是把一帧中的重复id的object去除
                prev_object_id_stack[t_i] = unique_object_ids

            # TODO: generate_flow_map: yhu, generate_flow_map_szwei: szwei
            flow_map, warp_mask = generate_flow_map_szwei(prev_object_stack,
                                            prev_object_id_stack,
                                            self.params['preprocess']['cav_lidar_range'],
                                            self.params['preprocess']['args']['voxel_size'],
                                            past_k=1)

            selected_cav_processed.update({'flow_gt': flow_map})  # (1, H, W, 2) 坐标网格 记录past0到cur的变换
            selected_cav_processed.update({'warp_mask': warp_mask}) # （1， C， H，W）0/1张量，标记变换后的object所在的区域

        return selected_cav_processed

    def find_common_id(self, objct_ids, cur_ids = None):
        '''
        objct_ids : List (k) k帧中每一帧pbject的id
        cur_ids: 当前的帧的object id, 为了训练流
        
        Return: 
        List 三帧中匹配完毕的object 索引， 三个元素， 每个元素是一个object在各自三帧中的索引
        '''
        id_past0 = objct_ids[0] # past0的所有object id
        id_past1 = objct_ids[1]
        id_past2 = objct_ids[2]
    
        dict_past0 = dict(zip(id_past0, range(len(id_past0))))  # 形成 4223 : 0 这样的形式
        dict_past1 = dict(zip(id_past1, range(len(id_past1)))) 
        dict_past2 = dict(zip(id_past2, range(len(id_past2)))) 


        if cur_ids is not None:
            dict_cur = dict(zip(cur_ids, range(len(cur_ids)))) 
            common_id = set(id_past0) & set(id_past1) & set(id_past2) & set(cur_ids) # 筛选出所有的 相同的 object id
            index_cur = torch.tensor([dict_cur[id] for id in common_id], dtype=torch.long)
        else:     
            common_id = set(id_past0) & set(id_past1) & set(id_past2) # 筛选出所有的 相同的 object id

        index_past0 = torch.tensor([dict_past0[id] for id in common_id], dtype=torch.long) # 长度都是M， M表示三帧中匹配到的object数量 表示的是一个object在三帧中各自的索引
        index_past1 = torch.tensor([dict_past1[id] for id in common_id], dtype=torch.long)
        index_past2 = torch.tensor([dict_past2[id] for id in common_id], dtype=torch.long)

        if cur_ids is not None:
            return [index_past0, index_past1, index_past2, index_cur]

        return [index_past0, index_past1, index_past2]


    @staticmethod
    def return_timestamp_key_async(cav_content, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # retrieve the correct index
        timestamp_key = list(cav_content.items())[timestamp_index][0]

        return timestamp_key

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

    def merge_past_k_features_to_dict(self, processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for cav_id in range(len(processed_feature_list)):# 遍历一个scenario下的每一个agent
            for time_id in range(self.k):
                for feature_name, feature in processed_feature_list[cav_id][time_id].items(): # 遍历距离某一个agent的某一帧 这里面是体素信息
                    if feature_name not in merged_feature_dict:
                        merged_feature_dict[feature_name] = []
                    if isinstance(feature, list):
                        merged_feature_dict[feature_name] += feature
                    else:
                        merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
        return merged_feature_dict

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)): # 遍历列表，也就是有几个agent
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
        return merged_feature_dict

    def collate_batch_train(self, batch):
        '''
        Parameters:
        ----------
        batch[i]['ego'] structure:
        {   
            'single_object_dict_stack': single_label_dict_stack,
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
            'sample_idx': idx,
            'cav_id_list': cav_id_list,
            'past_k_time_diffs': past_k_time_diffs_stack, 
                list of len(\sum_i^num_cav k_i), k_i represents the num of past frames of cav_i
            'flow_gt': [N, C, H, W]
        }
        '''
        for i in range(len(batch)):
            if batch[i] is None:
                return None
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        single_object_label = []

        pos_equal_one_single = []
        neg_equal_one_single = []
        targets_single = []

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        curr_processed_lidar_list = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        curr_lidar_pose_list = []
        past_k_lidar_pose_list = []
        past_k_label_list = []
        # store the time interval of each feature map
        past_k_time_diff = []
        past_k_sample_interval = []
        past_k_avg_time_delay = []
        past_k_avg_sample_interval = []
        past_k_avg_time_var = []
        pastk_2_past0_tr_mats = []
        # pairwise transformation matrix
        pairwise_t_matrix_list = []
        past_k_object_bbx_list = []
        cur_object_bbx_debug_list = []
        past_k_object_cav_num_list = []

        if self.is_generate_gt_flow:
            # flow gt
            flow_gt_list = []
            warp_mask_list = []

        # for debug use:
        sum_time_diff = 0.0
        sum_sample_interval = 0.0
        # time_consume = np.zeros_like(batch[0]['ego']['times'])

        if self.visualize:
            origin_lidar = []
        
        for i in range(len(batch)): # 遍历每一个样本，一个样本就是一个scenario
            ego_dict = batch[i]['ego']
            single_object_label.append(ego_dict['single_object_dict_stack'])

            pos_equal_one_single.append(ego_dict['single_object_dict_stack']['pos_equal_one'])
            neg_equal_one_single.append(ego_dict['single_object_dict_stack']['neg_equal_one'])
            targets_single.append(ego_dict['single_object_dict_stack']['targets'])

            curr_processed_lidar_list.append(ego_dict['curr_processed_lidar'])
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            curr_lidar_pose_list.append(ego_dict['curr_lidar_poses']) # ego_dict['curr_lidar_pose'] is np.ndarray [N,6]
            past_k_lidar_pose_list.append(ego_dict['past_k_lidar_poses']) # ego_dict['past_k_lidar_pose'] is np.ndarray [N,k,6]
            past_k_time_diff.append(ego_dict['past_k_time_diffs']) # ego_dict['past_k_time_diffs'] is np.array(), len=nxk
            past_k_sample_interval.append(ego_dict['past_k_sample_interval']) # ego_dict['past_k_sample_interval'] is np.array(), len=nxk
            past_k_avg_sample_interval.append(ego_dict['avg_sample_interval']) # ego_dict['avg_sample_interval'] is float
            past_k_avg_time_delay.append(ego_dict['avg_time_delay']) # ego_dict['avg_sample_interval'] is float
            try:
                past_k_avg_time_var.append(ego_dict['avg_var'])
            except KeyError:
                past_k_avg_time_var.append(-1.0)
            # avg_time_delay += ego_dict['avg_time_delay']
            # avg_sample_interval += ego_dict['avg_sample_interval']
            processed_lidar_list.append(ego_dict['processed_lidar']) # different cav_num, ego_dict['processed_lidar'] is list.
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            pastk_2_past0_tr_mats.append(ego_dict['pastk_2_past0_tr_mats'])
            # past_k_label_list.append(ego_dict['past_k_label_dicts'])

            past_k_object_bbx_list.append(ego_dict['past_k_object_bbx']) # List 每个元素为（一个scenario下所有object数， k， 7）
            past_k_object_cav_num_list += ego_dict['past_k_cav_object_num'] # List 原本ego_dict['past_k_cav_object_num']为列表，表示一个scenario下各个cav的匹配object数，长度为对应场景下的cav数，这里记为n_下标 则形如[n1, n2] batchsize=2
            cur_object_bbx_debug_list.append(ego_dict['cur_cav_object_bbx_debug']) # List 每个元素为（一个scenario下所有object数，  7）
            # print("一个scenario下cav的匹配object数,  ego_dict['past_k_cav_object_num']:  " , ego_dict['past_k_cav_object_num'])

            if self.is_generate_gt_flow:  
                flow_gt_list.append(ego_dict['flow_gt'])
                warp_mask_list.append(ego_dict['warp_mask'])

            # time_consume += ego_dict['times']
            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        
        past_k_object_bbx = torch.from_numpy(np.vstack(past_k_object_bbx_list)) # 一个batch中的所有object的三帧变化，（M_b， k, 7）
        past_k_object_cav_num = torch.from_numpy(np.array(past_k_object_cav_num_list)) # 记录一个batch中的所有cav的每一个的object数量 其长度和batch中的所有cav个数保持一致 
        cur_object_bbx_debug = torch.from_numpy(np.vstack(cur_object_bbx_debug_list)) # 一个batch中的所有object的cur，（M_b, 7）

        # single_object_label = self.post_processor.collate_batch(single_object_label)
        single_object_label = { "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                                "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                                 "targets": torch.cat(targets_single, dim=0)}
                                 
        # collate past k single view label from different batch [B, cav_num, k, 100, 252, 2]...
        past_k_single_label_torch_dict = self.post_processor.collate_batch(past_k_label_list)
        
        # collate past k time interval from different batch, (B, )
        past_k_time_diff = np.hstack(past_k_time_diff) # 本来是一个列表 长度为B，每一个元素长度为L*K，L不是固定长度，表示场景下的cav个数, 第二个轴上堆叠，
        past_k_time_diff= torch.from_numpy(past_k_time_diff)

        # collate past k sample interval from different batch, (B, )
        past_k_sample_interval = np.hstack(past_k_sample_interval)
        past_k_sample_interval = torch.from_numpy(past_k_sample_interval)

        past_k_avg_sample_interval = np.array(past_k_avg_sample_interval)
        avg_sample_interval = float(sum(past_k_avg_sample_interval) / len(past_k_avg_sample_interval))

        past_k_avg_time_delay = np.array(past_k_avg_time_delay)
        avg_time_delay = float(sum(past_k_avg_time_delay) / len(past_k_avg_time_delay))

        past_k_avg_time_var = np.array(past_k_avg_time_var)
        avg_time_var = float(sum(past_k_avg_time_var) / len(past_k_avg_time_var))

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        # （B, max_num)
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        
        curr_merged_feature_dict = self.merge_features_to_dict(curr_processed_lidar_list)
        curr_processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(curr_merged_feature_dict)
        # processed_lidar_list: list, len is 6. [batch_i] is OrderedDict, 3 keys: {'voxel_features': , ...}
        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        # [sum(record_len), C, H, W]
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        curr_lidar_pose = torch.from_numpy(np.concatenate(curr_lidar_pose_list, axis=0))
        # [[N1, k, 6], [N2, k, 6]...] -> [(N1+N2+...), k, 6]
        past_k_lidar_pose = torch.from_numpy(np.concatenate(past_k_lidar_pose_list, axis=0))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav, k, 4, 4)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        pastk_2_past0_tr_mats = torch.from_numpy(np.vstack(pastk_2_past0_tr_mats))

        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len

        if self.is_generate_gt_flow:
            # for flow
            flow_gt = torch.from_numpy(np.vstack(flow_gt_list)) # （N1+N2+...，H，W，2）一个batch中所有agent的gt flow， N1表示第一个scenario下的agent数目 统一将其和命名为N_b 则形状为(N_b, H, W, 2)
            label_torch_dict.update({'flow_gt': flow_gt})
            warp_mask = torch.from_numpy(np.vstack(warp_mask_list))
            label_torch_dict.update({'warp_mask': warp_mask}) # (N_b, C，H, W)

        # for debug use: 
        # time_consume = torch.from_numpy(time_consume)

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'single_object_label': single_object_label,
                                   'curr_processed_lidar': curr_processed_lidar_torch_dict,
                                   'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'single_past_dict': past_k_single_label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'pairwise_t_matrix': pairwise_t_matrix,
                                   'pastk_2_past0_tr_mats': pastk_2_past0_tr_mats,
                                   'curr_lidar_pose': curr_lidar_pose,
                                   'past_lidar_pose': past_k_lidar_pose,
                                   'past_k_time_interval': past_k_time_diff, # (所有帧的数量)
                                   'past_k_sample_interval': past_k_sample_interval,
                                   'past_k_object_bbx': past_k_object_bbx, # 一个batch中的所有object的三帧变化，（M_b， k, 7）
                                   'past_k_object_cav_num': past_k_object_cav_num, # 一维张量 形如（M1，M2....）其中存储每个cav的object数量 用于后续恢复形状对应到相应cav
                                   'cur_object_bbx_debug': cur_object_bbx_debug, 
                                   'avg_sample_interval': avg_sample_interval,
                                   'avg_time_delay': avg_time_delay,
                                   'avg_time_var': avg_time_var})
                                #    'times': time_consume})
        output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(self.anchor_box))})
        
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        if self.viz_bbx_flag: # 可视化每个样本的匹配框
            single_lidar = []
            single_object_bbx_center = []
            single_object_mask = []
            single_object_ids = []
            single_past_lidar = []
            for i in range(len(batch[0].keys())-1):
                single_lidar.append(torch.from_numpy(batch[0][i]['single_lidar']))
                single_past_lidar.append(torch.from_numpy(batch[0][i]['single_past_lidar']))
                single_object_bbx_center.append(torch.from_numpy(batch[0][i]['single_object_bbx_center']))
                single_object_ids.append(batch[0][i]['single_object_ids'])
            output_dict['ego'].update({
                'single_lidar_list': single_lidar,
                'single_object_bbx_center': single_object_bbx_center,
                'single_object_ids': single_object_ids,
                'single_past_lidar_list': single_past_lidar
            })
            
        # if self.params['preprocess']['core_method'] == 'SpVoxelPreprocessor' and \
        #     (output_dict['ego']['processed_lidar']['voxel_coords'][:, 0].max().int().item() + 1) != record_len.sum().int().item():
        #     return None

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        # transformation is only used in post process (no use.)
        # we all predict boxes in ego coord.
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = \
            torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch,
                                    'transformation_matrix_clean':
                                       transformation_matrix_clean_torch,})

        output_dict['ego'].update({
            "sample_idx": batch[0]['ego']['sample_idx'],
            "cav_id_list": batch[0]['ego']['cav_id_list']
        })

        # output_dict['ego'].update({'veh_frame_id': batch[0]['ego']['veh_frame_id']})

        return output_dict

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

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['curr']['params']['lidar_pose']
                t_list.append(x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
    
    def get_past_k_pairwise_transformation2ego(self, base_data_dict, ego_pose, max_cav):
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
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, self.k, 1, 1)) # (L, k, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                past_k_poses = []
                for time_id in range(self.k):
                    past_k_poses.append(x_to_world(cav_content['past_k'][time_id]['params']['lidar_pose']))
                t_list.append(past_k_poses) # Twx
            
            ego_pose = x_to_world(ego_pose)
            for i in range(len(t_list)): # different cav
                if i!=0 :
                    for j in range(len(t_list[i])): # different time
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[i][j], ego_pose)  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

    def generate_pred_bbx_frames(self, m_single, trans_mat_pastk_2_past0, past_time_diff, anchor_box):
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
        '''
        box_results = self.post_processor.single_post_process(m_single, trans_mat_pastk_2_past0, past_time_diff, anchor_box, self.k, self.num_roi_thres)
        return box_results

# if __name__ == '__main__':   
