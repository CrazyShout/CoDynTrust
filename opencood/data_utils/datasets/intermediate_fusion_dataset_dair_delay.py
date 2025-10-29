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
import torch
from torch.utils.data import DataLoader
import json
import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils

from opencood.data_utils.datasets import intermediate_fusion_dataset
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
from scipy import stats

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

class IntermediateFusionDatasetDAIRAsync(intermediate_fusion_dataset.IntermediateFusionDataset):
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

        self.strict_data = False
        if 'strict_data' in params and params['strict_data'] is True:
            print("===验证集使用严格策略, 即必须10*k历史帧严格存在===")
            self.strict_data = True

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        if "kd_flag" in params.keys():
            self.kd_flag = params['kd_flag']
        else:
            self.kd_flag = False

        if "box_align" in params.keys():
            self.box_align = True
            self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
            self.stage1_result = load_json(self.stage1_result_path)
            self.box_align_args = params['box_align']['args']
        
        else:
            self.box_align = False

        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
            
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None


        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        #这里root_dir是一个json文件！--> 代表一个split
        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']
        self.inf_idx2info = build_idx_to_info(
            load_json(osp.join(self.root_dir, "infrastructure-side/data_info.json"))
        )
        self.co_idx2info = build_idx_to_co_info(
            load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
        )
        self.inf_fid2veh_fid = build_inf_fid_to_veh_fid(load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
        )
        self.data_split = load_json(split_dir)
        self.data = []
        for veh_idx in self.data_split:
            if self.is_valid_id(veh_idx):
                self.data.append(veh_idx)

        print("ASync dataset with {} time delay initialized! {} samples totally!".format(self.binomial_n*self.binomial_p, len(self.data)))
    
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
            if delay_id not in self.inf_fid2veh_fid.keys():
                return False

        return True
    
        # # print('veh_frame_id: ',veh_frame_id,'\n')
        # frame_info = {}
        
        # frame_info = self.co_idx2info[veh_frame_id]
        # inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        # cur_inf_info = self.inf_idx2info[inf_frame_id]
        # if (
        #     int(inf_frame_id) - self.k < int(cur_inf_info["batch_start_id"])
        #     or id_to_str(int(inf_frame_id) - self.k) not in self.inf_idx2info
        # ):
        #     return False

        # return True

    def retrieve_base_data(self, idx):
        """
        Modified by sizhewei @ 2022/09/28
        Given the index, return the corresponding async data (time delay: self.k).

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.data[idx]
        # print('veh_frame_id: ',veh_frame_id,'\n')
        frame_info = {}
        system_error_offset = {}
        
        bernoulliDist = stats.bernoulli(self.binomial_p)
        trails = bernoulliDist.rvs(self.binomial_n)
        sample_interval = sum(trails)

        frame_info = self.co_idx2info[veh_frame_id]
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        # cur_inf_frame_id = inf_frame_id

        if self.strict_data is not True:
            cur_inf_info = self.inf_idx2info[inf_frame_id] # 取出路端信息 要判断两个：1、延迟减去后的帧是否存在，否为无延迟，2、延迟减去后的帧是否有车路帧，如果没有则倒退
            if (int(inf_frame_id) - sample_interval < int(cur_inf_info["batch_start_id"])): # 如果往前已经没有帧，则默认使用当前帧 ，设置延迟为0
                sample_interval = 0
            if sample_interval > 0:
                delay_id = id_to_str(int(inf_frame_id) - sample_interval)
                for _ in range(sample_interval): # 判断延迟帧是否有车路帧，无则回溯
                    if delay_id not in self.inf_fid2veh_fid.keys(): # 必须有车路对应
                        sample_interval -= 1
                    else:
                        break

        inf_frame_id = id_to_str(int(inf_frame_id) - sample_interval)

        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        #cav_id=0是车端，1是路边单元
        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir,frame_info['cooperative_label_path']))
        # print(data[0]['params']['vehicles'])
        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))

        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)

        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
        if self.clip_pc:
            data[0]['lidar_np'] = data[0]['lidar_np'][data[0]['lidar_np'][:,0]>0]
            
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[1]['params'] = OrderedDict()

        # data[1]['params']['vehicles'] = load_json(os.path.join(self.root_dir,frame_info['cooperative_label_path']))
        data[1]['params']['vehicles'] = [] # we only load cooperative label in vehicle side

        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))

        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)

        cur_inf_info = self.inf_idx2info[inf_frame_id]
        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir, \
            'infrastructure-side', cur_inf_info['pointcloud_path']))
        data[0]['veh_frame_id'] = veh_frame_id
        data[1]['veh_frame_id'] = inf_frame_id

        # 修正世界标签使用，不能使用延迟前的标签，应该用当前时间的标签 2024年7月19日 by xuyunjiang
        # 不需要了，因为世界标签由车端读取了
        # data[1]['curr'] = {}
        # data[1]['curr']['params'] = OrderedDict()
        # data[1]['curr']['params']['vehicles'] = []
        # virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(cur_inf_frame_id)+'.json'))
        # transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        # data[1]['curr']['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)
        # real_cur_inf_info = self.inf_idx2info[cur_inf_frame_id]
        # data[1]['curr']['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir, \
        #     'infrastructure-side', real_cur_inf_info['pointcloud_path']))
        # data[1]['curr']['veh_frame_id'] = cur_inf_frame_id

        return data

    def __getitem__(self, idx):
        # base_data_dict = self.retrieve_base_data(idx)
        base_data_dict = self.retrieve_base_data(idx)

        base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break
            
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0


        processed_features = []
        object_stack = []
        object_id_stack = []
        too_far = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        projected_lidar_clean_list = []
        cav_id_list = []

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)

            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue


            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
            cav_id_list.append(cav_id)


        ########## Added by Yifan Lu 2022.8.14 ##############
        # box align to correct pose.
        '''
        if self.box_align and str(idx) in self.stage1_result.keys():
            stage1_content = self.stage1_result[str(idx)]
            if stage1_content is not None:
                cav_id_list_stage1 = stage1_content['cav_id_list']
                
                pred_corners_list = stage1_content['pred_corner3d_np_list']
                pred_corners_list = [np.array(corners, dtype=np.float64) for corners in pred_corners_list]
                uncertainty_list = stage1_content['uncertainty_np_list']
                uncertainty_list = [np.array(uncertainty, dtype=np.float64) for uncertainty in uncertainty_list]
                stage1_lidar_pose_list = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list_stage1]
                stage1_lidar_pose = np.array(stage1_lidar_pose_list)


                refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                stage1_lidar_pose, 
                                                                uncertainty_list=uncertainty_list, 
                                                                **self.box_align_args)
                stage1_lidar_pose[:,[0,1,4]] = refined_pose
                stage1_lidar_pose_refined_list = stage1_lidar_pose.tolist() # updated lidar_pose_list
                for cav_id, lidar_pose_refined in zip(cav_id_list_stage1, stage1_lidar_pose_refined_list):
                    if cav_id not in cav_id_list:
                        continue
                    idx_in_list = cav_id_list.index(cav_id)
                    lidar_pose_list[idx_in_list] = lidar_pose_refined
                    base_data_dict[cav_id]['params']['lidar_pose'] = lidar_pose_refined
        '''     


        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]



            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose, 
                ego_lidar_pose_clean,
                idx)
                
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

            processed_features.append(
                selected_cav_processed['processed_features'])
            if self.kd_flag:
                projected_lidar_clean_list.append(
                    selected_cav_processed['projected_lidar_clean'])

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])





        ########## Added by Yifan Lu 2022.4.5 ################
        # filter those out of communicate range
        # then we can calculate get_pairwise_transformation
        for cav_id in too_far:
            base_data_dict.pop(cav_id)
        
        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)

        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
        ######################################################

        ############ for disconet ###########
        if self.kd_flag:
            stack_lidar_np = np.vstack(projected_lidar_clean_list)
            stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
            stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)

        # exclude all repetitive objects    
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)

        merged_feature_dict = self.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'lidar_poses_clean': lidar_poses_clean,
             'lidar_poses': lidar_poses})

        if self.kd_flag:
            processed_data_dict['ego'].update({'teacher_processed_lidar':
                stack_feature_processed})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})


        processed_data_dict['ego'].update({'sample_idx': idx,
                                            'cav_id_list': cav_id_list})

        # processed_data_dict['ego'].update({'veh_frame_id': base_data_dict[0]['veh_frame_id']})

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
