# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils
from opencood.utils.common_utils import limit_period

from collections import OrderedDict


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W'] # 504 
        H = self.params['anchor_args']['H'] # 200

        l = self.params['anchor_args']['l'] # 4.5
        w = self.params['anchor_args']['w'] # 2
        h = self.params['anchor_args']['h'] # 1.56
        r = self.params['anchor_args']['r'] # [0, 90]

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r] # 转换角度为弧度

        vh = self.params['anchor_args']['vh'] # voxel_size 0.4
        vw = self.params['anchor_args']['vw'] # 0.4

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]] # -100.8, 100.8
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]] # -40, 40

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride'] # 2
        else:
            feature_stride = 2


        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride) # 从 -100.8 + 0.4 到 100.8 - 0.4 生成 504//2 = 252 个数
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride) # 从 -40 + 0.4 到 40 - 0.4 生成 200//2 = 100 个数


        cx, cy = np.meshgrid(x, y) # 构建网格平面 形状都为 [100， 252]
        cx = np.tile(cx[..., np.newaxis], self.anchor_num) # center [100, 252, 2] 增加一个锚框数量的维度 这里搞清楚，cx是锚框的中心x坐标，和旋转无关，所以可以直接cp一份，下面的cy cz一样
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0 # 每个元素都是-1 形状 [100, 252, 2]

        w = np.ones_like(cx) * w # 缩放到对应的anchor box大小限制 [100, 252, 2]
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx) # 拷贝形状，三维数组，全1
        for i in range(self.anchor_num):
            r_[..., i] = r[i] # 切片，锁定前几维度，最后一维度选第i项，即存储锚框旋转角度

        if self.params['order'] == 'hwl': # pointpillar
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1) # (50, 176, 2, 7) <--原始备注  [100, 252, 2， 7]

        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)
            object [100, 7], 预设置的锚框 [100, 252, 2, 7] , object的mask [100]
        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']
        # (max_num)
        masks = kwargs['mask']

        # (H, W)
        feature_map_shape = anchors.shape[:2] # [100, 252]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)  # 锚框坐标结构[cx, cy, cz, h, w, l, r] 这里求出标准化因子 w**2 + l**2 开根号

        # (H, W, 2)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = \
            box_utils.boxes_to_corners_3d(gt_box_center_valid,
                                          self.params['order'])
        # (H*W*anchor_num, 8, 3)
        anchors_corner = \
            box_utils.boxes_to_corners_3d(anchors,
                                          order=self.params['order']) # 预设置的锚框也转变成八角表示
        # (H*W*anchor_num, 4)
        anchors_standup_2d = \
            box_utils.corner2d_to_standup_box(anchors_corner) # 返回锚框x,y的最小值和最大值 格式是[xmin,ymin,xmax,ymax]
        # (n, 4)
        gt_standup_2d = \
            box_utils.corner2d_to_standup_box(gt_box_corner_valid) # GT的同理

        # (H*W*anchor_n)
        iou = bbox_overlaps( # Cython函数，锚框和真值框之间的iou，结果是返回每个锚框的iou，形状为[H*W*anchor_num, n] 其中的任意一项[i, j] 表示的是第i个锚框和第j个真值框的重合度
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1) # 转置后[n， H*W*anchor_num]，选出最大值的索引 [n] 相当于每一个GT记录下最大iou的anchor id
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0 # 转置后的形状为[n, H*W*anchor_num]， 第一个维度全选，第二个维度选择出n个最大iou的anchor，合起来也就是说选出来n个也就是每个gt对应的最大iou是否是大于0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask] # 筛选，这两者形状是一致的

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = \
            np.where(iou >
                     self.params['target_args']['pos_threshold']) # iou大于0.6 分别返回符合要求的锚框索引和相应符合要求的gt 这两个返回的变量的形状应该是一致的如 [12] [12] 分别表示纵列索引
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(np.sum(iou <
                                 self.params['target_args']['neg_threshold'],
                                 axis=1) == iou.shape[1])[0] # iou 小于0.45 其中== iou.shape[1]判断的是每一行的所有元素是不是都是小于0.45，这返回的是bool值，然后np.where(...)[0]返回的是满足条件的一维数组，表示一行中所有元素都小于0.45的行索引
 
        id_pos = np.concatenate([id_pos, id_highest]) # 拼接起来 一个是iou大于0.6的锚框索引，一个是iou最大的锚框索引 如[12] 和 [7] 拼接成[19] 行索引
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt]) # 拼接起来 一个是iou大于0.6的gt索引，一个是iou最大的gt索引 如[12] 和 [7] 拼接成[19] 列索引
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index] # 以上两行是为了去重
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num)) # 等价于np.unravel_index(id_pos, (H, W, anchor_num)) 根据行id的索引，找到其在[H, W, 2]中对应的各个轴的索引
        pos_equal_one[index_x, index_y, index_z] = 1 # 置位 [H,W,2]

        # calculate the targets 总而言之target就是调整向量，就是计算有着较大IOU（极大或者超过阈值：0.6）的anchor在
        # 形状[H, W, anchor_num * 7]   bbx的七个值 每一个坐标都是：(gt的值 - 对应anchor的值) / 标准化因子 每一个长宽高则是：log(gt的值/anchor的值)这是为了大物体与小物体的相同偏差效果应该不同， 每一个角度都是gt值减去anchor值
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1 # 和所有的真值框的iou都小于0.45的将被标记

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0  # 避免既是 neg_equal_one 置1 又是 pos_equal_one 置1，也就是有的真值框和它对应的最大iou的锚框，这两者的iou仍然<0.45，这种情况下neg_equal_one显然不应该置1


        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict
    
    @staticmethod
    def merge_label_to_dict(label_k_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_k_list)):
            pos_equal_one.append(label_k_list[i]['pos_equal_one'])
            neg_equal_one.append(label_k_list[i]['neg_equal_one'])
            targets.append(label_k_list[i]['targets'])

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []
        # 原本是一个存放若干字典的列表，现在要合并，返回一个单一的字典
        for i in range(len(label_batch_list)): # 这是判断有几辆车
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    """
    Added by Sizhewei @ 2023-04-15
    Generate box results on each cav's past_0 time
    """
    def bandwidth_filter(self, input, num_box):
        if input['scores'].shape[0] <= num_box:
            return input
        
        topk_idx = torch.argsort(input['scores'], descending=True)[:num_box] # 降序排列
        output = {}
        output['scores'] = input['scores'][topk_idx]
        output['pred_box_3dcorner_tensor']  = input['pred_box_3dcorner_tensor'][topk_idx]
        output['pred_box_center_tensor'] = input['pred_box_center_tensor'][topk_idx]

        return output

    def single_post_process(self, m_single, trans_mat_pastk_2_past0, past_time_diff, anchor_box, num_sweeps=2, num_roi_thres=-1):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        m_single: Dict{'psm_single': (k, 2, H, W), 'rm_single': (k, 14, H, W), dm_single: (k, 4, H, W)}
        trans_mat_pastk_2_past0: (k, 4, 4) k帧中每一帧到第0帧的变换矩阵
        past_time_diff: (k) 对应 某一个agent 的k帧
        anchor_box: 预设置的锚框 (H, W, 2, 7)
        data_dict : dict
            
        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        {
            'past_k_time_diff' : (k, )
            [0] : {
                pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. (n, 8, 3)
                pred_box_center_tensor : (n, 7)
                scores: (n, )
            },
            ...
            [k-1] : { ... }
        }
        """
        self.k = num_sweeps
        self.num_roi_thres = num_roi_thres
        
        psm_single = m_single['psm_single'] # (k, 2, H, W)
        rm_single = m_single['rm_single'] # (k, 14, H, W)

        use_dir_flag = False
        if 'dm_single' in m_single.keys():
            use_dir_flag = True
            dm_single = m_single['dm_single'] # (k, 4, H, W)

        box_results = OrderedDict()
        # for cav_id, cav_content in data_dict.items():

        box_results.update({
            'past_k_time_diff': past_time_diff
        })
        # the transformation matrix to ego space
        # transformation_matrix = cav_content['transformation_matrix'] # no clean
        transformation_matrix = trans_mat_pastk_2_past0.to(torch.float32) # (k, 4, 4)

        # classification probability
        prob = psm_single  # k,2,100,352
        prob = torch.sigmoid(prob.permute(0, 2, 3, 1)) # (k, H, W, 2) anchor num==2 所以是每一种anchor的概率
        prob = prob.reshape(self.k, -1) # (k, HxWx2)

        # regression map
        reg = rm_single # (k,14,100,352)

        # convert regression map back to bounding box
        batch_box3d = self.delta_to_boxes3d(reg, anchor_box) # (k, HxWx2, 7) 将锚框按照预测偏移给变化好了
        mask = \
            torch.gt(prob, self.params['target_args']['score_threshold'])  # 0.25类似的数值，筛选出大于这个数值的部分，设为True (k, HxWx2)
        mask = mask.view(self.k, -1) # (k, HxWx2)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7) # (k, HxWx2, 7)

        # print('psm_single shape is  ',psm_single.shape)
        # print('rm_single shape is  ',rm_single.shape)
        # print('anchor_box shape is  ',anchor_box.shape)
        # print('batch_box3d shape is  ',batch_box3d.shape)

        # during validation/testing, the batch size should be 1
        # assert batch_box3d.shape[0] == 1

        if use_dir_flag:
            dir_offset = self.params['dir_args']['dir_offset'] # 这个是个弧度值 是π/4
            num_bins = self.params['dir_args']['num_bins']
            dm = dm_single # [N, H, W, 4]?? 应该是(K, 4, H, W) 单车检测的方向预测
        
        for i in range(self.k):  # 遍历所有的时间戳
            box_results[i] = OrderedDict()
            unit_trans_mat = transformation_matrix[i] # （4， 4）
            boxes3d = torch.masked_select(batch_box3d[i],
                                        mask_reg[i]).view(-1, 7) # 选择出符合要求（分类概率大于0.25）的锚框，假设复合要求的有M个，则是(M, 7)
            scores = torch.masked_select(prob[i], mask[i]) # （M）

            ########### adding dir classifier
            if use_dir_flag and len(boxes3d)!=0:
                dir_cls_preds = dm[i:i+1].permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask[i:i+1]] # （1，M，2）的张量 所有符合要求的锚框其方向预测
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0p    ？？形状应该是(1, M) 每个元素为0或者1，表示较大的那个的索引，如果是1则是大于0pi
                
                period = (2 * np.pi / num_bins) # pi 这个是周期
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi dir_labels中的0或者1其实就是代表着方向的类别
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]
            ###########################################

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                order=self.params['order']) # （M, 8，3） M是符合分类概率大于一定门槛的对应锚框个数
                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            unit_trans_mat) # 将空间坐标投影到第0帧 （M, 8，3）

                # backup_projected_boxes3d = projected_boxes3d.clone()
                # backup_scores = scores.clone()

                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d) # 转变成2d bbx 形式为(xmin, ymin, xmax, ymax)   （M, 4）
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1) # （M, 5） 也就是 (xmin, ymin, xmax, ymax，prob)

                # pred_box2d_list.append(boxes2d_score)
                # pred_box3d_list.append(projected_boxes3d)

                # scores = boxes2d_score[:, -1]

                # remove large bbx
                keep_index_1 = box_utils.remove_large_pred_bbx(projected_boxes3d) # 返回的是一种条件格式，即满足长宽大小小于等于6m
                keep_index_2 = box_utils.remove_bbx_abnormal_z(projected_boxes3d) # 返回的是一种条件格式，即满足z坐标在正常范围内[-1, 3]
                keep_index = torch.logical_and(keep_index_1, keep_index_2)

                projected_boxes3d = projected_boxes3d[keep_index] # 过滤异常的box （M, 8，3）- > （M', 8，3）
                scores = scores[keep_index] # 过滤异常的box （M）- > （M'）

                # STEP3
                # nms
                keep_index = box_utils.nms_rotated(projected_boxes3d,
                                                scores,
                                                self.params['nms_thresh']
                                                )
                pred_box3d_tensor = projected_boxes3d[keep_index] # NMS 后的box （M', 8，3）- > （M'', 8，3）
                # select cooresponding score
                scores = scores[keep_index]

                # filter out the prediction out of the range.
                range_mask = \
                    box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, self.params['gt_range'])
                pred_box_3dcorner_tensor = pred_box3d_tensor[range_mask, :, :]
                scores = scores[range_mask]
                pred_box_center_tensor = box_utils.corner_to_center_torch(pred_box_3dcorner_tensor, self.params['order']) # 转换成标准表示 （N, 7）

                assert scores.shape[0] == pred_box_3dcorner_tensor.shape[0]

                # if scores.shape[0] == 0: # 没有bbx剩下，那么保存之前剩的框
                #     pred_box_3dcorner_tensor = backup_projected_boxes3d
                #     scores = backup_scores
                #     pred_box_center_tensor = box_utils.corner_to_center_torch(pred_box_3dcorner_tensor, self.params['order'])

                try:
                    self.num_roi_thres = self.num_roi_thres # 这个应该是roi数量限制
                except KeyError:
                    print('Note! There is no num_roi_thres in the config file.')
                    self.num_roi_thres = -1
                if self.num_roi_thres != -1:
                    original_results = {
                        'pred_box_3dcorner_tensor': pred_box_3dcorner_tensor, 
                        'pred_box_center_tensor': pred_box_center_tensor,
                        'scores': scores
                    }
                    sorted_box_results = self.bandwidth_filter(original_results, self.num_roi_thres)
                    box_results[i].update({
                        'pred_box_3dcorner_tensor': sorted_box_results['pred_box_3dcorner_tensor'], 
                        'pred_box_center_tensor': sorted_box_results['pred_box_center_tensor'],
                        'scores': sorted_box_results['scores']
                    })
                else: 
                    box_results[i].update({
                        'pred_box_3dcorner_tensor': pred_box_3dcorner_tensor,  # （N, 8， 3）
                        'pred_box_center_tensor': pred_box_center_tensor, # （N, 7）
                        'scores': scores # （N） 置信度得分
                    })

            else:
                box_results[i].update({
                    'pred_box_3dcorner_tensor': torch.Tensor(0, 8, 3).to(scores.device), 
                    'pred_box_center_tensor': torch.Tensor(0, 7).to(scores.device),
                    'scores': torch.Tensor(0).to(scores.device)
                })

        return box_results
    
    def single_post_process_w_uncertainty(self, m_single, trans_mat_pastk_2_past0, past_time_diff, anchor_box, num_sweeps=2, num_roi_thres=-1, pairwise_t_matrix_past0_2_cur = None):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        m_single: Dict{'psm_single': (k, 2, H, W), 'rm_single': (k, 14, H, W), dm_single: (k, 4, H, W)}
        trans_mat_pastk_2_past0: (k, 4, 4) k帧中每一帧到第0帧的变换矩阵
        past_time_diff: (k) 对应 某一个agent 的k帧
        anchor_box: 预设置的锚框 (H, W, 2, 7)
        pairwise_t_matrix_past0_2_cur: (4, 4)
        data_dict : dict
            
        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        {
            'past_k_time_diff' : (k, )
            [0] : {
                pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. (n, 8, 3)
                pred_box_center_tensor : (n, 7)
                scores: (n, )
            },
            ...
            [k-1] : { ... }
        }
        """
        self.k = num_sweeps
        self.num_roi_thres = num_roi_thres
        
        psm_single = m_single['psm_single'] # (k, 2, H, W)
        rm_single = m_single['rm_single'] # (k, 14, H, W)

        predict_unc_cls = m_single['predict_unc_cls'] # (k, 2, H, W)
        predict_unc_reg = m_single['predict_unc_reg'] # (k, 2, H, W)

        # print("predict_unc_cls shape is", predict_unc_cls.shape)
        use_dir_flag = False
        if 'dm_single' in m_single.keys():
            use_dir_flag = True
            dm_single = m_single['dm_single'] # (k, 4, H, W)

        box_results = OrderedDict()
        # for cav_id, cav_content in data_dict.items():

        box_results.update({
            'past_k_time_diff': past_time_diff,
            'matrix_past0_2_cur': pairwise_t_matrix_past0_2_cur.to(torch.float32)
        })
        # the transformation matrix to ego space
        # transformation_matrix = cav_content['transformation_matrix'] # no clean
        transformation_matrix = trans_mat_pastk_2_past0.to(torch.float32) # (k, 4, 4)

        # classification probability
        prob = psm_single  # k,2,100,352
        prob = torch.sigmoid(prob.permute(0, 2, 3, 1)) # (k, H, W, 2) anchor num==2 所以是每一种anchor的概率
        prob = prob.reshape(self.k, -1) # (k, HxWx2)

        # regression map
        reg = rm_single # (k,14,100,352)

        # convert regression map back to bounding box
        batch_box3d = self.delta_to_boxes3d(reg, anchor_box) # (k, HxWx2, 7) 将锚框按照预测偏移给变化好了
        mask = \
            torch.gt(prob, self.params['target_args']['score_threshold'])  # 0.25类似的数值，筛选出大于这个数值的部分，设为True (k, HxWx2)
        mask = mask.view(self.k, -1) # (k, HxWx2)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7) # (k, HxWx2, 7)

        cls_data_unc = predict_unc_cls[:,:2,:,:] # (k, 2, H, W)
        cls_model_unc = predict_unc_cls[:,2:,:,:] # (k, 2, H, W)
        reg_data_unc = predict_unc_reg[:,:2,:,:] # (k, 2, H, W)
        reg_model_unc = predict_unc_reg[:,2:,:,:] # (k, 2, H, W)

        cls_data_unc = cls_data_unc.permute(0, 2, 3, 1).reshape(self.k, -1) # (k, 2HW)
        cls_model_unc = cls_model_unc.permute(0, 2, 3, 1).reshape(self.k, -1) # (k, 2HW)
        reg_data_unc = reg_data_unc.permute(0, 2, 3, 1).reshape(self.k, -1) # (k, 2HW)
        reg_model_unc = reg_model_unc.permute(0, 2, 3, 1).reshape(self.k, -1) # (k, 2HW)
        # predict_unc_cls = predict_unc_cls.permute(0, 2, 3, 1).reshape(self.k, -1) # (k, 2HW)
        # predict_unc_reg = predict_unc_reg.permute(0, 2, 3, 1).reshape(self.k, -1) # (k, 2HW)

        # print('psm_single shape is  ',psm_single.shape)
        # print('rm_single shape is  ',rm_single.shape)
        # print('anchor_box shape is  ',anchor_box.shape)
        # print('batch_box3d shape is  ',batch_box3d.shape)

        # during validation/testing, the batch size should be 1
        # assert batch_box3d.shape[0] == 1

        if use_dir_flag:
            dir_offset = self.params['dir_args']['dir_offset'] # 这个是个弧度值 是π/4
            num_bins = self.params['dir_args']['num_bins']
            dm = dm_single # [N, H, W, 4]?? 应该是(K, 4, H, W) 单车检测的方向预测
        
        for i in range(self.k):  # 遍历所有的时间戳
            box_results[i] = OrderedDict()
            unit_trans_mat = transformation_matrix[i] # （4， 4）
            boxes3d = torch.masked_select(batch_box3d[i],
                                        mask_reg[i]).view(-1, 7) # 选择出符合要求（分类概率大于0.25）的锚框，假设符合要求的有M个，则是(M, 7)
            scores = torch.masked_select(prob[i], mask[i]) # （M）

            u_cls_data = torch.masked_select(cls_data_unc[i], mask[i]) # （M）
            u_cls_model = torch.masked_select(cls_model_unc[i], mask[i]) # （M）
            u_reg_data = torch.masked_select(reg_data_unc[i], mask[i]) # （M）
            u_reg_model = torch.masked_select(reg_model_unc[i], mask[i]) # （M）
            # u_cls = torch.masked_select(predict_unc_cls[i], mask[i]) # （M）
            # u_reg = torch.masked_select(predict_unc_reg[i], mask[i]) # （M）

            # print("u_cls_data is ", u_cls_data[:20])
            # print("u_cls_model is ", u_cls_model[:20])
            
            ########### adding dir classifier
            if use_dir_flag and len(boxes3d)!=0:
                dir_cls_preds = dm[i:i+1].permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask[i:i+1]] # （1，M，2）的张量 所有符合要求的锚框其方向预测
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0p    ？？形状应该是(1, M) 每个元素为0或者1，表示较大的那个的索引，如果是1则是大于0pi
                
                period = (2 * np.pi / num_bins) # pi 这个是周期
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi dir_labels中的0或者1其实就是代表着方向的类别
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]
            ###########################################

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                order=self.params['order']) # （M, 8，3） M是符合分类概率大于一定门槛的对应锚框个数
                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            unit_trans_mat) # 将空间坐标投影到第0帧 （M, 8，3）

                # backup_projected_boxes3d = projected_boxes3d.clone()
                # backup_scores = scores.clone()

                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d) # 转变成2d bbx 形式为(xmin, ymin, xmax, ymax)   （M, 4）
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1) # （M, 5） 也就是 (xmin, ymin, xmax, ymax，prob)

                # pred_box2d_list.append(boxes2d_score)
                # pred_box3d_list.append(projected_boxes3d)

                # scores = boxes2d_score[:, -1]

                # remove large bbx
                keep_index_1 = box_utils.remove_large_pred_bbx(projected_boxes3d) # 返回的是一种条件格式，即满足长宽大小小于等于6m
                keep_index_2 = box_utils.remove_bbx_abnormal_z(projected_boxes3d) # 返回的是一种条件格式，即满足z坐标在正常范围内[-1, 3]
                keep_index = torch.logical_and(keep_index_1, keep_index_2)

                projected_boxes3d = projected_boxes3d[keep_index] # 过滤异常的box （M, 8，3）- > （M', 8，3）
                scores = scores[keep_index] # 过滤异常的box （M）- > （M'）

                u_cls_data = u_cls_data[keep_index]
                u_cls_model = u_cls_model[keep_index]
                u_reg_data = u_reg_data[keep_index]
                u_reg_model = u_reg_model[keep_index]
                # u_cls = u_cls[keep_index]
                # u_reg = u_reg[keep_index]

                # STEP3
                # nms
                keep_index = box_utils.nms_rotated(projected_boxes3d,
                                                scores,
                                                self.params['nms_thresh']
                                                )
                pred_box3d_tensor = projected_boxes3d[keep_index] # NMS 后的box （M', 8，3）- > （M'', 8，3）
                # select cooresponding score
                scores = scores[keep_index]
                u_cls_data = u_cls_data[keep_index]
                u_cls_model = u_cls_model[keep_index]
                u_reg_data = u_reg_data[keep_index]
                u_reg_model = u_reg_model[keep_index]
                # u_cls = u_cls[keep_index]
                # u_reg = u_reg[keep_index]

                # filter out the prediction out of the range.
                range_mask = \
                    box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, self.params['gt_range'])
                pred_box_3dcorner_tensor = pred_box3d_tensor[range_mask, :, :]
                scores = scores[range_mask]
                pred_box_center_tensor = box_utils.corner_to_center_torch(pred_box_3dcorner_tensor, self.params['order']) # 转换成标准表示 （N, 7）
                
                u_cls_data = u_cls_data[range_mask]
                u_cls_model = u_cls_model[range_mask]
                u_reg_data = u_reg_data[range_mask]
                u_reg_model = u_reg_model[range_mask]
                # u_cls = u_cls[range_mask]
                # u_reg = u_reg[range_mask]

                # print("单车检测到的分类不确定性得分: ", u_cls)
                # print("单车检测到的回归不确定性得分: ", u_reg)

                assert scores.shape[0] == pred_box_3dcorner_tensor.shape[0] == u_cls_data.shape[0] == u_cls_model.shape[0] == u_reg_data.shape[0] == u_reg_model.shape[0]

                # if scores.shape[0] == 0: # 没有bbx剩下，那么保存之前剩的框
                #     pred_box_3dcorner_tensor = backup_projected_boxes3d
                #     scores = backup_scores
                #     pred_box_center_tensor = box_utils.corner_to_center_torch(pred_box_3dcorner_tensor, self.params['order'])

                try:
                    self.num_roi_thres = self.num_roi_thres # 这个应该是roi数量限制
                except KeyError:
                    print('Note! There is no num_roi_thres in the config file.')
                    self.num_roi_thres = -1
                if self.num_roi_thres != -1:
                    original_results = {
                        'pred_box_3dcorner_tensor': pred_box_3dcorner_tensor, 
                        'pred_box_center_tensor': pred_box_center_tensor,
                        'scores': scores
                    }
                    sorted_box_results = self.bandwidth_filter(original_results, self.num_roi_thres)
                    box_results[i].update({
                        'pred_box_3dcorner_tensor': sorted_box_results['pred_box_3dcorner_tensor'], 
                        'pred_box_center_tensor': sorted_box_results['pred_box_center_tensor'],
                        'scores': sorted_box_results['scores'],
                        'u_cls_data': u_cls_data,
                        'u_cls_model': u_cls_model,
                        'u_reg_data': u_reg_data,
                        'u_reg_model': u_reg_model,
                    })
                else: 
                    box_results[i].update({
                        'pred_box_3dcorner_tensor': pred_box_3dcorner_tensor,  # （N, 8， 3）
                        'pred_box_center_tensor': pred_box_center_tensor, # （N, 7）
                        'scores': scores, # （N） 置信度得分
                        'u_cls_data': u_cls_data,
                        'u_cls_model': u_cls_model,
                        'u_reg_data': u_reg_data,
                        'u_reg_model': u_reg_model,
                        
                        # 'u_cls': u_cls,
                        # 'u_reg': u_reg
                    })

            else:
                box_results[i].update({
                    'pred_box_3dcorner_tensor': torch.Tensor(0, 8, 3).to(scores.device), 
                    'pred_box_center_tensor': torch.Tensor(0, 7).to(scores.device),
                    'scores': torch.Tensor(0).to(scores.device),
                    'u_cls_data': torch.Tensor(0).to(scores.device),
                    'u_cls_model': torch.Tensor(0).to(scores.device),
                    'u_reg_data': torch.Tensor(0).to(scores.device),
                    'u_reg_model': torch.Tensor(0).to(scores.device),
                    # 'u_cls': torch.Tensor(0).to(scores.device),
                    # 'u_reg': torch.Tensor(0).to(scores.device)
                })

        return box_results

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        for cav_id, cav_content in data_dict.items(): # no fusion的时候只会输入一个ego，而latefusion的时候则是场景下的所有cav
            assert cav_id in output_dict
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'] # no clean

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['psm'] # (N, 2, H, W) 一个scenario下的所有车的置信度图
            prob = F.sigmoid(prob.permute(0, 2, 3, 1)) # 归一化 （N，H，W，2）
            prob = prob.reshape(1, -1) # （1, NxHxWx2）

            # regression map
            reg = output_dict[cav_id]['rm'] # (N, 14, H, W)

            # convert regression map back to bounding box
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box) # (N, HxWx2, 7)
            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])# （1, NxHxWx2） 置信度筛选
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # adding dir classifier
            if 'dm' in output_dict[cav_id].keys() and len(boxes3d) !=0:
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']
                dm  = output_dict[cav_id]['dm'] # [N, H, W, 4]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                
                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, self.params['gt_range'])
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)?? should be (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7) # （N，2HW, 7）
        boxes3d = torch.zeros_like(deltas) # （N，2HW, 7）

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float() # （2HW, 7）
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2) # 勾股定理求2d框对角线 （2HW）
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2) # （N， 2， 2HW） 转置后 （N， 2HW， 2）
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1) # （N，2HW, 7）

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]] # 预测的deltas实际上是偏移量，对角线反应了锚框尺寸特征，因为不同尺寸下相同偏移可能造成的影响不同。例如，对于一个很小的锚框，10像素的偏移可能会导致它完全偏离目标区域。而对于一个很大的锚框，同样的10像素偏移可能几乎没有影响。为了解决这个问题，通常会将偏移量设计为相对于锚框尺寸的比例，而不是绝对的像素值。这样，偏移量就能够根据锚框的大小自适应缩放。
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]] # 长宽高用指数函数来调整：1.保证了正值，2. 保证了小偏移也有大变化，放大小变化 3. 更好更平滑的梯度分布 4. 在训练期间，模型学习预测长、宽、高的对数变化量。这是因为对数变换可以将较大范围的变化量转换为较小的区间，使得模型更容易学习。在推理时，使用指数函数可以将这些对数变化量转换回原始的尺寸空间。
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d #（N，2HW, 7）

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
