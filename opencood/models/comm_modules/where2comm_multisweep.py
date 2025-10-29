# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>, Sizhe Wei <sizhewei@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre'] # 0.01
        self.num_blocks_thres = 1
        if 'num_blocks_thres' in args:
            self.num_blocks_thres = args['num_blocks_thres']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size'] # 5
            c_sigma = args['gaussian_smooth']['c_sigma'] # 1.0
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix):
        """ 

        Parameters:
        -----------
        batch_confidence_maps:[(L1xk, 2, H, W), (L2xk, 2, H, W), ..., (L_B x k, 2, H, W)]
        confidence_maps (without regroup func): (B, 2, H, W)  B
        pairwise_t_matrix: (B,L,k,2,3)

        Returns:
        --------
        batch_communication_maps: list
            orignal confidence map x mask

        communication_masks: list / torch.Tensor(修改后是 list, 方便下游处理任务)
            mask, shape:(B, 1, H, W). 0 or 1

        communication_rates: float

        Memos:
        ------
        self.thre: threshold of objectiveness
        a_ji = (1 - q_i)*q_ji
        """
        
        '''
        B: batch_size
        L: max cav num
        k: num of past frames
        H, W: feature map height and weight H,W:100,352
        '''
        B, L, k, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        # print("____________________")
        # print('batch_confidence_maps shape is ', batch_confidence_maps[0].shape)
        # print("____________________")

        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        all_blocks = H*W # 对于V2XSET 是100*352 相当于像素点个数 这里要注意，原场景体素化后是200*702，然是rm的形状却是100， 352
        num_valid_thres = int(all_blocks * self.num_blocks_thres) # 乘以一个比例，也就是说最多默认是可以全传，但是这个比例可以限制
        for b in range(B):

            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors, (num_cav * k, 1, H, W)
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps) # (num_cav * k, 1, H, W)
            else:
                communication_maps = ori_communication_maps

            if self.num_blocks_thres<1 : # 这个参数用来限制总传输像素个数，会在object比较多的时候有用，保留置信度最高的像素块 ，这个参数默认设置为1
                curr_batch = communication_maps.shape[0] # num_cav * k
                communication_mask = (communication_maps>self.thre).to(torch.int) #  (num_cav * k, 1, H, W)
                valid_nums = torch.sum(communication_mask, dim=(-1, -2, -3))

                # print("communication_mask shape is ", communication_mask.shape)
                # print("valid_nums is ", valid_nums)
                # xxx
                final_comm_mask = torch.zeros_like(communication_mask)
                for i in range(curr_batch): # 遍历每一张特征图
                    if valid_nums[i] > num_valid_thres :
                        # 选择 comm_maps[i] 里面 top num_valid 的位置
                        tmp = communication_maps[i].reshape((1, -1))
                        _, idx = torch.topk(tmp, num_valid_thres)
                        # print(idx)
                        curr_mask = torch.zeros_like(tmp)
                        curr_mask[0, idx[0]] = 1
                        final_comm_mask[i] = curr_mask.reshape((1, H, W))
                    else:
                        final_comm_mask[i] = communication_mask[i]
                communication_mask = final_comm_mask
            else:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps>self.thre, ones_mask, zeros_mask) # (num_cav * k, 1, H, W) 大于阈值的像素置1

            # communication_rate = communication_mask[0].sum()/(H*W) # 第一个车的第一帧对应的置信度图中 置一的像素个数 / 总像素个数。即 ego 的通信数据 得到的是一个标量
            communication_rate = communication_mask[0].sum() # 第一个车的第一帧对应的置信度图中 置一的像素个数 / 总像素个数。即 ego 的通信数据 得到的是一个标量
            # print(communication_rate)
            communication_masks.append(communication_mask) #  列表，每个元素为(num_cav * k, 1, H, W)
            communication_rates.append(communication_rate) #  列表，存储标量
            batch_communication_maps.append(ori_communication_maps*communication_mask) # (confidence > thre 的位置以及偶数id的cav 是1，其余是0) * confidence
        communication_rates = sum(communication_rates)/B
        # communication_masks = torch.concat(communication_masks, dim=0) # confidence > thre 的位置以及偶数id的cav 是1，其余是0；然后concat到一起
        return batch_communication_maps, communication_masks, communication_rates 