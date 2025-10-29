import torch
import numpy as np
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
import sys
import os
import glob
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.box_utils import project_box3d
from collections import OrderedDict
from icecream import ic
from matplotlib import pyplot as plt

def draw_trajectory(pt_file, save_path):
    '''
    box_dict : {
        'single_detection_bbx': # dict, [0, 1, ... , N-1],  
            [i]: dict:{
                [0]/[1]/[2]: past 3 frames detection results:{ # 都是在[0]的各自坐标系下的检测结果。
                    pred_box_3dcorner_tensor
                    pred_box_center_tensor
                    scores
                }
            }
        }
        'lidar_pose_0': 过去第一帧, 所有车都的pose
        'lidar_pose_current': 当前帧, 所有车都的pose
        'compensated_results_list': # len=N-1, 每个non-ego补偿出来的box, each of which is [N_obj, 4, 3], 注意这里用的是4个点, 最后一个维度上是 xyz, z是多余的
        'matched_idx_list': matched_idx_list, # len=N-1, 表示每个non-ego的过去2/3帧的匹配id, each of which is [N_obj, 2or3], 比如 ['matched_idx_list'][0] shape(22,2) 表示过去第一帧的22个框与第二帧的22个框的索引的匹配情况
        'single_gt_box_tensor': # list, len=N, 表示ego与non-ego每辆车在current时刻的检测框结果, 例如box_dict['single_gt_box_tensor'][1]大小为[N_obj, 8, 3] 表示第二辆车在current时刻的检测框结果。在各自current时刻坐标系下
        'single_lidar': # list, len=N, 表示ego与non-ego每辆车在current时刻的lidar np
        'gt_range': # [-140.8, -40, -3, 140.8, 40, 1], 表示lidar的范围
    }

    TODO
    1. transform detection results (from past 1 frame coordinate) into current frame
    2. visualzie GT box
    3. visualize detection results
    4. draw lidar point cloud
    5. clustering the same object (matched_idx_list)
    '''
    data = torch.load(pt_file)

    agent_num = len(data["single_detection_bbx"]) # 一个场景下的agent数目
    left_hand = True
    pc_range = data['gt_range']
    print('agent_num is :', agent_num)
    print(pc_range)
    for agent_id in range(1, agent_num): # 这是一个场景下的所有车 ego 不参与循环
        # step0: initailize canvas
        canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=(int((pc_range[4]-pc_range[1])*10), int((pc_range[3]-pc_range[0])*10)),
                                                canvas_x_range=(pc_range[0], pc_range[3]), 
                                                canvas_y_range=(pc_range[1], pc_range[4]),
                                                left_hand=left_hand) 
        # step 1: transform detection results
        lidar_pose_past = data['lidar_pose_0'][agent_id].cpu().numpy() # past0的 pose （6）
        lidar_pose_current = data['lidar_pose_current'][agent_id].cpu().numpy() # cur 的pose （6）

        past_to_current = x1_to_x2(lidar_pose_past, lidar_pose_current)
        past_to_current = torch.from_numpy(past_to_current).to('cuda').float() # 转移到cuda
        boxes_past_in_past_coor_dict = data['single_detection_bbx'][agent_id]  # 取出场景下某一个agent的检测数据 三帧
        boxes_past_in_current_coor_dict = OrderedDict()

        for past_frame, detections in boxes_past_in_past_coor_dict.items(): # 遍历三帧检测结果，全部转到cur  view
            if past_frame == 'past_k_time_diff':
                continue
            boxes_past_in_current_coor_dict[past_frame] = OrderedDict()
            boxes_past_in_current_coor_dict[past_frame]['pred_box_3dcorner_tensor'] = \
                project_box3d(detections['pred_box_3dcorner_tensor'].float(), past_to_current) # （n, 8, 3）
            boxes_past_in_current_coor_dict[past_frame]['scores'] = detections['scores'] # (n)


        # step 2: draw GT box
        gt_box_np = data['single_gt_box_tensor'][agent_id].cpu().numpy()
        canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=['gt']*gt_box_np.shape[0])

        # step 3: visualze detection result, compensation, Establish inverted indices
        matched_idx_list = data['matched_idx_list']
        match_pairs = matched_idx_list[agent_id - 1] # 减去1是因为
        label_1 = [''] * boxes_past_in_current_coor_dict[1]['pred_box_3dcorner_tensor'].shape[0]
        label_0 = [''] * boxes_past_in_current_coor_dict[0]['pred_box_3dcorner_tensor'].shape[0]
        
        for pair_idx, pair in enumerate(match_pairs):
            idx_in_0 = pair[0]
            idx_in_1 = pair[1]
            label_1[idx_in_1] = pair_idx
            label_0[idx_in_0] = pair_idx

        ## past 2 frame (key is 1)
        pred_box_np_1 = boxes_past_in_current_coor_dict[1]['pred_box_3dcorner_tensor'].cpu().numpy()
        canvas.draw_boxes(pred_box_np_1,colors=(int(255*0.5),0,0), texts=label_1)

        ## past 1 frame (key is 0)
        pred_box_np_0 = boxes_past_in_current_coor_dict[0]['pred_box_3dcorner_tensor'].cpu().numpy()
        canvas.draw_boxes(pred_box_np_0,colors=(int(255*0.8),0,0), texts=label_0)

        ## compensate box
        compensate_box_np = data['compensated_results_list'][agent_id-1].cpu().numpy() # （N_obj, 4, 3）

        if compensate_box_np.shape[0] == 0:
            print("Error: no compensate box: ", pt_file, " ", agent_id)
            return
        compensate_box_np = np.concatenate([compensate_box_np, compensate_box_np], axis=1)
        compensate_box_np = project_box3d(compensate_box_np, past_to_current.cpu().numpy())
        canvas.draw_boxes(compensate_box_np,colors=(0, int(255*0.83), int(255*0.83)), texts=['cp']*compensate_box_np.shape[0])


        # step 4: draw lidar point cloud
        pcd_np = data['single_lidar'][agent_id].cpu().numpy()
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
        canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points

        print('gt_box_np is :', gt_box_np.shape)
        print('pred_box_np_1 is :', pred_box_np_1.shape)
        print('pred_box_np_0 is :', pred_box_np_0.shape)
        print('compensate_box_np is :', compensate_box_np.shape)
        print('pcd_np is :', pcd_np.shape)

        plt.axis("off")
        plt.imshow(canvas.canvas)
        plt.tight_layout()
        save_name = pt_file.split("/")[-1].rstrip(".pt")
        save_path_cav = os.path.join(save_path, f"{save_name}_{agent_id}.png")
        plt.savefig(save_path_cav, dpi=400)
        plt.close()
        print('save file name is : ', f"{save_path}{save_name}_{agent_id}.png")



def main():
    # data_dir = '/remote-home/share/sizhewei/logs/irv2v_where2comm_cobevflow_w_dir_finetune/vis_cobevflow_reverse_0.50/bbx_folder' #'/remote-home/share/sizhewei/logs/irv2v_where2comm_cobevflow_w_dir_finetune/vis_cobevflow_521_trick_reverse_new_0.50_0'
    data_dir = '/public/home/lilingzhi/xyj/logs/logs/v2xset_pointpillar_where2comm_maxfusion_resnet_2024_04_20_16_12_53/vis_ir_thre_0_d_20_0.00_noise_0_0_0_0_roi_-1/' #'/remote-home/share/sizhewei/logs/irv2v_where2comm_cobevflow_w_dir_finetune/vis_cobevflow_521_trick_reverse_new_0.50_0'
    files = glob.glob(os.path.join(data_dir, 'bbx_folder', '*.pt'))
    files.sort()
    # print("files sort completed!")

    save_path = os.path.join(data_dir, 'multi_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # print('files are:',files)
    for file in files:
        draw_trajectory(file, save_path)

if __name__ == "__main__":
    main()

