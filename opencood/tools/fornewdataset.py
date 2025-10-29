'''
Prediction model input data generation.
SizheWei 2023.4.23
'''

import os
import torch
from scipy import stats
from collections import OrderedDict

def retrieve_base_data(scenario_database, len_record, idx, binomial_n=10, binomial_p=0.1, k=3, is_no_shift=False, is_same_sample_interval=False):
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
                    'params': json_path,
                    'timestamp': string
                },
                past_k : {		                # (k) totally
                    [0]:{
                        'params': json_path,
                        'timestamp': string,
                        'time_diff': float,
                        'sample_interval': int
                    },
                    [1] : {},	(id-1)
                    ...,		
                    [k-1] : {} (id-(k-1))
                },
            }, 
            cav_id_2 : {		                # (k) totally
                'ego': false, 
                ...
            }, 
            ...
        }
    """
    sample_interval_exp = int(binomial_n * binomial_p) # 1
    # we loop the accumulated length list to get the scenario index
    scenario_index = 0
    for i, ele in enumerate(len_record):# 应该是遍历了所有场景
        if idx < ele:
            scenario_index = i
            break
    scenario_database = scenario_database[scenario_index]
    
    # 生成冻结分布函数
    bernoulliDist = stats.bernoulli(binomial_p) # 0.1概率

    data = OrderedDict()
    # 找到 current 时刻的 timestamp_index 这对于每辆车来讲都一样
    curr_timestamp_idx = idx if scenario_index == 0 else \
                    idx - len_record[scenario_index - 1] # 求得当前时间戳的索引
    curr_timestamp_idx = curr_timestamp_idx + binomial_n * k
    
    # load files for all CAVs
    for cav_id, cav_content in scenario_database.items():# 这时候已经具体到某一个场景 开始遍历场景中每一辆车
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

        timestamp_key = list(cav_content['regular'].items())[curr_timestamp_idx][0] # 取出时间戳字符串
        
        # 2.1 load curr params
        # json is faster than yaml
        json_file = cav_content['regular'][timestamp_key]['yaml'].replace("yaml", "json") # 取出描述文件名字符串
        json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")
        data[cav_id]['curr']['params'] = json_file

        # 2.3 store curr timestamp and time_diff
        data[cav_id]['curr']['timestamp'] = timestamp_key
        data[cav_id]['curr']['time_diff'] = 0.0
        data[cav_id]['curr']['sample_interval'] = 0

        # 3. past frames, for model input
        data[cav_id]['past_k'] = OrderedDict()
        latest_sample_stamp_idx = curr_timestamp_idx
        # past k frames, pose | lidar | label(for single view confidence map generator use)
        for i in range(k):
            # sample_interval
            if data[cav_id]['ego']:             # ego sample_interval = E(B(n, p))
                if i == 0: # ego-past-0 与 ego-curr 是一样的
                    data[cav_id]['past_k'][i] = data[cav_id]['curr']
                    continue
                sample_interval = sample_interval_exp # 1 期望值 也就是说 如果是ego的话 past0为当前帧，past1 past2 则是依次向前
                if sample_interval == 0:
                    sample_interval = 1
            else:                               # non-ego sample_interval ~ B(n, p)
                if sample_interval_exp==0 \
                    and is_no_shift \
                        and i == 0:
                    data[cav_id]['past_k'][i] = data[cav_id]['curr']
                    continue
                if is_same_sample_interval: # 相同采样间隔？默认为False
                    sample_interval = sample_interval_exp
                else:
                    # B(n, p)
                    trails = bernoulliDist.rvs(binomial_n) # 10次伯努利实验，概率为0.1 
                    sample_interval = sum(trails) # 随机值 0-10 
                if sample_interval==0:# 如果做十次实验，34.87%的概率会出现采样间隔为0
                    if i==0: # 检查past 0 的实际时间是否在curr 的后面
                        tmp_time_key = list(cav_content.items())[latest_sample_stamp_idx][0]# 取出当前时间戳的字符串
                        if dist_time(tmp_time_key, data[cav_id]['curr']['timestamp'])>0:# 如果past 0 时间戳大于 current时间戳 则采样间隔设置为1
                            sample_interval = 1
                    if i>0: # 过去的几帧不要重复
                        sample_interval = 1                

            # check the timestamp index
            data[cav_id]['past_k'][i] = {}
            latest_sample_stamp_idx -= sample_interval # 往前挪动若干个采样间隔
            timestamp_key = list(cav_content.items())[latest_sample_stamp_idx][0] # 找到对应的字符串
            # load the corresponding data into the dictionary
            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")
            data[cav_id]['past_k'][i]['params'] = json_file

            data[cav_id]['past_k'][i]['timestamp'] = timestamp_key # 这一帧对应的时间戳
            data[cav_id]['past_k'][i]['sample_interval'] = sample_interval
            data[cav_id]['past_k'][i]['time_diff'] = \
                dist_time(timestamp_key, data[cav_id]['curr']['timestamp']) # 时间差异永远是距离当前的时间间隔

    return data

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


# debug_path = '/remote-home/share/OPV2V_irregular_npy'
debug_path = '/dssg/home/acct-seecsh/seecsh/sizhewei/data_sftp'
scenario_database = torch.load(os.path.join(debug_path, 'scenario_database.pt'))
len_record = torch.load(os.path.join(debug_path, 'len_record.pt'))


unit_data = retrieve_base_data(scenario_database, len_record, 1)