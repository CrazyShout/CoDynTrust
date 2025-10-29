# -*- coding: utf-8 -*-
# Author: Runsheng Xu
# Modified: Sizhe Wei

import argparse
import os
import statistics
import sys
sys.path.append(os.getcwd())
import time

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import importlib
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange
import numpy as np
import random
import subprocess

run_test = True
# from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_irregular import illegal_path_list
compensation = True 

# def apply_dropout(m):
#     from torch import nn
#     print('type m is ',type(m))
#     if type(m) == nn.Dropout2d or type(m) == nn.Dropout:
#         m.train()

# def set_dropout_layers(model):
#     """
#     Function to enable specific dropout layers during test-time.
#     """
#     from torch import nn
#     for module in model.modules():
#         if isinstance(module, (nn.Dropout, nn.Dropout2d)):
#             module.train()

def worker_init_fn(worker_id):
    # 获取主进程的种子
    seed = torch.initial_seed() % 2**32
    # 为每个 worker 设定不同的随机种子
    train_deterministic(seed + worker_id)

def train_deterministic(myseed=42):
    '''
    "42" 出自道格拉斯·亚当斯(Douglas Adams)的小说《银河系漫游指南》(The Hitchhiker's Guide to the Galaxy)。
    在这部小说中, 超级计算机“Deep Thought”在被问到“生命、宇宙以及一切的终极答案”时, 经过七百五十万年的计算, 得出的答案是“42”
    '''
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


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--pretrained_path', default='',
                        help='The path of the model need to be fine tuned.')
    parser.add_argument('--device', '-d', default="cuda", help='cuda or cpu')
    parser.add_argument('--two_stage', help='whether to use two stage training', default=0, type=int)
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
    # train_deterministic()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    finetune_flag = False
    if 'is_finetune' in hypes:
        finetune_flag = hypes['is_finetune']
    if finetune_flag:
        finetune_time_interval = int(hypes['binomial_n'] * hypes['binomial_p'])
        print('### Finetune mode, only tune header. ###')

    print('### Dataset Building ... ###')
    start_time = time.time()
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=6,
                            # worker_init_fn=worker_init_fn,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=6,
                            # worker_init_fn=worker_init_fn,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    end_time = time.time()
    print("=== Time consumed: %.1f minutes. ===" % ((end_time - start_time)/60))
    start_time = time.time()
    
    print('### Creating Model ... ###')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    # lowest_val_loss = 0.355507
    lowest_val_epoch = -1    
    
    # load pre-train model for single view
    is_pre_trained = False
    if 'is_single_pre_trained' in hypes and hypes['is_single_pre_trained']['pre_train_flag']:
        is_pre_trained = True
    if is_pre_trained:
        pretrain_path = hypes['is_single_pre_trained']['pre_train_path']
        initial_epoch = hypes['is_single_pre_trained']['pre_train_epoch']
        pre_train_model = torch.load(os.path.join(pretrain_path, 'net_epoch%d.pth' % initial_epoch))
        diff_keys = {k:v for k, v in pre_train_model.items() if k not in model.state_dict()}
        if diff_keys:
            print(f"!!! PreTrained single model has keys: {diff_keys.keys()}, \
                which are not in the fusion model you have created!!!")
            raise ValueError("fusion model lack parameters! --from xyj at 2024/06/12")
        model.load_state_dict(pre_train_model, strict=False)
        print("### Pre-trained point pillar {} loaded successfully! ###".format(os.path.join(pretrain_path, 'net_epoch%d.pth' % initial_epoch)))
        fix = hypes['is_single_pre_trained']['pre_train_fix'] # 这个暂时用不到，因为我在训练单车的时候用的是fuse部分改的，导致所有的模型 parameter都在，然后会导致所有的参数都不需要反向传播
        # if fix:
        #     for name, value in model.named_parameters():
        #         if name in pre_train_model:
        #             value.requires_grad = False

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # lr scheduler setup
    num_steps = len(train_loader)# 分成了多少个批次

    # define the loss 
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model, is_pre_trained, num_steps)
    
    # for fine tune:
    if opt.pretrained_path:
        saved_path = opt.pretrained_path
        pretrained_model_dict = torch.load(saved_path, map_location='cpu')
        diff_keys = {k:v for k, v in pretrained_model_dict.items() if k not in model.state_dict()}
        if diff_keys:
            print(f"!!! PreTrained model has keys: {diff_keys.keys()}, \
                which are not in the model you have created!!!")
        diff_keys = {k:v for k, v in model.state_dict().items() if k not in pretrained_model_dict.keys()}
        if diff_keys:
            print(f"!!! Created model has keys: {diff_keys.keys()}, \
                which are not in the model you have trained!!!")
        model.load_state_dict(pretrained_model_dict, strict=False)
        for name, value in model.named_parameters():
            if name in pretrained_model_dict:
                value.requires_grad = False
    
    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model_diff(saved_path, model, finetune_flag)
        print("start from epoch: ", init_epoch)
        if finetune_flag:
            init_epoch = 0 # if finetune, we set the init_epoch to 10
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    else:
        init_epoch = 0
        log_path = '../logs' # Your log file path
        # if we train the model from scratch, we need to create a folder to save the model
        saved_path = train_utils.setup_train(hypes, log_path)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch=num_steps)

    end_time = time.time()
    print("=== Time consumed: %.1f minutes. ===" % ((end_time - start_time)/60))

    log_train_dir = os.path.join(saved_path,'log_train.log' )
    log_val_dir = os.path.join(saved_path,'log_val.log' )

    # record training
    writer = SummaryWriter(saved_path)

    start_time = time.time()
    print('### Training start! ###')
    epoches = hypes['train_params']['epoches']
    print('set max epochs is: ', epoches)
    supervise_single_flag = False
    if 'supervise_single_flag' in hypes['train_params'].keys():
        supervise_single_flag = hypes['train_params']['supervise_single_flag']
        print(f"=== supervise_single_flag: {supervise_single_flag} ===")

    ############ For DiscoNet ##############
    if "kd_flag" in hypes.keys():
        kd_flag = True
        teacher_model_name = hypes['kd_flag']['teacher_model'] # point_pillar_disconet_teacher
        teacher_model_config = hypes['kd_flag']['teacher_model_config']
        teacher_checkpoint_path = hypes['kd_flag']['teacher_path']

        # import the model
        model_filename = "opencood.models." + teacher_model_name
        model_lib = importlib.import_module(model_filename)
        teacher_model_class = None
        target_model_name = teacher_model_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                teacher_model_class = cls
        
        teacher_model = teacher_model_class(teacher_model_config)
        teacher_model.load_state_dict(torch.load(teacher_checkpoint_path), strict=False)
        
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
                teacher_model.to(device)

        teacher_model.eval()

    else:
        kd_flag = False

    sample_interval_all_epoch = 0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        # if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
        #     scheduler.step(epoch)
        # if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
        #     scheduler.step_update(epoch * num_steps + 0)
        if hasattr(opencood_train_dataset, "set_cur_epoch"):
            print("===记录当前epoch===")
            opencood_train_dataset.set_cur_epoch(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        
        pbar2 = tqdm(total=len(train_loader), leave=True, colour='green')

        sample_interval = 0
        i = 0
        for i, batch_data in enumerate(train_loader): 
            if batch_data is None:
                continue
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)

            batch_data['ego']['epoch'] = epoch

            # TODO: dataset parameter is only used for training flow module
            if opt.two_stage:
                ouput_dict = model(batch_data['ego'], opencood_train_dataset)
            else: 
                ouput_dict = model(batch_data['ego'])

            if kd_flag:
                teacher_output_dict = teacher_model(batch_data['ego'])
                ouput_dict.update(teacher_output_dict)

            # # only for SyncNet training
            # final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            # if i % 10 == 0:
            #     criterion.logging(epoch, i, len(train_loader), writer)

            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            
            if 'with_compensation' in hypes['model']['args'] and \
                hypes['model']['args']['with_compensation']:# 训练的时候是一起训练flow估计和单车 detector？
                if hypes['model']['args']['with_single_supervise']:
                    final_loss = ouput_dict['recon_loss'] 
                    single_det_loss = criterion(ouput_dict, batch_data['ego']['single_object_label'], mode = 'single')
                    detection_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                    final_loss = ouput_dict['recon_loss'] + single_det_loss + detection_loss
                else:
                    detection_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                    final_loss = ouput_dict['recon_loss'] + detection_loss
            else:
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict']) # 这个是协同场景下做损失
                # if i % 10 == 0:
                criterion.logging(epoch, i, len(train_loader), writer, pbar2, log_train_dir)
                if supervise_single_flag:
                    final_loss += criterion(ouput_dict, batch_data['ego']['single_object_label'], suffix="_single") # 这个是单车做损失
                    if i % 10 == 0:
                        criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")
            pbar2.update(1)
            # back-propagation
            final_loss.backward()
            optimizer.step()
            # if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            #     scheduler.step_update(epoch * num_steps + i)
        sample_interval /= i
        sample_interval_all_epoch += sample_interval

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            end_time = time.time()
            print('### %d th epoch trained, start validation! Time consumed %.2f ###' % (epoch, (end_time - start_time)/60))
            with torch.no_grad():
                for i, batch_data in tenumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    # TODO: dataset parameter is only used for training flow module
                    if opt.two_stage:
                        ouput_dict = model(batch_data['ego'], opencood_validate_dataset)
                    else:
                        ouput_dict = model(batch_data['ego'])

                    if kd_flag:
                        teacher_output_dict = teacher_model(batch_data['ego'])
                        ouput_dict.update(teacher_output_dict)

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            with open(log_val_dir, 'a') as file:
                print('At epoch %d, the validation loss is %f' % (epoch, valid_ave_loss), file=file)
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                save_checkpoint_prefix = 'net_epoch_bestval_at'
                if finetune_flag:
                    save_checkpoint_prefix = f'finetune_{finetune_time_interval}_' + save_checkpoint_prefix
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    f'{save_checkpoint_prefix}%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    f'{save_checkpoint_prefix}%d.pth' % (lowest_val_epoch))):
                        os.remove(os.path.join(saved_path,
                                    f'{save_checkpoint_prefix}%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        # scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealinglr':
            if epoch < hypes['lr_scheduler']['T_max']:
                print("===update lr===")
                scheduler.step() # warning问题 最好不要传递epoch了
            else:
                print("***learn rate fixed***")
        else:
            scheduler.step() # warning问题 最好不要传递epoch了

    end_time = time.time()
    print("Time consumed: %.1f" % ((end_time - start_time)/60))
    sample_interval_all_epoch /= max(epoches, init_epoch) - init_epoch
    print("Avg sample interval of all epochs: %.2f" % sample_interval_all_epoch) 
    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
