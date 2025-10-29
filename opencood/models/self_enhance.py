import torch
from torch import nn

class SELFEnhance(nn.Module):
    def __init__(self, dim, heads = 2) -> None:
        super().__init__()

        self.q_linear = nn.Linear(dim, dim)

    def to_qkv(self, x):
        return 0


    def forward(self, x):
        '''
        x: (BxNxk, C, H, W) 一个批次中所有cav的k帧
        step1: 添加时间编码，将其延迟作为编码加到特征图上
        step2: 计算出每个feature map的qkv (N, k, H, W, C') --> (N, H, W, head, k, C'/head)
        step3: 多头自注意力计算 得到结果 (N, H, W, head, k, C'/head) --> (N, H, W, k, C'/head)
        '''



        return x