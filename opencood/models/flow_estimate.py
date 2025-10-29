from numpy import record
import torch.nn as nn

from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world
from opencood.tools.matcher import Matcher
from collections import OrderedDict
import torch
import numpy as np

import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable
from opencood.utils import box_utils

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab): # 如果是对时间戳进行编码，那么输入应该是
        super(Embeddings, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(vocab,d_model),nn.ReLU(),nn.Linear(d_model,d_model))

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding_irregular(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000 ):
        super(PositionalEncoding_irregular, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model # 64维
        self.linear = nn.Linear(2*d_model,d_model)
         
    def forward(self, x, time): # 输入的序列以及query都会用这里将时间编码进入
        ### x:[batch, agent, k, d] time:[batch, agent, k] 时间就是距离
        if len(x.shape) > 3:
            last_shape = x.shape
            x = x.reshape(-1,x.shape[-2],x.shape[-1]) # （sum(cav), k, d）
            time = time.reshape(-1,time.shape[-1]) # (sum(cav), k)

        pe = torch.zeros(x.shape[0],time.shape[1], self.d_model).cuda() # （sum(cav), k, 64）
        position = time.unsqueeze(2) # 增加维度 (sum(cav), k, 1)
        div_term = torch.exp(torch.arange(0., self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).cuda()#相对位置公式
         
        pe[:,:, 0::2] = torch.sin(position * div_term)   #取奇数列
        pe[:,:, 1::2] = torch.cos(position * div_term)   #取偶数列

        # x = x + Variable(pe[:, :x.size(1)], requires_grad=False) # embedding 与positional相加
        
        x = self.linear(torch.cat((x,Variable(pe[:, :x.size(1)], requires_grad=False)),dim=-1))
        x = x.reshape(last_shape)
        return self.dropout(x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # N=2 将layer层重复两次
        self.N = N

        self.norm = LayerNorm(layer.size) #归一化

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for i in range(self.N):
            # print('x',x.shape)
            x = self.layers[i](x, mask) 

        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout): # size=64
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn #sublayer 1  注意力
        self.feed_forward = feed_forward #sublayer 2  维度恒等FFN
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #拷贝两个SublayerConnection，一个为了attention，一个是独自作为简单的神经网络
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # print('x',x.shape,'attn',self.self_attn(x,x,x,mask).shape)
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) #attention层 对应层1
        return self.sublayer[1](x, self.feed_forward) # 对应 层2
        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
 
    def forward(self, x):
        "Norm"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, sublayer):
        # print('sub',x.shape,self.dropout(sublayer(self.norm(x))).shape)
        return x + self.dropout(sublayer(self.norm(x))) #残差连接

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model,d_out, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size # 64
        self.src_attn = src_attn #解码的attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, src_key,memory, tgt_mask):
        m = memory
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) #self-attention
        x = self.sublayer[0](x, lambda x: self.src_attn(x, src_key, m, tgt_mask)) #解码  三个参数作为了qkv
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N): # layer为DecoderLayer对象
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # print('layersize',layer.size)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, src_key ,memory, tgt_mask): # x 为查询query  src_key和memory都是编码器输出的特征
        for layer in self.layers:
            x = layer(x, src_key, memory, tgt_mask) #添加编码的后的结果
        return self.norm(x)

def attention(query, key, value, mask=None, dropout=None): # 输入shape （batch， agent， h， k， 64//h）
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) # 
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    
    # print('score',scores.shape,'mask',mask.shape)
    if mask is not None:
        mask = mask.transpose(1,2).cuda()
        scores = scores.masked_fill(mask == 0, -1e9) #mask必须是一个ByteTensor 而且shape必须和 a一样 并且元素只能是 0或者1 ，是将 mask中为1的 元素所在的索引，在a中相同的的索引处替换为 value  ,mask value必须同为tensor
    
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, q_dim, k_dim,v_dim, d_model, dropout=0.1):# 注意力头个数为2 ， 后面的全部是64
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # d_v=d_k=d_model/h 
        self.h = h # heads 的数目文中为8
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linear_q = nn.Linear(q_dim,d_model)
        self.linear_k = nn.Linear(k_dim,d_model)
        self.linear_v = nn.Linear(v_dim,d_model)
        self.linear_out = nn.Linear(d_model,q_dim)
        self.attn = None  
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):# 输入的都是[batch, agent, k, d]
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        
        dim_to_keep = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k 


        query = self.linear_q(query).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3) # （batch， agent， k， 2， 64//2）
        key = self.linear_k(key).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)
        value = self.linear_v(value).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,# 进行attention softmax（q*k/d_k）* v
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(2, 3).contiguous() \
            .view(nbatches, dim_to_keep ,-1, self.h * self.d_k) # 还原序列[batch_size,len,d_model]

        return self.linear_out(x)

class Generator(nn.Module):
    def __init__(self, d_model, out_dim):
        
        super(Generator, self).__init__()
        
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        
        return self.proj(x)


class Motion_prediction(nn.Module):
    def __init__(self, encoder, decoder, embedding_src, embedding_tgt, position_src, position_tgt, generator, embed_dim):
        super(Motion_prediction, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = embedding_src
        self.tgt_embed = embedding_tgt # Embeddings(d_model, output_dim), # 输入是64 输出是2
        self.src_pe = position_src
        self.tgt_pe = position_tgt
        self.generator = generator

    def replicate_row_after(self, input_tensor, row_index, repeat_num):
        # 获取第i行的数据
        row_to_replicate = input_tensor[row_index:row_index+1, :].repeat(repeat_num, 1)
        
        # 拼接原始张量和复制的行
        output_tensor = torch.cat((input_tensor[:row_index+1, :], row_to_replicate, input_tensor[row_index+1:, :]), dim=0)
        
        return output_tensor       

    def extract_date_backup(self, data_dict):
        past_k_object_bbx = data_dict['past_k_object_bbx'] # 一个batch中的所有object的三帧变化，（M_b， k, 7） 注意，这些object全部已经投影到past0 的cav view 其中M_b表示batch中所有object个数
        past_k_time_interval = data_dict['past_k_time_interval'] # （N_b_l_k）（batch下所有帧数）：batch下所有帧到其cur的时间
        past_k_object_cav_num = data_dict['past_k_object_cav_num'] # 一维张量 形如（M1，M2....）其中存储每个cav的object数量 用于后续恢复形状对应到相应cav 长度为所有cav个数
        

        past_k_time_diff = past_k_time_interval.view(-1, 3) # 变为（N_b_l, k）即每一个cav的k帧 N_b_l表示所有的cav个数
        past_k_object_bbx = past_k_object_bbx[:,:,:2] # （M_b， k, 2） 只要 x, y 这里是bbx的中心点的x/y
        past_k_object_bbx = torch.flip(past_k_object_bbx, dims=[1]) # 倒置，本来是past0-past2，现在翻转为past2-past0 （M_b， k, 2）

        obj_coords_norm = past_k_object_bbx - past_k_object_bbx[:, -1:, :] # （M_b， k, 2）所有帧减去past0的x,y 其中(M_b, -1, 2)显然就是全0了

        past_k_time_diff = torch.flip(past_k_time_diff, dims=[1]) # (N_b_l， k) TODO: check if this is correct  顺序翻转，原来顺序依次为past0到cur时间间隔，past1到cur时间间隔，past2到cur时间间隔
        # past_k_time_diff_norm = past_k_time_diff - past_k_time_diff[:,-1] # (N_b_l， k)

        # print("_______________________")
        # print("past_k_time_diff shape is:  ", past_k_time_diff.shape)
        # print("past_k_object_bbx shape is:  ", past_k_object_bbx.shape)
        # print("_______________________")

        # N_b_l由于是batch下所有的帧数，而M_b是batch所有的object数，二者不对等，需要对时间间隔变量做扩展 转变成每个object的k帧延迟信息
        for i in range(past_k_time_diff.shape[0]): # 循环所有cav的个数
            if past_k_object_cav_num[i] < 1:
                print("error! a cav have less than 1 object!")
                return None
            past_k_time_diff = self.replicate_row_after(past_k_time_diff,i ,past_k_object_cav_num[i]-1) # 将每一行复制，由于每一行是一个cav下的时间间隔，对于其中的object是一样的，所以复制object 的数量-1
        if past_k_time_diff.shape[0] != obj_coords_norm.shape[0]: # 复制完后应该两者的形状一致 都是（M_b, k）
            print("duplicate error!")
            return None
        
        # print("++++++++++++++++++++++++")
        # print("past_k_time_diff shape is:  ", past_k_time_diff.shape)
        # print("past_k_object_bbx shape is:  ", past_k_object_bbx.shape)
        # print("++++++++++++++++++++++++")

        past_k_time_diff_norm = past_k_time_diff - past_k_time_diff[:,-1:] # (M_b， k)

        speed = torch.zeros_like(obj_coords_norm) # (N, k, 2)  下面的方法除法就是求速度，只求了第三帧到第二帧，第二帧到第一帧的速度 TODO 但是这个speed的放的顺序是不是有问题，需要check
        # speed[:, 1:, :] = torch.div((obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]), (past_k_time_diff[:,1:] - past_k_time_diff[:,:-1]).unsqueeze(-1)) # (M_b, k-1, 2) / (M_b, k-1, 1)
        speed[:, :-1, :] = torch.div((obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]), (past_k_time_diff[:,1:] - past_k_time_diff[:,:-1]).unsqueeze(-1)) # 放置顺序似乎原本有问题，原本是(0, past3到past2速度，past2到past1速度)
        obj_input = torch.cat([obj_coords_norm, speed], dim=-1) # (M_b, k, 4) 输入原来就是xy加上xy上的速度 TODO 这里似乎可以优化，因为速度和对应的距离并不对齐
        obj_input = obj_input.unsqueeze(0) # (1, M_b, k, 4)

        last_time_length = (past_k_time_diff_norm[:,-1] - past_k_time_diff_norm[:,-2]) # t1-t2 t1是past0到cur的时间间隔 t2是past1到cur的时间间隔 两者相减得到past0与past1的时间间隔 (M_b, )
        # print("last_time_length shape is: ",last_time_length.shape)

        query_list = []
        for i in range(last_time_length.shape[0]): # 遍历每一个object
            if last_time_length[i] == 0: # 也就是说过去两帧是重复的
                print("==== Warning! You met repeated package! ====")
                query_list.append(torch.zeros(obj_input.shape)[:,-1:,:1,:2].to(obj_input.device)) # (1, 1, 1, 2) 如果两帧实际的时间间隔相同 那就当做两帧他都没有移动，预测的偏移量应该是0
            else: # 只有算这个object其
                temp = obj_coords_norm[i:i+1, -1:, :] + \
                    (obj_coords_norm[i:i+1, -1:, :]-obj_coords_norm[i:i+1, -2:-1, :])*(0-past_k_time_diff[i:i+1, -1:]) / \
                        last_time_length[i] # past2到past1的偏移/时间差 * past0-past
                temp = temp.unsqueeze(0) # (1, 1, 1, 2) 查询使用的是past0到past1的偏移量除以两者的时间差，得到速度，再乘以T1，得到past0到cur可能的偏移量
                query_list.append(temp)
        query = torch.stack(query_list, dim=1).reshape(1, last_time_length.shape[0], 1, 2)
        # print("query shape is:  ", query.shape)
        # if last_time_length == 0:
        #     print("==== Warning! You met repeated package! ====")
        #     query = torch.zeros(obj_input.shape)[:,:,:1,:2].to(obj_input.device) # (1, M_b, 1, 2) 就是只要第一个元素也就是第三帧，且只要x,y 
        # else: 
        #     query = obj_coords_norm[:, -1:, :] + \
        #         (obj_coords_norm[:, -1:, :]-obj_coords_norm[:, -2:-1, :])*(0-past_k_time_diff[-1]) / \
        #             last_time_length
        #     query = query.unsqueeze(0) # (1, N, 1, 2) 查询使用的是past0到past1的偏移量除以两者的时间差，得到速度，再乘以T1，得到past0到cur可能的偏移量

        # target_time_diff = torch.tensor([-past_k_time_diff[:-1]]).to(obj_input.device) # (1,) # 这个就是时延T1 表示past0到cur的时间距离
        target_time_diff = -past_k_time_diff[:,-1].to(obj_input.device) # (1,) # 这个就是时延T1 表示past0到cur的时间距离
        # print("target_time_diff shape is : ", target_time_diff.shape)
        return obj_input.to(dtype=torch.double), query.to(dtype=torch.double), past_k_time_diff_norm.to(dtype=torch.double), target_time_diff.to(dtype=torch.double)

    def extract_date_backup1(self, data_dict):
        past_k_object_bbx = data_dict['past_k_object_bbx'] # 一个batch中的所有object的三帧变化，（M_b， k, 7） 注意，这些object全部已经投影到past0 的cav view 其中M_b表示batch中所有object个数
        past_k_time_interval = data_dict['past_k_time_interval'] # （N_b_l_k）（batch下所有帧数，每一个cav有三帧，所以总数=sum(cav)*3）：batch下所有帧到其cur的时间
        past_k_object_cav_num = data_dict['past_k_object_cav_num'] # 一维张量 形如（M1，M2....）其中存储每个cav的object数量 用于后续恢复形状对应到相应cav 长度为所有cav个数
        
        # print(past_k_object_bbx.shape) # torch.Size([117, 3, 7])
        # print(past_k_time_interval) # tensor([ -8., -12., -20.,  -6., -12., -16.,  -4., -12., -18., -12., -20., -30., -8., -14., -24.] 表示每一帧距离cur的时间间隔
        # print(past_k_object_cav_num) # tensor([ 7, 14, 27, 28, 41] 存储每个cav的object数量
        # exit9

        past_k_time_diff = past_k_time_interval.view(-1, 3) # 变为（sum(cav), k）即每一个cav的k帧 sum(cav) 表示所有的cav个数
        past_k_object_bbx = past_k_object_bbx[:,:,:2] # （M_b， k, 2） 只要 x, y 这里是bbx的中心点的x/y
        past_k_object_bbx = torch.flip(past_k_object_bbx, dims=[1]) # 倒置，本来是past0-past2，现在翻转为past2-past0 （M_b， k, 2）

        obj_coords_norm = past_k_object_bbx - past_k_object_bbx[:, -1:, :] # （M_b， k, 2）所有帧减去past0的x,y 其中(M_b, -1, 2)显然就是全0了

        past_k_time_diff = torch.flip(past_k_time_diff, dims=[1]) # (sum(cav)， k) TODO: check if this is correct  顺序翻转，原来顺序依次为past0到cur时间间隔，past1到cur时间间隔，past2到cur时间间隔
        # past_k_time_diff_norm = past_k_time_diff - past_k_time_diff[:,-1] # (N_b_l， k)

        # print("_______________________")
        # print("past_k_time_diff shape is:  ", past_k_time_diff.shape)
        # print("past_k_object_bbx shape is:  ", past_k_object_bbx.shape)
        # print("_______________________")

        # N_b_l由于是batch下所有的帧数，而M_b是batch所有的object数，二者不对等，需要对时间间隔变量做扩展 转变成每个object的k帧延迟信息
        j = 0
        for i in range(past_k_time_diff.shape[0]): # 循环所有cav的个数
            if past_k_object_cav_num[i] < 1: # 不足一辆cav
                print("error! a cav have less than 1 object!")
                return None
            past_k_time_diff = self.replicate_row_after(past_k_time_diff,j ,past_k_object_cav_num[i]-1) # 将每一行复制，由于每一行是一个cav下的时间间隔，对于其中的object是一样的，所以复制object 的数量-1
            j += past_k_object_cav_num[i]
        if past_k_time_diff.shape[0] != obj_coords_norm.shape[0]: # 复制完后应该两者的第一维度大小应该相同 （M_b, k） 表示每个bbx的k帧延迟情况
            print("duplicate error!")
            return None
        
        # print("++++++++++++++++++++++++")
        # print("past_k_time_diff shape is:  ", past_k_time_diff.shape)
        # print("past_k_object_bbx shape is:  ", past_k_object_bbx.shape)
        # print("++++++++++++++++++++++++")

        past_k_time_diff_norm = past_k_time_diff - past_k_time_diff[:,-1:] # (M_b， k) k维度 [T3-T1, T2-T1, 0] 注意 这里T3<T2<T1<=0

        speed = torch.zeros_like(obj_coords_norm) # (M_b, k, 2)  下面的方法除法就是求速度，只求了第三帧到第二帧，第二帧到第一帧的速度 TODO 但是这个speed的放的顺序是不是有问题，需要check
        # speed[:, 1:, :] = torch.div((obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]), (past_k_time_diff[:,1:] - past_k_time_diff[:,:-1]).unsqueeze(-1)) # (M_b, k-1, 2) / (M_b, k-1, 1)
        speed[:, :-1, :] = torch.div((obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]), (past_k_time_diff[:,1:] - past_k_time_diff[:,:-1]).unsqueeze(-1)) # 放置顺序似乎原本有问题，原本是(0, past3到past2速度，past2到past1速度)
        
        # 原本code中提到的特征工程的不太合理，四维分别是（pastk到past0的x偏移，pastk到past0的y偏移， pastk到pastk-1的x偏移速度，pastk到past-1的y偏移速度）        （k, 4）中显然(-1,4)全零
        distance = obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]
        speed = speed[:, :-1, :]
        # 修改后的特征编码为 四维分别是（pastk到pastk-1的x偏移，pastk到pastk-1的y偏移， pastk到pastk-1的x偏移速度，pastk到pastk-1的y偏移速度）   而且由于三帧中没有past0到cur的内容，所以这部分应该被去掉，变为(k-1, 4)
        
        # obj_input = torch.cat([obj_coords_norm, speed], dim=-1) # (M_b, k, 4) 输入原来就是xy加上xy上的速度 TODO 这里似乎可以优化，因为速度和对应的距离并不对齐
        obj_input = torch.cat([distance, speed], dim=-1) # (M_b, k-1, 4) 输入原来就是xy加上xy上的速度 TODO 这里似乎可以优化，因为速度和对应的距离并不对齐
        obj_input = obj_input.unsqueeze(0) # (1, M_b, k, 4)

        last_time_length = (past_k_time_diff_norm[:,-1] - past_k_time_diff_norm[:,-2]) # t1-t2 t1是past0到cur的时间间隔 t2是past1到cur的时间间隔 两者相减得到past0与past1的时间间隔 (M_b, )
        # print("last_time_length shape is: ",last_time_length.shape)

        query_list = []
        for i in range(last_time_length.shape[0]): # 遍历每一个object
            if last_time_length[i] == 0: # 也就是说过去两帧是重复的
                print("==== Warning! You met repeated package! ====")
                query_list.append(torch.zeros(obj_input.shape)[:,-1:,:1,:2].to(obj_input.device)) # (1, 1, 1, 2) 如果两帧实际的时间间隔相同 那就当做两帧他都没有移动，预测的偏移量应该是0
            else: # 只有算这个object其
                temp = obj_coords_norm[i:i+1, -1:, :] + \
                    (obj_coords_norm[i:i+1, -1:, :]-obj_coords_norm[i:i+1, -2:-1, :])*(0-past_k_time_diff[i:i+1, -1:]) / \
                        last_time_length[i] # past2到past1的偏移/时间差 * past0-past
                temp = temp.unsqueeze(0) # (1, 1, 1, 2) 查询使用的是past0到past1的偏移量除以两者的时间差，得到速度，再乘以T1，得到past0到cur可能的偏移量
                query_list.append(temp)
        query = torch.stack(query_list, dim=1).reshape(1, last_time_length.shape[0], 1, 2) # (1, M_b, 1, 2)

        # print("query shape is:  ", query.shape)
        # if last_time_length == 0:
        #     print("==== Warning! You met repeated package! ====")
        #     query = torch.zeros(obj_input.shape)[:,:,:1,:2].to(obj_input.device) # (1, M_b, 1, 2) 就是只要第一个元素也就是第三帧，且只要x,y 
        # else: 
        #     query = obj_coords_norm[:, -1:, :] + \
        #         (obj_coords_norm[:, -1:, :]-obj_coords_norm[:, -2:-1, :])*(0-past_k_time_diff[-1]) / \
        #             last_time_length
        #     query = query.unsqueeze(0) # (1, N, 1, 2) 查询使用的是past0到past1的偏移量除以两者的时间差，得到速度，再乘以T1，得到past0到cur可能的偏移量

        # target_time_diff = torch.tensor([-past_k_time_diff[:-1]]).to(obj_input.device) # (1,) # 这个就是时延T1 表示past0到cur的时间距离
        target_time_diff = -past_k_time_diff[:,-1].to(obj_input.device) # (M_b,) # 这个就是时延T1 表示past0到cur的时间距离
        # print("target_time_diff shape is : ", target_time_diff.shape)
        time_encode = past_k_time_diff[:,1:] - past_k_time_diff[:,:-1] # (M_b, 2) [T2-T3, T2-T1]
        torch.set_printoptions(precision=16)
        print("每一帧距离cur的时间间隔", past_k_time_diff[0])
        print("每一帧的object坐标", past_k_object_bbx[0])
        print('上两帧度量的平均速度乘以延迟时间', query[0,0,0,:])
        print('延迟时间', target_time_diff[0])
        print('前三帧两两延迟时间', time_encode[0])
        cur_bbx =  data_dict['cur_object_bbx_debug'][:, :2] # (M_b, 2)
        print('当cur_bbx的形状  ', cur_bbx.shape)

        print('当前的实际运动GT位置  ', cur_bbx[0])
        print('速度  ', speed[0])
        print('距离 ',distance[0])


        # return obj_input.to(dtype=torch.double), query.to(dtype=torch.double), past_k_time_diff_norm.to(dtype=torch.double), target_time_diff.to(dtype=torch.double)
        return obj_input.to(dtype=torch.double), query.to(dtype=torch.double), time_encode.to(dtype=torch.double), target_time_diff.to(dtype=torch.double)

    def extract_date(self, data_dict):
        past_k_object_bbx = data_dict['past_k_object_bbx'] # 一个batch中的所有object的三帧变化，（M_b， k, 7） 注意，这些object全部已经投影到past0 的cav view 其中M_b表示batch中所有object个数
        past_k_time_interval = data_dict['past_k_time_interval'] # （N_b_l_k）（batch下所有帧数，每一个cav有三帧，所以总数=sum(cav)*3）：batch下所有帧到其cur的时间
        past_k_object_cav_num = data_dict['past_k_object_cav_num'] # 一维张量 形如（M1，M2....）其中存储每个cav的object数量 用于后续恢复形状对应到相应cav 长度为所有cav个数

        # -- paper中的方法 2024年03月20日          
        past_k_object_bbx = past_k_object_bbx[:,:,[0,1,6]] # （M_b， k, 3） 只要 x, y ,yaw 这里是bbx的中心点的x/y
        obj_input = torch.flip(past_k_object_bbx, dims=[1]) # 倒置，本来是past0-past2，现在翻转为past2-past0 （M_b， k, 3）
        obj_input = obj_input.unsqueeze(0) # (1, M_b, k, 3)

        past_k_time_diff = past_k_time_interval.view(-1, 3) # 变为（sum(cav), k）即每一个cav的k帧 sum(cav) 表示所有的cav个数        
        past_k_time_diff = torch.flip(past_k_time_diff, dims=[1]) # (sum(cav)， k) TODO: check if this is correct  顺序翻转，原来顺序依次为past0到cur时间间隔，past1到cur时间间隔，past2到cur时间间隔
        # N_b_l由于是batch下所有的帧数，而M_b是batch所有的object数，二者不对等，需要对时间间隔变量做扩展 转变成每个object的k帧延迟信息
        j = 0
        for i in range(past_k_time_diff.shape[0]): # 循环所有cav的个数
            if past_k_object_cav_num[i] < 1: # 不足一辆cav
                print("error! a cav have less than 1 object!")
                return None
            past_k_time_diff = self.replicate_row_after(past_k_time_diff,j ,past_k_object_cav_num[i]-1) # 将每一行复制，由于每一行是一个cav下的时间间隔，对于其中的object是一样的，所以复制object 的数量-1
            j += past_k_object_cav_num[i]
        if past_k_time_diff.shape[0] != obj_input.shape[1]: # 复制完后应该两者的第一维度大小应该相同 （M_b, k） 表示每个bbx的k帧延迟情况
            print("duplicate error!")
            return None
        query = torch.zeros((1, past_k_time_diff.shape[0], 1, 3)).to(obj_input.device)
        future_time = torch.zeros((1, past_k_time_diff.shape[0], 1)).to(obj_input.device)

        
        return obj_input, query.to(dtype=torch.double), past_k_time_diff, future_time.to(dtype=torch.double)

        # -- end

        past_k_time_diff = past_k_time_interval.view(-1, 3) # 变为（sum(cav), k）即每一个cav的k帧 sum(cav) 表示所有的cav个数
        past_k_object_bbx = past_k_object_bbx[:,:,:2] # （M_b， k, 2） 只要 x, y 这里是bbx的中心点的x/y
        past_k_object_bbx = torch.flip(past_k_object_bbx, dims=[1]) # 倒置，本来是past0-past2，现在翻转为past2-past0 （M_b， k, 2）

        obj_coords_norm = past_k_object_bbx - past_k_object_bbx[:, -1:, :] # （M_b， k, 2）所有帧减去past0的x,y 其中(M_b, -1, 2)显然就是全0了

        past_k_time_diff = torch.flip(past_k_time_diff, dims=[1]) # (sum(cav)， k) TODO: check if this is correct  顺序翻转，原来顺序依次为past0到cur时间间隔，past1到cur时间间隔，past2到cur时间间隔
        # past_k_time_diff_norm = past_k_time_diff - past_k_time_diff[:,-1] # (N_b_l， k)

        # N_b_l由于是batch下所有的帧数，而M_b是batch所有的object数，二者不对等，需要对时间间隔变量做扩展 转变成每个object的k帧延迟信息
        j = 0
        for i in range(past_k_time_diff.shape[0]): # 循环所有cav的个数
            if past_k_object_cav_num[i] < 1: # 不足一辆cav
                print("error! a cav have less than 1 object!")
                return None
            past_k_time_diff = self.replicate_row_after(past_k_time_diff,j ,past_k_object_cav_num[i]-1) # 将每一行复制，由于每一行是一个cav下的时间间隔，对于其中的object是一样的，所以复制object 的数量-1
            j += past_k_object_cav_num[i]
        if past_k_time_diff.shape[0] != obj_coords_norm.shape[0]: # 复制完后应该两者的第一维度大小应该相同 （M_b, k） 表示每个bbx的k帧延迟情况
            print("duplicate error!")
            return None
        

        past_k_time_diff_norm = past_k_time_diff - past_k_time_diff[:,-1:] # (M_b， k) k维度 [T3-T1, T2-T1, 0] 注意 这里T3<T2<T1<=0

        speed = torch.zeros_like(obj_coords_norm) # (M_b, k, 2)  下面的方法除法就是求速度，只求了第三帧到第二帧，第二帧到第一帧的速度 TODO 但是这个speed的放的顺序是不是有问题，需要check
        # speed[:, 1:, :] = torch.div((obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]), (past_k_time_diff[:,1:] - past_k_time_diff[:,:-1]).unsqueeze(-1)) # (M_b, k-1, 2) / (M_b, k-1, 1)
        speed[:, :-1, :] = torch.div((obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]), (past_k_time_diff[:,1:] - past_k_time_diff[:,:-1]).unsqueeze(-1)) # 放置顺序似乎原本有问题，原本是(0, past3到past2速度，past2到past1速度)
        
        # 原本code中提到的特征工程的不太合理，四维分别是（pastk到past0的x偏移，pastk到past0的y偏移， pastk到pastk-1的x偏移速度，pastk到past-1的y偏移速度）        （k, 4）中显然(-1,4)全零
        distance = obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]
        speed = speed[:, :-1, :]
        # 修改后的特征编码为 四维分别是（pastk到pastk-1的x偏移，pastk到pastk-1的y偏移， pastk到pastk-1的x偏移速度，pastk到pastk-1的y偏移速度）   而且由于三帧中没有past0到cur的内容，所以这部分应该被去掉，变为(k-1, 4)
        
        # obj_input = torch.cat([obj_coords_norm, speed], dim=-1) # (M_b, k, 4) 输入原来就是xy加上xy上的速度 TODO 这里似乎可以优化，因为速度和对应的距离并不对齐
        obj_input = torch.cat([distance, speed], dim=-1) # (M_b, k-1, 4) 输入原来就是xy加上xy上的速度 TODO 这里似乎可以优化，因为速度和对应的距离并不对齐
        obj_input = obj_input.unsqueeze(0) # (1, M_b, k, 4)

        last_time_length = (past_k_time_diff_norm[:,-1] - past_k_time_diff_norm[:,-2]) # t1-t2 t1是past0到cur的时间间隔 t2是past1到cur的时间间隔 两者相减得到past0与past1的时间间隔 (M_b, )
        # print("last_time_length shape is: ",last_time_length.shape)

        query_list = []
        for i in range(last_time_length.shape[0]): # 遍历每一个object
            if last_time_length[i] == 0: # 也就是说过去两帧是重复的
                print("==== Warning! You met repeated package! ====")
                query_list.append(torch.zeros(obj_input.shape)[:,-1:,:1,:2].to(obj_input.device)) # (1, 1, 1, 2) 如果两帧实际的时间间隔相同 那就当做两帧他都没有移动，预测的偏移量应该是0
            else: # 只有算这个object其
                temp = obj_coords_norm[i:i+1, -1:, :] + \
                    (obj_coords_norm[i:i+1, -1:, :]-obj_coords_norm[i:i+1, -2:-1, :])*(0-past_k_time_diff[i:i+1, -1:]) / \
                        last_time_length[i] # past2到past1的偏移/时间差 * past0-past
                temp = temp.unsqueeze(0) # (1, 1, 1, 2) 查询使用的是past0到past1的偏移量除以两者的时间差，得到速度，再乘以T1，得到past0到cur可能的偏移量
                query_list.append(temp)
        query = torch.stack(query_list, dim=1).reshape(1, last_time_length.shape[0], 1, 2) # (1, M_b, 1, 2)

        # print("query shape is:  ", query.shape)
        # if last_time_length == 0:
        #     print("==== Warning! You met repeated package! ====")
        #     query = torch.zeros(obj_input.shape)[:,:,:1,:2].to(obj_input.device) # (1, M_b, 1, 2) 就是只要第一个元素也就是第三帧，且只要x,y 
        # else: 
        #     query = obj_coords_norm[:, -1:, :] + \
        #         (obj_coords_norm[:, -1:, :]-obj_coords_norm[:, -2:-1, :])*(0-past_k_time_diff[-1]) / \
        #             last_time_length
        #     query = query.unsqueeze(0) # (1, N, 1, 2) 查询使用的是past0到past1的偏移量除以两者的时间差，得到速度，再乘以T1，得到past0到cur可能的偏移量

        # target_time_diff = torch.tensor([-past_k_time_diff[:-1]]).to(obj_input.device) # (1,) # 这个就是时延T1 表示past0到cur的时间距离
        target_time_diff = -past_k_time_diff[:,-1].to(obj_input.device) # (M_b,) # 这个就是时延T1 表示past0到cur的时间距离
        # print("target_time_diff shape is : ", target_time_diff.shape)
        time_encode = past_k_time_diff[:,1:] - past_k_time_diff[:,:-1] # (M_b, 2) [T2-T3, T2-T1]
        torch.set_printoptions(precision=16)
        print("每一帧距离cur的时间间隔", past_k_time_diff[0])
        print("每一帧的object坐标", past_k_object_bbx[0])
        print('上两帧度量的平均速度乘以延迟时间', query[0,0,0,:])
        print('延迟时间', target_time_diff[0])
        print('前三帧两两延迟时间', time_encode[0])
        cur_bbx =  data_dict['cur_object_bbx_debug'][:, :2] # (M_b, 2)
        print('当cur_bbx的形状  ', cur_bbx.shape)

        print('当前的实际运动GT位置  ', cur_bbx[0])
        print('速度  ', speed[0])
        print('距离 ',distance[0])


        # return obj_input.to(dtype=torch.double), query.to(dtype=torch.double), past_k_time_diff_norm.to(dtype=torch.double), target_time_diff.to(dtype=torch.double)
        return obj_input.to(dtype=torch.double), query.to(dtype=torch.double), time_encode.to(dtype=torch.double), target_time_diff.to(dtype=torch.double)


    def regroup(self, x, record_len, k=1):
        '''
        x的形状为(B, C, H, W)
        record_len的形状为(B) 记录每一个样本场景下的agent个数
        k为保存的帧数

        '''
        cum_sum_len = torch.cumsum(record_len*k, dim=0) # 求累计和
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu()) # 分割批数据，返回的是List

        return split_x # List[p1()]

    def generate_flow_map(self, flow, bbox_list, scale=1.25, shape_list=None, align_corners=False, file_suffix=""):
        """
        Parameters
        -----------
        bbox_list: [num_cav, 4, 3] at cav coodinate system and lidar scale 最近的一帧的object 这些已经是做好match的部分 num_cav以下被我写作M
        flow:[num_cav, 2] at cav coodinate system and lidar scale
            bbox_list & flow : x and y are exactly image coordinate
            ------------> x
            |
            |
            |
            y
        scale: float, scale meters to voxel, feature_length / lidar_range_length = 1.25 or 2.5

        Returns
        -------
        updated_feature: feature after being warped by flow, [C, H, W]
        """
        # flow = torch.tensor([70, 0]).unsqueeze(0).to(feature)

        # only use x and y
        bbox_list = bbox_list[:, :, :2] # （M, 4, 2）

        # scale meters to voxel, feature_length / lidar_range_length = 1.25
        flow = flow * scale # （M, 2）
        bbox_list = bbox_list * scale # （M, 4, 2）

        flag_viz = False
        #######
        # store two parts of bbx: 1. original bbx, 2. 
        if flag_viz:
            viz_bbx_list = bbox_list
            fig, ax = plt.subplots(4, 1, figsize=(5,11))
            ######## viz-0: original feature, original bbx
            canvas_ori = viz_on_canvas(feature, bbox_list, scale=scale)
            plt.sca(ax[0])
            # plt.axis("off")
            plt.imshow(canvas_ori.canvas)
            ##########
        #######

        C, H, W = shape_list
        num_cav = bbox_list.shape[0] # （M， 4， 2） M个object
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(shape_list.device) # （1, H, W, 2）
        reserved_area = torch.zeros((C, H, W)).to(shape_list.device)  # C, H, W  全0张量
        if flow.shape[0] == 0 : 
            # reserved_area = torch.ones((C, H, W)).to(shape_list.device)  # C, H, W 
            return basic_warp_mat.squeeze(0),  reserved_area  # 返回不变的矩阵

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1    
        ------------
        '''
        flow_clone = flow.clone() # 拷贝  （M, 2） 之前使用flow.detach().clone()导致梯度传播中断

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1) # （M, 3, 3）设置恒等变换矩阵
        flow_clone = -2 * flow_clone / torch.tensor([W, H]).to(torch.float32).to(shape_list.device) # 类似于归一化，约束到[-1, 1]区间，这是因为 F.affine_grid 需要这样的范围的数值 （M, 2）
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone  # 
        
        cav_t_mat = affine_matrices[:, :2, :]   # M, 2, 3
        # print("cav_t_mat", cav_t_mat)

        cav_warp_mat = F.affine_grid(cav_t_mat, # 构建坐标网格 (M , H, W, 2)
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(shape_list.device) # .to() 统一数据格式 float32
        
        flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2  # 第0帧的object 的 x，y值 加上 flow 中运动信息 （M， 4 ， 2）
        ######### viz-1: original feature, original bbx and flowed bbx
        if flag_viz:
            viz_bbx_list = torch.cat((bbox_list, flowed_bbx_list), dim=0)
            canvas_hidden = viz_on_canvas(feature, viz_bbx_list, scale=scale)
            plt.sca(ax[1])
            # plt.axis("off") 
            plt.imshow(canvas_hidden.canvas)
        ##########

        x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1 # 最小x的值，缩小1应该是为了给边界框提供一点额外空间 (M)
        x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
        y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
        y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
        x_min_fid = (x_min + int(W/2)).to(torch.int) # 这是将边界框坐标从以图像中心为原点转换为以图像左上角为原点的坐标系 (M)
        x_max_fid = (x_max + int(W/2)).to(torch.int)
        y_min_fid = (y_min + int(H/2)).to(torch.int)
        y_max_fid = (y_max + int(H/2)).to(torch.int)

        for cav in range(num_cav): # 遍历每一辆agent 将 包含流矫正属性的坐标网格中对应部分赋值到基础变换矩阵
            basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] # 将计算得到的每个对象的仿射变换矩阵应用到一个基础变换矩阵上，用于更新特征图中相应区域的变换

        # generate mask
        x_min_ori = torch.min(bbox_list[:,:,0],dim=1)[0] # 第0帧object 的每个边界框的四个角确定
        x_max_ori = torch.max(bbox_list[:,:,0],dim=1)[0]
        y_min_ori = torch.min(bbox_list[:,:,1],dim=1)[0]
        y_max_ori = torch.max(bbox_list[:,:,1],dim=1)[0]
        x_min_fid_ori = (x_min_ori + int(W/2)).to(torch.int)
        x_max_fid_ori = (x_max_ori + int(W/2)).to(torch.int)
        y_min_fid_ori = (y_min_ori + int(H/2)).to(torch.int)
        y_max_fid_ori = (y_max_ori + int(H/2)).to(torch.int)
        # set original location as 0
        for cav in range(num_cav):
            reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 0 # object区域归零
        # set warped location as 1
        for cav in range(num_cav):
            reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1 # warped的object边界框区域置位1  （C，H， W）

        if self.viz_flag:
            single_reserved_area = torch.zeros_like(reserved_area)
            for cav in range(num_cav):
                single_reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 1
            return basic_warp_mat, reserved_area.unsqueeze(0), single_reserved_area.unsqueeze(0)

        return basic_warp_mat.squeeze(0), reserved_area # 返回形状（H， W， 2）， （C， H， W）

    def forward(self, data_dict, mask=None, viz_flag=None):
        '''
        obj_input: (1, N, k, 4) k帧中N个object的信息 最后一维度是 x, y 加其速度
        query: (1, N, 1, 2) 总体来看 这是将前两帧past0到past1的速度乘以past0到cur的时间 从而得到一个类似距离的东西
        past_k_time_diff_norm: (k,) 分别为(T3-T1, T2-T1, 0)
        target_time_diff:(1, ) T1 past0到cur的时间间隔

        input: torch.Size([1, 45, 3, 3])
        query: torch.Size([1, 45, 1, 3])
        time: torch.Size([45, 3])
        future_time: torch.Size([1, 45, 1])
        predictions: torch.Size([1, 45, 1, 3])
        '''
        self.viz_flag = viz_flag
        shape_list = torch.tensor((64, 200 ,704)).cuda() # 这个会影响最后生成的flow尺寸，需要和体素化后的H，W保持一致
        input, query, time, future_time = self.extract_date(data_dict)

        # print(input.shape)
        # print(query.shape)
        # print(time.shape)
        # print(future_time.shape)

        input = self.src_embed(input) # 编码 输入的应该是3维 （x, y，angle） 输出 64维
        input = self.src_pe(input, time) # 位置编码 将延迟编码进入  最后结果形状不变[batch, agent, k, 64]

        query = self.tgt_embed(query) # 查询应该用当前的时间戳 编码成64维度 [batch, agent, 1, 64]
        query = self.tgt_pe(query, future_time) # 位置编码

        input_features = self.encoder(input, mask)
        prediction_features = self.decoder(query, input_features, input_features,mask) # 做transformer 结果形状（batch, agent, 1, 64）
        predictions = self.generator(prediction_features) # 得到(1, N, 1, 3) 表示所有object的预测结果 包含了x,y值，角度值

        output_dict = {'preds_coop':predictions.squeeze(0).squeeze(1),
                       'gt_coop':data_dict['cur_object_bbx_debug'][:, [0,1,6]]}
        

        # print(output_dict['preds_coop'].shape)
        # print(output_dict['gt_coop'].shape) # 
        # exit9

        return output_dict

        # Motion prediction start
        ################################ input: [batch, agent, past_frames, dim]  mask: [batch, agent, past_frames, past_frames]
        input = self.src_embed(input) # 编码 输入的应该是3维 （x, y，angle） 输出 64维
        input = self.src_pe(input, time) # 位置编码 将延迟编码进入  最后结果形状不变[batch, agent, k, 64]

        query_copy = copy.deepcopy(query)
        query = self.tgt_embed(query) # 查询应该用当前的时间戳 编码成64维度 [batch, agent, 1, 64]
        query = self.tgt_pe(query, future_time) # 位置编码

        input_features = self.encoder(input, mask)
        
        prediction_features = self.decoder(query, input_features, input_features,mask) # 做transformer 结果形状（batch, agent, 1, 64）
        predictions = self.generator(prediction_features) + query_copy # 其实就是一个FFN，调节输出维度用 （1, N, 1, 2） 也就是所有object的预测偏移量

        flow = predictions.squeeze(0).squeeze(1) # (N 2)
        # print("predictions shape is : ", predictions.shape) 

        # end Motion prediction
        # 预测结果是（N, 2） N也就是所有object数量，现在要将其scatter到对应的 HxW画布上去，以方便后续计算flow map

        flow_list = self.regroup(flow, data_dict['past_k_object_cav_num']) # 第二个参数内容是每一个cav中含有的object数 返回一个List[cav1(N1, 2), (N2, 2)...]
        
        # print("data_dict['past_k_object_cav_num']的长度, 为所有cav的个数,为: ",len(data_dict['past_k_object_cav_num']))
        past0_object_bbx = data_dict['past_k_object_bbx'][:, 0,: ] # (M_b, 7) 取出所有object的past0帧的信息
        past0_object_bbx = box_utils.boxes_to_corners2d(past0_object_bbx, order='hwl') # （M_b， 4， 2） 转成2d的四角表示
        past0_obejct_bbx_list = self.regroup(past0_object_bbx, data_dict['past_k_object_cav_num']) # 返回一个list[cav1(N1, 4， 2)...]表示每一个cav的所有object分类好了
        # print("past0_obejct_bbx_list 的长度， 也是cav的个数", len(past0_obejct_bbx_list))

        flow_map_list = []
        flow_mask_list = []
        for cav_id, flow_cav in enumerate(flow_list): # 遍历所有cav的个数的次数 返回形状（H， W， 2）， （C， H， W）
            flow_map_cav, flow_mask_cav = self.generate_flow_map(flow_cav, past0_obejct_bbx_list[cav_id], scale=2.5, shape_list=shape_list) # 输入是（N， 2）N不固定，和 (N, 4, 2) 前者是预测偏移量，后者则是要施加偏移的object bev四角
            flow_map_list.append(flow_map_cav)
            flow_mask_list.append(flow_mask_cav)

        flow_map = torch.stack(flow_map_list, dim=0) # (num_cav, H, W, 2) 以cav为单位的flow
        flow_mask = torch.stack(flow_mask_list, dim=0)
        # print("flow_map 最终形成的流图：", flow_map.shape)
        # print("GT flow: ", data_dict['label_dict']['flow_gt'].shape)
        # print("flow 生成model 执行完毕一次！------------------------")
        output_dict = {'flow_preds':flow_map,
                        'flow_mask':flow_mask}
        return output_dict



class Motion_interaction(nn.Module):
    def __init__(self, encoder, embedding, generator, neighbor_threshold):
        super(Motion_interaction, self).__init__()
        self.encoder = encoder
        self.src_embed = embedding
        self.generator = generator
        self.neighbor_threshold = neighbor_threshold


    def forward(self, input):
        ####input:[batch,agent,2]
        mask = torch.cdist(input,input)

        mask = torch.where(mask<self.neighbor_threshold,1,0)
        input = input.unsqueeze(1)
        mask = mask.unsqueeze(1)

        input = self.src_embed(input)

        input_features = self.encoder(input, mask)
        predictions = self.generator(input_features)

        return predictions


def make_model(input_dim, output_dim, num_layers=2, d_model=64, d_ff=128, num_heads=2, dropout=0.1,neighbor_shreshold=10):
    c = copy.deepcopy # 深拷贝
    attn = MultiHeadedAttention(num_heads, d_model,d_model,d_model,d_model)
    attn_decoder = MultiHeadedAttention(num_heads,d_model,d_model,d_model,d_model)
    ff = PositionwiseFeedForward(d_model, d_model, d_ff,dropout)
    position = PositionalEncoding_irregular(d_model, dropout)

    model_prediction = Motion_prediction(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Embeddings(d_model, input_dim),
        Embeddings(d_model, output_dim), # 输入是64 输出是2
        c(position),
        c(position),
        Generator(d_model, output_dim),
        d_model)

    model_interaction = Motion_interaction(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Embeddings(d_model, input_dim), 
        Generator(d_model, output_dim),
        neighbor_shreshold)
    
    for p in model_prediction.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in model_interaction.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model_prediction, model_interaction

'''
运动预测模型 给定一个序列， 以时间编码作为query
输入：   input: (obejct_num, 4 , 3, 2)  (batch, agent, past_frames, dim)
        query: (obejct_num, 4 , 1, 2) 查询向量 paper中提到的为时间编码处理过的结果
        time: 时间间隔
        future_time:
        mask: (batch, agent, past_frames, past_frames)

'''


# m1, m2 = make_model(2,2)

# input = torch.ones((7,4,3,2)) # 猜测：输入的是7个obejct， 4个corner， 3帧， 2个坐标（x和y）
# prediction_query = torch.zeros((7,4,1,2)) # 猜测：预测的目标 7个obejct， 4个corner， 当前1帧， 2个坐标（x和y）
# time = torch.ones((7,4,3)) # 猜测： 时间 7个obejct， 4个corner， 3帧之间时间间隔
# future_time = torch.ones((7,4,1)) # 猜测：7个obejct， 4个corner， 距离当前的延迟
# x = m1(input,prediction_query,time,future_time)
# print(x.shape)
# x = x.squeeze(-2)
# print(x.shape)
# x = m2(x)
# print(x.shape)

if __name__ == "__main__":

    def replicate_row_after(input_tensor, row_index, repeat_num):
        # 获取第i行的数据
        row_to_replicate = input_tensor[row_index:row_index+1, :].repeat(repeat_num, 1)
        
        # 拼接原始张量和复制的行
        output_tensor = torch.cat((input_tensor[:row_index+1, :], row_to_replicate, input_tensor[row_index+1:, :]), dim=0)
        
        return output_tensor 

    past_k_time_diff = torch.range(1, 12).reshape(4,3)
    past_k_object_cav_num = [1, 2, 3, 4]

    print(past_k_time_diff.shape)
    print(past_k_time_diff)

    j = 0
    for i in range(past_k_time_diff.shape[0]): # 循环所有cav的个数
        if past_k_object_cav_num[i] < 1: # 不足一辆cav
            print("error! a cav have less than 1 object!")
        past_k_time_diff = replicate_row_after(past_k_time_diff,j ,past_k_object_cav_num[i]-1) # 将每一行复制，由于每一行是一个cav下的时间间隔，对于其中的object是一样的，所以复制object 的数量-1
        j += past_k_object_cav_num[i]

    print(past_k_time_diff.shape)
    print(past_k_time_diff)

