import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable

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
        self.size = size
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
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # print('layersize',layer.size)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, src_key ,memory, tgt_mask): # x 为查询query 
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

        

    def forward(self, input, query, time, future_time, mask=None):
        '''
        obj_input: (1, N, k, 4) k帧中N个object的信息 最后一维度是 x, y 加其速度
        query: (1, N, 1, 2) 总体来看 这是将前两帧past0到past1的速度乘以past0到cur的时间 从而得到一个类似距离的东西
        past_k_time_diff_norm: (k,) 分别为(T3-T1, T2-T1, 0)
        target_time_diff:(1, ) T1 past0到cur的时间间隔
        '''
        ################################ input: [batch, agent, past_frames, dim]  mask: [batch, agent, past_frames, past_frames]
        input = self.src_embed(input) # 编码 输入的应该是3维 （x, y，angle） 输出 64维
        input = self.src_pe(input, time) # 位置编码 将延迟编码进入  最后结果形状不变[batch, agent, k, 64]

        query = self.tgt_embed(query) # 查询应该用当前的时间戳 编码成64维度 [batch, agent, 1, 64]
        query = self.tgt_pe(query, future_time) # 位置编码

        input_features = self.encoder(input, mask)
        
        prediction_features = self.decoder(query, input_features, input_features,mask) # 做transformer 结果形状（batch, agent, 1, 64）
        predictions = self.generator(prediction_features) # 其实就是一个FFN，调节输出维度用 （1, N, 1, 2）
        return predictions



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