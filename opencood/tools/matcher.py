# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.autograd import Variable

from opencood.utils import box_utils


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(vocab,d_model),nn.ReLU(),nn.Linear(d_model,d_model))

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding_irregular(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000 ):
        super(PositionalEncoding_irregular, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        self.linear = nn.Linear(2*d_model,d_model)
         
    def forward(self, x, time):
        ### x:[batch, agent, k, d] time:[batch, agent, k]
        if len(x.shape) > 3:
            last_shape = x.shape
            x = x.reshape(-1,x.shape[-2],x.shape[-1])
            time = time.reshape(-1,time.shape[-1])

        pe = torch.zeros(x.shape[0],time.shape[1], self.d_model).cuda()
        position = time.unsqueeze(2) # 增加维度
        div_term = torch.exp(torch.arange(0., self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).cuda()#相对位置公式
         
        pe[:,:, 0::2] = torch.sin(position * div_term)   #取奇数列
        pe[:,:, 1::2] = torch.cos(position * div_term)   #取偶数列

        # x = x + Variable(pe[:, :x.size(1)], requires_grad=False) # embedding 与positional相加
        
        # print(x.shape)
        # print(x.size(1))
        # print(pe.shape)
        x = self.linear(torch.cat((x,Variable(pe[:, :x.size(1)], requires_grad=False)),dim=-1))
        x = x.reshape(last_shape)
        return self.dropout(x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
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
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn #sublayer 1 
        self.feed_forward = feed_forward #sublayer 2 
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
        x = self.sublayer[0](x, lambda x: self.src_attn(x, src_key, m, tgt_mask)) #解码
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # print('layersize',layer.size)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, src_key ,memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, src_key, memory, tgt_mask) #添加编码的后的结果
        return self.norm(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
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
    def __init__(self, h, q_dim, k_dim,v_dim, d_model, dropout=0.1):
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


    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        
        dim_to_keep = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k 


        query = self.linear_q(query).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)
        key = self.linear_k(key).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)
        value = self.linear_v(value).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,# 进行attention
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
        self.tgt_embed = embedding_tgt
        self.src_pe = position_src
        self.tgt_pe = position_tgt
        self.generator = generator

        

    def forward(self, input, query, time, future_time, mask=None):
        ################################ input: [batch, agent, past_frames, dim]  mask: [batch, agent, past_frames, past_frames]
        input = self.src_embed(input)
        input = self.src_pe(input, time)

        query = self.tgt_embed(query)
        query = self.tgt_pe(query, future_time)

        input_features = self.encoder(input, mask)
        
        prediction_features = self.decoder(query, input_features, input_features,mask)
        predictions = self.generator(prediction_features)
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
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, d_model,d_model,d_model,d_model)
    attn_decoder = MultiHeadedAttention(num_heads,d_model,d_model,d_model,d_model)
    ff = PositionwiseFeedForward(d_model, d_model, d_ff,dropout)
    position = PositionalEncoding_irregular(d_model, dropout)

    model_prediction = Motion_prediction(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Embeddings(d_model, input_dim),
        Embeddings(d_model, output_dim), 
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

# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

'''
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
'''

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stage='encoder'):
        super(BasicBlock, self).__init__()

        conv_block1 = nn.ModuleList()
        conv_block2 = nn.ModuleList()

        conv_block1.append(nn.Conv2d(in_channels, out_channels, 1))
        conv_block1.append(nn.ReLU())

        conv_block2.append(nn.Conv2d(out_channels, out_channels, 1))

        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)  # adjust residual
        self.relu = nn.ReLU()
        self.stage = stage

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        residual = self.conv3(residual)
        x = x + residual

        if self.stage == 'encoder':
            out = self.relu(x)
        elif self.stage == 'decoder':
            out = x
        else:
            raise NotImplementedError("stage should be encoder or decoder")
        return out

class Matcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, fusion, cost_dist: float = 1, cost_giou: float = 1, thre: float = 20, args = None):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_dist = cost_dist
        self.cost_giou = cost_giou
        self.thre = thre

        self.fusion = fusion

        self.distance_all = []
        # self.dataset = args['dataset']

        self.ego_mask = False
        if self.fusion =='flow_uncertainty':
            self.score_reliable = args['score_reliable']
            self.u_cls_data_reliable = args['u_cls_data_reliable']
            self.u_cls_model_reliable = args['u_cls_model_reliable']
            self.u_reg_data_reliable = args['u_reg_data_reliable']
            self.u_reg_model_reliable = args['u_reg_model_reliable']
            print(f"===score_reliable is {self.score_reliable}===")
            print(f"===u_cls_data_reliable is {self.u_cls_data_reliable}, u_cls_model_reliable is {self.u_cls_model_reliable}===")
            print(f"===u_reg_data_reliable is {self.u_reg_data_reliable}, u_reg_model_reliable is {self.u_reg_model_reliable}===")

            self.use_distance = False
            if self.use_distance:
                print("===DFTM 考虑与ego的距离===")
                self.score_generate = nn.Sequential(
                    BasicBlock(6, 12),
                    BasicBlock(12, 24),
                    BasicBlock(24, 1, stage='decoder'),
                    nn.Sigmoid()     
                )
            else:
                self.score_generate = nn.Sequential(
                    BasicBlock(5, 10),
                    BasicBlock(10, 20),
                    BasicBlock(20, 1, stage='decoder'),
                    nn.Sigmoid()     
                )
                if args is not None and args['ego_mask'] is True:
                    self.ego_mask = True
                    print("===ego开启Mask,且使用置信度===")
                    # self.score_generate_cur = nn.Sequential(
                    #     BasicBlock(5, 10),
                    #     BasicBlock(10, 20),
                    #     BasicBlock(20, 1, stage='decoder'),
                    #     nn.Sigmoid()     
                    # )
        self._initialize_weights()
        '''
        因为用不到运动预测了，直接使用线性外插，所以这里先注释
        if fusion=='flow':
            m1, m2 = make_model(input_dim=3, output_dim=3, num_layers=2) # num_layers指的是Encoder Decoder的个数
            self.compensate_motion = m1
            pre_train_weight = '/public/home/lilingzhi/xyj/logs/logs/v2xset_flow_estimate_2024_04_15_08_32_48/net_epoch_bestval_at33.pth'
            pretrained_model_dict = torch.load(pre_train_weight, map_location='cuda:0')
            diff_keys = {k:v for k, v in pretrained_model_dict.items() if k not in self.compensate_motion.state_dict()}
            if diff_keys:
                print(f"!!! PreTrained flow model has keys: {diff_keys.keys()}, \
                    which are not in the model you have created!!!")
            diff_keys = {k:v for k, v in self.compensate_motion.state_dict().items() if k not in pretrained_model_dict.keys()}
            if diff_keys:
                print(f"!!! Created flow model has keys: {diff_keys.keys()}, \
                    which are not in the model you have trained!!!")
            self.compensate_motion.load_state_dict(pretrained_model_dict, strict=False)
            self.compensate_motion.eval() # 评估模式，不参与梯度计算
        '''

            # for name, value in self.compensate_motion.named_parameters():
            #     if name in pretrained_model_dict:
            #         value.requires_grad = False

    # @torch.no_grad() 会临时禁止梯度计算
    def forward(self, input_dict, feature=None, shape_list=None, batch_id=0, viz_flag=False):
        self.viz_flag = viz_flag
        if self.fusion=='box':
            # return self.forward_box(input_dict, batch_id)
            return self.forward_box_w_dir(input_dict, batch_id)
        elif self.fusion=='feature':
            return self.forward_feature(input_dict, feature)
        # elif self.fusion=='flow':
        #     return self.forward_flow_multi_frames(input_dict, shape_list)
        elif self.fusion=='linear':
            return self.forward_flow(input_dict, shape_list)
        elif self.fusion=='flow': # TODO: flow_dir
            return self.forward_flow_dir(input_dict, shape_list)
            # return self.forward_flow_dir_w_uncertainty(input_dict, shape_list)
        elif self.fusion =='flow_uncertainty':
            return self.forward_flow_dir_w_uncertainty(input_dict, shape_list)
        else:
            print("Attention, fusion method must be in box or feature!")
   
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward_box(self, input_dict, batch_id):
        """ Performs the matching
        input_dict: 
            {
                'ego' : {
                    'past_k_time_diff' : 
                    [0] {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                    ... 
                    [k-1]
                    ['comp']{
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
                cav_id {}
            }
        """

        # output_dict_past = {}
        # output_dict_current = {}
        pred_bbox_list = []
        for cav, cav_content in input_dict.items():
            if cav == 'ego':
                estimated_box_center_current = cav_content[0]['pred_box_center_tensor']
                estimated_box_3dcorner_current = cav_content[0]['pred_box_3dcorner_tensor']
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]
                
                # TODO: norm
                # center_points_past1_norm = center_points_past1 - torch.mean(center_points_past1,dim=0,keepdim=True)
                # center_points_past2_norm = center_points_past2 - torch.mean(center_points_past2,dim=0,keepdim=True)
                # cost_mat_center = torch.cdist(center_points_past2_norm,center_points_past1_norm) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

                cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # cost_mat_iou = get_ious()
                cost_mat = cost_mat_center
                past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                past2_ids = dist_valid_past2[0][past2_ids]
                past1_ids = dist_valid_past1[0][past1_ids]
                # output_dict_past.update({car:past_ids})
                # output_dict_current.update({car:current_ids})

                matched_past2 = center_points_past2[past2_ids]
                matched_past1 = center_points_past1[past1_ids]

                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                if time_length == 0:
                    time_length = 1
                flow = (matched_past1 - matched_past2) / time_length

                # if flow.shape[0] != 0:
                #     print(f"max flow is {flow.max()}")

                estimate_position = matched_past1 + flow*(0-cav_content['past_k_time_diff'][0]) 

                # from copy import deepcopy
                # estimated_box_center_current = deepcopy(coord_past1['pred_box_center_tensor'].detach())
                estimated_box_center_current = coord_past1['pred_box_center_tensor'].detach().clone()
                estimated_box_center_current[past1_ids, :2] = estimate_position  # n, 7

                # estimated_box_center_current = torch.zeros_like(coord_past1['pred_box_center_tensor']).to(estimate_position.device)
                # estimated_box_center_current[past1_ids] += torch.cat([estimate_position, coord_past1['pred_box_center_tensor'][past1_ids][:,2:]], dim=-1)
                # no_past1_ids = [x for x in range(coord_past1['pred_box_center_tensor'].shape[0]) if x not in list(past1_ids)]
                # estimated_box_center_current[no_past1_ids] += coord_past1['pred_box_center_tensor'][no_past1_ids]

                estimated_box_3dcorner_current = box_utils.boxes_to_corners_3d(estimated_box_center_current, order='hwl')

            # debug use, update input dict adding estimated frame at cav-past0
            input_dict[cav]['comp'] = {}
            input_dict[cav]['comp'].update({
                'pred_box_center_tensor': estimated_box_center_current,
                'pred_box_3dcorner_tensor': estimated_box_3dcorner_current,
                'scores': cav_content[0]['scores']
            })

        return input_dict

    def forward_box_w_dir(self, input_dict, batch_id):
        """ Performs the matching
        input_dict: 
            {
                'ego' : {
                    'past_k_time_diff' : 
                    [0] {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                    ... 
                    [k-1]
                    ['comp']{
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
                cav_id {}
            }
        """

        # output_dict_past = {}
        # output_dict_current = {}
        pred_bbox_list = []
        for cav, cav_content in input_dict.items():
            if cav == 'ego':
                estimated_box_center_current = cav_content[0]['pred_box_center_tensor']
                estimated_box_3dcorner_current = cav_content[0]['pred_box_3dcorner_tensor']
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]

                cost_mat_center = torch.zeros((center_points_past2.shape[0], center_points_past1.shape[0])).to(center_points_past1.device)

                center_points_past1_repeat = center_points_past1.unsqueeze(0).repeat(center_points_past2.shape[0], 1, 1)
                center_points_past2_repeat = center_points_past2.unsqueeze(1).repeat(1, center_points_past1.shape[0], 1)

                delta_mat = center_points_past1_repeat - center_points_past2_repeat

                angle_mat = torch.atan2(delta_mat[:,:,1], delta_mat[:,:,0]) # [num_cav_past2,num_cav_past1]

                coord_past2_angle_reverse = coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0]).clone()

                visible_mat = torch.where((torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) < 0.785) | (torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

                visible_mat = torch.where(cost_mat_center<0.5, 1, visible_mat)

                tmp_thre = torch.tensor(1000.).to(torch.float32).to(visible_mat.device)
                cost_mat_center = torch.where(visible_mat==1, cost_mat_center, tmp_thre)

                if cost_mat_center.shape[1] == 0 or cost_mat_center.shape[0] == 0:
                    estimated_box_center_current = cav_content[0]['pred_box_center_tensor']
                    estimated_box_3dcorner_current = cav_content[0]['pred_box_3dcorner_tensor']

                else:
                    match = torch.min(cost_mat_center, dim=1)
                    match_to_keep = torch.where(match[0] < 5)

                    past2_ids = match_to_keep[0]
                    past1_ids = match[1][match_to_keep[0]]

                    coord_past2_angle_reverse += 3.1415926
                    coord_past2_angle_reverse[coord_past2_angle_reverse>3.1415926] -= 6.2831852

                    left_past2_id = [n for n in range(cost_mat_center.shape[0]) if n not in past2_ids]
                    left_past1_id = [n for n in range(cost_mat_center.shape[1]) if n not in past1_ids]

                    angle_mat_left = angle_mat[left_past2_id, :][:, left_past1_id]

                    coord_past2_angle_reverse_left = coord_past2_angle_reverse[left_past2_id, :][: ,left_past1_id]

                    visible_mat_left = torch.where((torch.abs(angle_mat_left-coord_past2_angle_reverse_left) < 0.785) | (torch.abs(angle_mat_left-coord_past2_angle_reverse_left) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]

                    cost_mat_center_left = torch.cdist(center_points_past2[left_past2_id], center_points_past1[left_past1_id]) # [num_cav_past2,num_cav_past1]
                    # cost_mat_center_left = cost_mat_center[left_past2_id, :][:, left_past1_id]

                    visible_mat_left = torch.where(cost_mat_center_left<0.5, 1, visible_mat_left)
                    cost_mat_center_left = torch.where(visible_mat_left==1, cost_mat_center_left, tmp_thre)

                    if cost_mat_center_left.shape[1] != 0 and cost_mat_center_left.shape[0] != 0:
                        match_left = torch.min(cost_mat_center_left, dim=1)
                        match_to_keep_left = torch.where(match_left[0] < 5)

                        if match_to_keep_left[0].shape[0] != 0:
                            past2_ids_left = match_to_keep_left[0]
                            past2_ids = torch.cat([past2_ids, torch.tensor(left_past2_id)[past2_ids_left].to(past2_ids.device)])
                            past1_ids_left = match_left[1][match_to_keep_left[0]]
                            past1_ids = torch.cat([past1_ids, torch.tensor(left_past1_id)[past1_ids_left].to(past1_ids.device)])

                    matched_past2 = center_points_past2[past2_ids]
                    matched_past1 = center_points_past1[past1_ids]

                    time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                    if time_length == 0:
                        time_length = 1
                    
                    flow = (matched_past1 - matched_past2) / time_length

                    # if flow.shape[0] != 0:
                    #     print(f"max flow is {flow.max()}")

                    estimate_position = matched_past1 + flow*(0-cav_content['past_k_time_diff'][0]) 

                    # from copy import deepcopy
                    # estimated_box_center_current = deepcopy(coord_past1['pred_box_center_tensor'].detach())
                    estimated_box_center_current = coord_past1['pred_box_center_tensor'].detach().clone()
                    estimated_box_center_current[past1_ids, :2] = estimate_position  # n, 7

                    estimated_box_3dcorner_current = box_utils.boxes_to_corners_3d(estimated_box_center_current, order='hwl')

            # debug use, update input dict adding estimated frame at cav-past0
            input_dict[cav]['comp'] = {}
            input_dict[cav]['comp'].update({
                'pred_box_center_tensor': estimated_box_center_current,
                'pred_box_3dcorner_tensor': estimated_box_3dcorner_current,
                'scores': cav_content[0]['scores']
            })

        return input_dict


    def forward_feature(self, input_dict, features_dict):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                'ego' / cav_id : {
                    [0] / [1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        pred_bbox_list = []
        
        ######## TODO: debug use
        # # pop up other cavs, only keep ego:
        # debug_dict = {'ego': input_dict['ego']}
        # input_dict = debug_dict.copy()
        # debug_features_dict = {'ego': features_dict['ego']}
        # features_dict = debug_features_dict.copy()
        ##############

        for cav, cav_content in input_dict.items():
            if cav == 'ego':
                updated_spatial_features_2d = features_dict[cav]['spatial_features_2d'][0]
                updated_spatial_features = features_dict[cav]['spatial_features'][0]
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

                cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # cost_mat_iou = get_ious()
                cost_mat = cost_mat_center
                past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                past2_ids = dist_valid_past2[0][past2_ids]
                past1_ids = dist_valid_past1[0][past1_ids]
                # output_dict_past.update({car:past_ids})
                # output_dict_current.update({car:current_ids})

                matched_past2 = center_points_past2[past2_ids]
                matched_past1 = center_points_past1[past1_ids]

                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                if time_length == 0:
                    time_length = 1
                flow = (matched_past1 - matched_past2) / time_length

                # TODO: flow * (0-past_k[0])
                flow = flow*(0-cav_content['past_k_time_diff'][0])
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,]
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl')
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                updated_spatial_features_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow, scale=1.25)
                updated_spatial_features = self.feature_warp(features_dict[cav]['spatial_features'][0], selected_box_3dcorner_past0, flow, scale=2.5)

                # debug use
                if debug_flag and flow.shape[0] != 0:
                    torch.save(updated_spatial_features_2d, viz_save_path+'/updated_feature.pt')
                ############

            features_dict[cav].update({
                'updated_spatial_features_2d': updated_spatial_features_2d
            })
            features_dict[cav].update({
                'updated_spatial_features': updated_spatial_features
            })

        return features_dict

    def forward_flow_multi_frames(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            matched_idx_list = []
            compensated_results_list = []
        for cav, cav_content in input_dict.items(): # 遍历每一辆车
            if cav == 0: # ego的处理
                # ego do not need warp
                C, H, W = shape_list
                basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                mask = torch.ones(1, C, H, W).to(shape_list)
                flow_map_list.append(basic_warp_mat)
                reserved_mask.append(mask)
            else:
                past_k_time_diff = cav_content['past_k_time_diff'] # （k） 每一帧到cur的时间间隔
                # TODO 有个问题，以下的这些bbx是已经投影了吗，投影到哪里了？ past0 的lidar pose？
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]
                coord_past3 = cav_content[2] # 过去三帧的信息全部取出

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2] # （ num_cav_past1 ， 7）-> （num_cav_past1， 2）
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]
                center_points_past3 = coord_past3['pred_box_center_tensor'][:,:2]

                self.thre_post_process = 10

                # past1 and past2 match
                cost_mat_center_a = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1] 计算2d下bev距离
                # original_cost_mat_center_a = cost_mat_center_a.clone()
                # cost_mat_center_a[cost_mat_center_a > self.thre_post_process] = 1000

                cost_mat_center_drop_2_a = torch.sum(torch.where(cost_mat_center_a > self.thre, 1, 0), dim=1) # 成本大于threshold的标记1，然后求和，得到（num_cav_past2）每一个元素表示距离past1和past2中某一个object的距离大于threshold的个数，最大值为num_cav_past1，表示全部超过范围
                dist_valid_past2_a = torch.where(cost_mat_center_drop_2_a < center_points_past1.shape[0]) # 和num_cav_past1相比 tuple（ (num_cav_past2') ） 返回的是元组，一个元素，是一维张量，满足条件的会将索引加入其中  说确定一个past2，在past1中有对应
                cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center_a > self.thre, 1, 0), dim=0) # 刚刚是在past2中挨个找一个past2对应每个past1的距离， 现在反过来 （num_cav_past1）
                dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0]) # 确定某一个past1，在past2中是否有其对应

                cost_mat_center_a = cost_mat_center_a[dist_valid_past2_a[0], :]
                cost_mat_center_a = cost_mat_center_a[:, dist_valid_past1[0]] # 将past1和past2互相都有对应的选出来 （M,M）这里M指的是两帧中能匹配上的数量
                
                cost_mat = cost_mat_center_a.clone()
                past2_ids_a, past1_ids = linear_sum_assignment(cost_mat.cpu()) # 寻找线性分配最优解 代价最小的num_cav_past2'索引和num_cav_past1'索引
                
                past2_ids_a = dist_valid_past2_a[0][past2_ids_a] # 从num_cav_past2'中筛选 （ num_cav_past2'' ）  dist_valid_past2_a[0]存储的是有合适匹配的num_cav_past2索引 一共有num_cav_past2'个 这一步就求出了在原来的num_cav_past2中的索引
                past1_ids = dist_valid_past1[0][past1_ids]

                if len(past2_ids_a)==0:
                    print('======= No matched boxes between latest 2 frames! =======')
                    C, H, W = shape_list
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                    mask = torch.ones(1, C, H, W).to(shape_list) # TODO: rethink ones or zeros
                    flow_map_list.append(basic_warp_mat)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([torch.tensor([]), torch.tensor([])], dim=1).to(shape_list.device))
                        compensated_results_list.append(torch.zeros(0, 8, 3).to(shape_list.device))
                    continue

                # past2 and past3 match
                cost_mat_center_b = torch.cdist(center_points_past3, center_points_past2) # [num_cav_past3,num_cav_past2]

                # original_cost_mat_center_b = cost_mat_center_b.clone()
                # cost_mat_center_b[cost_mat_center_b > self.thre_post_process] = 1000

                cost_mat_center_drop_3 = torch.sum(torch.where(cost_mat_center_b > self.thre, 1, 0), dim=1)
                dist_valid_past3 = torch.where(cost_mat_center_drop_3 < center_points_past2.shape[0])
                cost_mat_center_drop_2_b = torch.sum(torch.where(cost_mat_center_b > self.thre, 1, 0), dim=0)
                dist_valid_past2_b = torch.where(cost_mat_center_drop_2_b < center_points_past3.shape[0])

                cost_mat_center_b = cost_mat_center_b[dist_valid_past3[0], :]
                cost_mat_center_b = cost_mat_center_b[:, dist_valid_past2_b[0]]
                
                # cost_mat_iou = get_ious()
                cost_mat = cost_mat_center_b.clone()
                past3_ids, past2_ids_b = linear_sum_assignment(cost_mat.cpu())
                
                past3_ids = dist_valid_past3[0][past3_ids] # （num_cav_past3' ）筛选符合要求的能和past2匹配上的past3索引
                past2_ids_b = dist_valid_past2_b[0][past2_ids_b]
                # output_dict_past.update({car:past_ids})
                # output_dict_current.update({car:current_ids})

                # find the matched obj among three frames
                a_idx, b_idx = self.get_common_elements(past2_ids_a, past2_ids_b) # 通过past2匹配的前后两帧结果，找三帧之间的相同部分 

                # 最近两帧有匹配的，但是最近的三帧没有
                # there is no matched object in past frames
                if len(a_idx)==0 or len(b_idx)==0:
                    matched_past2 = center_points_past2[past2_ids_a]
                    matched_past1 = center_points_past1[past1_ids]

                    time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                    if time_length == 0:
                        time_length = 1
                    flow = (matched_past1 - matched_past2) / time_length

                    flow = flow*(0-cav_content['past_k_time_diff'][0])
                    selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,]
                    selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl')
                    flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                    flow_map_list.append(flow_map)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([past1_ids, past2_ids_a], dim=1))
                        selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                        selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 
                        compensated_results_list.append(selected_box_3dcorner_compensated)
                    continue

                # 三帧匹配结果输入预测模块
                matched_past1 = center_points_past1[past1_ids[a_idx]].unsqueeze(0) # 由于这是匹配成功的三帧 所以个数是一样的 （1， N， 2）
                matched_past2 = center_points_past2[past2_ids_a[a_idx]].unsqueeze(0)
                matched_past3 = center_points_past3[past3_ids[b_idx]].unsqueeze(0)

                obj_coords = torch.cat([matched_past3, matched_past2, matched_past1], dim=0) # （3， N， 2）
                obj_coords = obj_coords.permute(1, 0, 2) # (N, k, 2) 记录着K帧N个object的x/y坐标 倒着放也就是第三帧 第二帧 第一帧

                obj_coords_norm = obj_coords - obj_coords[:, -1:, :] # (N, k, 2)    (N, k, 2)- (N, 1, 2)  减去第一帧（past0）的x,y 得到前面两帧与第一帧的x/y偏移量
                past_k_time_diff = torch.flip(past_k_time_diff, dims=[0]) # (k,) TODO: check if this is correct  顺序翻转，原来顺序依次为past0到cur时间间隔，past1到cur时间间隔，past2到cur时间间隔
                past_k_time_diff_norm = past_k_time_diff - past_k_time_diff[-1] # (k,)

                speed = torch.zeros_like(obj_coords_norm) # (N, k, 2)  下面的方法除法就是求速度，只求了第三帧到第二帧，第二帧到第一帧的速度 TODO 但是这个speed的放的顺序是不是有问题，需要check
                speed[:, 1:, :] = torch.div((obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]), ((past_k_time_diff[1:] - past_k_time_diff[:-1]).unsqueeze(-1)).unsqueeze(0)) # (N, k-1, 2) / (1, k-1, 1)    

                obj_input = torch.cat([obj_coords_norm, speed], dim=-1) # (N, k, 4) 输入原来就是xy加上xy上的速度 

                obj_input = obj_input.unsqueeze(0) # (1, N, k, 4)
                
                last_time_length = (past_k_time_diff_norm[-1] - past_k_time_diff_norm[-2]) # t1-t2 t1是past0到cur的时间间隔 t2是past1到cur的时间间隔 两者相减得到past0与past1的时间间隔
                if last_time_length == 0:
                    print("==== Warning! You met repeated package! ====")
                    query = torch.zeros(obj_input.shape)[:,:,:1,:2].to(obj_input.device) # (1, N, 1, 2)
                else: 
                    query = obj_coords_norm[:, -1:, :] + \
                        (obj_coords_norm[:, -1:, :]-obj_coords_norm[:, -2:-1, :])*(0-past_k_time_diff[-1]) / \
                            last_time_length
                    query = query.unsqueeze(0) # (1, N, 1, 2) 查询使用的是past0到past1的偏移量除以两者的时间差，得到速度，再乘以T1，得到past0到cur可能的偏移量

                target_time_diff = torch.tensor([-past_k_time_diff[-1]]).to(obj_input.device) # (1,) # 这个就是时延T1 表示past0到cur的时间距离

                # target_time_diff = torch.tensor([-past_k_time_diff[0]]).to(obj_input.device) # (1,)
                '''
                运动预测模型的输入是：
                obj_input: (1, N, k, 4) k帧中N个object的信息 最后一维度是 x, y 加其速度
                query: (1, N, 1, 2) 总体来看 这是将前两帧past0到past1的速度乘以past0到cur的时间 从而得到一个类似距离的东西
                past_k_time_diff_norm: (k,) 分别为(T3-T1, T2-T1, 0)
                target_time_diff:(1, )
                '''
                compensated_coords_norm = self.compensate_motion(obj_input, query, past_k_time_diff_norm, target_time_diff) + query # （1, N, 1, 2）输出 注意到还加了一个平均速度下的预估偏移，所以其实输出的就是一个预估偏移

                flow = compensated_coords_norm.squeeze(0).squeeze(1) # (N, 2) 得到预测的N个object的坐标 不对 是中心点的偏移量预测

                # flow = flow*(0-cav_content['past_k_time_diff'][0])
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids[a_idx],] # 将past0的object取出（M， 7）
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl') # （M， 4， 2）

                if self.viz_flag and not(len(a_idx) < len(past2_ids_a)):
                    unit_matched_list = torch.stack([past1_ids[a_idx], past2_ids_a[a_idx], past3_ids[b_idx]], dim=1) # (N_obj, 3)
                    selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                    selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 

                # 两帧匹配成功 but三帧匹配失败 将两帧的结果进行插值 com 表示补集
                if len(a_idx) < len(past2_ids_a): 
                    com_past1_ids = [elem.item() for id, elem in enumerate(past1_ids) if id not in a_idx]
                    com_past2_ids = [elem.item() for id, elem in enumerate(past2_ids_a) if id not in a_idx]
                    matched_past1 = center_points_past1[com_past1_ids]
                    matched_past2 = center_points_past2[com_past2_ids]
                    
                    time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                    if time_length == 0:
                        time_length = 1
                    com_flow = (matched_past1 - matched_past2) / time_length

                    com_flow = com_flow*(0-cav_content['past_k_time_diff'][0])
                    com_selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][com_past1_ids,]
                    com_selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(com_selected_box_3dcenter_past0, order='hwl')
                    
                    flow = torch.cat([flow, com_flow], dim=0)
                    selected_box_3dcorner_past0 = torch.cat([selected_box_3dcorner_past0, com_selected_box_3dcorner_past0], dim=0)

                    # matched: 
                    # past1: past1_ids[a_idx] + com_past1_ids
                    # past2: past2_ids_a[a_idx] + com_past2_ids
                    # past3: past3_ids[b_idx]
                    if self.viz_flag:
                        tmp_past_1 = torch.cat([past1_ids[a_idx], torch.tensor(com_past1_ids).to(past1_ids)], dim=0)
                        tmp_past_2 = torch.cat([past2_ids_a[a_idx], torch.tensor(com_past2_ids).to(past1_ids)], dim=0)
                        unit_matched_list = torch.stack([tmp_past_1, tmp_past_2], dim=1)  # (N_obj, 2)
                        selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                        selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 

                if self.viz_flag:
                    matched_idx_list.append(unit_matched_list)
                    compensated_results_list.append(selected_box_3dcorner_compensated)
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2]
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]
        
        if self.viz_flag:
            return final_flow_map, reserved_mask, matched_idx_list, compensated_results_list
        
        return final_flow_map, reserved_mask

    def forward_flow(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            matched_idx_list = []
        for cav, cav_content in input_dict.items():
            if cav == 0:
                # ego do not need warp
                C, H, W = shape_list
                basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                mask = torch.ones(1, C, H, W).to(shape_list)
                flow_map_list.append(basic_warp_mat)
                reserved_mask.append(mask)
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1]

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2]
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

                self.thre_post_process = 10
                original_cost_mat_center = cost_mat_center.clone()
                cost_mat_center[cost_mat_center > self.thre_post_process] = 1000

                cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # cost_mat_iou = get_ious()
                cost_mat = cost_mat_center
                past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                past2_ids = dist_valid_past2[0][past2_ids]
                past1_ids = dist_valid_past1[0][past1_ids]
                
                ### a trick
                matched_cost = original_cost_mat_center[past2_ids, past1_ids]
                valid_mat_idx = torch.where(matched_cost < self.thre_post_process)
                past2_ids = past2_ids[valid_mat_idx[0]]
                past1_ids = past1_ids[valid_mat_idx[0]]
                ####################

                matched_past2 = center_points_past2[past2_ids]
                matched_past1 = center_points_past1[past1_ids]

                if self.viz_flag:
                    matched_idx_list.append(torch.stack([past1_ids, past2_ids], dim=1))

                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                if time_length == 0:
                    time_length = 1
                flow = (matched_past1 - matched_past2) / time_length

                flow = flow*(0-cav_content['past_k_time_diff'][0])
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,]
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl') # TODO: box order should be a parameter
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
                continue
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2]
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]

        if self.viz_flag:
            return final_flow_map, reserved_mask, matched_idx_list
        return final_flow_map, reserved_mask
        '''
                updated_spatial_features_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow, scale=1.25)
                updated_spatial_features = self.feature_warp(features_dict[cav]['spatial_features'][0], selected_box_3dcorner_past0, flow, scale=2.5)

                # debug use
                if debug_flag and flow.shape[0] != 0:
                    torch.save(updated_spatial_features_2d, viz_save_path+'/updated_feature.pt')
                ############

            features_dict[cav].update({
                'updated_spatial_features_2d': updated_spatial_features_2d
            })
            features_dict[cav].update({
                'updated_spatial_features': updated_spatial_features
            })

        return features_dict
        '''

    def forward_flow_dir(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                        u_cls: (n, )
                        u_reg: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            original_reserved_mask = []
            matched_idx_list = []
            compensated_results_list = []
        for cav, cav_content in input_dict.items(): # 遍历每一辆车
            if cav == 0:
                # ego do not need warp
                C, H, W = shape_list # 64 ，H， W
                basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32) # 创建一个标准仿射变换矩阵，即不做变换
                basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                mask = torch.ones(1, C, H, W).to(shape_list) # 创建的是全为1的矩阵
                flow_map_list.append(basic_warp_mat) 
                reserved_mask.append(mask)
                if self.viz_flag:
                    original_reserved_mask.append(mask)
            else:
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1] # 前两帧 即往前 第0帧 第1帧

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N0， 2） 第0帧的N0个object的x y坐标
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N1， 2）

                cost_mat_center = torch.zeros((center_points_past2.shape[0], center_points_past1.shape[0])).to(center_points_past1.device) # 代价矩阵 （N1, N0） 初始化为全0

                center_points_past1_repeat = center_points_past1.unsqueeze(0).repeat(center_points_past2.shape[0], 1, 1) # （1， N0, 2）-> （N1, N0, 2）
                center_points_past2_repeat = center_points_past2.unsqueeze(1).repeat(1, center_points_past1.shape[0], 1) #  (N1, 1, 2) -> (N1, N0, 2)

                delta_mat = center_points_past1_repeat - center_points_past2_repeat # (N1, N0, 2) 这也就求得了第1帧中的第j个object 到第0帧中第i个object的差值

                angle_mat = torch.atan2(delta_mat[:,:,1], delta_mat[:,:,0]) # [num_cav_past2,num_cav_past1] 通过反正切值求弧度值，atan2还能够根据输入xy的正负来判断象限从而精确判断角度 这一步的目的是表示出两个不同时间点检测到的目标的相对角度

                coord_past2_angle_reverse = coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0]).clone() # 角度信息，（N1, N0）
                # 标记可视范围内的车辆  也就是在车辆轴向方向正负四十五度范围内
                visible_mat = torch.where((torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) < 0.785) | (torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1] 计算两两之间的成对距离作为成本矩阵 （N1， N0）

                visible_mat = torch.where(cost_mat_center<0.5, 1, visible_mat) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况

                tmp_thre = torch.tensor(1000.).to(torch.float32).to(visible_mat.device)
                cost_mat_center = torch.where(visible_mat==1, cost_mat_center, tmp_thre) # （N1， N0） 如果在可视范围（角度或者距离合适），则添加其欧氏距离，否则直接无穷（1000）

                if cost_mat_center.shape[1] == 0 or cost_mat_center.shape[0] == 0: # 检查是否有任一维度为空，如果有，意味着没有可匹配的目标，即没有两个目标之间的距离可计算
                    C, H, W = shape_list
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                    mask = torch.ones(1, C, H, W).to(shape_list)
                    flow_map_list.append(basic_warp_mat)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([torch.tensor([]), torch.tensor([])], dim=1).to(shape_list.device))
                        compensated_results_list.append(torch.zeros(0, 8, 3).to(shape_list.device))
                        original_reserved_mask.append(mask)
                    continue

                match = torch.min(cost_mat_center, dim=1) # 返回一个元组 （val， index） 第一个元素形状是（N1）表示每一行对应的最小值，第二个元素形状相同，表示每一行对应最小值的索引
                match_to_keep = torch.where(match[0] < 5) # 找到最小距离小于5m的并记录其index  注意，torch.where只传一个参数的时候返回的是一个元组，其中就一个元素，为一维张量，长度不固定 为0到N1-1 比如说(1, 3)表示第1个和第3个满足条件
    
                past2_ids = match_to_keep[0] # 提取第一个元素 一维张量  也就是最小距离小于5m的N1索引
                past1_ids = match[1][match_to_keep[0]] # 最小距离小于5m的对应的N0的索引 
                # 以下操作是重复以上过程，首先计算可视范围过滤，范围小于正负45°或者距离小于0.5m   注意，在计算相对角度时，并没有考虑到object自身的角度，所以相对角度要减去自身角度，但是自身角度有可能出现正负
                coord_past2_angle_reverse += 3.1415926 # 角度加上了pi
                coord_past2_angle_reverse[coord_past2_angle_reverse>3.1415926] -= 6.2831852 # 这两行是为了确保角度值保持在 -π 到 π 的范围内， 这一步也就是考虑了负的角度，那接下来就要重新来考虑一遍

                left_past2_id = [n for n in range(cost_mat_center.shape[0]) if n not in past2_ids] # 遍历N1次 0 到 N1-1 如果不在past2_ids索引列表里的就存在这里
                left_past1_id = [n for n in range(cost_mat_center.shape[1]) if n not in past1_ids] # 遍历N0次 0 到 N0-1 不满足条件的N0索引

                angle_mat_left = angle_mat[left_past2_id, :][:, left_past1_id] # 从（N1， N0）中选出来不满足条件的相对角度

                coord_past2_angle_reverse_left = coord_past2_angle_reverse[left_past2_id, :][: ,left_past1_id] #  从（N1， N0）中选出来不满足条件的Object自身角度
                # 这是筛选出这些不满足条件的object中的可见性矩阵
                visible_mat_left = torch.where((torch.abs(angle_mat_left-coord_past2_angle_reverse_left) < 0.785) | (torch.abs(angle_mat_left-coord_past2_angle_reverse_left) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]
                
                cost_mat_center_left = torch.cdist(center_points_past2[left_past2_id], center_points_past1[left_past1_id]) # [num_cav_past2,num_cav_past1] 计算两两之间的欧氏距离
                # cost_mat_center_left = cost_mat_center[left_past2_id, :][:, left_past1_id]

                visible_mat_left = torch.where(cost_mat_center_left<0.5, 1, visible_mat_left) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况
                cost_mat_center_left = torch.where(visible_mat_left==1, cost_mat_center_left, tmp_thre) # 如果满足可见性，则填充欧式距离，否则填写1000 （N1', N0'）

                if cost_mat_center_left.shape[1] != 0 and cost_mat_center_left.shape[0] != 0: # 如果有欧式距离可以算
                    match_left = torch.min(cost_mat_center_left, dim=1) # 找最小距离对应的 返回元组
                    match_to_keep_left = torch.where(match_left[0] < 5) # 寻找最小距离还小于5m的结果 N1索引，外面套了一个元组

                    if match_to_keep_left[0].shape[0] != 0: # 如果确实有值，也就是说确实有满足条件的
                        past2_ids_left = match_to_keep_left[0] # 满足要求的N1索引
                        past2_ids = torch.cat([past2_ids, torch.tensor(left_past2_id)[past2_ids_left].to(past2_ids.device)]) # 将第二轮挑选的加入
                        past1_ids_left = match_left[1][match_to_keep_left[0]] # 满足要求的N0索引
                        past1_ids = torch.cat([past1_ids, torch.tensor(left_past1_id)[past1_ids_left].to(past1_ids.device)]) # 到这里为止已经获得了N1，与N0的索引根据这两个可以查询到对应的object

                # ##############################################
                # self.thre_post_process = 10
                # original_cost_mat_center = cost_mat_center.clone()
                # cost_mat_center[cost_mat_center > self.thre_post_process] = 1000

                # cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                # dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                # cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                # dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                # cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                # cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # # cost_mat_iou = get_ious()
                # cost_mat = cost_mat_center
                # past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                # past2_ids = dist_valid_past2[0][past2_ids]
                # past1_ids = dist_valid_past1[0][past1_ids]
                
                # ### a trick
                # matched_cost = original_cost_mat_center[past2_ids, past1_ids]
                # valid_mat_idx = torch.where(matched_cost < self.thre_post_process)
                # past2_ids = past2_ids[valid_mat_idx[0]]
                # past1_ids = past1_ids[valid_mat_idx[0]]
                # ####################

                matched_past2 = center_points_past2[past2_ids] #  取出对应的object   （M , 2）
                matched_past1 = center_points_past1[past1_ids] #  取出对应的object   （M , 2）

                # print(cav_content['past_k_time_diff'])
                # print(cav)
                # print(cav_content)
                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1] # 时间长度 这里的时间长度为T1-T2 也就是past0到past1的时间间隔 是个整数
                # print(cav_content['past_k_time_diff'])
                # print(time_length)
                # exit9

                if time_length == 0:
                    time_length = 1

                flow = (matched_past1 - matched_past2) / time_length # 两者想减得到距离，距离/时间为平均速度或者说“流速” (M, 2)

                flow = flow*(0-cav_content['past_k_time_diff'][0]) # 速度乘以距离得到预估的位移 past_k_time_diff中存放的是一个agent中对应帧与current帧的时间间隔 为 -τ， -τ-1，-τ-2.... 
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,] # 根据索引得到对应的bbx （M, 7） 然后下一行则是将3d bbx变为2d bbx （M, 7）-> （M, 8, 3）-> （M, 4, 3）
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl') # TODO: box order should be a parameter  

                if self.viz_flag:
                    matched_idx_list.append(torch.stack([past1_ids, past2_ids], dim=1))
                    selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                    selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 
                    compensated_results_list.append(selected_box_3dcorner_compensated)
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                if self.viz_flag:
                    flow_map, mask, single_ori_mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                    original_reserved_mask.append(single_ori_mask)
                else: # 返回的流场图，以及掩码， 流场图来源于坐标网格
                    flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list) # 返回形状（1， H， W， 2）， （1， C， H， W）
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
                continue
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2] 其实就是坐标网格 ego的话就是恒等变换形成的坐标网格，而其他agent则是将流的运动属性运用到
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]

        if self.viz_flag:
            original_reserved_mask = torch.concat(original_reserved_mask, dim=0)  # [N_b, C, H, W]
            return final_flow_map, reserved_mask, original_reserved_mask, matched_idx_list, compensated_results_list
        return final_flow_map, reserved_mask
        '''
                updated_spatial_features_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow, scale=1.25)
                updated_spatial_features = self.feature_warp(features_dict[cav]['spatial_features'][0], selected_box_3dcorner_past0, flow, scale=2.5)

                # debug use
                if debug_flag and flow.shape[0] != 0:
                    torch.save(updated_spatial_features_2d, viz_save_path+'/updated_feature.pt')
                ############

            features_dict[cav].update({
                'updated_spatial_features_2d': updated_spatial_features_2d
            })
            features_dict[cav].update({
                'updated_spatial_features': updated_spatial_features
            })

        return features_dict
        '''

# 无延迟衰减且未筛选那些其实没必要学习权重的bbx
    def forward_flow_dir_w_uncertainty_backup_no_delay(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    'matrix_past0_2_cur' : 当前cav past0到ego的空间变换矩阵
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                        u_cls_data: (n, )
                        u_cls_model: (n, )
                        u_reg_data: (n, )
                        u_reg_model: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            original_reserved_mask = []
            matched_idx_list = []
            compensated_results_list = []
        for cav, cav_content in input_dict.items(): # 遍历每一辆车
            if cav == 0:
                # ego do not need warp
                C, H, W = shape_list # 64 ，H， W
                basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32) # 创建一个标准仿射变换矩阵，即不做变换
                basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device) # （1， H， W， 2）
                mask = torch.ones(1, C, H, W).to(shape_list) # 创建的是全为1的矩阵

                # # ==计算到ego的距离
                # center_cur = cav_content[0]['pred_box_center_tensor'][:,:2] # n,2
                # distance2detector_cur = center_cur * 2.5
                # distance2detector_cur = distance2detector_cur.square().sum(dim=1).sqrt() # n, 距离ego的距离
                # distance2detector_cur = distance2detector_cur / (torch.sqrt(H**2 + W**2) / 2) # 最远距离判定
                # distance2detector_cur = distance2detector_cur.cpu().numpy().tolist()
                # self.distance_all += distance2detector_cur
                # # ==end
                # cur_res = cav_content[0]
                # scrore_cur = cur_res['scores']
                # u_cls_cur = cur_res['u_cls']
                # u_reg_cur = cur_res['u_reg']
                # scores_input = torch.stack((scrore_cur, u_cls_cur, u_reg_cur), dim=-1)
                # scores_input = scores_input.unsqueeze(-1).unsqueeze(-1) # n,3,1,1
                # scores_weighted = self.score_generate(scores_input) # n,1,1,1
                # scores_weighted = scores_weighted.squeeze(1).squeeze(1) # n ,1

                # flow_cur = torch.zeros((cur_res['scores'].shape[0],  2)).to(cur_res['scores'].device) # n,2 流

                # selected_box_3dcorner_cur = box_utils.boxes_to_corners2d(cur_res['pred_box_center_tensor'], order='hwl')
                # flow_map, mask = self.generate_flow_map_ego(flow_cur, selected_box_3dcorner_cur, scale=2.5, scores_weighted = scores_weighted, shape_list=shape_list) # 返回形状（1， H， W， 2）， （1， C， H， W）
                
                # flow_map_list.append(flow_map) 

                flow_map_list.append(basic_warp_mat) 
                reserved_mask.append(mask)

                if self.viz_flag:
                    original_reserved_mask.append(mask)
            else: 
                # 以下开始线性外插，但是我发现，由于只有两帧，因此它必须考虑两帧完全匹配的情况，由于最终warp以及mask都是针对past0的伪图，因此要排除一下past0中有物体而past1中没有的情况
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1] # 前两帧 即往前 第0帧 第1帧

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N0， 2） 第0帧的N0个object的x y坐标
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N1， 2）

                u_cls_1 = torch.stack((coord_past1['u_cls_data'], coord_past1['u_cls_model']), dim=-1) # (N0, 2)                
                u_reg_1 = torch.stack((coord_past1['u_reg_data'], coord_past1['u_reg_model']), dim=-1) # (N0, 2)     
                u_cls_2 = torch.stack((coord_past2['u_cls_data'], coord_past2['u_cls_model']), dim=-1) # (N0, 2)                
                u_reg_2 = torch.stack((coord_past2['u_reg_data'], coord_past2['u_reg_model']), dim=-1) # (N0, 2)  
                # print("coord_past1['u_cls_data'] shape is ", coord_past1['u_cls_data'].shape)
                # print("u_cls_1 shape is ", u_cls_1.shape)
                # print("u_cls1_data raw is ", coord_past1['u_cls_data'])
                # print("u_cls1_model raw is ", coord_past1['u_cls_model'])
                # u_cls_1 = coord_past1['u_cls'] # past 0 帧中的分类不确定性 （N0，）
                # u_reg_1 = coord_past1['u_reg']
                # u_cls_2 = coord_past2['u_cls']
                # u_reg_2 = coord_past2['u_reg']

                cost_mat_center = torch.zeros((center_points_past2.shape[0], center_points_past1.shape[0])).to(center_points_past1.device) # 代价矩阵 （N1, N0） 初始化为全0

                center_points_past1_repeat = center_points_past1.unsqueeze(0).repeat(center_points_past2.shape[0], 1, 1) # （1， N0, 2）-> （N1, N0, 2）
                center_points_past2_repeat = center_points_past2.unsqueeze(1).repeat(1, center_points_past1.shape[0], 1) #  (N1, 1, 2) -> (N1, N0, 2)

                delta_mat = center_points_past1_repeat - center_points_past2_repeat # (N1, N0, 2) 这也就求得了第1帧中的第j个object 到第0帧中第i个object的差值

                angle_mat = torch.atan2(delta_mat[:,:,1], delta_mat[:,:,0]) # [num_cav_past2,num_cav_past1] 通过反正切值求弧度值，atan2还能够根据输入xy的正负来判断象限从而精确判断角度 这一步的目的是表示出两个不同时间点检测到的目标的相对角度

                coord_past2_angle_reverse = coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0]).clone() # 角度信息，（N1, N0）
                # 标记可视范围内的车辆  也就是在车辆轴向方向正负四十五度范围内
                visible_mat = torch.where((torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) < 0.785) | (torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1] 计算两两之间的成对距离作为成本矩阵 （N1， N0）

                visible_mat = torch.where(cost_mat_center<0.5, 1, visible_mat) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况

                tmp_thre = torch.tensor(1000.).to(torch.float32).to(visible_mat.device)
                cost_mat_center = torch.where(visible_mat==1, cost_mat_center, tmp_thre) # （N1， N0） 如果在可视范围（角度或者距离合适），则添加其欧氏距离，否则直接无穷（1000）

                if cost_mat_center.shape[1] == 0 or cost_mat_center.shape[0] == 0: # 检查是否有任一维度为空，如果有，意味着没有可匹配的目标，即没有两个目标之间的距离可计算
                    C, H, W = shape_list
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                    mask = torch.ones(1, C, H, W).to(shape_list)
                    flow_map_list.append(basic_warp_mat)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([torch.tensor([]), torch.tensor([])], dim=1).to(shape_list.device))
                        compensated_results_list.append(torch.zeros(0, 8, 3).to(shape_list.device))
                        original_reserved_mask.append(mask)
                    continue

                match = torch.min(cost_mat_center, dim=1) # 返回一个元组 （val， index） 第一个元素形状是（N1）表示每一行对应的最小值，第二个元素形状相同，表示每一行对应最小值的索引
                match_to_keep = torch.where(match[0] < 5) # 找到最小距离小于5m的并记录其index  注意，torch.where只传一个参数的时候返回的是一个元组，其中就一个元素，为一维张量，长度不固定 为0到N1-1 比如说(1, 3)表示第1个和第3个满足条件
    
                past2_ids = match_to_keep[0] # 提取第一个元素 一维张量  也就是最小距离小于5m的N1索引
                past1_ids = match[1][match_to_keep[0]] # 最小距离小于5m的对应的N0的索引 
                # 以下操作是重复以上过程，首先计算可视范围过滤，范围小于正负45°或者距离小于0.5m   注意，在计算相对角度时，并没有考虑到object自身的角度，所以相对角度要减去自身角度，但是自身角度有可能出现正负
                coord_past2_angle_reverse += 3.1415926 # 角度加上了pi
                coord_past2_angle_reverse[coord_past2_angle_reverse>3.1415926] -= 6.2831852 # 这两行是为了确保角度值保持在 -π 到 π 的范围内， 这一步也就是考虑了负的角度，那接下来就要重新来考虑一遍

                left_past2_id = [n for n in range(cost_mat_center.shape[0]) if n not in past2_ids] # 遍历N1次 0 到 N1-1 如果不在past2_ids索引列表里的就存在这里
                left_past1_id = [n for n in range(cost_mat_center.shape[1]) if n not in past1_ids] # 遍历N0次 0 到 N0-1 不满足条件的N0索引

                angle_mat_left = angle_mat[left_past2_id, :][:, left_past1_id] # 从（N1， N0）中选出来不满足条件的相对角度

                coord_past2_angle_reverse_left = coord_past2_angle_reverse[left_past2_id, :][: ,left_past1_id] #  从（N1， N0）中选出来不满足条件的Object自身角度
                # 这是筛选出这些不满足条件的object中的可见性矩阵
                visible_mat_left = torch.where((torch.abs(angle_mat_left-coord_past2_angle_reverse_left) < 0.785) | (torch.abs(angle_mat_left-coord_past2_angle_reverse_left) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]
                
                cost_mat_center_left = torch.cdist(center_points_past2[left_past2_id], center_points_past1[left_past1_id]) # [num_cav_past2,num_cav_past1] 计算两两之间的欧氏距离
                # cost_mat_center_left = cost_mat_center[left_past2_id, :][:, left_past1_id]

                visible_mat_left = torch.where(cost_mat_center_left<0.5, 1, visible_mat_left) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况
                cost_mat_center_left = torch.where(visible_mat_left==1, cost_mat_center_left, tmp_thre) # 如果满足可见性，则填充欧式距离，否则填写1000 （N1', N0'）

                if cost_mat_center_left.shape[1] != 0 and cost_mat_center_left.shape[0] != 0: # 如果有欧式距离可以算
                    match_left = torch.min(cost_mat_center_left, dim=1) # 找最小距离对应的 返回元组
                    match_to_keep_left = torch.where(match_left[0] < 5) # 寻找最小距离还小于5m的结果 N1索引，外面套了一个元组

                    if match_to_keep_left[0].shape[0] != 0: # 如果确实有值，也就是说确实有满足条件的
                        past2_ids_left = match_to_keep_left[0] # 满足要求的N1索引
                        past2_ids = torch.cat([past2_ids, torch.tensor(left_past2_id)[past2_ids_left].to(past2_ids.device)]) # 将第二轮挑选的加入
                        past1_ids_left = match_left[1][match_to_keep_left[0]] # 满足要求的N0索引
                        past1_ids = torch.cat([past1_ids, torch.tensor(left_past1_id)[past1_ids_left].to(past1_ids.device)]) # 到这里为止已经获得了N1，与N0的索引根据这两个可以查询到对应的object

                # ##############################################
                # self.thre_post_process = 10
                # original_cost_mat_center = cost_mat_center.clone()
                # cost_mat_center[cost_mat_center > self.thre_post_process] = 1000

                # cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                # dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                # cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                # dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                # cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                # cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # # cost_mat_iou = get_ious()
                # cost_mat = cost_mat_center
                # past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                # past2_ids = dist_valid_past2[0][past2_ids]
                # past1_ids = dist_valid_past1[0][past1_ids]
                
                # ### a trick
                # matched_cost = original_cost_mat_center[past2_ids, past1_ids]
                # valid_mat_idx = torch.where(matched_cost < self.thre_post_process)
                # past2_ids = past2_ids[valid_mat_idx[0]]
                # past1_ids = past1_ids[valid_mat_idx[0]]
                # ####################

                matched_past2 = center_points_past2[past2_ids] #  取出对应的object   （M , 2）
                matched_past1 = center_points_past1[past1_ids] #  取出对应的object   （M , 2）
                matched_u_cls_1 = u_cls_1[past1_ids] # past 0 帧中的分类不确定性 （M，2）
                matched_u_reg_1 = u_reg_1[past1_ids]
                matched_u_cls_2 = u_cls_2[past2_ids]
                matched_u_reg_2 = u_reg_2[past2_ids]

                # C, H, W = shape_list # 64 ，H， W
                # trans_matrix = cav_content['matrix_past0_2_cur'] # 4,4
                # matched_3d = box_utils.boxes_to_corners_3d(coord_past1['pred_box_center_tensor'][past1_ids], order='hwl')
                # projected_boxes3d = box_utils.project_box3d(matched_3d, trans_matrix) # 将空间坐标投影到第0帧 （M, 8，3）
                # projected_center_past0 = box_utils.corner_to_center_torch(projected_boxes3d, order='hwl') # （M， 7）
                # projected_center_past0 = projected_center_past0[:, :2] # cur view 下对应的object位置
                # distance2detector = projected_center_past0 * 2.5 # 转移到体素距离
                # distance2detector = distance2detector.square().sum(dim=1).sqrt() # 得到距离检测器的距离
                # distance2detector = distance2detector / (torch.sqrt(H**2 + W**2) / 2)

                # distance2detector = distance2detector.cpu().numpy().tolist()
                # self.distance_all += distance2detector
                # for id in remain_past1:


                # print(cav_content['past_k_time_diff'])
                # print(cav)
                # print(cav_content)
                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1] # 时间长度 这里的时间长度为T1-T2 也就是past0到past1的时间间隔 是个整数
                # exit9

                if time_length == 0:
                    time_length = 1

                # 不确定性需要在线性插值的同时进行传播
                # 1、分类不确定性得分是通过分类偏差比来衡量，（M,）大于一定阈值：tp均值+std 取两帧object最大值，小于一定阈值：fp均值-std 取两帧object最小值，其余情况取两帧均值
                # 2、回归不确定性是通过方差并且标准化得到，（M，）
                scores_average = (cav_content[0]['scores'][past1_ids] + cav_content[1]['scores'][past2_ids]) / 2
                u_cls_average = (matched_u_cls_1 + matched_u_cls_2) / 2 # （M，2）
                u_reg_average = (matched_u_reg_1 + matched_u_reg_2) / 2

                scores_raw = scores_average
                u_cls_compensate = u_cls_average
                u_reg_compensate = u_reg_average
                # scores_raw = cav_content[0]['scores'][past1_ids]
                # u_cls_compensate = matched_u_cls_1
                # u_reg_compensate = matched_u_reg_1

                # print("u_cls_compensate shape is ", u_cls_compensate.shape)
                # print("u_reg_compensate shape is ", u_reg_compensate.shape)
                # print("u_cls_data is", u_cls_compensate[:,0])
                # print("u_cls_model is", u_cls_compensate[:,1])
                # print("u_reg_data is", u_reg_compensate[:,0])
                # print("u_reg_model is", u_reg_compensate[:,1])
                # print("scores_raw is", scores_raw)
                # xxx
                
                # 调整传播方式
                # if u_cls_average >= 0.7891:
                #     u_cls_compensate = torch.max(matched_u_cls_1, matched_u_cls_2)
                # elif u_cls_average <= 0.5114:
                #     u_cls_compensate = torch.min(matched_u_cls_1, matched_u_cls_2)
                # else:
                #     u_cls_compensate = u_cls_average

                # u_reg_compensate = u_reg_average + time_length * 0.01
                    


                # def print_requires_grad(module, prefix=''):
                #     for name, param in module.named_parameters(recurse=False):
                #         print(f"{prefix}Parameter {name} requires_grad: {param.requires_grad}")
                #     for name, sub_module in module.named_children():
                #         print_requires_grad(sub_module, prefix + '  ')

                # for idx, layer in enumerate(self.score_generate):
                #     print(f"Layer {idx} ({layer.__class__.__name__}):")
                #     print_requires_grad(layer, '  ')

                scores_input = torch.stack((scores_raw, u_cls_compensate[:, 0], u_cls_compensate[:, 1], u_reg_compensate[:, 0], u_reg_compensate[:, 1]), dim=-1)

                scores_input = scores_input.unsqueeze(-1).unsqueeze(-1) # n,3,1,1
                scores_weighted = self.score_generate(scores_input) # n,1,1,1
                scores_weighted = scores_weighted.squeeze(1).squeeze(1) # n ,1
                # 是否需要线性变换映射到0.2-1
                # scores_weighted = 0.5 * scores_weighted + 0.5

 
                flow = (matched_past1 - matched_past2) / time_length # 两者想减得到距离，距离/时间为平均速度或者说“流速” (M, 2)

                flow = flow*(0-cav_content['past_k_time_diff'][0]) # 速度乘以距离得到预估的位移 past_k_time_diff中存放的是一个agent中对应帧与current帧的时间间隔 为 -τ， -τ-1，-τ-2.... 
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,] # 根据索引得到对应的bbx （M, 7） 然后下一行则是将3d bbx变为2d bbx （M, 7）-> （M, 8, 3）-> （M, 4, 3）
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl') # TODO: box order should be a parameter  

                if self.viz_flag:
                    matched_idx_list.append(torch.stack([past1_ids, past2_ids], dim=1))
                    selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                    selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 
                    compensated_results_list.append(selected_box_3dcorner_compensated)
                
                # 给一种稀有情况打补丁：即past0中检测到且置信度很高的物体，
                '''
                remain_past1_ids = [id for id in range(center_points_past1.shape[0]) if id not in past1_ids] # past0中匹配到了，然后这是选出剩余的部分
                # print('center_points_past1.shape[0] is ', center_points_past1.shape[0])
                # print('past1_ids is ', past1_ids)
                # print('remain_past1_ids is ', remain_past1_ids)
                remain_past1 = coord_past1['pred_box_center_tensor'][remain_past1_ids,] # (remains, 7)
                remain_past1_scores = cav_content[0]['scores'][remain_past1_ids] # (remains,)
                condition = (remain_past1_scores > 0.6653) & (u_reg_1[remain_past1_ids] <= -0.2271) & (u_cls_1[remain_past1_ids] >= 0.7891)
                indices_score_confirm = torch.where(condition)
                # print("选中的bbx其分类不确定性为: ", matched_u_cls_1)
                # print("选中的bbx其回归不确定性为: ", matched_u_reg_1)
                # print("剩余的bbx其分类不确定性为: ", u_cls_1[remain_past1_ids])
                # print("剩余的bbx其回归不确定性为: ", u_reg_1[remain_past1_ids])
                # print('indices_score_confirm is ', indices_score_confirm)
                # print('高置信度bbx的置信度为:  ', remain_past1_scores[indices_score_confirm])
                remain_past1_confirm = remain_past1[indices_score_confirm] # 这就筛选出past0中没有匹配成功但是置信度又很高的部分 （n, 7）
                if remain_past1_confirm.shape[0] != 0:
                    print("===fuck! boom====")
                    print("有高置信度bbx未匹配,数目为: ", remain_past1_confirm.shape[0])
                    print("高可信但未匹配的bbx的回归不确定性为: ", u_reg_1[remain_past1_ids][indices_score_confirm])
                    print("高可信但未匹配的bbx的分类不确定性为: ", u_cls_1[remain_past1_ids][indices_score_confirm])
                    zero_flow = torch.zeros((remain_past1_confirm.shape[0],  2)).to(center_points_past1.device) # n,2 流
                    one_weight = torch.ones((remain_past1_confirm.shape[0],  1)).to(center_points_past1.device)
                    remain_past1_confirm_3dcorner = box_utils.boxes_to_corners2d(remain_past1_confirm, order='hwl') # n,7 -> n, 4, 3

                    flow = torch.cat([flow, zero_flow], dim=0)
                    selected_box_3dcorner_past0 = torch.cat([selected_box_3dcorner_past0, remain_past1_confirm_3dcorner], dim=0)
                    scores_weighted = torch.cat([scores_weighted, one_weight], dim=0)
                    # scores_weighted = torch.ones_like(scores_weighted)
                '''

                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                if self.viz_flag:
                    flow_map, mask, single_ori_mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, scores_weighted = scores_weighted, shape_list=shape_list)
                    original_reserved_mask.append(single_ori_mask)
                else: # 返回的流场图，以及掩码， 流场图来源于坐标网格
                    flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, scores_weighted = scores_weighted, shape_list=shape_list) # 返回形状（1， H， W， 2）， （1， C， H， W）
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
                continue
        
        # distance_mean = np.mean(self.distance_all)
        # distance_std = np.std(self.distance_all)
        # distance_max = np.max(self.distance_all)
        # distance_min = np.min(self.distance_all)
        # print("distance len is ", len(self.distance_all))
        # print("distance_mean is ", distance_mean)
        # print("distance_std is ", distance_std)
        # print("distance_max is ", distance_max)
        # print("distance_min is ", distance_min)
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2] 其实就是坐标网格 ego的话就是恒等变换形成的坐标网格，而其他agent则是将流的运动属性运用到
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]

        if self.viz_flag:
            original_reserved_mask = torch.concat(original_reserved_mask, dim=0)  # [N_b, C, H, W]
            return final_flow_map, reserved_mask, original_reserved_mask, matched_idx_list, compensated_results_list
        return final_flow_map, reserved_mask

    def forward_flow_dir_w_uncertainty(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    'matrix_past0_2_cur' : 当前cav past0到ego的空间变换矩阵
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                        u_cls_data: (n, )
                        u_cls_model: (n, )
                        u_reg_data: (n, )
                        u_reg_model: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            original_reserved_mask = []
            matched_idx_list = []
            compensated_results_list = []
        for cav, cav_content in input_dict.items(): # 遍历每一辆车
            if cav == 0:
                # ego do not need warp
                C, H, W = shape_list # 64 ，H， W
                if self.ego_mask is not True:
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32) # 创建一个标准仿射变换矩阵，即不做变换
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device) # （1， H， W， 2）
                    mask = torch.ones(1, C, H, W).to(shape_list) # 创建的是全为1的矩阵

                    # # ==计算到ego的距离
                    # center_cur = cav_content[0]['pred_box_center_tensor'][:,:2] # n,2
                    # distance2detector_cur = center_cur * 2.5
                    # distance2detector_cur = distance2detector_cur.square().sum(dim=1).sqrt() # n, 距离ego的距离
                    # distance2detector_cur = distance2detector_cur / (torch.sqrt(H**2 + W**2) / 2) # 最远距离判定
                    # distance2detector_cur = distance2detector_cur.cpu().numpy().tolist()
                    # self.distance_all += distance2detector_cur
                    # # ==end
                    # cur_res = cav_content[0]
                    # scrore_cur = cur_res['scores']
                    # u_cls_cur = cur_res['u_cls']
                    # u_reg_cur = cur_res['u_reg']
                    # scores_input = torch.stack((scrore_cur, u_cls_cur, u_reg_cur), dim=-1)
                    # scores_input = scores_input.unsqueeze(-1).unsqueeze(-1) # n,3,1,1
                    # scores_weighted = self.score_generate(scores_input) # n,1,1,1
                    # scores_weighted = scores_weighted.squeeze(1).squeeze(1) # n ,1

                    # flow_cur = torch.zeros((cur_res['scores'].shape[0],  2)).to(cur_res['scores'].device) # n,2 流

                    # selected_box_3dcorner_cur = box_utils.boxes_to_corners2d(cur_res['pred_box_center_tensor'], order='hwl')
                    # flow_map, mask = self.generate_flow_map_ego(flow_cur, selected_box_3dcorner_cur, scale=2.5, scores_weighted = scores_weighted, shape_list=shape_list) # 返回形状（1， H， W， 2）， （1， C， H， W）
                    
                    # flow_map_list.append(flow_map) 

                    flow_map_list.append(basic_warp_mat) 
                    reserved_mask.append(mask)

                    if self.viz_flag:
                        original_reserved_mask.append(mask)
                else: 
                    # ==========从这里开始ego也mask的实验===============
                    coord_cur = cav_content[0]
                    center_points_cur = coord_cur['pred_box_center_tensor'][:,:2] # n,2
                    score_cur = coord_cur['scores']
                    u_cls_cur = torch.stack((coord_cur['u_cls_data'], coord_cur['u_cls_model']), dim=-1) # (n, 2)                
                    u_reg_cur = torch.stack((coord_cur['u_reg_data'], coord_cur['u_reg_model']), dim=-1) # (n, 2)     
                    scores_weighted = torch.zeros((center_points_cur.shape[0], )).to(center_points_cur.device) # (n,)

                    condition = (score_cur >= self.score_reliable) & (u_cls_cur[:, 0] >= self.u_cls_data_reliable) & (u_cls_cur[:, 1] >= self.u_cls_model_reliable) & (u_reg_cur[:, 0] <= self.u_reg_data_reliable) & (u_reg_cur[:, 1] <= self.u_reg_model_reliable)
                    # condition = (score_cur > 0.6469) & (u_cls_cur[:, 0] >= 0.8123) & (u_cls_cur[:, 1] >= 0.8229) & (u_reg_cur[:, 0] <= -0.5297) & (u_reg_cur[:, 1] <= -0.3940)
                    scores_weighted[condition] = 1.0
                    # print("置信度：", score_cur)
                    # print("数据不确定分类", u_cls_cur[:,0])
                    # print("模型不确定分类", u_cls_cur[:,1])
                    # print("数据不确定回归", u_reg_cur[:,0])
                    # print("模型不确定回归", u_reg_cur[:,1])
                    # print("一开始将高可靠的直接选出来：", scores_weighted)
                    condition_zero = (scores_weighted == 0)
                    if condition_zero.sum() != 0:
                        score_cur = score_cur[condition_zero]
                        u_cls_cur = u_cls_cur[condition_zero, ...]
                        u_reg_cur = u_reg_cur[condition_zero, ...]

                        scores_input = torch.stack((score_cur, u_cls_cur[:, 0], u_cls_cur[:, 1], u_reg_cur[:, 0], u_reg_cur[:, 1]), dim=-1)

                        scores_input = scores_input.unsqueeze(-1).unsqueeze(-1) # n,3,1,1
                        pred_score = self.score_generate(scores_input) # n,1,1,1
                        pred_score = pred_score.squeeze(1).squeeze(1) # n ,1
                        scores_weighted[condition_zero] = pred_score[...,0]

                    # print("scores_weighted is ", scores_weighted)
                    flow_cur = torch.zeros((center_points_cur.shape[0],  2)).to(center_points_cur.device) # n,2 流
                    # print("flow_cur shape is ", flow_cur.shape)

                    selected_box_3dcenter_cur = coord_cur['pred_box_center_tensor']# （n, 7） 然后下一行则是将3d bbx变为2d bbx （n, 7）-> （n, 8, 3）-> （n, 4, 3）
                    selected_box_3dcorner_cur = box_utils.boxes_to_corners2d(selected_box_3dcenter_cur, order='hwl') # TODO: box order should be a parameter  
                    # print("selected_box_3dcorner_cur shape is ", selected_box_3dcorner_cur.shape)
                    # xxx
                    if self.viz_flag:
                        flow_map, mask, single_ori_mask = self.generate_flow_map(flow_cur, selected_box_3dcorner_cur, scale=2.5, scores_weighted = scores_weighted, shape_list=shape_list)
                        original_reserved_mask.append(single_ori_mask)
                    else: # 返回的流场图，以及掩码， 流场图来源于坐标网格
                        flow_map, mask = self.generate_flow_map(flow_cur, selected_box_3dcorner_cur, scale=2.5, scores_weighted = scores_weighted, shape_list=shape_list) # 返回形状（1， H， W， 2）， （1， C， H， W）
                    flow_map_list.append(flow_map)
                    reserved_mask.append(mask)

            else: 
                # 以下开始线性外插，但是我发现，由于只有两帧，因此它必须考虑两帧完全匹配的情况，由于最终warp以及mask都是针对past0的伪图，因此要排除一下past0中有物体而past1中没有的情况
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1] # 前两帧 即往前 第0帧 第1帧

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N0， 2） 第0帧的N0个object的x y坐标
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N1， 2）

                u_cls_1 = torch.stack((coord_past1['u_cls_data'], coord_past1['u_cls_model']), dim=-1) # (N0, 2)                
                u_reg_1 = torch.stack((coord_past1['u_reg_data'], coord_past1['u_reg_model']), dim=-1) # (N0, 2)     
                u_cls_2 = torch.stack((coord_past2['u_cls_data'], coord_past2['u_cls_model']), dim=-1) # (N0, 2)                
                u_reg_2 = torch.stack((coord_past2['u_reg_data'], coord_past2['u_reg_model']), dim=-1) # (N0, 2)  
                # print("coord_past1['u_cls_data'] shape is ", coord_past1['u_cls_data'].shape)
                # print("u_cls_1 shape is ", u_cls_1.shape)
                # print("u_cls1_data raw is ", coord_past1['u_cls_data'])
                # print("u_cls1_model raw is ", coord_past1['u_cls_model'])
                # u_cls_1 = coord_past1['u_cls'] # past 0 帧中的分类不确定性 （N0，）
                # u_reg_1 = coord_past1['u_reg']
                # u_cls_2 = coord_past2['u_cls']
                # u_reg_2 = coord_past2['u_reg']

                cost_mat_center = torch.zeros((center_points_past2.shape[0], center_points_past1.shape[0])).to(center_points_past1.device) # 代价矩阵 （N1, N0） 初始化为全0

                center_points_past1_repeat = center_points_past1.unsqueeze(0).repeat(center_points_past2.shape[0], 1, 1) # （1， N0, 2）-> （N1, N0, 2）
                center_points_past2_repeat = center_points_past2.unsqueeze(1).repeat(1, center_points_past1.shape[0], 1) #  (N1, 1, 2) -> (N1, N0, 2)

                delta_mat = center_points_past1_repeat - center_points_past2_repeat # (N1, N0, 2) 这也就求得了第1帧中的第j个object 到第0帧中第i个object的差值

                angle_mat = torch.atan2(delta_mat[:,:,1], delta_mat[:,:,0]) # [num_cav_past2,num_cav_past1] 通过反正切值求弧度值，atan2还能够根据输入xy的正负来判断象限从而精确判断角度 这一步的目的是表示出两个不同时间点检测到的目标的相对角度

                coord_past2_angle_reverse = coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0]).clone() # 角度信息，（N1, N0）
                # 标记可视范围内的车辆  也就是在车辆轴向方向正负四十五度范围内
                visible_mat = torch.where((torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) < 0.785) | (torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1] 计算两两之间的成对距离作为成本矩阵 （N1， N0）

                visible_mat = torch.where(cost_mat_center<0.5, 1, visible_mat) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况

                tmp_thre = torch.tensor(1000.).to(torch.float32).to(visible_mat.device)
                cost_mat_center = torch.where(visible_mat==1, cost_mat_center, tmp_thre) # （N1， N0） 如果在可视范围（角度或者距离合适），则添加其欧氏距离，否则直接无穷（1000）

                if cost_mat_center.shape[1] == 0 or cost_mat_center.shape[0] == 0: # 检查是否有任一维度为空，如果有，意味着没有可匹配的目标，即没有两个目标之间的距离可计算
                    C, H, W = shape_list
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                    mask = torch.ones(1, C, H, W).to(shape_list)
                    flow_map_list.append(basic_warp_mat)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([torch.tensor([]), torch.tensor([])], dim=1).to(shape_list.device))
                        compensated_results_list.append(torch.zeros(0, 8, 3).to(shape_list.device))
                        original_reserved_mask.append(mask)
                    continue

                match = torch.min(cost_mat_center, dim=1) # 返回一个元组 （val， index） 第一个元素形状是（N1）表示每一行对应的最小值，第二个元素形状相同，表示每一行对应最小值的索引
                match_to_keep = torch.where(match[0] < 5) # 找到最小距离小于5m的并记录其index  注意，torch.where只传一个参数的时候返回的是一个元组，其中就一个元素，为一维张量，长度不固定 为0到N1-1 比如说(1, 3)表示第1个和第3个满足条件
    
                past2_ids = match_to_keep[0] # 提取第一个元素 一维张量  也就是最小距离小于5m的N1索引
                past1_ids = match[1][match_to_keep[0]] # 最小距离小于5m的对应的N0的索引 
                # 以下操作是重复以上过程，首先计算可视范围过滤，范围小于正负45°或者距离小于0.5m   注意，在计算相对角度时，并没有考虑到object自身的角度，所以相对角度要减去自身角度，但是自身角度有可能出现正负
                coord_past2_angle_reverse += 3.1415926 # 角度加上了pi
                coord_past2_angle_reverse[coord_past2_angle_reverse>3.1415926] -= 6.2831852 # 这两行是为了确保角度值保持在 -π 到 π 的范围内， 这一步也就是考虑了负的角度，那接下来就要重新来考虑一遍

                left_past2_id = [n for n in range(cost_mat_center.shape[0]) if n not in past2_ids] # 遍历N1次 0 到 N1-1 如果不在past2_ids索引列表里的就存在这里
                left_past1_id = [n for n in range(cost_mat_center.shape[1]) if n not in past1_ids] # 遍历N0次 0 到 N0-1 不满足条件的N0索引

                angle_mat_left = angle_mat[left_past2_id, :][:, left_past1_id] # 从（N1， N0）中选出来不满足条件的相对角度

                coord_past2_angle_reverse_left = coord_past2_angle_reverse[left_past2_id, :][: ,left_past1_id] #  从（N1， N0）中选出来不满足条件的Object自身角度
                # 这是筛选出这些不满足条件的object中的可见性矩阵
                visible_mat_left = torch.where((torch.abs(angle_mat_left-coord_past2_angle_reverse_left) < 0.785) | (torch.abs(angle_mat_left-coord_past2_angle_reverse_left) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]
                
                cost_mat_center_left = torch.cdist(center_points_past2[left_past2_id], center_points_past1[left_past1_id]) # [num_cav_past2,num_cav_past1] 计算两两之间的欧氏距离
                # cost_mat_center_left = cost_mat_center[left_past2_id, :][:, left_past1_id]

                visible_mat_left = torch.where(cost_mat_center_left<0.5, 1, visible_mat_left) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况
                cost_mat_center_left = torch.where(visible_mat_left==1, cost_mat_center_left, tmp_thre) # 如果满足可见性，则填充欧式距离，否则填写1000 （N1', N0'）

                if cost_mat_center_left.shape[1] != 0 and cost_mat_center_left.shape[0] != 0: # 如果有欧式距离可以算
                    match_left = torch.min(cost_mat_center_left, dim=1) # 找最小距离对应的 返回元组
                    match_to_keep_left = torch.where(match_left[0] < 5) # 寻找最小距离还小于5m的结果 N1索引，外面套了一个元组

                    if match_to_keep_left[0].shape[0] != 0: # 如果确实有值，也就是说确实有满足条件的
                        past2_ids_left = match_to_keep_left[0] # 满足要求的N1索引
                        past2_ids = torch.cat([past2_ids, torch.tensor(left_past2_id)[past2_ids_left].to(past2_ids.device)]) # 将第二轮挑选的加入
                        past1_ids_left = match_left[1][match_to_keep_left[0]] # 满足要求的N0索引
                        past1_ids = torch.cat([past1_ids, torch.tensor(left_past1_id)[past1_ids_left].to(past1_ids.device)]) # 到这里为止已经获得了N1，与N0的索引根据这两个可以查询到对应的object

                # ##############################################
                # self.thre_post_process = 10
                # original_cost_mat_center = cost_mat_center.clone()
                # cost_mat_center[cost_mat_center > self.thre_post_process] = 1000

                # cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                # dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                # cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                # dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                # cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                # cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # # cost_mat_iou = get_ious()
                # cost_mat = cost_mat_center
                # past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                # past2_ids = dist_valid_past2[0][past2_ids]
                # past1_ids = dist_valid_past1[0][past1_ids]
                
                # ### a trick
                # matched_cost = original_cost_mat_center[past2_ids, past1_ids]
                # valid_mat_idx = torch.where(matched_cost < self.thre_post_process)
                # past2_ids = past2_ids[valid_mat_idx[0]]
                # past1_ids = past1_ids[valid_mat_idx[0]]
                # ####################

                matched_past2 = center_points_past2[past2_ids] #  取出对应的object   （M , 2）
                matched_past1 = center_points_past1[past1_ids] #  取出对应的object   （M , 2）
                matched_u_cls_1 = u_cls_1[past1_ids] # past 0 帧中的分类不确定性 （M，2）
                matched_u_reg_1 = u_reg_1[past1_ids]
                matched_u_cls_2 = u_cls_2[past2_ids]
                matched_u_reg_2 = u_reg_2[past2_ids]

                # 需要距离？
                if self.use_distance:
                    C, H, W = shape_list # 64 ，H， W
                    trans_matrix = cav_content['matrix_past0_2_cur'] # 4,4
                    matched_3d = box_utils.boxes_to_corners_3d(coord_past1['pred_box_center_tensor'][past1_ids], order='hwl')
                    projected_boxes3d = box_utils.project_box3d(matched_3d, trans_matrix) # 将空间坐标投影cur ego （M, 8，3）
                    projected_center_past0 = box_utils.corner_to_center_torch(projected_boxes3d, order='hwl') # （M， 7）
                    projected_center_past0 = projected_center_past0[:, :2] # cur view 下对应的object位置
                    distance2detector = projected_center_past0 * 2.5 # 转移到体素距离 除以0.4
                    distance2detector = distance2detector.square().sum(dim=1).sqrt() # 得到距离ego检测器的距离
                    distance2detector = distance2detector / (torch.sqrt(H**2 + W**2) / 2) # 归一化

                # distance2detector = distance2detector.cpu().numpy().tolist()
                # self.distance_all += distance2detector
                # for id in remain_past1:


                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1] # 时间长度 这里的时间长度为T1-T2 也就是past0到past1的时间间隔 是个整数

                if time_length == 0:
                    time_length = 1

                # print("cav_content['past_k_time_diff'] is ", cav_content['past_k_time_diff'])
                # print("time_length is ", time_length)
                # xxx
                # 不确定性需要在线性插值的同时进行传播
                # 1、分类不确定性得分是通过分类偏差比来衡量，（M,）大于一定阈值：tp均值+std 取两帧object最大值，小于一定阈值：fp均值-std 取两帧object最小值，其余情况取两帧均值
                # 2、回归不确定性是通过方差并且标准化得到，（M，）
                scores_average = (cav_content[0]['scores'][past1_ids] + cav_content[1]['scores'][past2_ids]) / 2
                u_cls_average = (matched_u_cls_1 + matched_u_cls_2) / 2 # （M，2）
                u_reg_average = (matched_u_reg_1 + matched_u_reg_2) / 2

                scores_raw = scores_average
                u_cls_compensate = u_cls_average
                u_reg_compensate = u_reg_average
                # scores_raw = cav_content[0]['scores'][past1_ids]
                # u_cls_compensate = matched_u_cls_1
                # u_reg_compensate = matched_u_reg_1

                # print("u_cls_compensate shape is ", u_cls_compensate.shape)
                # print("u_reg_compensate shape is ", u_reg_compensate.shape)
                # print("u_cls_data is", u_cls_compensate[:,0])
                # print("u_cls_model is", u_cls_compensate[:,1])
                # print("u_reg_data is", u_reg_compensate[:,0])
                # print("u_reg_model is", u_reg_compensate[:,1])
                # print("scores_raw is", scores_raw)
                # xxx
                
                # 调整传播方式
                # if u_cls_average >= 0.7891:
                #     u_cls_compensate = torch.max(matched_u_cls_1, matched_u_cls_2)
                # elif u_cls_average <= 0.5114:
                #     u_cls_compensate = torch.min(matched_u_cls_1, matched_u_cls_2)
                # else:
                #     u_cls_compensate = u_cls_average

                # u_reg_compensate = u_reg_average + time_length * 0.01
                    


                # def print_requires_grad(module, prefix=''):
                #     for name, param in module.named_parameters(recurse=False):
                #         print(f"{prefix}Parameter {name} requires_grad: {param.requires_grad}")
                #     for name, sub_module in module.named_children():
                #         print_requires_grad(sub_module, prefix + '  ')

                # for idx, layer in enumerate(self.score_generate):
                #     print(f"Layer {idx} ({layer.__class__.__name__}):")
                #     print_requires_grad(layer, '  ')
                # print("scores_raw is ", scores_raw)
                # print("u_cls_compensate[:, 0] is ", u_cls_compensate[:, 0])
                # print("u_cls_compensate[:, 1] is ", u_cls_compensate[:, 1])
                # print("u_reg_compensate[:, 0] is ", u_reg_compensate[:, 0])
                # print("u_reg_compensate[:, 1] is ", u_reg_compensate[:, 1])
                scores_weighted = torch.zeros((scores_raw.shape[0], )).to(scores_raw.device) # (M,)
                condition = (scores_raw >= self.score_reliable) & (u_cls_compensate[:, 0] >= self.u_cls_data_reliable) & (u_cls_compensate[:, 1] >= self.u_cls_model_reliable) & (u_reg_compensate[:, 0] <= self.u_reg_data_reliable) & (u_reg_compensate[:, 1] <= self.u_reg_model_reliable)
                # dair-v2x
                # condition = (scores_raw > 0.6469) & (u_cls_compensate[:, 0] >= 0.8123) & (u_cls_compensate[:, 1] >= 0.8229) & (u_reg_compensate[:, 0] <= -0.5297) & (u_reg_compensate[:, 1] <= -0.3940)
                # v2xset
                # condition = (scores_raw > 0.6652) & (u_cls_compensate[:, 0] >= 0.7887) & (u_cls_compensate[:, 1] >= 0.8097) & (u_reg_compensate[:, 0] <= -0.2251) & (u_reg_compensate[:, 1] <= -0.1764)
                scores_weighted[condition] = 1.0
                # print("一开始将高可靠的直接选出来：", scores_weighted)
                condition_zero = (scores_weighted == 0)
                if condition_zero.sum() != 0:
                    scores_raw = scores_raw[condition_zero]
                    u_cls_compensate = u_cls_compensate[condition_zero, ...]
                    u_reg_compensate = u_reg_compensate[condition_zero, ...]
                    if self.use_distance:
                        distance2detector = distance2detector[condition_zero, ...] # object距离ego的距离

                        scores_input = torch.stack((distance2detector, scores_raw, u_cls_compensate[:, 0], u_cls_compensate[:, 1], u_reg_compensate[:, 0], u_reg_compensate[:, 1]), dim=-1)
                    else:
                        scores_input = torch.stack((scores_raw, u_cls_compensate[:, 0], u_cls_compensate[:, 1], u_reg_compensate[:, 0], u_reg_compensate[:, 1]), dim=-1)

                    scores_input = scores_input.unsqueeze(-1).unsqueeze(-1) # n,3,1,1
                    pred_score = self.score_generate(scores_input) # n,1,1,1
                    pred_score = pred_score.squeeze(1).squeeze(1) # n ,1
                    scores_weighted[condition_zero] = pred_score[...,0]
                    # scores_weighted = scores_weighted.squeeze(1).squeeze(1) # n ,1
                # 是否需要线性变换映射到0.2-1
                # scores_weighted = 0.5 * scores_weighted + 0.5
                # print("scores_weighted is ", scores_weighted)
                delay_factor = 0-cav_content['past_k_time_diff'][0]
                if self.score_reliable != 0.6469: # v2xset以及opv2v的时间是偶数，双倍，因此要除以二
                    delay_factor = torch.div(delay_factor, 2, rounding_mode='trunc')
                # print("delay_factor is ", delay_factor)
                delay_factor = torch.clamp(delay_factor, min=0.0)
                delay_factor = torch.exp(-0.02 * delay_factor) # 得分需要乘以一个e的-kt 随着延迟会衰减
                # delay_factor = 1 / (1 + 0.0044 * delay_factor**2)
                # print("delay_factor is ", delay_factor)
                # xx
                # delay_factor = torch.exp(-0.02 * delay_factor) # 得分需要乘以一个e的-kt 随着延迟会衰减


                # print("delay_factor is ", delay_factor)
                # print("scores_weighted is ", scores_weighted)

                scores_weighted = delay_factor * scores_weighted
                # print("after decay scores_weighted is ", scores_weighted)
                
                # end
                flow = (matched_past1 - matched_past2) / time_length # 两者想减得到距离，距离/时间为平均速度或者说“流速” (M, 2)

                flow = flow*(0-cav_content['past_k_time_diff'][0]) # 速度乘以距离得到预估的位移 past_k_time_diff中存放的是一个agent中对应帧与current帧的时间间隔 为 -τ， -τ-1，-τ-2.... 
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,] # 根据索引得到对应的bbx （M, 7） 然后下一行则是将3d bbx变为2d bbx （M, 7）-> （M, 8, 3）-> （M, 4, 3）
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl') # TODO: box order should be a parameter  

                if self.viz_flag:
                    matched_idx_list.append(torch.stack([past1_ids, past2_ids], dim=1))
                    selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                    selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 
                    compensated_results_list.append(selected_box_3dcorner_compensated)
                
                # 给一种稀有情况打补丁：即past0中检测到且置信度很高的物体，
                '''
                remain_past1_ids = [id for id in range(center_points_past1.shape[0]) if id not in past1_ids] # past0中匹配到了，然后这是选出剩余的部分
                # print('center_points_past1.shape[0] is ', center_points_past1.shape[0])
                # print('past1_ids is ', past1_ids)
                # print('remain_past1_ids is ', remain_past1_ids)
                remain_past1 = coord_past1['pred_box_center_tensor'][remain_past1_ids,] # (remains, 7)
                remain_past1_scores = cav_content[0]['scores'][remain_past1_ids] # (remains,)
                condition = (remain_past1_scores > 0.6653) & (u_reg_1[remain_past1_ids] <= -0.2271) & (u_cls_1[remain_past1_ids] >= 0.7891)
                indices_score_confirm = torch.where(condition)
                # print("选中的bbx其分类不确定性为: ", matched_u_cls_1)
                # print("选中的bbx其回归不确定性为: ", matched_u_reg_1)
                # print("剩余的bbx其分类不确定性为: ", u_cls_1[remain_past1_ids])
                # print("剩余的bbx其回归不确定性为: ", u_reg_1[remain_past1_ids])
                # print('indices_score_confirm is ', indices_score_confirm)
                # print('高置信度bbx的置信度为:  ', remain_past1_scores[indices_score_confirm])
                remain_past1_confirm = remain_past1[indices_score_confirm] # 这就筛选出past0中没有匹配成功但是置信度又很高的部分 （n, 7）
                if remain_past1_confirm.shape[0] != 0:
                    print("===fuck! boom====")
                    print("有高置信度bbx未匹配,数目为: ", remain_past1_confirm.shape[0])
                    print("高可信但未匹配的bbx的回归不确定性为: ", u_reg_1[remain_past1_ids][indices_score_confirm])
                    print("高可信但未匹配的bbx的分类不确定性为: ", u_cls_1[remain_past1_ids][indices_score_confirm])
                    zero_flow = torch.zeros((remain_past1_confirm.shape[0],  2)).to(center_points_past1.device) # n,2 流
                    one_weight = torch.ones((remain_past1_confirm.shape[0],  1)).to(center_points_past1.device)
                    remain_past1_confirm_3dcorner = box_utils.boxes_to_corners2d(remain_past1_confirm, order='hwl') # n,7 -> n, 4, 3

                    flow = torch.cat([flow, zero_flow], dim=0)
                    selected_box_3dcorner_past0 = torch.cat([selected_box_3dcorner_past0, remain_past1_confirm_3dcorner], dim=0)
                    scores_weighted = torch.cat([scores_weighted, one_weight], dim=0)
                    # scores_weighted = torch.ones_like(scores_weighted)
                '''

                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                if self.viz_flag:
                    flow_map, mask, single_ori_mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, scores_weighted = scores_weighted, shape_list=shape_list)
                    original_reserved_mask.append(single_ori_mask)
                else: # 返回的流场图，以及掩码， 流场图来源于坐标网格
                    flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, scores_weighted = scores_weighted, shape_list=shape_list) # 返回形状（1， H， W， 2）， （1， C， H， W）
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
                continue
        
        # distance_mean = np.mean(self.distance_all)
        # distance_std = np.std(self.distance_all)
        # distance_max = np.max(self.distance_all)
        # distance_min = np.min(self.distance_all)
        # print("distance len is ", len(self.distance_all))
        # print("distance_mean is ", distance_mean)
        # print("distance_std is ", distance_std)
        # print("distance_max is ", distance_max)
        # print("distance_min is ", distance_min)
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2] 其实就是坐标网格 ego的话就是恒等变换形成的坐标网格，而其他agent则是将流的运动属性运用到
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]

        if self.viz_flag:
            original_reserved_mask = torch.concat(original_reserved_mask, dim=0)  # [N_b, C, H, W]
            return final_flow_map, reserved_mask, original_reserved_mask, matched_idx_list, compensated_results_list
        return final_flow_map, reserved_mask
        '''
                updated_spatial_features_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow, scale=1.25)
                updated_spatial_features = self.feature_warp(features_dict[cav]['spatial_features'][0], selected_box_3dcorner_past0, flow, scale=2.5)

                # debug use
                if debug_flag and flow.shape[0] != 0:
                    torch.save(updated_spatial_features_2d, viz_save_path+'/updated_feature.pt')
                ############

            features_dict[cav].update({
                'updated_spatial_features_2d': updated_spatial_features_2d
            })
            features_dict[cav].update({
                'updated_spatial_features': updated_spatial_features
            })

        return features_dict
        '''
    def forward_flow_dir_backup(self, input_dict, shape_list):
        """
        Parameters:
        -----------
        input_dict : The dictionary containing the box detections on each frame of each cav.
            dict : { 
                cav_idx : {
                    'past_k_time_diff':
                    [0], [1], ..., [k-1] : {
                        pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                        pred_box_center_tensor : n, 7
                        scores: (n, )
                    }
                }
            }
        features_dict : The dictionary containing the output of the model.
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                }
            }

        Returns:
        --------
        features_dict :
            dict : {
                'ego' / cav_id : {
                    'spatial_features_2d'
                    'spatial_features'
                    'psm'
                    'rm'
                    'updated_spatial_features_2d'
                    'updated_spatial_features'
                }
            }
        """
        flow_map_list = []
        reserved_mask = []
        if self.viz_flag:
            original_reserved_mask = []
            matched_idx_list = []
            compensated_results_list = []
        for cav, cav_content in input_dict.items(): # 遍历每一辆车
            if cav == 0: # 场景下第一个agent必是ego
                # ego do not need warp
                C, H, W = shape_list # 64 ，H， W
                basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32) # 创建一个标准仿射变换矩阵，即不做变换
                basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                mask = torch.ones(1, C, H, W).to(shape_list) # 创建的是全为1的矩阵
                flow_map_list.append(basic_warp_mat) 
                reserved_mask.append(mask)
                if self.viz_flag:
                    original_reserved_mask.append(mask)
            else:
                past_k_time_diff = cav_content['past_k_time_diff'] # （k） 每一帧到cur的时间间隔
                # print("cav_content['past_k_time_diff'] shape is :", cav_content['past_k_time_diff'].shape)

                # if past_k_time_diff.shape[0] != 3:
                #     print('error past_k_time_diff shape error!:', past_k_time_diff)
                coord_past1 = cav_content[0]
                coord_past2 = cav_content[1] # 前两帧 即往前 第0帧 第1帧
                coord_past3 = cav_content[2] # 前两帧 即往前 第0帧 第1帧

                center_points_past1 = coord_past1['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N0， 2） 第0帧的N0个object的x y坐标 
                center_points_past2 = coord_past2['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N1， 2）
                center_points_past3 = coord_past3['pred_box_center_tensor'][:,:2] # 取出 x y 值 （N2， 2）

                # 首先是past1与past0匹配
                cost_mat_center = torch.zeros((center_points_past2.shape[0], center_points_past1.shape[0])).to(center_points_past1.device) # 代价矩阵 （N1, N0） 初始化为全0

                # 开始通过角度来设置可视范围
                center_points_past1_repeat = center_points_past1.unsqueeze(0).repeat(center_points_past2.shape[0], 1, 1) # （1， N0, 2）-> （N1, N0, 2）
                center_points_past2_repeat = center_points_past2.unsqueeze(1).repeat(1, center_points_past1.shape[0], 1) #  (N1, 1, 2) -> (N1, N0, 2)

                delta_mat = center_points_past1_repeat - center_points_past2_repeat # (N1, N0, 2) 这也就求得了第1帧中的第j个object 到第0帧中第i个object的差值

                angle_mat = torch.atan2(delta_mat[:,:,1], delta_mat[:,:,0]) # [num_cav_past2,num_cav_past1] 通过反正切值求弧度值，atan2还能够根据输入xy的正负来判断象限从而精确判断角度 这一步的目的是表示出两个不同时间点检测到的目标的相对角度

                coord_past2_angle_reverse = coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0]).clone() # 角度信息，（N1, N0）
                # 标记可视范围内的车辆  也就是在车辆轴向方向正负四十五度范围内 
                visible_mat = torch.where((torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) < 0.785) | (torch.abs(angle_mat-coord_past2['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1] 计算两两之间的成对欧式距离作为成本矩阵 （N1， N0）

                visible_mat = torch.where(cost_mat_center<0.5, 1, visible_mat) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况

                tmp_thre = torch.tensor(1000.).to(torch.float32).to(visible_mat.device)
                cost_mat_center = torch.where(visible_mat==1, cost_mat_center, tmp_thre) # （N1， N0） 如果在可视范围（角度或者距离合适），则添加其欧氏距离，否则直接无穷（1000）

                if cost_mat_center.shape[1] == 0 or cost_mat_center.shape[0] == 0: # 检查是否有任一维度为空，如果有，意味着没有可匹配的目标，即没有两个目标之间的距离可计算
                    C, H, W = shape_list
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                    mask = torch.ones(1, C, H, W).to(shape_list)
                    flow_map_list.append(basic_warp_mat)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([torch.tensor([]), torch.tensor([])], dim=1).to(shape_list.device))
                        compensated_results_list.append(torch.zeros(0, 8, 3).to(shape_list.device))
                        original_reserved_mask.append(mask)
                    continue

                match = torch.min(cost_mat_center, dim=1) # 返回一个元组 （val， index） 第一个元素形状是（N1）表示每一行对应的最小值，第二个元素形状相同，表示每一行对应最小值的列索引
                match_to_keep = torch.where(match[0] < 5) # 找到最小距离小于5m的并记录其index（0到N1-1 也就是说是past1的的匹配索引）  注意，torch.where只传一个参数的时候返回的是一个元组，其中就一个元素，为一维张量，长度不固定 为0到N1-1 比如说(1, 3)表示第1个和第3个满足条件
    
                past2_ids_a = match_to_keep[0] # 提取第一个元素 一维张量  也就是最小距离小于5m的N1索引
                past1_ids = match[1][match_to_keep[0]] # match[1]本来就是存储最小距离的past0索引，一共有N1个，表示每个past1和其对应的最小距离，这一步就是筛选最小距离小于5m的对应的N0的索引 
                # 以下操作是重复以上过程，首先计算可视范围过滤，范围小于正负45°或者距离小于0.5m   注意，在计算相对角度时，并没有考虑到object自身的角度，所以相对角度要减去自身角度，但是自身角度有可能出现正负
                coord_past2_angle_reverse += 3.1415926 # 角度加上了pi
                coord_past2_angle_reverse[coord_past2_angle_reverse>3.1415926] -= 6.2831852 # 这两行是为了确保角度值保持在 -π 到 π 的范围内， 这一步也就是考虑了负的角度，那接下来就要重新来考虑一遍

                left_past2_id = [n for n in range(cost_mat_center.shape[0]) if n not in past2_ids_a] # 遍历N1次 0 到 N1-1 如果不在past2_ids索引列表里的就存在这里
                left_past1_id = [n for n in range(cost_mat_center.shape[1]) if n not in past1_ids] # 遍历N0次 0 到 N0-1 不满足条件的N0索引

                angle_mat_left = angle_mat[left_past2_id, :][:, left_past1_id] # 从（N1， N0）中选出来不满足条件的相对角度

                coord_past2_angle_reverse_left = coord_past2_angle_reverse[left_past2_id, :][: ,left_past1_id] #  从（N1， N0）中选出来不满足条件的Object自身角度
                # 这是筛选出这些不满足条件的object中的可见性矩阵
                visible_mat_left = torch.where((torch.abs(angle_mat_left-coord_past2_angle_reverse_left) < 0.785) | (torch.abs(angle_mat_left-coord_past2_angle_reverse_left) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]
                
                cost_mat_center_left = torch.cdist(center_points_past2[left_past2_id], center_points_past1[left_past1_id]) # [num_cav_past2,num_cav_past1] 计算两两之间的欧氏距离
                # cost_mat_center_left = cost_mat_center[left_past2_id, :][:, left_past1_id]

                visible_mat_left = torch.where(cost_mat_center_left<0.5, 1, visible_mat_left) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况
                cost_mat_center_left = torch.where(visible_mat_left==1, cost_mat_center_left, tmp_thre) # 如果满足可见性，则填充欧式距离，否则填写1000 （N1', N0'）

                if cost_mat_center_left.shape[1] != 0 and cost_mat_center_left.shape[0] != 0: # 如果有欧式距离可以算
                    match_left = torch.min(cost_mat_center_left, dim=1) # 找最小距离对应的 返回元组
                    match_to_keep_left = torch.where(match_left[0] < 5) # 寻找最小距离还小于5m的结果 N1索引，外面套了一个元组

                    if match_to_keep_left[0].shape[0] != 0: # 如果确实有值，也就是说确实有满足条件的
                        past2_ids_left = match_to_keep_left[0] # 满足要求的N1索引
                        past2_ids_a = torch.cat([past2_ids_a, torch.tensor(left_past2_id)[past2_ids_left].to(past2_ids_a.device)]) # 将第二轮挑选的加入
                        past1_ids_left = match_left[1][match_to_keep_left[0]] # 满足要求的N0索引
                        past1_ids = torch.cat([past1_ids, torch.tensor(left_past1_id)[past1_ids_left].to(past1_ids.device)]) # 到这里为止已经获得了N1，与N0的索引根据这两个可以查询到对应的object


                # 再来past2与past0匹配
                cost_mat_center = torch.zeros((center_points_past3.shape[0], center_points_past2.shape[0])).to(center_points_past2.device) # 代价矩阵 （N2, N1） 初始化为全0

                # 开始通过角度来设置可视范围
                center_points_past2_repeat = center_points_past2.unsqueeze(0).repeat(center_points_past3.shape[0], 1, 1) # （1， N1, 2）-> （N2, N1, 2）
                center_points_past3_repeat = center_points_past3.unsqueeze(1).repeat(1, center_points_past2.shape[0], 1) #  (N2, 1, 2) -> (N2, N1, 2)

                delta_mat = center_points_past2_repeat - center_points_past3_repeat # (N2, N1, 2) 这也就求得了第1帧中的第j个object 到第0帧中第i个object的差值

                angle_mat = torch.atan2(delta_mat[:,:,1], delta_mat[:,:,0]) # [num_cav_past3,num_cav_past2] 通过反正切值求弧度值，atan2还能够根据输入xy的正负来判断象限从而精确判断角度 这一步的目的是表示出两个不同时间点检测到的目标的相对角度

                coord_past3_angle_reverse = coord_past3['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past2.shape[0]).clone() # 角度信息，（N1, N0）
                # 标记可视范围内的车辆  也就是在车辆轴向方向正负四十五度范围内 
                visible_mat = torch.where((torch.abs(angle_mat-coord_past3['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past2.shape[0])) < 0.785) | (torch.abs(angle_mat-coord_past3['pred_box_center_tensor'][:,6].unsqueeze(1).repeat(1, center_points_past2.shape[0])) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]

                cost_mat_center = torch.cdist(center_points_past3, center_points_past2) # [num_cav_past3,num_cav_past2] 计算两两之间的成对欧式距离作为成本矩阵 （N2， N1）

                visible_mat = torch.where(cost_mat_center<0.5, 1, visible_mat) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况

                tmp_thre = torch.tensor(1000.).to(torch.float32).to(visible_mat.device)
                cost_mat_center = torch.where(visible_mat==1, cost_mat_center, tmp_thre) # （N2， N1） 如果在可视范围（角度或者距离合适），则添加其欧氏距离，否则直接无穷（1000）

                if cost_mat_center.shape[1] == 0 or cost_mat_center.shape[0] == 0: # 检查是否有任一维度为空，如果有，意味着没有可匹配的目标，即没有两个目标之间的距离可计算
                    C, H, W = shape_list
                    basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
                    basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=False).to(shape_list.device)
                    mask = torch.ones(1, C, H, W).to(shape_list)
                    flow_map_list.append(basic_warp_mat)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([torch.tensor([]), torch.tensor([])], dim=1).to(shape_list.device))
                        compensated_results_list.append(torch.zeros(0, 8, 3).to(shape_list.device))
                        original_reserved_mask.append(mask)
                    continue

                match = torch.min(cost_mat_center, dim=1) # 返回一个元组 （val， index） 第一个元素形状是（N2）表示每一行对应的最小值，第二个元素形状相同，表示每一行对应最小值的列索引
                match_to_keep = torch.where(match[0] < 5) # 找到最小距离小于5m的并记录其index（0到N2-1 也就是说是past2的的匹配索引）  注意，torch.where只传一个参数的时候返回的是一个元组，其中就一个元素，为一维张量，长度不固定 为0到N1-1 比如说(1, 3)表示第1个和第3个满足条件
    
                past3_ids = match_to_keep[0] # 提取第一个元素 一维张量  也就是最小距离小于5m的N1索引
                past2_ids_b = match[1][match_to_keep[0]] # match[1]本来就是存储最小距离的past1索引，一共有N2个，表示每个past2和其对应的最小距离，这一步就是筛选最小距离小于5m的对应的N2的索引 
                # 以下操作是重复以上过程，首先计算可视范围过滤，范围小于正负45°或者距离小于0.5m   注意，在计算相对角度时，并没有考虑到object自身的角度，所以相对角度要减去自身角度，但是自身角度有可能出现正负
                coord_past3_angle_reverse += 3.1415926 # 角度加上了pi
                coord_past3_angle_reverse[coord_past3_angle_reverse>3.1415926] -= 6.2831852 # 这两行是为了确保角度值保持在 -π 到 π 的范围内， 这一步也就是考虑了负的角度，那接下来就要重新来考虑一遍

                left_past3_id = [n for n in range(cost_mat_center.shape[0]) if n not in past3_ids] # 遍历N2次 0 到 N2-1 如果不在past3_ids索引列表里的就存在这里
                left_past2_id = [n for n in range(cost_mat_center.shape[1]) if n not in past2_ids_b] # 遍历N1次 0 到 N1-1 不满足条件的N1索引

                angle_mat_left = angle_mat[left_past3_id, :][:, left_past2_id] # 从（N2， N1）中选出来不满足条件的相对角度

                coord_past3_angle_reverse_left = coord_past3_angle_reverse[left_past3_id, :][: ,left_past2_id] #  从（N2， N1）中选出来不满足条件的Object自身角度
                # 这是筛选出这些不满足条件的object中的可见性矩阵
                visible_mat_left = torch.where((torch.abs(angle_mat_left-coord_past3_angle_reverse_left) < 0.785) | (torch.abs(angle_mat_left-coord_past3_angle_reverse_left) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]
                
                cost_mat_center_left = torch.cdist(center_points_past3[left_past3_id], center_points_past2[left_past2_id]) # [num_cav_past3,num_cav_past2] 计算两两之间的欧氏距离
                # cost_mat_center_left = cost_mat_center[left_past2_id, :][:, left_past1_id]

                visible_mat_left = torch.where(cost_mat_center_left<0.5, 1, visible_mat_left) # 使得当两个目标非常接近时（距离小于0.5），无论它们之间的角度如何，它们都被认为是彼此可见的。这种方法可能用于处理那些距离很近但由于角度差异而在原始角度条件下可能不被认为是可见的情况
                cost_mat_center_left = torch.where(visible_mat_left==1, cost_mat_center_left, tmp_thre) # 如果满足可见性，则填充欧式距离，否则填写1000 （N1', N0'）

                if cost_mat_center_left.shape[1] != 0 and cost_mat_center_left.shape[0] != 0: # 如果有欧式距离可以算
                    match_left = torch.min(cost_mat_center_left, dim=1) # 找最小距离对应的 返回元组
                    match_to_keep_left = torch.where(match_left[0] < 5) # 寻找最小距离还小于5m的结果 N1索引，外面套了一个元组

                    if match_to_keep_left[0].shape[0] != 0: # 如果确实有值，也就是说确实有满足条件的
                        past3_ids_left = match_to_keep_left[0] # 满足要求的N1索引
                        past3_ids = torch.cat([past3_ids, torch.tensor(left_past3_id)[past3_ids_left].to(past3_ids.device)]) # 将第二轮挑选的加入
                        past2_ids_left = match_left[1][match_to_keep_left[0]] # 满足要求的N0索引
                        past2_ids_b = torch.cat([past2_ids_b, torch.tensor(left_past2_id)[past2_ids_left].to(past2_ids_b.device)]) # 到这里为止已经获得了N1，与N0的索引根据这两个可以查询到对应的object
                
                # find the matched obj among three frames
                a_idx, b_idx = self.get_common_elements(past2_ids_a, past2_ids_b) # 通过past2匹配的前后两帧结果，找三帧之间的相同部分

                # 最近两帧有匹配的，但是最近的三帧没有 这种情况下会用past0和past1的平均速度乘上延迟时间得到预估的偏移
                # there is no matched object in past frames
                if len(a_idx)==0 or len(b_idx)==0:
                    matched_past2 = center_points_past2[past2_ids_a]
                    matched_past1 = center_points_past1[past1_ids]

                    time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                    if time_length == 0:
                        time_length = 1
                    flow = (matched_past1 - matched_past2) / time_length

                    flow = flow*(0-cav_content['past_k_time_diff'][0])
                    selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,]
                    selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl')
                    if self.viz_flag:
                        flow_map, mask, single_ori_mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                        original_reserved_mask.append(single_ori_mask)
                    else:
                        flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                    flow_map_list.append(flow_map)
                    reserved_mask.append(mask)
                    if self.viz_flag:
                        matched_idx_list.append(torch.stack([past1_ids, past2_ids_a], dim=1)) # (N_obj, 2) 2帧中所有object的id
                        selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                        selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 
                        compensated_results_list.append(selected_box_3dcorner_compensated) # （N, 4, 2）
                    continue

                center_points_past1_recover = coord_past1['pred_box_center_tensor'][:,[0,1,6]] # 取出 x y 值 （N0， 3） 第0帧的N0个object的x y坐标 heading
                center_points_past2_recover = coord_past2['pred_box_center_tensor'][:,[0,1,6]] # 取出 x y 值 （N1， 3）
                center_points_past3_recover = coord_past3['pred_box_center_tensor'][:,[0,1,6]] # 取出 x y 值 （N2， 3）
                # 三帧匹配结果输入预测模块
                matched_past1 = center_points_past1_recover[past1_ids[a_idx]].unsqueeze(0) # 由于这是匹配成功的三帧 所以个数是一样的 （1， N， 3）
                matched_past2 = center_points_past2_recover[past2_ids_a[a_idx]].unsqueeze(0)
                matched_past3 = center_points_past3_recover[past3_ids[b_idx]].unsqueeze(0)
                obj_coords = torch.cat([matched_past3, matched_past2, matched_past1], dim=0) # （3， N， 3）
                obj_coords = obj_coords.permute(1, 0, 2) # (N, k, 3) 记录着K帧N个object的x/y坐标 倒着放也就是第三帧 第二帧 第一帧
                obj_input = obj_coords.unsqueeze(0) # (1, N, k, 3)

                past_k_time_diff = torch.flip(past_k_time_diff, dims=[0]) # (k,)
                past_k_time_diff = past_k_time_diff.unsqueeze(0).repeat(obj_input.shape[1], 1) # (N, k)

                query = torch.zeros((1, past_k_time_diff.shape[0], 1, 3)).to(obj_input.device) # (1, N, 1, 3)
                future_time = torch.zeros((1, past_k_time_diff.shape[0], 1)).to(obj_input.device) # (1, N, 1)
                # print('obj_input shape  is ',obj_input.shape)
                # print('query shape  is ',query.shape)
                # print('past_k_time_diff shape  is ',past_k_time_diff.shape)
                # print('future_time shape  is ',future_time.shape)

                predictions = self.compensate_motion(obj_input, query, past_k_time_diff, future_time) # (1, N, 1, 3)

                predictions = predictions.squeeze(0).squeeze(1) # (N, 3)

                center_points_past1_recover = coord_past1['pred_box_center_tensor'][:,[2,3,4,5]] # 取出 z，dx dy dz 值 （N0， 4）
                matched_past1 = center_points_past1_recover[past1_ids[a_idx]] # 由于这是匹配成功的三帧 所以个数是一样的 （N， 4）
                predictions = torch.cat([predictions, matched_past1], dim=1) # (N, 7) 恢复bbx center格式

                # print('matched_past1 shape  is ',matched_past1.shape)
                # print('predictions shape  is ',predictions.shape)

                predictions = predictions[:, [0,1,3,4,5,6,2]]

                center_points_past1_recover = coord_past1['pred_box_center_tensor'] #  （N0， 7）
                matched_past1 = center_points_past1_recover[past1_ids[a_idx]] # 由于这是匹配成功的三帧 所以个数是一样的 （N， 7）
                flow = predictions[:, :2] - matched_past1[:, :2] # （N， 2）
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(matched_past1, order='hwl') # (N , 4, 3)

                if self.viz_flag and not(len(a_idx) < len(past2_ids_a)): # past0和past1的匹配数大于总匹配数，那说明有遗漏未匹配
                    unit_matched_list = torch.stack([past1_ids[a_idx], past2_ids_a[a_idx], past3_ids[b_idx]], dim=1) # (N_obj, 3) 三帧中所有object的id
                    selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                    selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1)  # past0的所有object施加flow后的补偿结果 （N, 4, 2）

                # 两帧匹配成功 but三帧匹配失败 将两帧的结果进行插值 com 表示补集
                # 这时候要插入是因为显然past2帧有没检测到的，而past1 past0有匹配上的，这时直接用平均速度来计算其flow 一旦出现这个情况，那在可视化的时候也只会标记两帧
                if len(a_idx) < len(past2_ids_a): 
                    com_past1_ids = [elem.item() for id, elem in enumerate(past1_ids) if id not in a_idx] # 找出past1 past0匹配上而past2中未被匹配上的bbx
                    com_past2_ids = [elem.item() for id, elem in enumerate(past2_ids_a) if id not in a_idx]
                    matched_past1 = center_points_past1[com_past1_ids]
                    matched_past2 = center_points_past2[com_past2_ids]
                    
                    time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1]
                    if time_length == 0:
                        time_length = 1
                    com_flow = (matched_past1 - matched_past2) / time_length

                    com_flow = com_flow*(0-cav_content['past_k_time_diff'][0]) # (n_com, 2)
                    com_selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][com_past1_ids,]
                    com_selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(com_selected_box_3dcenter_past0, order='hwl')
                    
                    flow = torch.cat([flow, com_flow], dim=0)
                    selected_box_3dcorner_past0 = torch.cat([selected_box_3dcorner_past0, com_selected_box_3dcorner_past0], dim=0)

                    # matched: 
                    # past1: past1_ids[a_idx] + com_past1_ids
                    # past2: past2_ids_a[a_idx] + com_past2_ids
                    # past3: past3_ids[b_idx]
                    if self.viz_flag:
                        tmp_past_1 = torch.cat([past1_ids[a_idx], torch.tensor(com_past1_ids).to(past1_ids)], dim=0)
                        tmp_past_2 = torch.cat([past2_ids_a[a_idx], torch.tensor(com_past2_ids).to(past1_ids)], dim=0)
                        unit_matched_list = torch.stack([tmp_past_1, tmp_past_2], dim=1)  # (N_obj, 2)
                        selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                        selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 

                if self.viz_flag:
                    matched_idx_list.append(unit_matched_list) # List 元素有 (N_obj, 3)  也有 (N_obj, 2) 前者 表示三帧匹配成功，后者 表示存在不完全匹配问题，只显示最近两帧
                    compensated_results_list.append(selected_box_3dcorner_compensated)

                if self.viz_flag:
                    flow_map, mask, single_ori_mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                    original_reserved_mask.append(single_ori_mask)
                else: # 返回的流场图，以及掩码， 流场图来源于坐标网格
                    flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list) # 返回形状（1， H， W， 2）， （1， C， H， W）
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2] 其实就是坐标网格 ego的话就是恒等变换形成的坐标网格，而其他agent则是将流的运动属性运用到
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]

        if self.viz_flag:
            original_reserved_mask = torch.concat(original_reserved_mask, dim=0)  # [N_b, C, H, W] 标记了所有的object
            return final_flow_map, reserved_mask, original_reserved_mask, matched_idx_list, compensated_results_list
        return final_flow_map, reserved_mask
        '''   
    
                # ##############################################
                # self.thre_post_process = 10
                # original_cost_mat_center = cost_mat_center.clone()
                # cost_mat_center[cost_mat_center > self.thre_post_process] = 1000

                # cost_mat_center_drop_2 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=1)
                # dist_valid_past2 = torch.where(cost_mat_center_drop_2 < center_points_past1.shape[0])
                # cost_mat_center_drop_1 = torch.sum(torch.where(cost_mat_center > self.thre, 1, 0), dim=0)
                # dist_valid_past1 = torch.where(cost_mat_center_drop_1 < center_points_past2.shape[0])

                # cost_mat_center = cost_mat_center[dist_valid_past2[0], :]
                # cost_mat_center = cost_mat_center[:, dist_valid_past1[0]]
                
                # # cost_mat_iou = get_ious()
                # cost_mat = cost_mat_center
                # past2_ids, past1_ids = linear_sum_assignment(cost_mat.cpu())
                
                # past2_ids = dist_valid_past2[0][past2_ids]
                # past1_ids = dist_valid_past1[0][past1_ids]
                
                # ### a trick
                # matched_cost = original_cost_mat_center[past2_ids, past1_ids]
                # valid_mat_idx = torch.where(matched_cost < self.thre_post_process)
                # past2_ids = past2_ids[valid_mat_idx[0]]
                # past1_ids = past1_ids[valid_mat_idx[0]]
                # ####################

                matched_past2 = center_points_past2[past2_ids] #  取出对应的object   （M , 2）
                matched_past1 = center_points_past1[past1_ids] #  取出对应的object   （M , 2）
                # print(cav_content['past_k_time_diff'])
                # print(cav)
                # print(cav_content)
                time_length = cav_content['past_k_time_diff'][0] - cav_content['past_k_time_diff'][1] # 时间长度 这里的时间长度为T1-T2 也就是past0到past1的时间间隔 是个整数
                # exit9

                if time_length == 0:
                    time_length = 1
                flow = (matched_past1 - matched_past2) / time_length # 两者想减得到距离，距离/时间为平均速度或者说“流速” (M, 2)

                flow = flow*(0-cav_content['past_k_time_diff'][0]) # 速度乘以距离得到预估的位移 past_k_time_diff中存放的是一个agent中对应帧与current帧的时间间隔 为 -τ， -τ-1，-τ-2.... 
                selected_box_3dcenter_past0 = coord_past1['pred_box_center_tensor'][past1_ids,] # 根据索引得到对应的bbx （M, 7） 然后下一行则是将3d bbx变为2d bbx （M, 7）-> （M, 8, 3）-> （M, 4, 3）
                selected_box_3dcorner_past0 = box_utils.boxes_to_corners2d(selected_box_3dcenter_past0, order='hwl') # TODO: box order should be a parameter  

                if self.viz_flag:
                    matched_idx_list.append(torch.stack([past1_ids, past2_ids], dim=1))
                    selected_box_3dcorner_compensated = selected_box_3dcorner_past0.clone()
                    selected_box_3dcorner_compensated[:, :, :-1] += flow.unsqueeze(1).repeat(1, 4, 1) 
                    compensated_results_list.append(selected_box_3dcorner_compensated)
                
                # debug use
                debug_flag = False
                if debug_flag and flow.shape[0] != 0:
                    viz_save_path = '/DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/vis_debug'
                    torch.save(features_dict[cav]['spatial_features_2d'][0], viz_save_path+'/features_2d.pt')
                    torch.save(features_dict[cav]['spatial_features'][0], viz_save_path+'/features.pt')
                    torch.save(selected_box_3dcorner_past0, viz_save_path+'/bbx_list.pt')
                    torch.save(flow, viz_save_path+'/flow.pt')
                    print(f"===saved, max flow is {flow.max()}===")
                ############
                
                if self.viz_flag:
                    flow_map, mask, single_ori_mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list)
                    original_reserved_mask.append(single_ori_mask)
                else: # 返回的流场图，以及掩码， 流场图来源于坐标网格
                    flow_map, mask = self.generate_flow_map(flow, selected_box_3dcorner_past0, scale=2.5, shape_list=shape_list) # 返回形状（1， H， W， 2）， （1， C， H， W）
                flow_map_list.append(flow_map)
                reserved_mask.append(mask)
                continue
        
        final_flow_map = torch.concat(flow_map_list, dim=0) # [N_b, H, W, 2] 其实就是坐标网格 ego的话就是恒等变换形成的坐标网格，而其他agent则是将流的运动属性运用到
        reserved_mask = torch.concat(reserved_mask, dim=0)  # [N_b, C, H, W]

        if self.viz_flag:
            original_reserved_mask = torch.concat(original_reserved_mask, dim=0)  # [N_b, C, H, W]
            return final_flow_map, reserved_mask, original_reserved_mask, matched_idx_list, compensated_results_list
        return final_flow_map, reserved_mask
        '''

        '''
                updated_spatial_features_2d = self.feature_warp(features_dict[cav]['spatial_features_2d'][0], selected_box_3dcorner_past0, flow, scale=1.25)
                updated_spatial_features = self.feature_warp(features_dict[cav]['spatial_features'][0], selected_box_3dcorner_past0, flow, scale=2.5)

                # debug use
                if debug_flag and flow.shape[0] != 0:
                    torch.save(updated_spatial_features_2d, viz_save_path+'/updated_feature.pt')
                ############

            features_dict[cav].update({
                'updated_spatial_features_2d': updated_spatial_features_2d
            })
            features_dict[cav].update({
                'updated_spatial_features': updated_spatial_features
            })

        return features_dict
        '''

    def generate_flow_map_ego(self, flow, bbox_list, scale=1.25, scores_weighted=None, shape_list=None, align_corners=False, file_suffix=""):
        """
        Parameters
        -----------
        feature: [C, H, W] at voxel scale
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

        C, H, W = shape_list
        num_cav = bbox_list.shape[0] # M
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(shape_list.device) # （1, H, W, 2）
        reserved_area = torch.ones((C, H, W)).to(shape_list.device)  # C, H, W  全0张量 修改为全1张量 因为没有flow如果掩码设置为全0，那会将原来的特征图全部掩蔽，但是可能是匹配失败或者ROI Generator的问题
        if flow.shape[0] == 0 : 
            reserved_area = torch.ones((C, H, W)).to(shape_list.device)  # C, H, W 
            if self.viz_flag:
                return basic_warp_mat,  reserved_area.unsqueeze(0), reserved_area.unsqueeze(0)  # 返回不变的矩阵
            return basic_warp_mat,  reserved_area.unsqueeze(0)  # 返回不变的矩阵

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1 
        ------------
        '''
        flow_clone = flow.detach().clone() # 拷贝  （M, 2）

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1) # （M, 3, 3）设置恒等变换矩阵
        flow_clone = -2 * flow_clone / torch.tensor([W, H]).to(torch.float32).to(shape_list.device) # 类似于归一化，约束到[-1, 1]区间，这是因为 F.affine_grid 需要这样的范围的数值
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
        # print("cav_t_mat", cav_t_mat)

        cav_warp_mat = F.affine_grid(cav_t_mat, # 构建坐标网格 (M , H, W, 2)
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(shape_list.device) # .to() 统一数据格式 float32
        
        flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2  # 第0帧的object 的 x，y值 加上 flow 中运动信息 （M， 4 ， 2）

        x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1 # 最小x的值，缩小1应该是为了给边界框提供一点额外空间 (M)
        x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
        y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
        y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
        x_min_fid = (x_min + int(W/2)).to(torch.int) # 这是将边界框坐标从以图像中心为原点转换为以图像左上角为原点的坐标系 (M)
        x_max_fid = (x_max + int(W/2)).to(torch.int)
        y_min_fid = (y_min + int(H/2)).to(torch.int)
        y_max_fid = (y_max + int(H/2)).to(torch.int)

        for cav in range(num_cav): # 遍历每一辆agent
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
            reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 1 # object区域归零
        # set warped location as 1
        for cav in range(num_cav):
            if scores_weighted is not None:
                scores_weighted_cav = scores_weighted[cav]
                reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = scores_weighted_cav # warped的object边界框区域置位1  （C，H， W）
            else:
                reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1 # warped的object边界框区域置位1  （C，H， W）

        return basic_warp_mat, reserved_area.unsqueeze(0) # 返回形状（1， H， W， 2）， （1， C， H， W）

    def generate_flow_map(self, flow, bbox_list, scale=1.25, scores_weighted=None, shape_list=None, align_corners=False, file_suffix=""):
        """
        Parameters
        -----------
        feature: [C, H, W] at voxel scale
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
        num_cav = bbox_list.shape[0] # M
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(shape_list.device) # （1, H, W, 2）
        reserved_area = torch.zeros((C, H, W)).to(shape_list.device)  # C, H, W  全0张量 修改为全1张量 因为没有flow如果掩码设置为全0，那会将原来的特征图全部掩蔽，但是可能是匹配失败或者ROI Generator的问题
        if flow.shape[0] == 0 : 
            reserved_area = torch.ones((C, H, W)).to(shape_list.device)  # C, H, W 
            if self.viz_flag:
                return basic_warp_mat,  reserved_area.unsqueeze(0), reserved_area.unsqueeze(0)  # 返回不变的矩阵
            return basic_warp_mat,  reserved_area.unsqueeze(0)  # 返回不变的矩阵

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1 
        ------------
        '''
        flow_clone = flow.detach().clone() # 拷贝  （M, 2）

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1) # （M, 3, 3）设置恒等变换矩阵
        flow_clone = -2 * flow_clone / torch.tensor([W, H]).to(torch.float32).to(shape_list.device) # 类似于归一化，约束到[-1, 1]区间，这是因为 F.affine_grid 需要这样的范围的数值，此外，负号的原因是要构建逆映射
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
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

        for cav in range(num_cav): # 遍历每一辆agent
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
            if scores_weighted is not None:
                scores_weighted_cav = scores_weighted[cav]
                reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = scores_weighted_cav # warped的object边界框区域置位1  （C，H， W）
            else:
                reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1 # warped的object边界框区域置位1  （C，H， W）

        if self.viz_flag:
            single_reserved_area = torch.zeros_like(reserved_area) # （C，H， W）
            for cav in range(num_cav):
                single_reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 1 # object部分标记
            return basic_warp_mat, reserved_area.unsqueeze(0), single_reserved_area.unsqueeze(0)

        return basic_warp_mat, reserved_area.unsqueeze(0) # 返回形状（1， H， W， 2）， （1， C， H， W）
        '''
        ##################################### below is not used
        final_feature = F.grid_sample(feature.unsqueeze(0), basic_warp_mat, align_corners=align_corners)[0]
        
        ####### viz-2: warped feature, flowed box and warped 
        if flag_viz:
            p_0 = torch.stack((x_min, y_min), dim=1).to(torch.int)
            p_1 = torch.stack((x_min, y_max), dim=1).to(torch.int)
            p_2 = torch.stack((x_max, y_max), dim=1).to(torch.int)
            p_3 = torch.stack((x_max, y_min), dim=1).to(torch.int)
            warp_area_bbox_list = torch.stack((p_0, p_1, p_2, p_3), dim=1)
            viz_bbx_list = torch.cat((flowed_bbx_list, warp_area_bbox_list), dim=0)
            canvas_new = viz_on_canvas(final_feature, viz_bbx_list, scale=scale)
            plt.sca(ax[2]) 
            # plt.axis("off") 
            plt.imshow(canvas_new.canvas)
        ############## 

        reserved_area = torch.ones_like(feature)  # C, H, W
        x_min_ori = torch.min(bbox_list[:,:,0],dim=1)[0]
        x_max_ori = torch.max(bbox_list[:,:,0],dim=1)[0]
        y_min_ori = torch.min(bbox_list[:,:,1],dim=1)[0]
        y_max_ori = torch.max(bbox_list[:,:,1],dim=1)[0]
        x_min_fid_ori = (x_min_ori + int(W/2)).to(torch.int)
        x_max_fid_ori = (x_max_ori + int(W/2)).to(torch.int)
        y_min_fid_ori = (y_min_ori + int(H/2)).to(torch.int)
        y_max_fid_ori = (y_max_ori + int(H/2)).to(torch.int)
        # set original location as 0
        for cav in range(num_cav):
            reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 0
        # set warped location as 1
        for cav in range(num_cav):
            reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
        final_feature = final_feature * reserved_area

        ####### viz-3: mask area out of warped bbx
        if flag_viz:
            partial_feature_one = torch.zeros_like(feature)  # C, H, W
            for cav in range(num_cav):
                partial_feature_one[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
            masked_final_feature = partial_feature_one * final_feature
            canvas_hidden = viz_on_canvas(masked_final_feature, warp_area_bbox_list, scale=scale)
            plt.sca(ax[3]) 
            # plt.axis("off") 
            plt.imshow(canvas_hidden.canvas)
        ##############

        ####### viz: draw figures
        if flag_viz:
            plt.tight_layout()
            plt.savefig(f'result_canvas_{file_suffix}.jpg', transparent=False, dpi=400)
            plt.clf()

            fig, axes = plt.subplots(2, 1, figsize=(4, 4))
            major_ticks_x = np.linspace(0,350,8)
            minor_ticks_x = np.linspace(0,350,15)
            major_ticks_y = np.linspace(0,100,3)
            minor_ticks_y = np.linspace(0,100,5)
            for i, ax in enumerate(axes):
                plt.sca(ax); #plt.axis("off")
                ax.set_xticks(major_ticks_x); ax.set_xticks(minor_ticks_x, minor=True)
                ax.set_yticks(major_ticks_y); ax.set_yticks(minor_ticks_y, minor=True)
                ax.grid(which='major', color='w', linewidth=0.4)
                ax.grid(which='minor', color='w', linewidth=0.2, alpha=0.5)
                if i==0:
                    plt.imshow(torch.max(feature, dim=0)[0].cpu())
                else:
                    plt.imshow(torch.max(final_feature, dim=0)[0].cpu())
            plt.tight_layout()
            plt.savefig(f'result_features_{file_suffix}.jpg', transparent=False, dpi=400)
            plt.clf()
        #######

        return final_feature
        '''

    # return shape: (H, W)
    def create_weighted_mask(self, height, width, center_x, center_y, sigma, k=0.1, n=4):
        # 平滑函数生成mask
        # import numpy as np
        # x = np.arange(0, width, 1, float)
        # y = np.arange(0, height, 1, float)
        # y = y[:, np.newaxis]
        # x0 = center_x
        # y0 = center_y
        # distance_square = (x - x0)**2 + (y - y0)**2
        # distance = np.sqrt(distance_square)
        # mask = np.where(distance <= sigma, 
        #                 1.0, 
        #                 np.exp(-((distance - sigma) / (k * sigma))**n))
        # return torch.from_numpy(mask).float()
        grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid_x = grid_x.float() # (H, W)
        grid_y = grid_y.float()
        distance = ((grid_x - center_x)**2 + (grid_y - center_y)**2).sqrt() # (H,W)
        mask = torch.where(distance <= sigma, 
                        1.0, 
                        torch.exp(-((distance - sigma) / (k * sigma))**n))
        return mask # (H,W)

    def create_smooth_warp_grid(self, center, radius, flow, feature_map_size, n=4, k=0.1):
        """
        创建一个平滑warp网格, 用于将特征图的像素平滑地位移到新的位置。

        参数:注意，因为有坐标网格，因此以下所有坐标描述都要是以左上角为原点
        center (tuple): 目标物体的中心点 (M, x_c, y_c)。注意, 这里必须要是flowed后的结果, 因为grid是要求一个逆映射
        radius (float): 用于计算平滑mask的半径, 一般为bbx对角线的一半。
        flow (torch.Tensor): 目标物体的位移 (M, 2)。
        feature_map_size (tuple): 特征图的尺寸 (height, width)。

        返回:
        grid (torch.Tensor): 平滑warp网格, 用于`torch.nn.functional.grid_sample`函数。
                            形状为 (height, width, 2)。
        """

        # 获取特征图的高度和宽度
        c, h, w = feature_map_size

        # 创建网格坐标，grid_x和grid_y形状均为 (height, width)
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid_x = grid_x.float().to(flow.device)  # 转换为浮点数并迁移到相同设备 （H，W)
        grid_y = grid_y.float().to(flow.device)  # 转换为浮点数并迁移到相同设备 （H，W)

        # 初始化平滑warp网格
        delta_x = torch.zeros_like(grid_x) # （H，W)
        delta_y = torch.zeros_like(grid_y) # （H，W)
        weight_mask = torch.zeros((h, w)).to(flow.device)

        # 遍历每个agent的flow，应用平滑位移
        for i in range(flow.shape[0]):
            center_cav = center[i]

            # 计算每个像素点到中心点的距离，distance形状为 (height, width)
            distance = ((grid_x - center_cav[0])**2 + (grid_y - center_cav[1])**2).sqrt()

            # 计算平滑mask，mask形状为 (height, width)
            sigma = radius
            mask = torch.where(distance <= sigma, 
                        torch.tensor(1.0).to(flow.device), 
                        torch.tensor(0.0).to(flow.device))

            reserved_mask = torch.where(distance <= sigma, 
                            1.0, 
                            torch.exp(-((distance - sigma) / (k * sigma))**n)) # （H，W）
            weight_mask += reserved_mask

            # 当前agent的flow
            flow_x, flow_y = flow[i]

            # 计算平滑位移
            delta_x -= mask * flow_x # （H，W) delta_x其中记录了所有的object的x的偏移量，有object的地方才会被移动 这是基于已经flowed的object，因此是减去flow
            delta_y -= mask * flow_y # （H，W) delta_y其中记录了所有的object的y的偏移量

        # 将调整后的坐标标准化到 [-1, 1] 区间，符合`grid_sample`的要求
        grid_x = (grid_x + delta_x) / h * 2
        grid_y = (grid_y + delta_y) / w * 2

        # 将网格坐标组合在一起，形状为 (height, width, 2)
        grid = torch.stack((grid_y, grid_x), dim=-1) # @TODO 这里的顺序注意一下
        weight_mask = weight_mask.reshape(1, h, w).repeat(c, 1, 1).unsqueeze(0) # 1,C,H, W
        return grid, weight_mask

    def generate_smooth_flow_map(self, flow, bbox_list, scale=1.25, shape_list=None, align_corners=False, file_suffix=""):
        """
        生成平滑warp网格并应用到特征图上。

        参数:
        flow (torch.Tensor): 目标物体的位移 (num_cav, 2)。
        bbox_list (torch.Tensor): 边界框列表 (num_cav, 4, 2)。
        scale (float): 缩放因子。
        shape_list (tuple): 特征图的形状 (C, H, W)。
        align_corners (bool): 对齐角落参数。

        返回:
        updated_feature: 特征图经过warp后的结果 [C, H, W]。
        """
        # 提取bbox的x和y坐标
        bbox_list = bbox_list[:, :, :2]  # (M, 4, 2)

        # 按比例缩放flow和bbox_list
        flow = flow * scale  # (M, 2)
        bbox_list = bbox_list * scale  # (M, 4, 2)

        C, H, W = shape_list
        num_cav = bbox_list.shape[0]  # M

        # 计算每个目标物体的中心点
        centers = bbox_list.mean(dim=1)  # (M, 2)

        # 计算bbx的对角线长度的一半，作为半径 @TODO 这里要改掉，直接用标准形式的dx dy的平方和开根号
        radius = ((bbox_list[:, 1, 0] - bbox_list[:, 0, 0])**2 + (bbox_list[:, 2, 1] - bbox_list[:, 0, 1])**2).sqrt() / 2

        # 初始化warp矩阵
        basic_mat = torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(shape_list.device)  # (1, H, W, 2)

        for i in range(num_cav):
            center = centers[i]
            rad = radius[i]
            flow_single = flow[i].unsqueeze(0) # (1,2)
            smooth_warp_grid = self.create_smooth_warp_grid(center, rad, flow_single, (H, W))
            basic_warp_mat[0] = smooth_warp_grid

        # 生成平滑的warp特征图
        warped_feature = F.grid_sample(feature, basic_warp_mat.unsqueeze(0), align_corners=align_corners)
        
        # 生成掩码
        reserved_area = torch.zeros((C, H, W)).to(shape_list.device)  # (C, H, W)
        for cav in range(num_cav):
            x_min = int(torch.min(bbox_list[cav, :, 0]) + W / 2)
            x_max = int(torch.max(bbox_list[cav, :, 0]) + W / 2)
            y_min = int(torch.min(bbox_list[cav, :, 1]) + H / 2)
            y_max = int(torch.max(bbox_list[cav, :, 1]) + H / 2)
            reserved_area[:, y_min:y_max, x_min:x_max] = 1

        return warped_feature, reserved_area.unsqueeze(0)  # 返回形状 (1, C, H, W)


    def generate_flow_map_smooth(self, flow, bbox_list, bbox_center, bbox_half_diag, bbox_unc, scale=1.25, shape_list=None, align_corners=False, file_suffix=""):
        """
        Parameters
        -----------
        feature: [C, H, W] at voxel scale
        bbox_list: [num_cav, 4, 3] at cav coodinate system and lidar scale 最近的一帧的object 这些已经是做好match的部分 num_cav以下被我写作M
        bbox_center: [num_cav, 2] bbx中心位置
        bbox_half_diag: [num_cav, ] bbx对角线的一半
        bbox_unc: [num_cav, ] bbx的预测不确定分数 0-1
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
        bbox_center = bbox_center * scale
        bbox_half_diag = bbox_half_diag * scale
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
        num_cav = bbox_list.shape[0] # M

        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(shape_list.device) # （1, H, W, 2）
        reserved_area = torch.zeros((C, H, W)).to(shape_list.device)  # C, H, W  全0张量 修改为全1张量 因为没有flow如果掩码设置为全0，那会将原来的特征图全部掩蔽，但是可能是匹配失败或者ROI Generator的问题
        if flow.shape[0] == 0 : 
            reserved_area = torch.ones((C, H, W)).to(shape_list.device)  # C, H, W 
            if self.viz_flag:
                return basic_warp_mat,  reserved_area.unsqueeze(0), reserved_area.unsqueeze(0)  # 返回不变的矩阵
            return basic_warp_mat,  reserved_area.unsqueeze(0)  # 返回不变的矩阵

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1 
        ------------
        '''
        flow_clone = flow.detach().clone() # 拷贝  （M, 2）

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1) # （M, 3, 3）设置恒等变换矩阵
        flow_clone = -2 * flow_clone / torch.tensor([W, H]).to(torch.float32).to(shape_list.device) # 类似于归一化，约束到[-1, 1]区间，这是因为 F.affine_grid 需要这样的范围的数值
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
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
        x_min_fid = (x_min + int(W/2)).to(torch.int) # 这是将边界框坐标从以图像中心为原点转换为以图像左上角为原点的坐标系 (M)  这里表示的都是补偿过后的位置
        x_max_fid = (x_max + int(W/2)).to(torch.int)
        y_min_fid = (y_min + int(H/2)).to(torch.int)
        y_max_fid = (y_max + int(H/2)).to(torch.int)

        bbox_center[..., 0] = (bbox_center[..., 0] + int(W / 2)).to(torch.int) # 坐标原点转移到左上角
        bbox_center[..., 1] = (bbox_center[..., 1] + int(H / 2)).to(torch.int)

        bbox_center_flowed = bbox_center + flow # M,2 + M,2

        for cav in range(num_cav): # 遍历每一辆agent
            grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid_x = grid_x.float().to(shape_list.device)
            grid_y = grid_y.float().to(shape_list.device)
            distance = ((grid_x - bbox_center[cav][0])**2 + (grid_y - bbox_center[cav][1])**2).sqrt() # (H, W)
            mask = (distance <= bbox_half_diag[cav]).float() # (H, W)
            basic_warp_mat[..., 0] = torch.where(mask == 1.0, cav_warp_mat[cav,:,:,0], basic_warp_mat[..., 0])
            basic_warp_mat[..., 1] = torch.where(mask == 1.0, cav_warp_mat[cav,:,:,1], basic_warp_mat[..., 1])
            # basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] # 将计算得到的每个对象的仿射变换矩阵应用到一个基础变换矩阵上，用于更新特征图中相应区域的变换

        # generate mask
            
        basic_warp_mat, weight_mask = self.create_smooth_warp_grid(bbox_center_flowed, bbox_half_diag)
        weight_mask = torch.zeros((H, W))
        for i in range(num_cav):
            unc_score = bbox_unc[i]
            sigma = bbox_half_diag[i]
            weighted_mask = self.create_weighted_mask(H, W, bbox_center[i, 0], bbox_center[i, 1], sigma) # (H, W)
            weight_mask += unc_score * weighted_mask
        weight_mask = weight_mask.reshape(1, H, W).repeat(C, 1, 1).unsqueeze(0)
        return basic_warp_mat, weight_mask
        # generate mask
        # x_min_ori = torch.min(bbox_list[:,:,0],dim=1)[0] # 第0帧object 的每个边界框的四个角确定
        # x_max_ori = torch.max(bbox_list[:,:,0],dim=1)[0]
        # y_min_ori = torch.min(bbox_list[:,:,1],dim=1)[0]
        # y_max_ori = torch.max(bbox_list[:,:,1],dim=1)[0]
        # x_min_fid_ori = (x_min_ori + int(W/2)).to(torch.int)
        # x_max_fid_ori = (x_max_ori + int(W/2)).to(torch.int)
        # y_min_fid_ori = (y_min_ori + int(H/2)).to(torch.int)
        # y_max_fid_ori = (y_max_ori + int(H/2)).to(torch.int)
        # # set original location as 0
        # for cav in range(num_cav):
        #     reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 0 # object区域归零
        # # set warped location as 1
        # for cav in range(num_cav):
        #     reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1 # warped的object边界框区域置位1  （C，H， W）

        # if self.viz_flag:
        #     single_reserved_area = torch.zeros_like(reserved_area) # （C，H， W）
        #     for cav in range(num_cav):
        #         single_reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 1 # object部分标记
        #     return basic_warp_mat, reserved_area.unsqueeze(0), single_reserved_area.unsqueeze(0)

        return basic_warp_mat, reserved_area.unsqueeze(0) # 返回形状（1， H， W， 2）， （1， C， H， W）

    def feature_warp(self, feature, bbox_list, flow, scale=1.25, align_corners=False, file_suffix=""):
        """
        Parameters
        -----------
        feature: [C, H, W] at voxel scale
        bbox_list: [num_cav, 4, 3] at cav coodinate system and lidar scale
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

        if flow.shape[0] == 0 : 
            return feature

        # only use x and y
        bbox_list = bbox_list[:, :, :2]

        # scale meters to voxel, feature_length / lidar_range_length = 1.25
        flow = flow * scale
        bbox_list = bbox_list * scale

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

        C, H, W = feature.size()
        num_cav = bbox_list.shape[0]
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(feature)

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1 
        ------------
        '''
        flow_clone = flow.detach().clone()

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1)
        flow_clone = -2 * flow_clone / torch.tensor([feature.shape[2], feature.shape[1]]).to(feature)
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
        # print("cav_t_mat", cav_t_mat)

        cav_warp_mat = F.affine_grid(cav_t_mat,
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(feature) # .to() 统一数据格式 float32
        
        flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2
        ######### viz-1: original feature, original bbx and flowed bbx
        if flag_viz:
            viz_bbx_list = torch.cat((bbox_list, flowed_bbx_list), dim=0)
            canvas_hidden = viz_on_canvas(feature, viz_bbx_list, scale=scale)
            plt.sca(ax[1])
            # plt.axis("off") 
            plt.imshow(canvas_hidden.canvas)
        ##########

        x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1
        x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
        y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
        y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
        x_min_fid = (x_min + int(W/2)).to(torch.int)
        x_max_fid = (x_max + int(W/2)).to(torch.int)
        y_min_fid = (y_min + int(H/2)).to(torch.int)
        y_max_fid = (y_max + int(H/2)).to(torch.int)

        for cav in range(num_cav):
            basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]]

        final_feature = F.grid_sample(feature.unsqueeze(0), basic_warp_mat, align_corners=align_corners)[0]
        
        ####### viz-2: warped feature, flowed box and warped 
        if flag_viz:
            p_0 = torch.stack((x_min, y_min), dim=1).to(torch.int)
            p_1 = torch.stack((x_min, y_max), dim=1).to(torch.int)
            p_2 = torch.stack((x_max, y_max), dim=1).to(torch.int)
            p_3 = torch.stack((x_max, y_min), dim=1).to(torch.int)
            warp_area_bbox_list = torch.stack((p_0, p_1, p_2, p_3), dim=1)
            viz_bbx_list = torch.cat((flowed_bbx_list, warp_area_bbox_list), dim=0)
            canvas_new = viz_on_canvas(final_feature, viz_bbx_list, scale=scale)
            plt.sca(ax[2]) 
            # plt.axis("off") 
            plt.imshow(canvas_new.canvas)
        ############## 

        reserved_area = torch.ones_like(feature)  # C, H, W
        x_min_ori = torch.min(bbox_list[:,:,0],dim=1)[0]
        x_max_ori = torch.max(bbox_list[:,:,0],dim=1)[0]
        y_min_ori = torch.min(bbox_list[:,:,1],dim=1)[0]
        y_max_ori = torch.max(bbox_list[:,:,1],dim=1)[0]
        x_min_fid_ori = (x_min_ori + int(W/2)).to(torch.int)
        x_max_fid_ori = (x_max_ori + int(W/2)).to(torch.int)
        y_min_fid_ori = (y_min_ori + int(H/2)).to(torch.int)
        y_max_fid_ori = (y_max_ori + int(H/2)).to(torch.int)
        # set original location as 0
        for cav in range(num_cav):
            reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 0
        # set warped location as 1
        for cav in range(num_cav):
            reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
        final_feature = final_feature * reserved_area

        ####### viz-3: mask area out of warped bbx
        if flag_viz:
            partial_feature_one = torch.zeros_like(feature)  # C, H, W
            for cav in range(num_cav):
                partial_feature_one[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
            masked_final_feature = partial_feature_one * final_feature
            canvas_hidden = viz_on_canvas(masked_final_feature, warp_area_bbox_list, scale=scale)
            plt.sca(ax[3]) 
            # plt.axis("off") 
            plt.imshow(canvas_hidden.canvas)
        ##############

        ####### viz: draw figures
        if flag_viz:
            plt.tight_layout()
            plt.savefig(f'result_canvas_{file_suffix}.jpg', transparent=False, dpi=400)
            plt.clf()

            fig, axes = plt.subplots(2, 1, figsize=(4, 4))
            major_ticks_x = np.linspace(0,350,8)
            minor_ticks_x = np.linspace(0,350,15)
            major_ticks_y = np.linspace(0,100,3)
            minor_ticks_y = np.linspace(0,100,5)
            for i, ax in enumerate(axes):
                plt.sca(ax); #plt.axis("off")
                ax.set_xticks(major_ticks_x); ax.set_xticks(minor_ticks_x, minor=True)
                ax.set_yticks(major_ticks_y); ax.set_yticks(minor_ticks_y, minor=True)
                ax.grid(which='major', color='w', linewidth=0.4)
                ax.grid(which='minor', color='w', linewidth=0.2, alpha=0.5)
                if i==0:
                    plt.imshow(torch.max(feature, dim=0)[0].cpu())
                else:
                    plt.imshow(torch.max(final_feature, dim=0)[0].cpu())
            plt.tight_layout()
            plt.savefig(f'result_features_{file_suffix}.jpg', transparent=False, dpi=400)
            plt.clf()
        #######

        return final_feature

    def backup_feature_warp(self, feature, bbox_list, flow, scale=1.25, align_corners=False):
        """
        Parameters
        -----------
        feature: [C, H, W] at voxel scale
        bbox_list: [num_cav, 4, 3] at cav coodinate system and lidar scale
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

        if flow.shape[0] == 0 : 
            return feature

        # only use x and y
        bbox_list = bbox_list[:, :, :2]

        # scale meters to voxel, feature_length / lidar_range_length = 1.25
        flow = flow * scale
        bbox_list = bbox_list * scale

        # # store two parts of bbx: 1. original bbx, 2. 
        # viz_bbx_list = bbox_list
        # fig, ax = plt.subplots(4, 1, figsize=(5,11))
        
        # ######## viz-0: original feature, original bbx
        # canvas_ori = viz_on_canvas(feature, bbox_list)
        # plt.sca(ax[0])
        # # plt.axis("off")
        # plt.imshow(canvas_ori.canvas)
        # ##########

        C, H, W = feature.size()
        num_cav = bbox_list.shape[0]
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(feature)

        '''
        create affine matrix:
        ------------
        1  0  -2*t_y/W
        0  1  -2*t_x/H
        0  0    1 
        ------------
        '''
        flow_clone = flow.detach().clone()

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1)
        flow_clone = -2 * flow_clone / torch.tensor([feature.shape[2], feature.shape[1]]).to(feature)
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
        # print("cav_t_mat", cav_t_mat)

        cav_warp_mat = F.affine_grid(cav_t_mat,
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(feature) # .to() 统一数据格式 float32
        
        ######### viz-1: original feature, original bbx and flowed bbx
        flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2
        # viz_bbx_list = torch.cat((bbox_list, flowed_bbx_list), dim=0)
        # canvas_hidden = viz_on_canvas(feature, viz_bbx_list)
        # plt.sca(ax[1])
        # # plt.axis("off") 
        # plt.imshow(canvas_hidden.canvas)
        ##########

        x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1
        x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
        y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
        y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
        x_min_fid = (x_min + 176).to(torch.int) # TODO: 这里面的176需要重新考虑
        x_max_fid = (x_max + 176).to(torch.int)
        y_min_fid = (y_min + 50).to(torch.int)  # TODO: 这里面的50需要重新考虑
        y_max_fid = (y_max + 50).to(torch.int)

        for cav in range(num_cav):
            basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]]

        final_feature = F.grid_sample(feature.unsqueeze(0), basic_warp_mat, align_corners=align_corners)[0]
        
        ####### viz-2: warped feature, flowed box and warped 
        # p_0 = torch.stack((x_min, y_min), dim=1).to(torch.int)
        # p_1 = torch.stack((x_min, y_max), dim=1).to(torch.int)
        # p_2 = torch.stack((x_max, y_max), dim=1).to(torch.int)
        # p_3 = torch.stack((x_max, y_min), dim=1).to(torch.int)
        # warp_area_bbox_list = torch.stack((p_0, p_1, p_2, p_3), dim=1)
        # viz_bbx_list = torch.cat((flowed_bbx_list, warp_area_bbox_list), dim=0)
        # canvas_new = viz_on_canvas(final_feature, viz_bbx_list)
        # plt.sca(ax[2]) 
        # # plt.axis("off") 
        # plt.imshow(canvas_new.canvas)
        ############## 

        ####### viz-3: mask area out of warped bbx
        # partial_feature_one = torch.zeros_like(feature)  # C, H, W
        # for cav in range(num_cav):
        #     partial_feature_one[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1
        # masked_final_feature = partial_feature_one * final_feature
        # canvas_hidden = viz_on_canvas(masked_final_feature, warp_area_bbox_list)
        # plt.sca(ax[3]) 
        # # plt.axis("off") 
        # plt.imshow(canvas_hidden.canvas)
        ##############

        # plt.tight_layout()
        # plt.savefig('result_canvas.jpg', transparent=False, dpi=400)
        # plt.clf()

        # fig, axes = plt.subplots(2, 1, figsize=(4, 4))
        # major_ticks_x = np.linspace(0,350,8)
        # minor_ticks_x = np.linspace(0,350,15)
        # major_ticks_y = np.linspace(0,100,3)
        # minor_ticks_y = np.linspace(0,100,5)
        # for i, ax in enumerate(axes):
        #     plt.sca(ax); #plt.axis("off")
        #     ax.set_xticks(major_ticks_x); ax.set_xticks(minor_ticks_x, minor=True)
        #     ax.set_yticks(major_ticks_y); ax.set_yticks(minor_ticks_y, minor=True)
        #     ax.grid(which='major', color='w', linewidth=0.4)
        #     ax.grid(which='minor', color='w', linewidth=0.2, alpha=0.5)
        #     if i==0:
        #         plt.imshow(torch.max(feature, dim=0)[0].cpu())
        #     else:
        #         plt.imshow(torch.max(final_feature, dim=0)[0].cpu())
        # plt.tight_layout()
        # plt.savefig('result_features.jpg', transparent=False, dpi=400)
        # plt.clf()

        return final_feature

    def get_common_elements(self, A, B):# 输入是第二帧和第一帧匹配上的past2 id  还有第二帧和第三帧匹配上的past2 id
        common_elements_A = []
        common_elements_B = []
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                if a == b:
                    common_elements_A.append(i)
                    common_elements_B.append(j)
        return common_elements_A, common_elements_B # 两者都在的才会返回


def get_center_points(corner_points):
    corner_points2d = corner_points[:,:4,:2]

    centers_x = torch.mean(corner_points2d[:,:,0],dim=1,keepdim=True)

    centers_y = torch.mean(corner_points2d[:,:,1],dim=1,keepdim=True)

    return torch.cat((centers_x,centers_y), dim=1)

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)