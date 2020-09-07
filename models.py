import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *

def make_mlp(dim_list):
    '''
    批量生成全连接层
    :param dim_list: 维度列表
    :return: 一系列线性层的感知机
    '''
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def get_noise(shape):
    '''
    生成高斯噪声
    :param shape: 噪声的shape
    :return:
    '''
    return torch.randn(*shape).cuda()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.h_dim = H_DIM  # 64 与人数一致
        self.embedding_dim = EMBEDDING_DIM   # 16 未知

        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)        # 输入的维度16，隐状态维度64，层数1，LSTM编码
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)           # 输入2维，输出embedding_dim

    def init_hidden(self, batch):
        '''
        隐层状态初始化，细胞状态初始化
        :param batch:
        :return:
        '''
        h = torch.zeros(1, batch, self.h_dim).cuda()        # 1*bat*64
        c = torch.zeros(1, batch, self.h_dim).cuda()
        return (h, c)

    def forward(self, obs_traj):

        padded = len(obs_traj.shape) == 4    # 观测轨迹的维度数目4
        npeds = obs_traj.size(1)             # 人数，观测轨迹张量的第二个维度64
        total = npeds * (MAX_PEDS if padded else 1)         # 64*64

        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))   # 展成n*2输入线性嵌入层,输出16 8*3*64*16
        obs_traj_embedding = obs_traj_embedding.view(-1, total, self.embedding_dim)  # 调整维度（-1，4096，16）
        state = self.init_hidden(total)        # 以total初始化隐藏状态 1*4096*64
        output, state = self.encoder(obs_traj_embedding, state)         # 输入编码器输出到state中64
        final_h = state[0]      # 取隐藏状态
        if padded:
            final_h = final_h.view(npeds, MAX_PEDS, self.h_dim)     # 64*64*64
        else:
            final_h = final_h.view(npeds, self.h_dim)
        return final_h      # 最后一步的输出

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.seq_len = PRED_LEN         # 解码长度 12
        self.h_dim = H_DIM              # 64
        self.embedding_dim = EMBEDDING_DIM      # 16

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)   # 输入为16 ，输出为64 ，1层隐藏层
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)   # 输入2输出16
        self.hidden2pos = nn.Linear(self.h_dim, 2)                  # 转换为坐标序列 输入64，输出2

    def forward(self, last_pos, last_pos_rel, state_tuple):
        npeds = last_pos.size(0)        # 行人编号
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)        # 输入64*2，输出维度64*16
        decoder_input = decoder_input.view(1, npeds, self.embedding_dim)        # 调整维度1*64*16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)      # 输出1*64*64的隐状态和细胞状态
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))              # 将out变为64*64再输入坐标转换64*2
            curr_pos = rel_pos + last_pos   # 当前坐标为坐标差和最后坐标之和
            embedding_input = rel_pos       # 64*2

            decoder_input = self.spatial_embedding(embedding_input)     # 输入64*2输出64*16
            decoder_input = decoder_input.view(1, npeds, self.embedding_dim)   # 变为 1*64*16
            pred_traj_fake_rel.append(rel_pos.view(npeds, -1))   # 预测轨迹加入rel_pos  12*64*2
            last_pos = curr_pos  # 当前坐标作为最后坐标

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)     # 按0维拼接
        return pred_traj_fake_rel       # 12*64*2

class PhysicalAttention(nn.Module):
    def __init__(self):
        super(PhysicalAttention, self).__init__()

        self.L = ATTN_L   # 900
        self.D = ATTN_D   # 512
        self.D_down = ATTN_D_DOWN  # 16
        self.bottleneck_dim = BOTTLENECK_DIM  # 32
        self.embedding_dim = EMBEDDING_DIM    # 16

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)     # 2-16
        self.pre_att_proj = nn.Linear(self.D, self.D_down)        # 512-16

        mlp_pre_dim = self.embedding_dim + self.D_down    # 32
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)   # 32-512-32

        self.attn = nn.Linear(self.L*self.bottleneck_dim, self.L)     # 900*32--900

    def forward(self, vgg, end_pos):

        npeds = end_pos.size(0)
        end_pos = end_pos[:, 0, :]        # n*2
        curr_rel_embedding = self.spatial_embedding(end_pos)  # n*16
        curr_rel_embedding = curr_rel_embedding.view(-1, 1, self.embedding_dim).repeat(1, self.L, 1)  # n*900*16

        vgg = vgg.view(-1, self.D)    # x*512
        features_proj = self.pre_att_proj(vgg)        # x*16   x=n*900
        features_proj = features_proj.view(-1, self.L, self.D_down)   # y*900*16

        mlp_h_input = torch.cat([features_proj, curr_rel_embedding], dim=2)   # n*900*32
        attn_h = self.mlp_pre_attn(mlp_h_input.view(-1, self.embedding_dim+self.D_down))  # -1，32--32
        attn_h = attn_h.view(npeds, self.L, self.bottleneck_dim)  # n*900*32

        attn_w = F.softmax(self.attn(attn_h.view(npeds, -1)), dim=1)  # n*28800--n*900
        attn_w = attn_w.view(npeds, self.L, 1)      # n*900*1

        attn_h = torch.sum(attn_h * attn_w, dim=1)      # n*900*32**
        return attn_h

class SocialAttention(nn.Module):
    def __init__(self):
        super(SocialAttention, self).__init__()

        self.h_dim = H_DIM          # 64
        self.bottleneck_dim = BOTTLENECK_DIM    # 瓶颈维度 32
        self.embedding_dim = EMBEDDING_DIM      # 16

        mlp_pre_dim = self.embedding_dim + self.h_dim       # 16+64=80
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]     # 80，512，32

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)       # 2--16
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)                 # 80-512-32
        self.attn = nn.Linear(MAX_PEDS*self.bottleneck_dim, MAX_PEDS)   # 6输入64*32输出64

    def repeat(self, tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, end_pos):

        npeds = h_states.size(0)            # 64
        curr_rel_pos = end_pos[:, :, :] - end_pos[:, 0:1, :]        # 不知道
        curr_rel_embedding = self.spatial_embedding(curr_rel_pos.view(-1, 2))   # 转换为坐标
        curr_rel_embedding = curr_rel_embedding.view(npeds, MAX_PEDS, self.embedding_dim)       # 64*64*16

        mlp_h_input = torch.cat([h_states, curr_rel_embedding], dim=2)      # 64*64*80
        attn_h = self.mlp_pre_attn(mlp_h_input.view(-1, self.embedding_dim+self.h_dim))     # 变为4096*80--4096*32
        attn_h = attn_h.view(npeds, MAX_PEDS, self.bottleneck_dim)          # 64*64*32
        
        attn_w = F.softmax(self.attn(attn_h.view(npeds, -1)), dim=1)        # 64*2048进行softmax行和为1
        attn_w = attn_w.view(npeds, MAX_PEDS, 1)        # 64*64*32

        attn_h = torch.sum(attn_h * attn_w, dim=1)      # 乘积求和      # 64*32 中间的64被缩并
        return attn_h

class TrajectoryGenerator(nn.Module):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = OBS_LEN      # 8
        self.pred_len = PRED_LEN    # 12
        self.mlp_dim = MLP_DIM      # 64
        self.h_dim = H_DIM          # 64
        self.embedding_dim = EMBEDDING_DIM  # 16
        self.bottleneck_dim = BOTTLENECK_DIM    # 32
        self.noise_dim = NOISE_DIM      # 8

        self.encoder = Encoder()
        self.sattn = SocialAttention()
        # self.pattn = PhysicalAttention()
        self.decoder = Decoder()

        # input_dim = self.h_dim + 2*self.bottleneck_dim      # 128
        input_dim = self.h_dim + self.bottleneck_dim  # 96
        mlp_decoder_context_dims = [input_dim, self.mlp_dim, self.h_dim - self.noise_dim]       # 96,64,56
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)       # 96--56

    def add_noise(self, _input):
        npeds = _input.size(0)      # 64
        noise_shape = (self.noise_dim,)     # 8,
        z_decoder = get_noise(noise_shape)  # 8
        vec = z_decoder.view(1, -1).repeat(npeds, 1)        # 64*8
        return torch.cat((_input, vec), dim=1)              # 64*56 +64*8 =64*64

    def forward(self, obs_traj, obs_traj_rel):

        npeds = obs_traj_rel.size(1)    # 64
        final_encoder_h = self.encoder(obs_traj_rel)    # 64*64*64

        end_pos = obs_traj[-1, :, :, :]  # 3*64*2
        attn_s = self.sattn(final_encoder_h, end_pos)       # 64*32
        # attn_p = self.pattn(vgg_list, end_pos)
        # mlp_decoder_context_input = torch.cat([final_encoder_h[:, 0, :], attn_s, attn_p], dim=1)
        mlp_decoder_context_input = torch.cat([final_encoder_h[:, 0, :], attn_s], dim=1)        # 64*96
        noise_input = self.mlp_decoder_context(mlp_decoder_context_input)       # 64*56
        decoder_h = self.add_noise(noise_input)   # 64*64
        decoder_h = torch.unsqueeze(decoder_h, 0)  # 1*64*64

        decoder_c = torch.zeros(1, npeds, self.h_dim).cuda()    # 1*64*64 细胞状态为空，隐藏状态为编码器隐藏状态加上注意力
        state_tuple = (decoder_h, decoder_c)

        last_pos = obs_traj[-1, :, 0, :]        # 64*2
        last_pos_rel = obs_traj_rel[-1, :, 0, :]    # 64*2
        pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
        return pred_traj_fake_rel

class TrajectoryDiscriminator(nn.Module):
    def __init__(self):
        super(TrajectoryDiscriminator, self).__init__()

        self.mlp_dim = MLP_DIM      # 64
        self.h_dim = H_DIM          # 64

        self.encoder = Encoder()    # 调用编码器
        real_classifier_dims = [self.h_dim, self.mlp_dim, 1]   # 64，64，1
        self.real_classifier = make_mlp(real_classifier_dims)  # 感知机64--1

    def forward(self, traj, traj_rel):

        final_h = self.encoder(traj_rel)   # 编码隐藏状态64*64*64
        scores = self.real_classifier(final_h)  # 64*64*1
        return scores
class EndpointDecoder(nn.Module):
    def __init__(self):
        super(EndpointDecoder, self).__init__()
        self.h_dim = H_DIM  # 64
        self.embedding_dim = EMBEDDING_DIM      # 16

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)     # lstm 解码器
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.hidden2pos = nn.Linear(self.h_dim, 2)
    def forward(self, last_pos, obs_speed, state_tuple):
        npeds = last_pos.size(0)        # 行人编号
        decoder_input = self.spatial_embedding(obs_speed)        # 输入64*2，输出维度64*16 输入过去的速度
        decoder_input = decoder_input.view(1, npeds, self.embedding_dim)        # 调整维度1*64*16

        output, state_tuple = self.decoder(decoder_input, state_tuple)      # 输出1*64*64  根据速度和注意力解码
        rel_pos = self.hidden2pos(output.view(-1, self.h_dim))              # 解码输出变为位移差  64*2
        pre_end_pos = rel_pos + last_pos   # 终点坐标为坐标差和最后坐标之和
        return pre_end_pos      # 64*2

class EndpointGenerate(nn.Module):
    def __init__(self):
        super(EndpointGenerate, self).__init__()
        self.obs_len = OBS_LEN  # 8
        self.pred_len = PRED_LEN  # 12
        self.mlp_dim = MLP_DIM  # 64
        self.h_dim = H_DIM  # 64
        self.embedding_dim = EMBEDDING_DIM  # 16
        self.bottleneck_dim = BOTTLENECK_DIM  # 32
        self.noise_dim = NOISE_DIM  # 8
        self.encoder = Encoder()
        self.sattn = SocialAttention()      # 社会注意
        # self.pattn = PhysicalAttention()
        self.decoder = EndpointDecoder()

        input_dim = self.h_dim + self.bottleneck_dim  # 96
        mlp_decoder_context_dims = [input_dim, self.mlp_dim, self.h_dim - self.noise_dim]  # 96,64,56
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)  # 96--56
    def add_noise(self, _input):
        npeds = _input.size(0)      # 64
        noise_shape = (self.noise_dim,)     # 8,
        z_decoder = get_noise(noise_shape)  # 8
        vec = z_decoder.view(1, -1).repeat(npeds, 1)        # 64*8
        return torch.cat((_input, vec), dim=1)              # 64*56 +64*8 =64*64
    def forward(self,obs_traj,obs_traj_rel):
        '''
        参数暂时包括轨迹和坐标差
        :param obs_traj:
        :param obs_traj_rel:
        :return:
        '''
        npeds = obs_traj_rel.size(1)  # 64
        final_encoder_h = self.encoder(obs_traj_rel)  # 64*64*64

        end_pos = obs_traj[-1, :, :, :]  # 64*2*8
        attn_s = self.sattn(final_encoder_h, end_pos)  # 64*32  计算最后一个时刻的注意力（暂时使用社会注意力填充）
        # attn_p = self.pattn(vgg_list, end_pos)
        # mlp_decoder_context_input = torch.cat([final_encoder_h[:, 0, :], attn_s, attn_p], dim=1)
        mlp_decoder_context_input = torch.cat([final_encoder_h[:, 0, :], attn_s], dim=1)  # 64*96
        noise_input = self.mlp_decoder_context(mlp_decoder_context_input)  # 64*56
        decoder_h = self.add_noise(noise_input)  # 64*64
        decoder_h = torch.unsqueeze(decoder_h, 0)  # 1*64*64

        decoder_c = torch.zeros(1, npeds, self.h_dim).cuda()  # 1*64*64 细胞状态为空，隐藏状态为编码器隐藏状态加上注意力
        state_tuple = (decoder_h, decoder_c)

        last_pos = obs_traj[-1, :, 0, :]  # 64*2
        obs_speed = obs_traj_rel[:, :, 0, :]  # 8*n*2
        obs_speed = torch.sum(obs_speed, dim=0)/OBS_LEN
        obs_speed = obs_speed*PRED_LEN      # 得出PRED_LEN之后的位移和方向
        pre_end = self.decoder(last_pos, obs_speed, state_tuple)
        return pre_end
class EndPointDiscriminator(nn.Module):
    def __init__(self):
        super(EndPointDiscriminator,self).__init__()
        self.mip_dim = MLP_DIM      # 64
        self.h_dim = H_DIM          # 64

        self.encoder = Encoder()    # 调用编码器
        real_classifar_dims = [self.h_dim, self.mlp_dim,1]          # 64-64-1
        self.real_classfar = make_mlp(real_classifar_dims)      # 64--1

    def forward(self, traj_rel):

        finl_h = self.encoder(traj_rel)     # 轨迹编码
        scores = self.real_classfar(finl_h)         # 64*64*1
        return scores








# print("this is Encoder")
# print(e)
# print("this is Decoder")
# print(d)
# print("this is SocialAttention")
# print(s)
# print("this is TrajectoryGenerator")
# print(G)
# print("this is TrajectoryDiscriminator")
# print(D)
