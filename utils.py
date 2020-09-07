import os
import math
import random
import numpy as np
import torch.nn.functional as f
import torch

from constants import *

def get_dset_path(dset_name, dset_type):
    '''
    获取数据集路径
    :param dset_name:
    :param dset_type:
    :return:
    '''
    return os.path.join('datasets', dset_name, dset_type)

def relative_to_abs(rel_traj, start_pos):
    rel_traj = rel_traj.permute(1, 0, 2)    # 64*12*2
    displacement = torch.cumsum(rel_traj, dim=1)    # 沿第二个维度求和
    start_pos = torch.unsqueeze(start_pos, dim=1)   # 变为3*1*2
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)        # 12*64*2

def bce_loss(input, target):
    neg_abs = -input.abs()          # 取负值
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)    # 真实数据在1左右
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)     # 预测数据在0左右
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def l2_loss(pred_traj, pred_traj_gt, mode='average'):
    seq_len, batch, _ = pred_traj.size()        # 12 64 2
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2  # 64*12*2，做均方误差
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / (seq_len * batch)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)   # 64*1

def displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    '''
    计算位移误差总和
    :param pred_traj: 预测结果 12*n*2
    :param pred_traj_gt: 实际路径 12*n*2
    :param mode:
    :return:
    '''
    seq_len, _, _ = pred_traj.size()    # 12
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)   # 3*12*2
    loss = loss**2      # 平方差
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)   # n*1 平方根
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(
    pred_pos, pred_pos_gt, mode='sum'
):
    '''
    计算FDE
    :param pred_pos: 终点
    :param pred_pos_gt: 真实终点
    :param mode:
    :return:
    '''
    loss = pred_pos_gt - pred_pos
    loss = loss**2 # 平方差
    loss = torch.sqrt(loss.sum(dim=1))  # 求和取平方根
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
