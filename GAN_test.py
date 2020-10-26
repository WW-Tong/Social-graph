import gc
import os
import math
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

# from data import data_loader
from utils import get_dset_path
from utils import relative_to_abs
from utils import gan_g_loss, gan_d_loss, l2_loss, displacement_error, final_displacement_error
from model import TrajectoryGenerator, TrajectoryDiscriminator
from utils import *
from constants import *

def evaluate_helper(error):
    error = torch.stack(error, dim=1)       # V *20
    error = torch.sum(error, dim=0)         # 20
    error = torch.min(error)                # 最小的一个
    return error

def evaluate(loader, generator):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list) = batch
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,n_l,l_m,V_obs,A_obs,V_pre,A_pre,vgg_list) = batch
            V_obs=V_obs.permute(0,3,1,2)
            V_pre=V_pre.permute(0,3,1,2)

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)      # 求出计算的总人数，

            for _ in range(NUM_SAMPLES):        # 20次
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel,V_obs,A_obs, vgg_list)        # TVC 
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0, :, :, -1])     # TVC+VC
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))        # 添加过滤
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[0,:,:,-1], mode='raw'))
            # V*20
            ade_sum = evaluate_helper(ade)
            fde_sum = evaluate_helper(fde)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * PRED_LEN)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde

def load_and_evaluate(generator, version):
    print("Initializing {} dataset".format(version))
    path = get_dset_path('eth', version)
    # path = get_dset_path('univ', version)
    train_dset = TrajectoryDataset(path)
    loader = DataLoader(
        train_dset,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=0)
    # _, loader = data_loader(path)
    ade, fde = evaluate(loader, generator)
    print('{} Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(version, DATASET_NAME, PRED_LEN, ade, fde))
# torch.cuda.set_device()
model = torch.load("wmodel.pt",map_location='cuda:0')
generator = TrajectoryGenerator()
generator.load_state_dict(model['g'])
generator.cuda()
generator.eval()

# load_and_evaluate(generator, 'train')
# load_and_evaluate(generator, 'val')
load_and_evaluate(generator, 'test')

