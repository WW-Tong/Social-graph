import gc
import os
import math
import random
import numpy as np
from collections import defaultdict
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from data import data_loader
from utils import get_dset_path
from utils import relative_to_abs
from utils import gan_g_loss, gan_d_loss, l2_loss, displacement_error, final_displacement_error
from model import TrajectoryGenerator, TrajectoryDiscriminator
from utils import *
from constants import *
from tensorboardX import SummaryWriter
from torch import autograd
from torch.autograd import Variable

def init_weights(m):
    classname = m.__class__.__name__            # 初始化权重
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes():
    return torch.cuda.LongTensor, torch.cuda.FloatTensor

def main():
    train_path = './datasets/'+DATASET_NAME+'/val'
    val_path = './datasets/'+DATASET_NAME+'/test'   # datasets/val/name 
    long_dtype, float_dtype = get_dtypes()
    writer=SummaryWriter()
    print("Initializing train dataset")
    train_dset = TrajectoryDataset(
        train_path,
        obs_len=8,
        pred_len=12,
        skip=1,norm_lap_matr=True)
    train_loader = DataLoader(
        train_dset,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=0)
    print("Initializing val dataset")
    dset_val = TrajectoryDataset(
        val_path,
        obs_len=8,
        pred_len=12,
        skip=1,norm_lap_matr=True)

    val_loader = DataLoader(
        dset_val,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =False,
        num_workers=1)

    iterations_per_epoch = len(train_dset)    # 数据集长度除以步长  每个Epoch的bantch数
    NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS)     # 500 epoch
    print('There are {} iterations per epoch'.format(iterations_per_epoch))     # 每个Epoch的迭代次数

    generator = TrajectoryGenerator()               # 生成器 及初始化
    generator.apply(init_weights)
    generator.type(float_dtype).train()     

    optimizer_g = optim.Adam(generator.parameters(), lr=G_LR)       # 


    t, epoch = 0, 0
    t0 = None
    min_ade = None
    test_step=0
    while t < NUM_ITERATIONS:
        gc.collect()        # 清理内存
        g_steps_left = G_STEPS  # 1
        epoch += 1
        print('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            losses_g = generator_step(batch, generator,optimizer_g)
            if t % PRINT_EVERY == 0:        # 250个序列 输出一下评价
                test_step=test_step+1
                print('t = {} / {}'.format(t + 1, NUM_ITERATIONS))
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))
                    writer.add_scalar(k,v,test_step)

                print('Checking stats on val ...')
                metrics_val = check_accuracy(val_loader, generator)
                writer.add_scalar('ade',metrics_val['ade'],test_step)
                writer.add_scalar('fde',metrics_val['fde'],test_step)
                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                if min_ade is None or metrics_val['ade'] < min_ade:
                    min_ade = metrics_val['ade']
                    checkpoint = {'t': t, 'g': generator.state_dict(), 'g_optim': optimizer_g.state_dict()}
                    print("Saving checkpoint to model.pt")
                    torch.save(checkpoint, " G-model.pt")
                    print("Done.")

            t += 1
            # d_steps_left = D_STEPS
            # g_steps_left = G_STEPS
            if t >= NUM_ITERATIONS:
                
                break
    writer.close()

def gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    # nepeds=real_data.size(0)
    alpha = torch.rand(real_data.shape).cuda()
    # alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    # alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    # if use_cuda:
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates,interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
    return gradient_penalty

def discriminator_step(batch, generator, discriminator, d_loss_fn, optimizer_d):
    # discriminator.parameters().req
    # for p in discriminator.parameters():  # reset requires_grad
    #     p.requires_grad = True 
    with torch.backends.cudnn.flags(enabled=False):

        optimizer_d.zero_grad()
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,n_l,l_m,V_obs,A_obs,V_pre,A_pre,vgg_list) = batch
        V_obs=V_obs.permute(0,3,1,2)
        V_pre=V_pre.permute(0,3,1,2)
        # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch
        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)

        # generator_out = generator(obs_traj, obs_traj_rel, vgg_list)
        generator_out = generator(obs_traj, obs_traj_rel,V_obs,A_obs,vgg_list)
        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0, :, :, -1])  # V C

        traj_real = torch.cat([obs_traj[0], pred_traj_gt[0]], dim=2)      # 1*3*2*20 轨迹序列

        traj_real_rel = torch.cat([obs_traj_rel[0], pred_traj_gt_rel[0]], dim=2)  # 1*3*2*20 轨迹差值序列 N V C T
        pred_traj_fake=pred_traj_fake.permute(1,2,0)    # T V C——V C T
        # pred_traj_fake=pred_traj_fake.unsqueeze(dim=0)
        pred_traj_fake_rel=pred_traj_fake_rel.permute(1,2,0)    #  T V C——V C T
        # pred_traj_fake_rel=pred_traj_fake_rel.unsqueeze(dim=0)
        traj_fake = torch.cat([obs_traj[0], pred_traj_fake], dim=2)        # 观测轨迹加上预测 V C T
        traj_fake_rel = torch.cat([obs_traj_rel[0], pred_traj_fake_rel], dim=2)    # 观测差加上预测差 V C T

        scores_fake = discriminator(traj_fake_rel, traj_fake_rel)       # 计算鉴别分数输入 VCT编码
        scores_real = discriminator(traj_real_rel, traj_real_rel)       # 输入 V C T 输出 V 1

        # data_loss = d_loss_fn(scores_real, scores_fake)         # BCE
        data_loss = scores_fake.mean()-scores_real.mean()
        losses['D_data_loss'] = data_loss.item()
        # loss += data_loss
        loss=data_loss + gradient_penalty(discriminator,traj_fake_rel,traj_fake_rel)
        losses['D_total_loss'] = loss.item()
        # data_loss.backward()
        loss.backward()           # derivative for _cudnn_rnn_backward is not implemented
        optimizer_d.step()
        return losses

    

def generator_step(batch, generator, optimizer_g):
    optimizer_g.zero_grad()
    batch = [tensor.cuda() for tensor in batch]
    # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list) = batch
    # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,n_l,l_m,V_obs,A_obs,V_pre,A_pre,vgg_list) = batch
    V_obs=V_obs.permute(0,3,1,2)
    V_pre=V_pre.permute(0,3,1,2)
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    
    g_l2_loss_rel = []
    for _ in range(BEST_K):
        # generator_out = generator(obs_traj, obs_traj_rel, vgg_list)
        generator_out =generator(obs_traj, obs_traj_rel,V_obs,A_obs,vgg_list)       # 生成坐标差
        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0, :, :, -1])     # 12*3*2 TVC

        g_l2_loss_rel.append(l2_loss(       # 生成坐标差和真实坐标差 n*1
            pred_traj_fake_rel,     # T V C
            pred_traj_gt_rel,       # N V C T
            mode='raw'))
    # 生成了K条轨迹并计算损失 K V
    npeds = obs_traj.size(1)    # V
    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)     # 1
    g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)  # 拼接张量 得v k
    _g_l2_loss_rel = torch.sum(g_l2_loss_rel, dim=0)     # 求和 得 k

    _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / (npeds*PRED_LEN)   # 取最小的然后取平均
    g_l2_loss_sum_rel += _g_l2_loss_rel
    losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
    # loss += g_l2_loss_sum_rel           # 生成轨迹的损失
    # pred_traj_fake=pred_traj_fake.permute(1,2,0)        # TVC——VCT
    # pred_traj_fake_rel=pred_traj_fake_rel.permute(1,2,0)
    # traj_fake = torch.cat([obs_traj[0], pred_traj_fake], dim=2)     # VCT T=20
    # traj_fake_rel = torch.cat([obs_traj_rel[0], pred_traj_fake_rel], dim=2)
    
    # scores_fake = discriminator(traj_fake_rel, traj_fake_rel)       # 生成轨迹鉴别分数
    # # discriminator_loss = g_loss_fn(scores_fake)
    # discriminator_loss= scores_fake.mean()
    # loss += discriminator_loss          # 加入鉴别器损失
    # loss=g_l2_loss_sum_rel-discriminator_loss
    # losses['G_discriminator_loss'] = discriminator_loss.item()
    # losses['G_total_loss'] = loss.item()
    loss += g_l2_loss_sum_rel 
    
    loss.backward()
    optimizer_g.step()

    return losses

def check_accuracy(loader, generator, limit=False):
    
    d_losses = []   #
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error = []     # ADE FDE
    f_disp_error = []
    total_traj = 0

    mask_sum = 0
    generator.eval()
    with torch.no_grad():   #
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list) = batch
            # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,n_l,l_m,V_obs,A_obs,V_pre,A_pre,vgg_list) = batch
            V_obs=V_obs.permute(0,3,1,2)
            V_pre=V_pre.permute(0,3,1,2)
            # pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, vgg_list)
            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel,V_obs,A_obs,vgg_list)     # T V C
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0, :, :, -1])     # T V C——V C

            g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, mode='sum')
            g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, mode='sum')

            ade = displacement_error(pred_traj_fake, pred_traj_gt)      # TVC NVCT 
            fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[0,:,:,-1])        # VC  VC

            # traj_real = torch.cat([obs_traj[0], pred_traj_gt[0]], dim=2)        # V C T=20
            # traj_real_rel = torch.cat([obs_traj_rel[0], pred_traj_gt_rel[0]], dim=2)    # V C T =20
            # pred_traj_fake=pred_traj_fake.permute(1,2,0)
            # pred_traj_fake_rel=pred_traj_fake_rel.permute(1,2,0)    # V C T
            # traj_fake = torch.cat([obs_traj[0], pred_traj_fake], dim=2)
            # traj_fake_rel = torch.cat([obs_traj_rel[0], pred_traj_fake_rel], dim=2)     # V C T

            # scores_fake = discriminator(traj_fake_rel, traj_fake_rel)
            # scores_real = discriminator(traj_real_rel, traj_real_rel)

            # d_loss = d_loss_fn(scores_real, scores_fake)
            # d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            mask_sum += (pred_traj_gt.size(1) * PRED_LEN)       # V
            total_traj += pred_traj_gt.size(1)
            if limit and total_traj >= NUM_SAMPLES_CHECK:
                break

    # metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * PRED_LEN)
    metrics['fde'] = sum(f_disp_error) / total_traj
    generator.train()
    return metrics

main()