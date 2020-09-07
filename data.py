import os
import math
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, Dataset

from constants import *

def read_file(path, delim='\t'):
    '''
    读取数据文件为ndarray
    :param path: 文件路径
    :param delim: 文件行中的分隔符
    :return: 返回ndarray格式的数据
    '''
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def collate(data):
    '''
    把一批dataset的实例转换为包含miniBatch的张量
    :param data:
    :return:Batch张量
    '''
    # obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, vgg_list = zip(*data)
    obs_seq, pred_seq, obs_seq_rel, pred_seq_rel= zip(*data)  # 解压数据
    obs_seq = torch.cat(obs_seq, dim=0).permute(3, 0, 1, 2)     # 观测轨迹变为 8*n*64*2
    pred_seq = torch.cat(pred_seq, dim=0).permute(2, 0, 1)      # 12*3*2
    obs_seq_rel = torch.cat(obs_seq_rel, dim=0).permute(3, 0, 1, 2)
    pred_seq_rel = torch.cat(pred_seq_rel, dim=0).permute(2, 0, 1)
    # vgg_list = torch.cat(vgg_list, dim=0).repeat(obs_seq.size(1), 1, 1)  # 第0维重复n次
    # return tuple([obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, vgg_list])
    return tuple([obs_seq, pred_seq, obs_seq_rel, pred_seq_rel])
def data_loader(path):
    dset = TrajDataset(path)
    loader = DataLoader(dset, 1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)
    return dset, loader

class TrajDataset(Dataset):
    def __init__(self, data_dir):

        super(TrajDataset, self).__init__()
        all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path[0] != "." and path.endswith(".txt")]
        # 路径下的所有TXT文件
        num_peds_in_seq = []    # 行人编号序列
        seq_list = []
        seq_list_rel = []
        seq_len = OBS_LEN + PRED_LEN        # 8+12  8+8 
        fet_map = {}
        fet_list = []

        for path in all_files:
            data = read_file(path)  # 读取文件
            frames = np.unique(data[:, 0]).tolist()  # 获取帧号集合（不重复）
            # os.pat
            # hkl_path = os.path.splitext(path)[0] + ".pkl"   # 获取数据文件的文件名和路径，读取对应的pkl文件
            # with open(hkl_path, 'rb') as handle:
            #     new_fet = pickle.load(handle, encoding='bytes')
            # fet_map[hkl_path] = torch.from_numpy(new_fet)

            frame_data = [data[frame == data[:, 0], :] for frame in frames]     # 按帧取信息,非对称。
            num_sequences = len(frames) - seq_len + 1           # 总帧数减去预测过程的长度（16/20）得到序列个数
            for idx in range(0, num_sequences+1):

                curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)   # 取连续的seq_len帧  123*4
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])                       # 连续帧中的行人编号16
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, seq_len))            # n*2*20
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))                # n*2*20
                
                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):   # 遍历id 找出id在seq_len里的帧
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]  # id为ped_id的行人的帧 2*4
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)      # 四舍五入
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx      # 观测点到该行人出现的第一帧的距离
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1   # 观测点到该行人出现的最后一帧的距离
                    if pad_end - pad_front != seq_len:                      # 该行人出现的时间少于20
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])        # 取x,y并转置
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)         # 存放结果2*n
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]  # 后时刻减去前时刻的坐标差
                    curr_seq[num_peds_considered, :, pad_front:pad_end] = curr_ped_seq    # 16*2*20
                    curr_seq_rel[num_peds_considered, :, pad_front:pad_end] = rel_curr_ped_seq  # 坐标差
                    num_peds_considered += 1



                if num_peds_considered > 1:
                    num_peds_in_seq.append(num_peds_considered)     # 当前序列中被考虑的人数

                    curr_seq_exp = np.zeros((num_peds_considered, MAX_PEDS, 2, seq_len))   # 3*64*2*20
                    curr_seq_rel_exp = np.zeros((num_peds_considered, MAX_PEDS, 2, seq_len))    # 同型
                    for i in range(num_peds_considered):                # 在这个序列中被考虑的人
                        curr_seq_exp[i, 0, :, :] = curr_seq[i]
                        curr_seq_exp[i, 1:i+1, :, :] = curr_seq[0:i]
                        curr_seq_exp[i, i+1:num_peds_considered, :, :] = curr_seq[i+1:num_peds_considered]

                        dists = (curr_seq_exp[i, :] - curr_seq_exp[i, 0]) ** 2  # 其余行人与i的欧式距离
                        dists = np.sum(np.sum(dists, axis=2), axis=1)           # 对2维度求和，然后对1维度求和
                        idxs = np.argsort(dists)                                # 排序返回索引顺序
                        curr_seq_exp[i, :] = curr_seq_exp[i, :][idxs]


                        curr_seq_rel_exp[i, 0, :, :] = curr_seq_rel[i]
                        curr_seq_rel_exp[i, 1:i+1, :, :] = curr_seq_rel[0:i]
                        curr_seq_rel_exp[i, i+1:num_peds_considered, :, :] = curr_seq_rel[i+1:num_peds_considered]
                        curr_seq_rel_exp[i, :] = curr_seq_rel_exp[i, :][idxs]

                    seq_list.append(curr_seq_exp[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel_exp[:num_peds_considered])
                    # fet_list.append(hkl_path)     # 每个人对应一个hkl_path

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :, :OBS_LEN]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, 0, :, OBS_LEN:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :, :OBS_LEN]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, 0, :, OBS_LEN:]).type(torch.float)

        # self.fet_map = fet_map
        # self.fet_list = fet_list

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            # self.fet_map[self.fet_list[index]]
        ]
        return out


da,loade= data_loader("./datasets/eth/train")
for  step,(obs,pre,bos,pred) in enumerate(loade):
    print(obs.shape)
    print(pre.shape)
    print(bos.shape)
    print(pred.shape)
    print("ok")
    continue



