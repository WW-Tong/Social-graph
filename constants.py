import numpy as np
import torch
import math
OBS_LEN = 8
PRED_LEN = 12
NUM_WORKERS = 0

MLP_DIM = 64
H_DIM = 64
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32
NOISE_DIM = 8

DATASET_NAME = 'eth'
NUM_ITERATIONS = 20000
NUM_EPOCHS = 200
G_LR = 5e-4
D_LR = 2e-4
G_STEPS = 1
D_STEPS = 2

MAX_PEDS = 64
BEST_K = 20
PRINT_EVERY = 100
NUM_SAMPLES = 20
NUM_SAMPLES_CHECK = 5000

DROPOUT = 0.2

ATTN_L = 196
ATTN_D = 512
ATTN_D_DOWN = 16

# a=torch.ones(3,4)
# b=torch.rand(a.shape)
# print(b.size())
# # s=[]
# inp=torch.randn(2,3)
# ina=torch.randn(2,3)
# inb=torch.randn(2,3)
# s.append(inp)
# s.append(ina)
# s.append(inb)
# s=torch.stack(s, dim=2) 
# print(s)
# print(s.shape)
# print(s.reshape(6,3).shape)
