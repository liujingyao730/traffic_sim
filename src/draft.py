'''
这就是一个用来乱写乱动的文件整体完成后会删掉
'''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np

import argparse
import os
import time
import pickle

import conf
from utils import batchGenerator
from model import MD_lstm_cell
from model import TP_lstm
import dataProcess as dp 
import train

parser = argparse.ArgumentParser()
    
# 网络结构
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--lane_gate_size', type=int, default=4)
parser.add_argument('--output_hidden_size', type=int, default=16)
parser.add_argument('--t_predict', type=int, default=7)
parser.add_argument('--temporal_length', type=int, default=11)
parser.add_argument('--spatial_length', type=int, default=5)

# 训练参数
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--save_every', type=int, default=500)
parser.add_argument('--learing_rate', type=float, default=0.003)
parser.add_argument('--decay_rate', type=float, default=0.95)
parser.add_argument('--lambda_param', type=float, default=0.0005)
parser.add_argument('--use_cuda', action='store_true', default=True)
parser.add_argument('--flow_loss_weight', type=float, default=1)
parser.add_argument('--grad_clip', type=float, default=10.)

# 数据参数
parser.add_argument('--sim_step', type=float, default=0.1)
parser.add_argument('--delta_T', type=int, default=7)
parser.add_argument('--cycle', type=int, default=100)
parser.add_argument('--green_pass', type=int, default=52)
parser.add_argument('--yellow_pass', type=int, default=55)

# 模型相关
parser.add_argument('--model_prefix', type=str, default='8-17')

args = parser.parse_args()

dg = batchGenerator(["300", "400", "500"], args)
for i in range(50):
    dg.generateBatchRandomForBucket()
    print(dg.CurrentSequences.shape)