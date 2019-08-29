'''
这就是一个用来乱写乱动的文件整体完成后会删掉
'''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
parser.add_argument('--embedding_size', type=int, default=8)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--lane_gate_size', type=int, default=4)
parser.add_argument('--output_hidden_size', type=int, default=16)
parser.add_argument('--t_predict', type=int, default=7)
parser.add_argument('--temporal_length', type=int, default=11)
parser.add_argument('--spatial_length', type=int, default=5)

# 训练参数
parser.add_argument('--batch_size', type=int, default=20)
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
parser.add_argument('--model_prefix', type=str, default='new_data_new_time')
parser.add_argument('--use_epoch', type=int, default=49)
parser.add_argument('--test_batchs', type=int, default=500)

args = parser.parse_args()

model_prefix = args.model_prefix
test_batchs = args.test_batchs
use_epoch = args.use_epoch

load_directory = os.path.join(conf.logPath, model_prefix)
    
def checkpath(epoch):
    return os.path.join(load_directory, str(epoch)+'.tar')

file = checkpath(use_epoch)
checkpoint = torch.load(file)
argsfile = open(os.path.join(load_directory, 'config.pkl'), 'rb')
args = pickle.load(argsfile)

dg = batchGenerator(["500", "1000", "2000"], args)
dg.generateBatchRandomForBucket()

model = TP_lstm(args)
model.load_state_dict(checkpoint['state_dict'])
if args.use_cuda:
    model = model.cuda()

number = []

for i in range(13):
    number.append(np.array([]))

for i in range(test_batchs):
    
    dg.generateBatchRandomForBucket()

    data = torch.Tensor(dg.CurrentSequences)
    laneT = torch.Tensor(dg.CurrentLane)
    target = torch.Tensor(dg.CurrentOutputs)

    if args.use_cuda:
        data = data.cuda()
        laneT = laneT.cuda()
        target = target.cuda()

    output = model(data, laneT)

    output = output.view(-1)
    target = target[:, :, :, 0].view(-1)

    for i in range(13):
        number[i] = np.concatenate((number[i], output[torch.eq(target, i)].cpu().detach().numpy()))

for i in range(13):
    print("target {}, median value: {:.3f}, mean value: {:.3f}, std: {:.3f}".format(i, np.median(number[i]), np.mean(number[i]), np.std(number[i])))
'''
x = np.array(range(13))
y = np.array(number)
plt.figure()
plt.title('输入输出分布', fontproperties='SimHei')
plt.boxplot(y, labels=x)
plt.show()
'''