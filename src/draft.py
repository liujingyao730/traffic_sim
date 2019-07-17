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
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--lane_gate_size', type=int, default=4)
parser.add_argument('--output_hidden_size', type=int, default=16)
parser.add_argument('--t_predict', type=int, default=20)

args = parser.parse_args()
model = TP_lstm(args)
lane = Variable(torch.tensor([1]).float())
init_input = Variable(torch.randn(5, 3))
temporal_input = Variable(torch.randn(23, 3))
target = Variable(torch.randn(5, 4, 3))
criterion = torch.nn.MSELoss()
output = model(temporal_input, init_input, lane)
loss = criterion(output, target)
loss.backward()
print(output.shape)
#prefix = train.trainmd(conf.args)
#refix = "multi_Dimension_LSTM_mdLSTM_123456"
#train.testmd(prefix)

#prefix = "multi_Dimension_LSTM_mdLSTM_123456_to_mix123456"
#dp.bucketResult(prefix)
'''
result = conf.resultPath + "/multi_Dimension_LSTM_mdLSTM_123456_to_mix123456_result.npy"
target = conf.resultPath + "/multi_Dimension_LSTM_mdLSTM_123456_to_mix123456_target.npy"

r = np.load(result)
t = np.load(target)

print(r.shape)
print(t.shape)
'''