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

'''
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--lane_gate_size', type=int, default=4)
parser.add_argument('--output_hidden_size', type=int, default=16)
parser.add_argument('--t_predict', type=int, default=7)
parser.add_argument('--temporal_length', type=int, default=11)
parser.add_argument('--spatial_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--sim_step', type=float, default=0.1)
parser.add_argument('--delta_T', type=int, default=7)
args = parser.parse_args()

model = TP_lstm(args)
model.eval()
test_generator1 = batchGenerator(
            ['300_3'], 
            batchSize=30, 
            simTimeStep=args.sim_step,
            seqLength=args.temporal_length,
            seqPredict=args.t_predict,
            deltaT=args.delta_T
            )
t0 = time.time()
test_generator1.generateBatchForBucket()
t1 = time.time()
print(t1-t0)
'''
'''
init_data = torch.tensor(test_generator.CurrentSequences[:, :, :args.t_predict, :]).float()
temporal_data = torch.tensor(test_generator.CurrentSequences[:, 0, args.t_predict:, :]).float()
data = Variable(torch.tensor(test_generator.CurrentSequences).float())
laneT = torch.tensor(test_generator.CurrentLane).float()
target = Variable(torch.tensor(test_generator.CurrentOutputs).float())

criterion = torch.nn.MSELoss()
#output = model.infer(temporal_data, init_data, laneT)
t1 = time.time()
output = model(data, Variable(laneT))
t2 = time.time()
output = criterion(target[:, :, :, 0], output)
t3 = time.time()
output.backward()
t4 = time.time()
print(t2-t1, t3-t2, t4-t3)
'''
#prefix = train.trainmd(conf.args)
#refix = "multi_Dimension_LSTM_mdLSTM_123456"
#train.testmd(prefix)

