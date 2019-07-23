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
parser.add_argument('--t_predict', type=int, default=7)
parser.add_argument('--temporal_length', type=int, default=13)
parser.add_argument('--spatial_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--sim_step', type=float, default=0.1)
parser.add_argument('--delta_T', type=int, default=7)
args = parser.parse_args()

model = TP_lstm(args)
model.eval()
test_generator = batchGenerator(
            ['300_3'], 
            batchSize=args.batch_size, 
            simTimeStep=args.sim_step,
            seqLength=args.temporal_length,
            seqPredict=args.t_predict,
            deltaT=args.delta_T
            )
test_generator.generateBatchForBucket()
init_data = torch.tensor(test_generator.CurrentSequences[:, 0, :]).float()
temporal_data = torch.tensor(test_generator.CurrentSequences[0, 1:, 1]).float()
data = Variable(torch.tensor(test_generator.CurrentSequences).float())
laneT = torch.tensor(test_generator.CurrentLane).float()
target = torch.tensor(test_generator.CurrentOutputs)
criterion = torch.nn.MSELoss()
output = model.infer(temporal_data, init_data, laneT)
output = model(data, Variable(laneT))
print(output.shape)
#prefix = train.trainmd(conf.args)
#refix = "multi_Dimension_LSTM_mdLSTM_123456"
#train.testmd(prefix)
