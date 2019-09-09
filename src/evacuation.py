import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from torchnet import meter
import numpy as np 
import sklearn.metrics as metrics
from tqdm import tqdm
import torchvision.models as models
import pyecharts as pe
import pandas as pd

import argparse
import os
import time
import pickle

from utils import batchGenerator
from model import TP_lstm
from model import MD_lstm_cell
from model import FCNet
from model import loss_function
import conf

class sim_cell(nn.Module):

    # cell2 只是去掉了原版本里的batch，每次只能处理一个路段

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        
        self.sigma = torch.nn.Sigmoid()

        self.spatial_gate = torch.nn.Linear(2*hidden_size, hidden_size)
        self.spatial_embedding = torch.nn.Linear(2*hidden_size, hidden_size)

    def forward(self, inputs, h_s_t, c_s_t, h_after_t, h_before_t):

        #inputs: tensor [batch_size, spatial_size, input_size] 当前节点此时刻的输入
        #h_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的隐层状态
        #c_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的细胞状态
        #h_after_t: tensor [batch_size, spatial_size, hidden_size] 下一个节点前一个时刻的隐层状态
        #h_before_t: tensor [batch_size, spatial_size, hidden_size] 前一个节点前一个时刻的隐层状态
       
        [spatial_size, hidden_size] = h_s_t.shape

        spatial_info = torch.cat((h_after_t, h_before_t), dim=1)
        
        #处理batch 因为batch内部的不同路段的不同节点在这里都是独立的，所以可以分开来
        gate_controller = spatial_info.clone()
        spatial_info = self.spatial_embedding(spatial_info)
        gate_controller = self.spatial_gate(gate_controller)
        spatial_info = spatial_info * gate_controller

        h_hat, c_s_tp = self.cell(inputs, (h_s_t, c_s_t))
        
        #空间门
        h_s_tp = h_hat + spatial_info 

        return h_s_tp, c_s_tp

class segment:

    def __init__(self, args, init_data=None):

        self.use_epoch = args.use_epoch
        self.model_prefix = args.model_prefix
        self.spatial_length = args.spatial_length

        load_directory = os.path.join(conf.logPath, self.model_prefix)

        file = os.path.join(load_directory, str(self.use_epoch)+'.tar')
        checkpoint = torch.load(file)
        argsfile = open(os.path.join(load_directory, 'config.pkl'), 'rb')
        args = pickle.load(argsfile)
        
        self.use_cuda = args.use_cuda
        self.hidden_size = args.hidden_size
        tp_lstm = TP_lstm(args)
        tp_lstm.load_state_dict(checkpoint['state_dict'])
        self.model = sim_cell(args.input_size, args.hidden_size)
        self.model.load_state_dict(tp_lstm.cell.state_dict())
        self.output_layer = FCNet(layerSize=[args.hidden_size, args.output_hidden_size, 1])
        self.output_layer.load_state_dict(tp_lstm.output_layer.state_dict())
        self.hidden_state = torch.zeros(self.spatial_length, args.hidden_size)
        self.predict_input = torch.zeros(self.spatial_length, args.input_size)

        if self.use_cuda:
            self.model = self.model.cuda()
            self.output_layer = self.output_layer.cuda()
            self.hidden_state = self.hidden_state.cuda()
        
        if not init_data is None:

            if self.use_cuda:
                init_data = init_data.cuda()
            
            self.predict_input = init_data[:, -1, :]
            cell_state = init_data.data.new(self.spatial_length, args.hidden_size).fill_(0).float()
            hidden_state = init_data.data.new(self.spatial_length, args.hidden_size).fill_(0).float()
            hidden_state_after = init_data.data.new(self.spatial_length, args.hidden_size).fill_(0).float()
            hidden_state_before = init_data.data.new(self.spatial_length, args.hidden_size).fill_(0).float()
        
            zero_hidden = init_data.data.new(1, args.hidden_size).fill_(0).float()
            [spatial, temporal, _] = init_data.shape
            for time in range(temporal):
                hidden_state, cell_state = self.model(init_data[:, time, :], 
                    hidden_state, cell_state, hidden_state_after, hidden_state_before)
                hidden_state_after = torch.cat((hidden_state[1:, :], zero_hidden))
                hidden_state_before = torch.cat((zero_hidden, hidden_state[:self.spatial_length-1, :]))
        else:
            virtual_input = torch.zeros(self.spatial_length, args.input_size)
            if self.use_cuda:
                virtual_input = virtual_input.cuda()
            hidden_state, cell_state = self.model(virtual_input,
                self.hidden_state, self.hidden_state, self.hidden_state, self.hidden_state)
        
        self.hidden_state = hidden_state
        self.cell_state = cell_state


    def simulation(self, input_data):

        [temporal_length, _] = input_data.shape
        output = self.predict_input.data.new(self.spatial_length, temporal_length).fill_(0).float()
        zero_hidden = self.predict_input.data.new(1, self.hidden_size).fill_(0).float()
        hidden_state = self.hidden_state
        predict_input = self.predict_input
        cell_state = self.cell_state
        if self.use_cuda:
            input_data = input_data.cuda()
        
        for time in range(temporal_length):
            
            outflow = self.output_layer(hidden_state)

            # output是下一时刻的输出，所以与当前时刻要错后一位
            if time > 0:
                output[:, time-1] = outflow.view(self.spatial_length)
            
            # 每个节点的输入是由第一个节点的输入与预测得到的从第一个节点到倒数第二个节点的流出值
            inflow = torch.cat((input_data[time, 1].view(1,1), outflow[:self.spatial_length-1]))
            
            # 每个节点的车辆数是由每个节点前一个时刻的车辆数以及输入和输出车辆数计算得到的
            number = predict_input[:, 2].view(self.spatial_length, 1) - outflow + inflow
            
            # 拼接得到下一时刻的输入
            predict_input = torch.cat((outflow, inflow, number), 1)
            hidden_state_after = torch.cat((hidden_state[1:, :], zero_hidden))
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:self.spatial_length-1, :]))
            hidden_state, cell_state = self.model(predict_input, hidden_state, cell_state, hidden_state_after, hidden_state_before)

        outflow = self.output_layer(hidden_state)
        output[:, -1] = outflow.view(self.spatial_length)

        return output
            

    
parser = argparse.ArgumentParser()
    
# 网络结构
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--embedding_size', type=int, default=8)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--lane_gate_size', type=int, default=4)
parser.add_argument('--output_hidden_size', type=int, default=16)
parser.add_argument('--t_predict', type=int, default=4)
parser.add_argument('--temporal_length', type=int, default=8)
parser.add_argument('--spatial_length', type=int, default=5)

# 训练参数
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--save_every', type=int, default=500)
parser.add_argument('--learing_rate', type=float, default=0.003)
parser.add_argument('--decay_rate', type=float, default=0.95)
parser.add_argument('--lambda_param', type=float, default=0.0005)
parser.add_argument('--use_cuda', action='store_true', default=True)
parser.add_argument('--flow_loss_weight', type=float, default=0)
parser.add_argument('--grad_clip', type=float, default=10.)

# 数据参数
parser.add_argument('--sim_step', type=float, default=0.1)
parser.add_argument('--delta_T', type=int, default=10)
parser.add_argument('--cycle', type=int, default=100)
parser.add_argument('--green_pass', type=int, default=52)
parser.add_argument('--yellow_pass', type=int, default=55)
parser.add_argument('--mask_level', type=int, default=3)

# 模型相关
parser.add_argument('--model_prefix', type=str, default='multi_dimension')

args = parser.parse_args()

dg = batchGenerator(['300'], args)
dg.generateBatchForBucket()
init_data = torch.Tensor(dg.CurrentSequences)[1, :, :4, :]

parser = argparse.ArgumentParser()

# 模型相关
parser.add_argument('--model_prefix', type=str, default='cell_2')
parser.add_argument('--spatial_length', type=int, default=6)
parser.add_argument('--use_epoch', type=int, default=49)

# 测试相关
parser.add_argument('--test_batchs', type=int, default=50)

args = parser.parse_args()

seg = segment(args, init_data)
input_data = torch.Tensor(dg.CurrentSequences)[1, 0, 4:, :]
print(seg.simulation(input_data))
