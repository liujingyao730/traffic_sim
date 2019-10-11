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
import matplotlib.pyplot as plt

import argparse
import os
import time
import pickle
import yaml

from utils import batchGenerator
from model import TP_lstm
from model import MD_lstm_cell
from model import FCNet
from model import loss_function
import conf

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="eva")

    args = parser.parse_args()
    eva(args)

def eva(args):

    with open(os.path.join(conf.configPath, args.config+'.yaml'), encoding='UTF-8') as config:
        args = yaml.load(config)

    model_prefix = args["model_prefix"]
    use_epoch = args["use_epoch"]
    
    load_directory = os.path.join(conf.logPath, args["model_prefix"])
    eva_prefix = args["eva_prefix"]
    eva_generator = batchGenerator(eva_prefix, args)
    eva_generator.CurrentTime = 0
    eva_generator.CurrentEdgePoint = 1

    eva_generator.generateNewMatrix()

    data = torch.tensor(eva_generator.CurrentSequences).float()
    init_data = data[:, :args["t_predict"], :]
    input_data = data[0, args["t_predict"]:, :]
    target = torch.tensor(eva_generator.CurrentOutputs)[:, :, 2].float()

    seg = segment(args, init_data=init_data)
    output = seg.simulation(input_data)

    number_before = data[:, args["t_predict"]:, 2]
    In = data[0, args["t_predict"]:, 1].view(1, args["temporal_length"]-args["t_predict"])
    inflow = torch.cat((In, output[:args["spatial_length"]-1,:]), 0)
    number_caculate = number_before + inflow - output
    
    real_flow = torch.sum(target, dim=0).numpy()
    target_flow = torch.sum(number_caculate, dim=0).detach().numpy()
    print(metrics.explained_variance_score(real_flow, target_flow))
    print(metrics.r2_score(real_flow, target_flow))
    print(metrics.mean_absolute_error(real_flow, target_flow))
    x = range(len(real_flow))
    plt.plot(x, real_flow, 's-', color='r', label='real')
    plt.plot(x, target_flow, 'o-', color='g', label='predict')
    plt.xlabel('time')
    plt.ylabel('num_vehicle')
    plt.legend(loc='best')
    plt.ylim(ymin=0)
    plt.ylim(ymax=200)
    plt.show()

    plt.plot(x, real_flow-target_flow)
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('error with time')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(311)
    im = ax.imshow((target-number_caculate).detach().numpy(), cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title('error')
    ax = fig.add_subplot(312)
    im = ax.imshow(target.detach().numpy(), cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title('ground truth')
    ax = fig.add_subplot(313)
    im = ax.imshow(number_caculate.detach().numpy(), cmap=plt.cm.hot_r, vmin=0, vmax=9)
    plt.colorbar(im)
    plt.title('simulate result')
    plt.show()

'''
class sim_cell(nn.Module):

    #cell1

    def __init__(self, input_size, hidden_size):

        #目前的结构是在传统LSTM cell之外，对隐层状态施加一个门控制，这里我们叫做空间门，后续可能会要改
        #input_size: int 输入的维度
        #hidden_size: int 隐层状态的维度

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        
        self.sigma = torch.nn.Sigmoid()

        #self.spatial_embedding = torch.nn.Linear(2*hidden_size, hidden_size)
        self.after_embedding = torch.nn.Linear(hidden_size, hidden_size)
        self.before_embedding = torch.nn.Linear(hidden_size, hidden_size)
         

    def forward(self, inputs, h_s_t, c_s_t, h_after_t, h_before_t):

        #inputs: tensor [batch_size, spatial_size, input_size] 当前节点此时刻的输入
        #h_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的隐层状态
        #c_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的细胞状态
        #h_after_t: tensor [batch_size, spatial_size, hidden_size] 下一个节点前一个时刻的隐层状态
        #h_before_t: tensor [batch_size, spatial_size, hidden_size] 前一个节点前一个时刻的隐层状态

        [spatial_size, hidden_size] = h_s_t.shape

        h_hat_after = self.after_embedding(h_after_t)
        h_hat_before = self.before_embedding(h_before_t)

        h_hat, c_s_tp = self.cell(inputs, (h_s_t, c_s_t))

        #输出结果
        h_s_tp = h_hat + h_hat_after + h_hat_before

        return h_s_tp, c_s_tp
'''
'''
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
'''

class sim_cell(nn.Module):

    #cell3

    def __init__(self, input_size, hidden_size):

        #input_size: int 输入的维度
        #hidden_size: int 隐层状态的维度

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        
        self.sigma = torch.nn.Sigmoid()

        self.spatial_forget = torch.nn.Linear(2*hidden_size, hidden_size)
        self.spatial_input = torch.nn.Linear(2*hidden_size, hidden_size)
         

    def forward(self, inputs, h_s_t, c_s_t, h_after_t, h_before_t):

        #inputs: tensor [batch_size, spatial_size, input_size] 当前节点此时刻的输入
        #h_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的隐层状态
        #c_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的细胞状态
        #h_after_t: tensor [batch_size, spatial_size, hidden_size] 下一个节点前一个时刻的隐层状态
        #h_before_t: tensor [batch_size, spatial_size, hidden_size] 前一个节点前一个时刻的隐层状态

        [spatial_size, hidden_size] = h_s_t.shape

        spatial_gate = torch.cat((h_after_t, h_before_t), dim=1) 

        spatial_f = self.spatial_forget(spatial_gate)
        spatial_f = self.sigma(spatial_f)
        spatial_i = self.spatial_input(spatial_gate)
        spatial_i = self.sigma(spatial_i)

        c_s_t = c_s_t * spatial_f
        h_s_t = h_s_t * spatial_i

        h_s_tp, c_s_tp = self.cell(inputs, (h_s_t, c_s_t))

        return h_s_tp, c_s_tp

'''
class sim_cell(nn.Module):

    #cell4

    def __init__(self, input_size, hidden_size):

        #目前的结构是在传统LSTM cell之外，对隐层状态施加一个门控制，这里我们叫做空间门，后续可能会要改
        #input_size: int 输入的维度
        #hidden_size: int 隐层状态的维度

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        
        self.sigma = torch.nn.Sigmoid()

        self.after_gate = torch.nn.Linear(input_size+hidden_size, hidden_size)
        self.before_gate = torch.nn.Linear(input_size+hidden_size, hidden_size)
         

    def forward(self, inputs, h_s_t, c_s_t, h_after_t, h_before_t):

        #inputs: tensor [batch_size, spatial_size, input_size] 当前节点此时刻的输入
        #h_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的隐层状态
        #c_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的细胞状态
        #h_after_t: tensor [batch_size, spatial_size, hidden_size] 下一个节点前一个时刻的隐层状态
        #h_before_t: tensor [batch_size, spatial_size, hidden_size] 前一个节点前一个时刻的隐层状态

        [spatial_size, hidden_size] = h_s_t.shape
        [_, input_size] = inputs.shape

        cell_input = torch.cat((h_s_t, inputs), dim=1)

        i_after = self.after_gate(cell_input)
        i_before = self.before_gate(cell_input)
        i_after = self.sigma(i_after)
        i_before = self.sigma(i_before)

        h_hat, c_s_tp = self.cell(inputs, (h_s_t, c_s_t))

        #输出结果
        h_s_tp = h_hat + h_after_t * i_after + h_before_t * i_before

        return h_s_tp, c_s_tp
'''

class segment:

    def __init__(self, args, init_data=None):

        self.use_epoch = args["use_epoch"]
        self.model_prefix = args["model_prefix"]
        self.spatial_length = args["spatial_length"]
        
        
        load_directory = os.path.join(conf.logPath, args["model_prefix"])
        file = os.path.join(load_directory, str(args["use_epoch"])+'.tar')
        checkpoint = torch.load(file)
        
        self.use_cuda = args["use_cuda"]
        self.hidden_size = args["hidden_size"]
        tp_lstm = TP_lstm(args)
        tp_lstm.load_state_dict(checkpoint['state_dict'])
        self.model = sim_cell(args["input_size"], args["hidden_size"])
        self.model.load_state_dict(tp_lstm.cell.state_dict())
        self.output_layer = FCNet(layerSize=[args["hidden_size"], args["output_hidden_size"], 1])
        self.output_layer.load_state_dict(tp_lstm.output_layer.state_dict())
        self.hidden_state = torch.zeros(self.spatial_length, args["hidden_size"])
        self.predict_input = torch.zeros(self.spatial_length, args["input_size"])

        if self.use_cuda:
            self.model = self.model.cuda()
            self.output_layer = self.output_layer.cuda()
            self.hidden_state = self.hidden_state.cuda()
        
        if not init_data is None:

            if self.use_cuda:
                init_data = init_data.cuda()
            
            self.predict_input = init_data[:, -1, :]
            cell_state = init_data.data.new(self.spatial_length, args["hidden_size"]).fill_(0).float()
            hidden_state = init_data.data.new(self.spatial_length, args["hidden_size"]).fill_(0).float()
            hidden_state_after = init_data.data.new(self.spatial_length, args["hidden_size"]).fill_(0).float()
            hidden_state_before = init_data.data.new(self.spatial_length, args["hidden_size"]).fill_(0).float()
        
            zero_hidden = init_data.data.new(1, args["hidden_size"]).fill_(0).float()
            [spatial, temporal, _] = init_data.shape
            for time in range(temporal):
                hidden_state, cell_state = self.model(init_data[:, time, :], 
                    hidden_state, cell_state, hidden_state_after, hidden_state_before)
                hidden_state_after = torch.cat((hidden_state[1:, :], zero_hidden))
                hidden_state_before = torch.cat((zero_hidden, hidden_state[:self.spatial_length-1, :]))
        else:
            virtual_input = torch.zeros(self.spatial_length, args["input_size"])
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

        if self.use_cuda:
            output = output.cpu()

        return output
            

if __name__ == "__main__":
    main()
