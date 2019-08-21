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
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties

import argparse
import os
import time
import pickle

from utils import batchGenerator
from model import TP_lstm
from model import loss_function
import conf


def main():
    
    parser = argparse.ArgumentParser()
    
    # 网络结构
    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--emmbedding_size', type=int, default=8)
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
    parser.add_argument('--model_prefix', type=str, default='8-21-1')
    parser.add_argument('--use_epoch', type=int, default=0)

    # 测试相关
    parser.add_argument('--test_batchs', type=int, default=100)

    args = parser.parse_args()
    test(args)

def test(args):

    model_prefix = args.model_prefix
    test_prefix = conf.args["test_prefix"]
    test_generator = batchGenerator(test_prefix, args)
    test_generator.generateBatchRandomForBucket()
    spatial_length = test_generator.CurrentSequences.shape[1]

    model = TP_lstm(args)

    load_directory = os.path.join(conf.logPath, model_prefix)
    
    def checkpath(epoch):
        return os.path.join(load_directory, str(epoch)+'.tar')

    result_file = os.path.join(conf.resultPath, model_prefix, '.txt')

    file = checkpath(args.use_epoch)
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['state_dict'])

    flow_loss_meter = loss_function()
    mes_loss_meter = torch.nn.MSELoss()
    result = torch.zeros(spatial_length, args.temporal_length-args.t_predict)
    flow_result = torch.zeros(spatial_length, args.temporal_length-args.t_predict)

    if args.use_cuda:
        model = model.cuda()
        flow_loss_meter = flow_loss_meter.cuda()
        mes_loss_meter = mes_loss_meter.cuda()
        result = result.cuda()
        flow_result = flow_result.cuda()

    for batch in range(args.test_batchs):

        test_generator.generateBatchRandomForBucket()

        data = torch.tensor(test_generator.CurrentSequences).float()
        init_data = data[:, :, :args.t_predict, :]
        temporal_data = data[:, 0, args.t_predict:, :]
        laneT = torch.tensor(test_generator.CurrentLane).float()
        target = torch.tensor(test_generator.CurrentOutputs).float()

        if args.use_cuda:
            data = data.cuda()
            init_data = init_data.cuda()
            temporal_data = temporal_data.cuda()
            laneT = laneT.cuda()
            target = target.cuda()

        output = model.infer(temporal_data, init_data, laneT)
        
        number_current = target[:, :, :, 2]
        number_before = data[:, :, args.t_predict:, 2]
        In = data[:, 0, args.t_predict:, 1].view(-1, 1, args.temporal_length-args.t_predict)
        inflow = torch.cat((In, output[:, :spatial_length-1,:]), 1)
        number_caculate = number_before + inflow - output
        
        mes_loss = target[:, :, :, 0] - output
        flow_loss = number_current - number_caculate

        result = result + torch.mean(mes_loss, 0, keepdim=True).squeeze(0)
        flow_result = flow_result + torch.mean(flow_loss, 0, keepdim=True).squeeze(0)

    result = result / args.test_batchs
    flow_result = flow_result / args.test_batchs
    flow_result = flow_result.cpu().detach().numpy()
    result = result.cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_yticks(range(result.shape[0]))
    ax.set_yticklabels(range(result.shape[0]))
    ax.set_xticks(range(result.shape[1]))
    ax.set_xticklabels(range(result.shape[1]))

    im = ax.imshow(result, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title('meam square error')

    ax = fig.add_subplot(122)
    ax.set_yticks(range(flow_result.shape[0]))
    ax.set_yticklabels(range(flow_result.shape[0]))
    ax.set_xticks(range(flow_result.shape[1]))
    ax.set_xticklabels(range(flow_result.shape[1]))

    im = ax.imshow(flow_result, cmap=plt.cm.hot_r)

    plt.colorbar(im)
    plt.title('mean flow square error')

    plt.show()

    
if __name__ == '__main__':
    main()