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

    # 模型相关
    parser.add_argument('--model_prefix', type=str, default='cell2_with_mask6')
    parser.add_argument('--use_epoch', type=int, default=16)

    # 测试相关
    parser.add_argument('--test_batchs', type=int, default=50)

    args = parser.parse_args()
    visualization(args)

def visualization(args, laneNumber=conf.laneNumber):

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

    test_prefix = conf.args["test_prefix"]
    test_generator = batchGenerator(test_prefix, args)
    test_generator.generateBatchRandomForBucket(laneNumber)
    spatial_length = test_generator.CurrentSequences.shape[1]

    model = TP_lstm(args)
    model.load_state_dict(checkpoint['state_dict'])

    flow_loss_meter = loss_function()
    mes_loss_meter = torch.nn.MSELoss()
    result = torch.zeros(spatial_length, args.temporal_length-args.t_predict)
    flow_result = torch.zeros(spatial_length, args.temporal_length-args.t_predict)
    abs_result = torch.zeros(spatial_length, args.temporal_length-args.t_predict)
    abs_flow_result = torch.zeros(spatial_length, args.temporal_length-args.t_predict)

    if args.use_cuda:
        model = model.cuda()
        flow_loss_meter = flow_loss_meter.cuda()
        mes_loss_meter = mes_loss_meter.cuda()
        result = result.cuda()
        flow_result = flow_result.cuda()
        abs_result = abs_result.cuda()
        abs_flow_result = abs_flow_result.cuda()

    for batch in range(test_batchs):

        test_generator.generateBatchRandomForBucket(laneNumber)

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
        #output = model(data, laneT)

        number_current = target[:, :, :, 2]
        number_before = data[:, :, args.t_predict:, 2]
        In = data[:, 0, args.t_predict:, 1].view(-1, 1, args.temporal_length-args.t_predict)
        inflow = torch.cat((In, output[:, :spatial_length-1,:]), 1)
        number_caculate = number_before + inflow - output
        
        mes_loss = target[:, :, :, 0] - output
        flow_loss = number_current - number_caculate

        abs_mes = torch.abs(mes_loss)
        abs_flow_loss = torch.abs(flow_loss)

        result = result + torch.mean(mes_loss, 0, keepdim=True).squeeze(0)
        flow_result = flow_result + torch.mean(flow_loss, 0, keepdim=True).squeeze(0)
        abs_result = abs_result + torch.mean(abs_mes, 0, keepdim=True).squeeze(0)
        abs_flow_result = abs_flow_result + torch.mean(abs_flow_loss, 0, keepdim=True).squeeze(0)


    result = result / test_batchs
    flow_result = flow_result / test_batchs
    abs_result = abs_result / test_batchs
    abs_flow_result = abs_flow_result / test_batchs
    flow_result = flow_result.cpu().detach().numpy()
    result = result.cpu().detach().numpy()
    abs_result = abs_result.cpu().detach().numpy()
    abs_flow_result = abs_flow_result.cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(141)
    ax.set_yticks(range(result.shape[0]))
    ax.set_yticklabels(range(result.shape[0]))
    ax.set_xticks(range(result.shape[1]))
    ax.set_xticklabels(range(result.shape[1]))

    im = ax.imshow(result, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title('error')

    ax = fig.add_subplot(143)
    ax.set_yticks(range(flow_result.shape[0]))
    ax.set_yticklabels(range(flow_result.shape[0]))
    ax.set_xticks(range(flow_result.shape[1]))
    ax.set_xticklabels(range(flow_result.shape[1]))

    im = ax.imshow(flow_result, cmap=plt.cm.hot_r)

    plt.colorbar(im)
    plt.title('flow error')

    ax = fig.add_subplot(142)
    ax.set_yticks(range(abs_result.shape[0]))
    ax.set_yticklabels(range(abs_result.shape[0]))
    ax.set_xticks(range(abs_result.shape[1]))
    ax.set_xticklabels(range(abs_result.shape[1]))

    im = ax.imshow(abs_result, cmap=plt.cm.hot_r)

    plt.colorbar(im)
    plt.title('abs error')

    ax = fig.add_subplot(144)
    ax.set_yticks(range(abs_flow_result.shape[0]))
    ax.set_yticklabels(range(abs_flow_result.shape[0]))
    ax.set_xticks(range(abs_flow_result.shape[1]))
    ax.set_xticklabels(range(abs_flow_result.shape[1]))

    im = ax.imshow(abs_flow_result, cmap=plt.cm.hot_r)

    plt.colorbar(im)
    plt.title('abs flow error')


    plt.show()
    
if __name__ == '__main__':
    main()