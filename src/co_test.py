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
import matplotlib.animation as anim
import matplotlib

import argparse
import os
import time
import pickle
import yaml

from model import TP_lstm
from model import loss_function
from net_model import network_model
from data import traffic_data
import data as d
import conf

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="co_eva")

    args = parser.parse_args()
    test(args)

def test(args):

    with open(os.path.join(conf.configPath, args.config+'.yaml'), encoding='UTF-8') as config:
        args = yaml.load(config)

    model_prefix = args["model_prefix"]
    use_epoch = args["use_epoch"]

    load_directory = os.path.join(conf.logPath, args["model_prefix"])
    file = os.path.join(load_directory, str(use_epoch)+'.tar')
    checkpoint = torch.load(file)
    model = network_model(args)
    model.load_state_dict(checkpoint['state_dict'])
    
    eva_prefix = args["eva_prefix"]
    topology_index = args["topology_index"]

    data_set = traffic_data(args, data_prefix=args["eva_prefix"][0], 
                            mod='cooperate', topology=d.co_topology[topology_index])

    co_data = data_set[0].unsqueeze(0)

    if args["use_cuda"]:
        co_data = co_data.cuda()
        model = model.cuda()

    [_, temporal, spatial, input_size] = co_data.shape
    inputs = co_data[:, :args["t_predict"]+1, :, :]
    target = co_data[:, args["t_predict"]+1:, :, :]
    numbers_before = inputs[:, args["t_predict"]:, :, 2]
    outputs = inputs.data.new(1, args["temporal_length"]-args["t_predict"], spatial, 1).fill_(0).float()
    In = inputs.data.new(1, args["temporal_length"]-args["t_predict"], spatial, 1).fill_(0).float()
    numbers_caculate = inputs.data.new(1, args["temporal_length"]-args["t_predict"], spatial, 1).fill_(0).float()

    bucket_number = data_set.bucket_number
    init_bucket = [0, bucket_number[1]+1]
    non_flow_from = [bucket_number[1], bucket_number[3], 
                    bucket_number[5], bucket_number[6]]
    non_flow_in = [0, bucket_number[1]+1, 
                    bucket_number[3], bucket_number[6]]
    flow_from = [i for i in range(spatial) if i not in non_flow_from]
    flow_in = [i for i in range(spatial) if i not in non_flow_in]

    In[:, :, init_bucket, 0] += target[:, :, init_bucket, 1]
    h_all = None
    c_all = None
    topology_struct = None

    tt = time.time()

    for t in range(args["t_predict"]):
        input_data = inputs[:, t, :, :]
        output, [h_all, c_all], topology_struct = model.infer(input_data, bucket_number, 
                                                        [h_all, c_all], topology_struct)

    input_data = inputs[:, args["t_predict"], :, :]

    for t in range(args["temporal_length"]-args["t_predict"]):

        output, [h_all, c_all], topology_struct=model.infer(input_data, bucket_number,
                                                        [h_all, c_all], topology_struct)

        In[:, t, flow_in, 0] += output[:, flow_from, 0]
        In[:, t, bucket_number[6]] += output[:, bucket_number[1], 0] + output[:, bucket_number[3], 0]
        In[:, t, bucket_number[4]] += output[:, bucket_number[6], 0]
        numbers_caculate[0, t, :, 0] = co_data[0, t+args["t_predict"], :, 2] + In[0, t, :, 0] - output[0, :, 0]
        outputs[:, t, :, :] += output
        input_data = torch.cat((output, In[:, t, :, :], numbers_caculate[:, t, :, :]), 2)
    
    ttt = time.time()

    print(ttt - tt)
    
    if args["use_cuda"]:
        target = target.cpu()
        numbers_caculate = numbers_caculate.cpu()
        outputs = outputs.cpu()

    target = target.detach().numpy()
    numbers_caculate = numbers_caculate.detach().numpy()
    outputs = outputs.detach().numpy()

    print(metrics.mean_absolute_error(target[0, :, :, 0].flatten(), outputs[0, :, :, 0].flatten()))
    print(metrics.mean_absolute_error(target[0, :, :, 2].flatten(), numbers_caculate[0, :, :, 0].flatten()))
    print(metrics.explained_variance_score(target[0, :, :, 0].flatten(), outputs[0, :, :, 0].flatten()))
    print(metrics.explained_variance_score(target[0, :, :, 2].flatten(), numbers_caculate[0, :, :, 0].flatten()))

    major_real_number = target[0, :, :bucket_number[1]+1, 2]
    major_predict_number = numbers_caculate[0, :, :bucket_number[1]+1, 0]
    minor_real_number = target[0, :, bucket_number[1]+1:bucket_number[4], 2]
    minor_predict_number = numbers_caculate[0, :, bucket_number[1]+1:bucket_number[4], 0]
    end_real_number = target[0, :, bucket_number[4]:-1, 2]
    end_predict_number = numbers_caculate[0, :, bucket_number[4]:-1, 0]
    
    major_flow_real = major_real_number.sum(axis=1)
    major_flow_pred = major_predict_number.sum(axis=1)
    minor_flow_real = minor_real_number.sum(axis=1)
    minor_flow_pred = minor_predict_number.sum(axis=1)
    end_flow_real = end_real_number.sum(axis=1)
    end_flow_pred = end_predict_number.sum(axis=1)

    x = range(len(end_flow_pred))

    plt.figure(13, figsize=(18, 4))
    plt.subplot(131)
    plt.plot(x, major_flow_real, 's-', color='r', label='real')
    plt.plot(x, major_flow_pred, 'o-', color='g', label='predict')
    plt.xlabel('time')
    plt.ylabel('num_vehicle')
    plt.legend(loc='best')
    plt.title('major segment')
    
    plt.subplot(132)
    plt.plot(x, minor_flow_real, 's-', color='r', label='real')
    plt.plot(x, minor_flow_pred, 'o-', color='g', label='predict')
    plt.xlabel('time')
    plt.ylabel('num_vehicle')
    plt.legend(loc='best')
    plt.title('minor segment')

    plt.subplot(133)
    plt.plot(x, end_flow_real, 's-', color='r', label='real')
    plt.plot(x, end_flow_pred, 'o-', color='g', label='predict')
    plt.xlabel('time')
    plt.ylabel('num_vehicle')
    plt.legend(loc='best')
    plt.title('end segment')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    heat = fig.add_subplot(311)
    im = heat.imshow(major_real_number.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("marjor ground truth")
    heat = fig.add_subplot(312)
    im = heat.imshow(major_predict_number.T, cmap=plt.cm.hot_r, vmin=0)
    plt.colorbar(im)
    plt.title("major predict")
    heat = fig.add_subplot(313)
    im = heat.imshow(major_real_number.T-major_predict_number.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("major error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    heat = fig.add_subplot(311)
    im = heat.imshow(minor_real_number.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("minor ground truth")
    heat = fig.add_subplot(312)
    im = heat.imshow(minor_predict_number.T, cmap=plt.cm.hot_r, vmin=0)
    plt.colorbar(im)
    plt.title("minor predict")
    heat = fig.add_subplot(313)
    im = heat.imshow(minor_real_number.T-minor_predict_number.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("minor error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    heat = fig.add_subplot(311)
    im = heat.imshow(end_real_number.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("end ground truth")
    heat = fig.add_subplot(312)
    im = heat.imshow(end_predict_number.T, cmap=plt.cm.hot_r, vmin=0)
    plt.colorbar(im)
    plt.title("end predict")
    heat = fig.add_subplot(313)
    im = heat.imshow(end_real_number.T-end_predict_number.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("end error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()
        
if __name__ == "__main__":
    main()