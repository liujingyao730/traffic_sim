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
import seaborn as sns

import argparse
import os
import time
import pickle
import yaml

from model import TP_lstm
from model import loss_function
from unit_net_model import uni_network_model 
from net_model import network_model
from separate_seg_model import sp_network_model
from data import traffic_data
import data as d
import conf

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="co_eva")
    parser.add_argument("--test_index", type=int, default=2)

    args = parser.parse_args()
    test(args)

def test(args):

    test_index = args.test_index

    with open(os.path.join(conf.configPath, args.config+'.yaml'), encoding='UTF-8') as config:
        args = yaml.load(config)

    model_prefix = args["model_prefix"]
    use_epoch = args["use_epoch"]
    start_time = args["time"]
    start_time = int(start_time / args["sim_step"])

    load_directory = os.path.join(conf.logPath, args["model_prefix"])
    file = os.path.join(load_directory, str(use_epoch)+'.tar')
    checkpoint = torch.load(file)
    #model = network_model(args)
    model = sp_network_model(args)
    model.load_state_dict(checkpoint['state_dict'])
    
    eva_prefix = args["eva_prefix"]
    topology_index = args["topology_index"]

    data_set = traffic_data(args, data_prefix=args["eva_prefix"][test_index], 
                            mod='cooperate', topology=d.co_topology[topology_index[test_index]])

    co_data = data_set[start_time].unsqueeze(0)

    if args["use_cuda"]:
        co_data = co_data.cuda()
        model = model.cuda()

    [_, temporal, spatial, input_size] = co_data.shape
    target = co_data[:, args["t_predict"]+1:, :, :]

    bucket_number = data_set.bucket_number

    tt = time.time()
    outputs, _ = model.infer(co_data, bucket_number)
    ttt = time.time()

    print("simlation time ", ttt - tt)
    print("real time ", (args["temporal_length"]-args["t_predict"])*args["delta_T"])
    
    if args["use_cuda"]:
        target = target.cpu()
        outputs = outputs.cpu()
    
    ordered_output = []
    for i in range(10):
        ordered_output.append(np.array([]))

    output = outputs[0, :, :, 0].view(-1)
    target_output = target[0, :, :, 0].view(-1)
    for i in range(10):
        ordered_output[i] = output[torch.eq(target_output, i)].detach().numpy()
    sns.violinplot(data=np.array(ordered_output))
    plt.show()
    
    target = target.detach().numpy()
    outputs = outputs.detach().numpy()

    print("output mean absolute error ", metrics.mean_absolute_error(target[0, :, :, 0].flatten(), outputs[0, :, :, 0].flatten()))
    print("number mean absolute error ", metrics.mean_absolute_error(target[0, :, :, 2].flatten(), outputs[0, :, :, 2].flatten()))
    print("output explained variance score ", metrics.explained_variance_score(target[0, :, :, 0].flatten(), outputs[0, :, :, 0].flatten()))
    print("number explained variance score ", metrics.explained_variance_score(target[0, :, :, 2].flatten(), outputs[0, :, :, 2].flatten()))

    major_real_number = target[0, :, :bucket_number[1]+1, 2]
    major_predict_number = outputs[0, :, :bucket_number[1]+1, 2]
    minor_real_number = target[0, :, bucket_number[1]+1:bucket_number[4], 2]
    minor_predict_number = outputs[0, :, bucket_number[1]+1:bucket_number[4], 2]
    end_real_number = target[0, :, bucket_number[4]:-1, 2]
    end_predict_number = outputs[0, :, bucket_number[4]:-1, 2]
    
    major_flow_real = major_real_number.sum(axis=1)
    major_flow_pred = major_predict_number.sum(axis=1)
    minor_flow_real = minor_real_number.sum(axis=1)
    minor_flow_pred = minor_predict_number.sum(axis=1)
    end_flow_real = end_real_number.sum(axis=1)
    end_flow_pred = end_predict_number.sum(axis=1)

    print("major r2_score ", metrics.r2_score(major_flow_real, major_flow_pred))
    print("minor r2_score ", metrics.r2_score(minor_flow_real, minor_flow_pred))
    print("end r2_score ", metrics.r2_score(end_flow_real, end_flow_pred))

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
    im = heat.imshow(major_predict_number.T, cmap=plt.cm.hot_r, vmin=0, vmax=10)
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
    im = heat.imshow(minor_predict_number.T, cmap=plt.cm.hot_r, vmin=0, vmax=10)
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
    im = heat.imshow(end_predict_number.T, cmap=plt.cm.hot_r, vmin=0, vmax=10)
    plt.colorbar(im)
    plt.title("end predict")
    heat = fig.add_subplot(313)
    im = heat.imshow(end_real_number.T-end_predict_number.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("end error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()
    
    fig = plt.figure(figsize=(2, 6))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(233)
    ax3 = fig.add_subplot(235)
    ims = []
    for i in x:
        im1 = ax1.imshow(major_real_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0, vmax=10)
        im2 = ax2.imshow(minor_real_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0, vmax=10)
        im3 = ax3.imshow(end_real_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0, vmax=10)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.subplots_adjust(left=0, top=1, right=1, bottom=0, wspace=0, hspace=0)
        ims.append([im1, im2, im3])
    ani = anim.ArtistAnimation(fig, ims, interval=100, blit=False)
    ani.save(os.path.join(conf.picsPath, args["model_prefix"]+'_'+str(test_index)+'_real.gif'), writer="imagemagick")
    #plt.show()

    fig = plt.figure(figsize=(2, 6))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(233)
    ax3 = fig.add_subplot(235)
    ims = []
    for i in x:
        im1 = ax1.imshow(major_predict_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0, vmax=10)
        im2 = ax2.imshow(minor_predict_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0, vmax=10)
        im3 = ax3.imshow(end_predict_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0, vmax=10)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.subplots_adjust(left=0, top=1, right=1, bottom=0, wspace=0, hspace=0)
        ims.append([im1, im2, im3])
    ani = anim.ArtistAnimation(fig, ims, interval=100, blit=False)
    ani.save(os.path.join(conf.picsPath, args["model_prefix"]+'_'+str(test_index)+'_pred.gif'), writer="imagemagick")
    #plt.show()
        
if __name__ == "__main__":
    main()