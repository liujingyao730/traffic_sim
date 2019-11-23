import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from torchnet import meter
import numpy as np 
import sklearn.metrics as metrics
from tqdm import tqdm
import torchvision.models as models
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib
import seaborn as sns

import argparse
import os
import time
import pickle
import yaml
from tqdm import tqdm

from utils import batchGenerator
from model import TP_lstm
from model import loss_function
from net_model import network_model
from unit_net_model import uni_network_model as unlstm
from data import traffic_data
import data as d
import conf

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="co_eva")
    parser.add_argument("--test_index", type=int, default=0)
    parser.add_argument("--time", type=float, default=111.1)

    args = parser.parse_args()
    test(args)

def test(args):

    test_index = args.test_index
    start_time = args.time

    with open(os.path.join(conf.configPath, args.config+'.yaml'), encoding='UTF-8') as config:
        args = yaml.load(config)

    start_time = int(start_time / args["sim_step"])
    model_prefix = args["model_prefix"]
    use_epoch = args["use_epoch"]

    load_directory = os.path.join(conf.logPath, args["model_prefix"])
    file = os.path.join(load_directory, str(use_epoch)+'.tar')
    checkpoint = torch.load(file)
    model = network_model(args)
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
    #outputs, _ = model.infer(co_data, bucket_number)
    ttt = time.time()

    print("simlation time ", ttt - tt)
    print("real time ", (args["temporal_length"]-args["t_predict"])*args["delta_T"])
    
    if args["use_cuda"]:
        target = target.cpu()
    
    target = target.detach().numpy()

    major_real_number = target[0, :, :bucket_number[1]+1, 2]
    minor_real_number = target[0, :, bucket_number[1]+1:bucket_number[4], 2]
    end_real_number = target[0, :, bucket_number[4]:-1, 2]
    
    major_flow_real = major_real_number.sum(axis=1)
    minor_flow_real = minor_real_number.sum(axis=1)
    end_flow_real = end_real_number.sum(axis=1)
    x = range(len(end_flow_real))
    fig = plt.figure(figsize=(2, 6))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(233)
    ax3 = fig.add_subplot(235)
    ims = []
    for i in x:
        im1 = ax1.imshow(major_real_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0)
        im2 = ax2.imshow(minor_real_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0)
        im3 = ax3.imshow(end_real_number[i, :, np.newaxis], cmap=plt.cm.hot_r, vmin=0)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.subplots_adjust(left=0, top=1, right=1, bottom=0, wspace=0, hspace=0)
        ims.append([im1, im2, im3])
    ani = anim.ArtistAnimation(fig, ims, interval=100, blit=False)
    #ani.save(os.path.join(conf.picsPath, args["model_prefix"]+'_'+str(test_index)+'_real.gif'), writer="imagemagick")
    plt.show()


        
if __name__ == "__main__":
    main()