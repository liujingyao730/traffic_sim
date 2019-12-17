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
from seg_model import continuous_seg
from data import traffic_data
import data as d
import conf

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="single_test")
    parser.add_argument("--test_index", type=int, default=0)

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
    model = continuous_seg(args)
    model.load_state_dict(checkpoint['state_dict'])
    
    eva_prefix = args["eva_prefix"]

    data_set = traffic_data(args, data_prefix=args["eva_prefix"], 
                            mod='seg', topology=[args["seg"]])

    co_data = data_set[start_time].unsqueeze(0)

    if args["use_cuda"]:
        co_data = co_data.cuda()
        model = model.cuda()

    [_, temporal, spatial, input_size] = co_data.shape
    target = co_data[:, args["t_predict"]+1:, :, :]

    tt = time.time()
    outputs = model.infer(co_data)
    ttt = time.time()

    output = outputs[0, :, :, 2].cpu().detach().numpy()
    target = target[0, :, :, 2].cpu().detach().numpy()

    predict_flow = output.sum(axis=1)
    real_flow = target.sum(axis=1)
    x = range(len(real_flow))

    plt.figure(13, figsize=(6, 4))
    plt.plot(x, real_flow, 's-', color='r', label='real')
    plt.plot(x, predict_flow, 'o-', color='g', label='predict')
    plt.xlabel('time')
    plt.ylabel('num_vehicle')
    plt.legend(loc='best')
    plt.title('flow with time')
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    heat = fig.add_subplot(311)
    im = heat.imshow(target.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("ground truth")
    heat = fig.add_subplot(312)
    im = heat.imshow(output.T, cmap=plt.cm.hot_r, vmax=15)
    plt.colorbar(im)
    plt.title("simulation result")
    heat = fig.add_subplot(313)
    im = heat.imshow(output.T-target.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()

if __name__ == "__main__":
    main()