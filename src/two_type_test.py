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

from model import loss_function
from seg_model import basic_model
from seg_model import attn_model
from seg_model import attn_model_ad
from seg_model import two_type_attn_model
from data import traffic_data
from data import two_type_data
import data as d
import conf
import CTM_utils as ctm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="two_type_test")

    args = parser.parse_args()
    test(args)

def test(args):

    with open(os.path.join(conf.configPath, args.config+'.yaml'), encoding='UTF-8') as config:
        args = yaml.load(config)

    model_prefix = args["model_prefix"]
    use_epoch = args["use_epoch"]
    start_time = args["time"]
    start_time = int(start_time / args["sim_step"])

    load_directory = os.path.join(conf.logPath, args["model_prefix"])
    file = os.path.join(load_directory, str(use_epoch)+'.tar')
    checkpoint = torch.load(file)
    model = two_type_attn_model(args)
    model.load_state_dict(checkpoint['state_dict'])
    
    eva_prefix = args["eva_prefix"]

    data_set = two_type_data(args, data_prefix=args["eva_prefix"], topology=args["seg"])

    co_data = torch.Tensor(data_set[start_time]).float().unsqueeze(0)[:, :, :10, :]

    if args["use_cuda"]:
        co_data = co_data.cuda()
        model = model.cuda()

    [_, temporal, spatial, input_size] = co_data.shape
    target = co_data[:, args["t_predict"]+1:, :, :]

    tt = time.time()
    outputs = model.infer(co_data)
    ttt = time.time()

    print(ttt - tt)

    outputs = outputs.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    outputs = outputs * data_set.std
    outputs = outputs + data_set.mean
    target = target * data_set.std
    target = target + data_set.mean
    
    res = outputs[:, :, :, [0, 3]] - target[:, :, :, [0, 3]]
    output = outputs[:, :, :, [2, 5]].squeeze(0).sum(axis=2)
    target = target[:, :, :, [2, 5]].squeeze(0).sum(axis=2)

    prefix = "change"
    seg = 1
    cargs = {}
    cargs["number_vclass"] = 2
    cargs["cell_length"] = 50
    cargs["time_slot"] = 5
    cargs["lane_number"] = 6
    cargs["vlength"] = [4.5, 13]
    cargs["vspeed"] = [13.8, 10]
    cargs["cell_number"] = 30

    cargs["phi"] = [1, 1]
    cargs["sigma"] = 0.7
    cargs["over_take_factor"] = [1, 0.8]
    cargs["congest_factor"] = [1, 0.8]
    cargs["q"] = 18

    ctm_output, _ = ctm.caclutation_error(cargs, data_set)

    target_flow = target.sum(axis=1)
    output_flow = output.sum(axis=1)

    print("cell metrics ")
    print("MAE ", metrics.mean_absolute_error(target, output))
    print("R2 ", metrics.r2_score(target, output))
    print("EVR ", metrics.explained_variance_score(target, output))

    print("seg metrics")
    print("MAE ", metrics.mean_absolute_error(target_flow, output_flow))
    print("R2 ", metrics.r2_score(target_flow, output_flow))
    print("EVR ", metrics.explained_variance_score(target_flow, output_flow))

    print("res metrics")
    print("mean ", np.mean(res))
    print("median", np.median(res))
    print("min", np.min(res))
    print("max", np.max(res))

    predict_flow = output.sum(axis=1)
    real_flow = target.sum(axis=1)
    x = range(len(real_flow))

    plt.figure(13, figsize=(6, 4))
    plt.plot(x, real_flow, '-', color='k', label='ground truth')
    plt.plot(x, predict_flow, '-', color='r', label='R-CTM')
    plt.plot(x, ctm_output[:755, ].sum(axis=1), '-', color='g', label="FM-CTM")
    plt.xlabel('time')
    plt.ylabel('num_vehicle')
    plt.legend(loc='best')
    plt.title('5% HV rate')
    plt.show()

    np.save("with_all.npy", output)
    #np.save("targe.npy", target)

    fig = plt.figure(figsize=(15, 10))
    heat = fig.add_subplot(311)
    im = heat.imshow(target[190:300, 2:-2].T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("ground truth", fontsize=20)
    heat = fig.add_subplot(312)
    im = heat.imshow(output[190:300, 2:-2].T, cmap=plt.cm.hot_r, vmin=0, vmax=20)
    plt.colorbar(im)
    plt.title("R-CTM", fontsize=20)
    heat = fig.add_subplot(313)
    im = heat.imshow(output.T-target.T, cmap=plt.cm.hot_r)
    im = heat.imshow(ctm_output[190:300, 2:-2].T, cmap=plt.cm.hot_r, vmin=0, vmax=20)
    plt.colorbar(im)
    plt.title("FM-CTM", fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()

if __name__ == "__main__":
    main()