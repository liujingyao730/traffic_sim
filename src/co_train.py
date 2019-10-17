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
import random

import argparse
import os
import time
import pickle
import yaml
from tqdm import tqdm

from utils import batchGenerator
from model import TP_lstm
from model import loss_function
from model import network_model
from data import traffic_data
import data as d
import conf

'''路口与路段之间需要联合训练
'''


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="co_train")

    args = parser.parse_args()
    train(args)

def train(args):

    with open(os.path.join(conf.configPath, args.config+'.yaml'), encoding='UTF-8') as config:
        args = yaml.load(config)

    log_directory = os.path.join(conf.logPath, args["model_prefix"])
    log_curve_file = open(os.path.join(log_directory, "log_curve.txt"), "w+")

    def checkpoint_path(x):
        return os.path.join(log_directory, str(x)+'.tar')

    model = network_model(args)

    criterion = loss_function()
    sim_error_criterion = torch.nn.ReLU()
    pool = torch.nn.AvgPool1d(3, stride=1, padding=1)

    if args["use_cuda"]:
        model = model.cuda()
        criterion = criterion.cuda()
        sim_error_criterion = sim_error_criterion.cuda()
        pool = pool.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=args["lambda_param"])

    acc_meter = meter.AverageValueMeter()
    flow_loss_meter = meter.AverageValueMeter()
    last_loss_meter = meter.AverageValueMeter()
    last_flow_loss_meter = meter.AverageValueMeter()

    best_acc_epoch = 0
    best_acc = float('inf')
    best_flow_epoch = 0
    best_flow_loss = float('inf')

    topology = d.co_topology
    seg = d.seg
    inter = d.inter_node
    
    print("****************training epoch beginning**********************")
    for epoch in range(args["num_epochs"]):
        acc_meter.reset()
        
        for prefix in args["prefix"]:
            data_set = traffic_data(args, data_prefix=prefix, 
                        mod='cooperate', topology=d.co_topology)
            
            dataloader = torch.utils.data.DataLoader(data_set, 
                                                    batch_size=args["batch_size"], 
                                                    num_workers=1)

            for i, co_data in tqdm(enumerate(dataloader)):

                model.zero_grad()
                optimizer.zero_grad()
                
                


if __name__ == "__main__":
    main()