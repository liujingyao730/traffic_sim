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

    parser.add_argument("--config", type=str, default="model_test")
    parser.add_argument("--test_index", type=int, default=0)

    args = parser.parse_args()
    test(args)

def test(args):

    test_index = args.test_index

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

    data_set = traffic_data(args, data_prefix=args["eva_prefix"][test_index], 
                            mod='cooperate', topology=d.co_topology[topology_index[test_index]])

    dd = torch.utils.data.DataLoader(data_set, batch_size=100, num_workers=1)

    for i, data in enumerate(dd):

        [batch_size, temporal, spatial, inputs_size] = data.shape

        inputs = data[:, :-1, :, :]
        target = data[:, args["t_predict"]:, :, :]

        
        
        break

if __name__ == "__main__":
    main()