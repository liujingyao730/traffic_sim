'''
这就是一个用来乱写乱动的文件整体完成后会删掉
'''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os
import time
import pickle
import yaml

import conf
from utils import batchGenerator
from model import MD_lstm_cell
from model import TP_lstm
import dataProcess as dp 
import train
'''

label = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
real = np.array([82149, 41055, 31698, 34835, 27731, 14730, 9844, 7457, 6697, 6212, 3464, 1554, 468])

number_total = sum(real)
number_rate = real / number_total

print(number_rate)
plt.plot(label, number_rate, 's-', color='r', label='number rate')

flow = label*real
flow_total = sum(flow)
flow_rate = flow / flow_total
plt.plot(label, flow_rate, 'o-', color='g', label='flow rate')

plt.xlabel('number of vehicle')
plt.ylabel('rate')
plt.legend(loc='best')
plt.show()

a = flow_rate[1:]/number_rate[1:]
a = a / 0.39887318
'''
parser = argparse.ArgumentParser()  
parser.add_argument("--config", type=str, default="test")
args = parser.parse_args()

with open(os.path.join(conf.configPath, args.config+'.yaml'), encoding='UTF-8') as config:
        args = yaml.load(config)

model_prefix = args["model_prefix"]
test_batchs = args["test_batchs"]
use_epoch = args["use_epoch"]

load_directory = os.path.join(conf.logPath, model_prefix)
    
def checkpath(epoch):
    return os.path.join(load_directory, str(epoch)+'.tar')

file = checkpath(use_epoch)
checkpoint = torch.load(file)

dg = batchGenerator(["500", "1000", "2000"], args)
dg.generateBatchRandomForBucket()

model = TP_lstm(args)
model.load_state_dict(checkpoint['state_dict'])
if args["use_cuda"]:
    model = model.cuda()

number = []

for i in range(13):
    number.append(np.array([]))

for i in range(test_batchs):
    
    dg.generateBatchRandomForBucket()

    data = torch.Tensor(dg.CurrentSequences)
    laneT = torch.Tensor(dg.CurrentLane)
    target = torch.Tensor(dg.CurrentOutputs)

    if args["use_cuda"]:
        data = data.cuda()
        laneT = laneT.cuda()
        target = target.cuda()

    output = model(data, laneT)

    output = output.view(-1)
    target = target[:, :, :, 0].view(-1)

    for i in range(13):
        number[i] = np.concatenate((number[i], output[torch.eq(target, i)].cpu().detach().numpy()))

for i in range(13):
    print("target {}, median value: {:.2f}, mean value: {:.2f}, std: {:.2f}, len {}".format(i, np.median(number[i]), np.mean(number[i]), np.std(number[i]), len(number[i])))

sns.violinplot(data=np.array(number[:6]))
plt.show()
sns.violinplot(data=np.array(number[6:]))
plt.show()