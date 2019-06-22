'''
这就是一个用来乱写乱动的文件整体完成后会删掉
'''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import pandas as pd

import conf
from utils import batchGenerator
from model import BasicLSTM
from model import stackedLSTM
from model import mixBasicLSTM
from model import mdLSTM
import dataProcess as dp 
import train

bg = batchGenerator(["300_1", "300_2", "300_3"], simTimeStep=conf.args["testSimStep"], batchSize=3)
model = mdLSTM(conf.args)
result = pd.DataFrame(columns=[1,2,3,4,5])
traget = pd.DataFrame(columns=[1,2,3,4,5])

for i in range(30):
    bg.generateBatchRandomForBucket()
    inputs = Variable(torch.tensor(bg.CurrentSequences)).float()
    laneT = Variable(torch.tensor(bg.CurrentLane)).float()
    output,_ = model(inputs, laneT)
    traget = traget.append(pd.DataFrame(bg.CurrentOutputs, columns=[1,2,3,4,5]), ignore_index=True)
    result = result.append(pd.DataFrame(output.detach().numpy(), columns=[1,2,3,4,5]), ignore_index=True)