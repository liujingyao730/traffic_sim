'''
这就是一个用来乱写乱动的文件整体完成后会删掉
'''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import conf
from utils import batchGenerator
from model import BasicLSTM
from model import stackedLSTM
from model import mixBasicLSTM
from model import mdLSTM
import dataProcess as dp 
import train


inputdata = torch.rand(2,10,5,3)
lane = torch.Tensor([1,1])
model = mdLSTM(conf.args)

print(model(inputdata, lane))