import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import conf
from utils import batchGenerator
from model import BasicLSTM
from model import stackedLSTM
from model import mixBasicLSTM
import dataProcess as dp 
import train

conf.sequenceLength = 20
conf.deltaT = 15

conf.args["modelFilePrefix"] = "500"
conf.args["prefix"] = ["500_1", "500_2"]
conf.args["testFilePrefix"] = ["500_3"]
prefix = train.train(conf.args, [1,2,3,4,5,6], "mixBasicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "mixBasicLSTM")


conf.deltaT = 20

prefix = train.train(conf.args, [1,2,3,4,5,6], "mixBasicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "mixBasicLSTM")