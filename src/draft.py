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


conf.args["prefix"] = ["300_1", "300_2", "500_1", "500_2", "800_1", "800_2"]
conf.args["testFilePrefix"] = ["300_3", "500_3", "800_3"]
conf.args["modelFilePrefix"] = "multi_Dimension_LSTM"


prefix = train.trainmd(conf.args)
train.testmd(prefix)

#prefix = "multi_Dimension_LSTM_mdLSTM_123456_to_mix123456"
#dp.bucketResult(prefix)

'''
bg = batchGenerator(prefix=conf.args["prefix"], 
        batchSize=conf.args["batchSize"], simTimeStep=conf.args["trainSimStep"])
bg.generateBatchForBucket()
data = Variable(torch.Tensor(bg.CurrentSequences))
laneT = Variable(torch.Tensor(bg.CurrentLane))
target = Variable(torch.Tensor(bg.CurrentOutputs))
train_model = mdLSTM(conf.args)
test_model = mdLSTM(conf.args, test_mod=True)

output_train, _ = train_model(data, laneT)
output_test, _ = test_model(data, laneT)
print(output_train.shape)
print(output_test.shape)
'''