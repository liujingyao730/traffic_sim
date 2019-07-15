'''
这就是一个用来乱写乱动的文件整体完成后会删掉
'''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np

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
#prefix = "multi_Dimension_LSTM_mdLSTM_123456"
train.testmd(prefix)

#prefix = "multi_Dimension_LSTM_mdLSTM_123456_to_mix123456"
#dp.bucketResult(prefix)
'''
result = conf.resultPath + "/multi_Dimension_LSTM_mdLSTM_123456_to_mix123456_result.npy"
target = conf.resultPath + "/multi_Dimension_LSTM_mdLSTM_123456_to_mix123456_target.npy"

r = np.load(result)
t = np.load(target)

print(r.shape)
print(t.shape)
'''