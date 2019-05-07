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
import dataProcess as dp 
import train

# dp.edgeRecord(conf.netDebug, conf.fcdDebug, conf.carInDebug, conf.carOutDebug, conf.numberDebug)
# dataCheck(conf.carInDebug, conf.carOutDebug, conf.numberDebug)
# dp.laneNumber(conf.netDebug, conf.laneNumberDebug)

#dp.edgeRecord(conf.netDebug, conf.fcdDefualt)
#dp.laneNumber(conf.netDebug)
#dp.resultBoxplot("mixBasicLSTM_mix123456")

'''
bg = batchGenerator(conf.args["prefix"])
bg.generateBatch()
inputs = Variable(torch.Tensor(bg.CurrentSequences))
lane = Variable(torch.Tensor([bg.CurrentLane]))
model = stackedLSTM(conf.args)
output, hidden = model(lane, inputs)
'''

#train.train(conf.args, [1,2,3,4,5], "mixBasicLSTM")
#train.test(conf.args, [1,2,3,4,5], "mixBasicLSTM")
conf.args["version"] = "0423"
train.test("data_basicLSTM_123456", conf.args, [1])
train.test("data_basicLSTM_123456", conf.args, [2])
train.test("data_basicLSTM_123456", conf.args, [3])
train.test("data_basicLSTM_123456", conf.args, [4])
train.test("data_basicLSTM_123456", conf.args, [5])
train.test("data_basicLSTM_123456", conf.args, [6])
train.test("data_basicLSTM_123456", conf.args, [1,2,3,4,5])
train.test("data_basicLSTM_123456", conf.args, [1,2,3,4,5,6])

train.test("data_basicLSTM_12345", conf.args, [6])
train.test("data_basicLSTM_12345", conf.args, [5])
train.test("data_basicLSTM_12345", conf.args, [4])
train.test("data_basicLSTM_12345", conf.args, [3])
train.test("data_basicLSTM_12345", conf.args, [2])
train.test("data_basicLSTM_12345", conf.args, [1])
train.test("data_basicLSTM_12345", conf.args, [1,2,3,4,5])

'''
fcd = conf.firstStageFcd + "/0.5_0.5_7200_0.1.xml"
carIn = conf.midDataPath + "/dataCarIn.csv"
carOut = conf.midDataPath + "/dataCarOut.csv"
number = conf.midDataPath + "/dataNumber.csv"
dp.edgeRecord(conf.netDebug, fcd, carIn, carOut, number)
'''

