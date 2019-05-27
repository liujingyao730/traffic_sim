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

'''
length = "300_1"
fcd = conf.fcd(length, "varibleLength")
carIn = conf.midDataName(length)
carOut = conf.midDataName(length, "carOut")
number = conf.midDataName(length, "number")
dp.edgeRecord(conf.netDebug, fcd, carIn, carOut, number)
'''

# dataCheck(conf.carInDebug, conf.carOutDebug, conf.numberDebug)
# dp.laneNumber(conf.netDebug, conf.laneNumberDebug)

#dp.edgeRecord(conf.netDebug, conf.fcdDefualt)
#dp.laneNumber(conf.netDebug)
#conf.args["version"] = "0423"
#dp.resultBoxplot("data_basicLSTM_3_to_mix1")

'''
bg = batchGenerator(conf.args["prefix"])
model = BasicLSTM(conf.args)
bg.generateBatch()
data = Variable(torch.Tensor(bg.CurrentSequences))
laneT = Variable(torch.Tensor(bg.CurrentLane))
output, hidden = model(laneT, data)
print(output)    
'''

prefix = train.train(conf.args, [1,2,3,4,5,6], "mixBasicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "mixBasicLSTM")



'''
fcd = conf.firstStageFcd + "/0.5_0.5_7200_0.1.xml"
carIn = conf.midDataPath + "/dataCarIn.csv"
carOut = conf.midDataPath + "/dataCarOut.csv"
number = conf.midDataPath + "/dataNumber.csv"
dp.edgeRecord(conf.netDebug, fcd, carIn, carOut, number)
'''

