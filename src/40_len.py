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
import dataProcess as dp 
import train

conf.args["seqLength"] = 40

prefix = train.train(conf.args, [1,2,3,4,5,6], "mixBasicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "mixBasicLSTM")

conf.args["modelFilePrefix"] = "500"
conf.args["prefix"] = ["500_1", "500_2"]
conf.args["testFilePrefix"] = ["500_3"]
prefix = train.train(conf.args, [1,2,3,4,5,6], "mixBasicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "mixBasicLSTM")

conf.args["modelFilePrefix"] = "1000"
conf.args["prefix"] = ["1000_1", "1000_2"]
conf.args["testFilePrefix"] = ["1000_3"]
prefix = train.train(conf.args, [1,2,3,4,5,6], "mixBasicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "mixBasicLSTM")

conf.args["modelFilePrefix"] = "1500"
conf.args["prefix"] = ["1500_1", "1500_2"]
conf.args["testFilePrefix"] = ["1500_3"]
prefix = train.train(conf.args, [1,2,3,4,5,6], "mixBasicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "mixBasicLSTM")

conf.args["modelFilePrefix"] = "300"
conf.args["prefix"] = ["300_1", "300_2"]
conf.args["testFilePrefix"] = ["300_3"]
prefix = train.train(conf.args, [1,2,3,4,5,6], "basicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "basicLSTM")

conf.args["modelFilePrefix"] = "500"
conf.args["prefix"] = ["500_1", "500_2"]
conf.args["testFilePrefix"] = ["500_3"]
prefix = train.train(conf.args, [1,2,3,4,5,6], "basicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "basicLSTM")

conf.args["modelFilePrefix"] = "1000"
conf.args["prefix"] = ["1000_1", "1000_2"]
conf.args["testFilePrefix"] = ["1000_3"]
prefix = train.train(conf.args, [1,2,3,4,5,6], "basicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "basicLSTM")

conf.args["modelFilePrefix"] = "1500"
conf.args["prefix"] = ["1800_1", "1800_2"]
conf.args["testFilePrefix"] = ["1800_3"]
prefix = train.train(conf.args, [1,2,3,4,5,6], "basicLSTM")
train.test(prefix, conf.args, [1,2,3,4,5,6], "basicLSTM")

