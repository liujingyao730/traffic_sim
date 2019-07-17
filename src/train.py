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

import argparse
import os
import time
import pickle

from utils import batchGenerator
from model import TP_lstm
import conf



def trainmd(args=conf.args, lane=[1, 2, 3, 4, 5, 6]):

    dataFilePrefix = args["prefix"]
    modelFilePrefix = args["modelFilePrefix"]
    datagenerator = batchGenerator(dataFilePrefix, 
            batchSize=args["batchSize"], simTimeStep=args["trainSimStep"])

    model = mdLSTM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if args["useCuda"]:
        model.cuda()
        criterion.cuda()
    lossMeter = meter.AverageValueMeter()

    laneNumber = len(lane)

    for epoch in range(args["epoch"]):
        lossMeter.reset()
        datagenerator.setFilePoint(0)
        i = 0
        while datagenerator.generateBatchForBucket(lane):
            data = Variable(torch.Tensor(datagenerator.CurrentSequences))
            laneT = Variable(torch.Tensor(datagenerator.CurrentLane))
            target = Variable(torch.Tensor(datagenerator.CurrentOutputs))
            data.squeeze_(0)
            laneT.squeeze_(0)
            target.squeeze_(0)
            if args["useCuda"]: 
                data = data.cuda()
                laneT = laneT.cuda()
                target = target.cuda()
            optimizer.zero_grad()

            output, _ = model(data, laneT)
            output = output.view(-1, args["seqLength"]-args["seqPredict"])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            lossMeter.add(loss.item())

            if i % args["plotEvery"] == 0:
                print("epoch: ", epoch, "  batch:  ", i, "  loss  ", lossMeter.value()[0], 
                " current time is ", datagenerator.CurrentTime, 
                " current file is ", datagenerator.filePrefixList[datagenerator.prefixPoint])
            i += 1

    prefix = modelFilePrefix + "_mdLSTM" 
    if lane:
        prefix = prefix + "_"
        for l in lane:
            prefix = prefix + str(l)
    torch.save(model.state_dict(), conf.modelName(prefix))

    return prefix



def trainmdRandom(args=conf.args, lane=[1, 2, 3, 4, 5, 6]):

    dataFilePrefix = args["prefix"]
    modelFilePrefix = args["modelFilePrefix"]
    datagenerator = batchGenerator(dataFilePrefix, 
            batchSize=args["batchSize"], simTimeStep=args["trainSimStep"])

    model = mdLSTM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if args["useCuda"]:
        model.cuda()
        criterion.cuda()
    lossMeter = meter.AverageValueMeter()

    laneNumber = len(lane)

    for epoch in range(args["epoch"]):
        lossMeter.reset()
        datagenerator.setFilePoint(0)
        i = 0
        for batch in range(args["batchNum"]):
            datagenerator.generateBatchRandomForBucket(lane)
            data = Variable(torch.Tensor(datagenerator.CurrentSequences))
            laneT = Variable(torch.Tensor(datagenerator.CurrentLane))
            target = Variable(torch.Tensor(datagenerator.CurrentOutputs))
            data.squeeze_(0)
            laneT.squeeze_(0)
            target.squeeze_(0)
            if args["useCuda"]: 
                data = data.cuda()
                laneT = laneT.cuda()
                target = target.cuda()
            optimizer.zero_grad()

            output, _ = model(data, laneT)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            lossMeter.add(loss.item())

            if i % args["plotEvery"] == 0:
                print("epoch: ", epoch, "  batch:  ", i, "  loss  ", lossMeter.value()[0])
            i += 1

    prefix = modelFilePrefix + "_mdLSTM" 
    if lane:
        prefix = prefix + "_"
        for l in lane:
            prefix = prefix + str(l)
    torch.save(model.state_dict(), conf.modelName(prefix))

    return prefix


def testmd(modelprefix, args=conf.args, lane=[1,2,3,4,5,6]):

    dataFilePrefix = args["prefix"]
    testFilePrefix = args["testFilePrefix"]
    modelfile = conf.modelName(modelprefix)

    model = mdLSTM(args, test_mod=False)
    state_dict = torch.load(modelfile) 
    model.load_state_dict(state_dict)
    model.eval()
    testData = batchGenerator(testFilePrefix, 
        simTimeStep=args["testSimStep"], batchSize=args["batchSize"])
    result1 = pd.DataFrame(columns=list(range(15)))
    result2 = pd.DataFrame(columns=list(range(15)))
    result3 = pd.DataFrame(columns=list(range(15)))
    result4 = pd.DataFrame(columns=list(range(15)))
    target1 = pd.DataFrame(columns=list(range(15)))
    target2 = pd.DataFrame(columns=list(range(15)))
    target3 = pd.DataFrame(columns=list(range(15)))
    target4 = pd.DataFrame(columns=list(range(15)))

    for i in range(args["testBatch"]):

        testData.generateBatchRandomForBucket()
        laneT = torch.Tensor(testData.CurrentLane)
        inputData = torch.Tensor(testData.CurrentSequences)
        target = np.array(testData.CurrentOutputs)
        [SpatialLength, temporalLength, inputSize] = inputData.size()
        
        if args["useCuda"]:
            laneT = laneT.cuda()
            inputData = inputData.cuda()
            model.cuda()
            
        output, _ = model.infer(inputData, laneT)
        if args["useCuda"]:
            output = output.cpu()
        output = output.detach().numpy()
        
        result1 = result1.append(pd.DataFrame(output[:, 0]).T, ignore_index=True)
        result2 = result2.append(pd.DataFrame(output[:, 1]).T, ignore_index=True)
        result3 = result3.append(pd.DataFrame(output[:, 2]).T, ignore_index=True)
        result4 = result4.append(pd.DataFrame(output[:, 3]).T, ignore_index=True)
        target1 = target1.append(pd.DataFrame(target[:, 0]).T, ignore_index=True)
        target2 = target2.append(pd.DataFrame(target[:, 1]).T, ignore_index=True)
        target3 = target3.append(pd.DataFrame(target[:, 2]).T, ignore_index=True)
        target4 = target4.append(pd.DataFrame(target[:, 3]).T, ignore_index=True)

    result1.to_csv(conf.csvName(modelprefix+"_result_time1"))
    result2.to_csv(conf.csvName(modelprefix+"_result_time2"))
    result3.to_csv(conf.csvName(modelprefix+"_result_time3"))
    result4.to_csv(conf.csvName(modelprefix+"_result_time4"))
    target1.to_csv(conf.csvName(modelprefix+"_target_time1"))
    target2.to_csv(conf.csvName(modelprefix+"_target_time2"))
    target3.to_csv(conf.csvName(modelprefix+"_target_time3"))
    target4.to_csv(conf.csvName(modelprefix+"_target_time4"))

    return 
    