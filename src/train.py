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

from utils import batchGenerator
from model import BasicLSTM
from model import stackedLSTM
import conf

def basicLSTMtrain(args=conf.args, lane=None):

    dataFilePrefix = args["prefix"]
    datagenerator = batchGenerator(dataFilePrefix, simTimeStep=args["trainSimStep"])

    model = BasicLSTM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if args["useCuda"]:
        model.cuda()
        criterion.cuda()
    lossMeter = meter.AverageValueMeter()

    if lane:
        j = 0
        laneNumber = len(lane)

    for epoch in range(args["epoch"]):
        lossMeter.reset()
        for i in range(args["batchNum"]):
            if lane:
                datagenerator.generateBatchLane(lane[j])
                j = (j + 1) % laneNumber
            else:
                datagenerator.generateBatch()
            data = Variable(torch.Tensor(datagenerator.CurrentSequences))
            laneT = Variable(torch.Tensor([datagenerator.CurrentLane]))
            target = Variable(torch.Tensor(datagenerator.CurrentOutputs))
            if args["useCuda"]: 
                data = data.cuda()
                laneT = laneT.cuda()
                target = target.cuda()
            optimizer.zero_grad()

            output, _ = model(laneT, data)
            output.squeeze_(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            lossMeter.add(loss.item())

            if (1+i) % args["plotEvery"] == 0:
                print("epoch: ", epoch, "  batch:  ", i, "  loss  ", lossMeter.value()[0])

    prefix = dataFilePrefix + "_basicLSTM"
    if lane:
        prefix = prefix + "_"
        for l in lane:
            prefix = prefix + str(l)
    torch.save(model.state_dict(), conf.modelName(prefix))

def basicLSTMtest(args=conf.args, lane=None):

    dataFilePrefix = args["prefix"]
    testFilePrefix = args["testFilePrefix"]
    model = BasicLSTM(args)
    prefix = dataFilePrefix + "_basicLSTM"
    if lane:
        prefix = prefix + "_"
        for l in lane:
            prefix = prefix + str(l)
    state_dict = torch.load(conf.modelName(prefix)) 
    model.load_state_dict(state_dict)
    model.eval()
    testData = batchGenerator(testFilePrefix, simTimeStep=args["testSimStep"])
    target = np.array([])
    result = np.array([])

    if lane:
        j = 0
        laneNumber = len(lane)

    for i in range(args["testBatch"]):

        if lane:
            testData.generateBatchLane(lane[j])
            j = (j + 1) % laneNumber
        testData.generateBatch()
        laneT = Variable(torch.Tensor([testData.CurrentLane]))
        inputData = Variable(torch.Tensor(testData.CurrentSequences))
        [batchSize, seqLength, embeddingSize] = inputData.size()
        
        if args["useCuda"]:
            laneT = laneT.cuda()
            inputData = inputData.cuda()
            model.cuda()
            
        output, _ = model(laneT, inputData)
        output = output.view(batchSize, -1).data.cpu().numpy()
        result = np.append(result, output)
        target = np.append(target, np.array(testData.CurrentOutputs))

    print("r2_sroce : ", metrics.r2_score(target, result))
    print("mean_absolute_error : ", metrics.mean_absolute_error(target, result))

    prefix = "basic_mix"
    for l in lane:
        prefix = prefix + str(l)
    csvPath = conf.csvName(prefix)
    df = [result, target]
    df = pd.DataFrame(df, columns=["result", "target"])
    df.to_csv(csvPath)
    print("result saved as ", csvPath)

    return csvPath

def stackedLSTMtrain(args=conf.args, lane=None):

    dataFilePrefix = args["prefix"]
    datagenerator = batchGenerator(dataFilePrefix, simTimeStep=args["trainSimStep"])

    model = stackedLSTM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if args["useCuda"]:
        model.cuda()
        criterion.cuda()
    lossMeter = meter.AverageValueMeter()

    if lane:
        j = 0
        laneNumber = len(lane)

    for epoch in range(args["epoch"]):
        lossMeter.reset()
        for i in range(args["batchNum"]):
            if lane:
                datagenerator.generateBatchLane(lane[j])
                j = (j + 1) % laneNumber
            else:
                datagenerator.generateBatch()
            data = Variable(torch.Tensor(datagenerator.CurrentSequences))
            lane = Variable(torch.Tensor([datagenerator.CurrentLane]))
            target = Variable(torch.Tensor(datagenerator.CurrentOutputs))
            if args["useCuda"]: 
                data = data.cuda()
                lane = lane.cuda()
                target = target.cuda()
            optimizer.zero_grad()

            output, _ = model(lane, data)
            output.squeeze_(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            lossMeter.add(loss.item())

            if (1+i) % args["plotEvery"] == 0:
                print("epoch: ", epoch, "  batch:  ", i, "  loss  ", lossMeter.value()[0])

    prefix = dataFilePrefix + "_stackedLSTM"
    if lane:
        prefix = prefix + "_"
        for l in lane:
            prefix = prefix + str(l)
    torch.save(model.state_dict(), conf.modelName(prefix))

def stackedLSTMtest(args=conf.args, lane=None):

    dataFilePrefix = args["prefix"]
    testFilePrefix = args["testFilePrefix"]
    model = stackedLSTM(args)
    state_dict = torch.load(conf.modelName(dataFilePrefix + "_stackedLSTM")) 
    model.load_state_dict(state_dict)
    model.eval()
    testData = batchGenerator(testFilePrefix, simTimeStep=args["testSimStep"])
    target = np.array([])
    result = np.array([])

    if lane:
        j = 0
        laneNumber = len(lane)

    for i in range(args["testBatch"]):
        
        if lane:
            datagenerator.generateBatchLane(lane[j])
            j = (j + 1) % laneNumber
        else:
            datagenerator.generateBatch()
        lane = Variable(torch.Tensor([testData.CurrentLane]))
        inputData = Variable(torch.Tensor(testData.CurrentSequences))
        [batchSize, seqLength, embeddingSize] = inputData.size()
        
        if args["useCuda"]:
            lane = lane.cuda()
            inputData = inputData.cuda()
            model.cuda()
            
        output, _ = model(lane, inputData)
        output = output.view(batchSize, -1).data.cpu().numpy()
        result = np.append(result, output)
        target = np.append(target, np.array(testData.CurrentOutputs))

    print("r2_sroce : ", metrics.r2_score(target, result))
    print("mean_absolute_error : ", metrics.mean_absolute_error(target, result))

    prefix = "stacked_mix"
    for l in lane:
        prefix = prefix + str(l)
    csvPath = conf.csvName(prefix)
    df = [result, target]
    df = pd.DataFrame(df, columns=["result", "target"])
    df.to_csv(csvPath)
    print("result saved as ", csvPath)

    return csvPath

def stackedLSTMtestLane(lane, model=None, testData=None, args = conf.args):

    if model == None:
        dataFilePrefix = args["prefix"]
        testFilePrefix = args["testFilePrefix"]
        model = stackedLSTM(args)
        state_dict = torch.load(conf.modelName(dataFilePrefix + "_stackedLSTM")) 
        model.load_state_dict(state_dict)
        model.eval()
        testData = batchGenerator(testFilePrefix, simTimeStep=args["testSimStep"])
    
    target = np.array([])
    result = np.array([])
    j = 0
    laneNumber = len(lane)

    for i in range(args["testBatch"]):
        
        l = lane[j]
        j = (j + 1) % laneNumber
        testData.generateBatchLane(l)
        Lane = Variable(torch.Tensor([testData.CurrentLane]))
        inputData = Variable(torch.Tensor(testData.CurrentSequences))
        [batchSize, seqLength, embeddingSize] = inputData.size()
        
        if args["useCuda"]:
            Lane = Lane.cuda()
            inputData = inputData.cuda()
            model.cuda()
            
        output, _ = model(Lane, inputData)
        output = output.view(batchSize, -1).data.cpu().numpy()
        result = np.append(result, output)
        target = np.append(target, np.array(testData.CurrentOutputs))

    print("lane", lane, " r2_sroce : ", metrics.r2_score(target, result))
    print("lane", lane, "mean_absolute_error : ", metrics.mean_absolute_error(target, result))
    
    title = "stacked_lane_" + str(lane)
    scatter = pe.Scatter(title=title)
    scatter.add("baseline", target, target)
    scatter.add("model", target, result)
    picturePath = conf.picsName("stackedLSTM_"+str(lane))
    scatter.render(path=picturePath)
    print("picture saved as ",picturePath)

    return model, testData