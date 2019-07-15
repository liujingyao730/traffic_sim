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
from model import mixBasicLSTM
from model import mdLSTM
import conf

def train(args=conf.args, lane=[1, 2, 3, 4, 5, 6], modelType="basicLSTM"):

    dataFilePrefix = args["prefix"]
    modelFilePrefix = args["modelFilePrefix"]
    datagenerator = batchGenerator(dataFilePrefix, 
            batchSize=args["batchSize"] * len(args["gpu_id"]), simTimeStep=args["trainSimStep"])

    if modelType == "basicLSTM":
        model = BasicLSTM(args)
    elif modelType == "stackedLSTM":
        model = stackedLSTM(args)
    elif modelType == "mixBasicLSTM":
        model = mixBasicLSTM(args)
    elif modelType == "mdLSTM":
        model = mdLSTM(args)
    else:
        print("there is no such model type as ", modelType)
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if args["useCuda"]:
        model.cuda()
        criterion.cuda()
        if len(args["gpu_id"]) > 1:
            model = torch.nn.DataParallel(model, device_ids=args["gpu_id"])
    lossMeter = meter.AverageValueMeter()

    laneNumber = len(lane)

    for epoch in range(args["epoch"]):
        lossMeter.reset()
        datagenerator.setFilePoint(0)
        i = 0
        while datagenerator.generateBatch(lane):
            data = Variable(torch.Tensor(datagenerator.CurrentSequences))
            laneT = Variable(torch.Tensor(datagenerator.CurrentLane))
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

            if i % args["plotEvery"] == 0:
                print("epoch: ", epoch, "  batch:  ", i, "  loss  ", lossMeter.value()[0], 
                " current time is ", datagenerator.CurrentTime, 
                " current file is ", datagenerator.filePrefixList[datagenerator.prefixPoint])
            i += 1

    prefix = modelFilePrefix + "_" + modelType
    if lane:
        prefix = prefix + "_"
        for l in lane:
            prefix = prefix + str(l)
    torch.save(model.state_dict(), conf.modelName(prefix))

    return prefix

def test(modelprefix, args=conf.args, lane=[1,2,3,4,5,6], modelType="basicLSTM"):

    dataFilePrefix = args["prefix"]
    testFilePrefix = args["testFilePrefix"]
    modelfile = conf.modelName(modelprefix)

    if modelType == "basicLSTM":
        model = BasicLSTM(args)
    elif modelType == "stackedLSTM":
        model = stackedLSTM(args)
    elif modelType == "mixBasicLSTM":
        model = mixBasicLSTM(args)
    else:
        print("there is no such type as ", modelType)
        return

    state_dict = torch.load(modelfile) 
    model.load_state_dict(state_dict)
    model.eval()
    testData = batchGenerator(testFilePrefix, simTimeStep=args["testSimStep"])
    target = np.array([])
    result = np.array([])

    j = 0
    laneNumber = len(lane)

    for i in range(args["testBatch"]):

        testData.generateBatchRandom()
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

    prefix = modelprefix + "_to_mix"
    for l in lane:
        prefix = prefix + str(l)
    csvPath = conf.csvName(prefix)
    df = [result, target]
    df = pd.DataFrame(df, index=["result", "target"])
    df.to_csv(csvPath)
    print("result saved as ", csvPath)

    return prefix

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

    model = mdLSTM(args, test_mod=True)
    state_dict = torch.load(modelfile) 
    model.load_state_dict(state_dict)
    model.eval()
    testData = batchGenerator(testFilePrefix, simTimeStep=args["testSimStep"], batchSize=args["batchSize"])
    target = np.array([])
    result = np.array([])

    for i in range(args["testBatch"]):

        testData.generateBatchRandomForBucket()
        laneT = Variable(torch.Tensor(testData.CurrentLane))
        inputData = Variable(torch.Tensor(testData.CurrentSequences))
        [SpatialLength, temporalLength, inputSize] = inputData.size()
        
        if args["useCuda"]:
            laneT = laneT.cuda()
            inputData = inputData.cuda()
            model.cuda()
            
        output, _ = model(inputData, laneT)
        if args["useCuda"]:
            output = output.cpu()
        
        target = np.append(target, testData.CurrentOutputs)
        result = np.append(result, list(output))

    prefix = modelprefix + "_to_mix"
    for l in lane:
        prefix = prefix + str(l)
    PathR = conf.resultPath + "/" + prefix + "_result.npy"
    PathT = conf.resultPath + "/" + prefix + "_target.npy"
    np.save(PathR, result)
    np.save(PathT, target)
    print("result saved as ", PathR)
    print("target saved as ", PathT)

    return prefix
    