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
import conf

def train(args=conf.args, lane=None, modelType="basicLSTM"):

    dataFilePrefix = args["prefix"]
    datagenerator = batchGenerator(dataFilePrefix, simTimeStep=args["trainSimStep"])

    if modelType == "basicLSTM":
        model = BasicLSTM(args)
    elif modelType == "stackedLSTM":
        model = stackedLSTM(args)
    elif modelType == "mixBasicLSTM":
        model = mixBasicLSTM(args)
    else:
        print("there is no such model type as ", modelType)
        return
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

    prefix = dataFilePrefix + "_" + modelType
    if lane:
        prefix = prefix + "_"
        for l in lane:
            prefix = prefix + str(l)
    torch.save(model.state_dict(), conf.modelName(prefix))

def test(args=conf.args, lane=None, modelType="basicLSTM"):

    dataFilePrefix = args["prefix"]
    testFilePrefix = args["testFilePrefix"]
    prefix = dataFilePrefix + "_" + modelType

    if modelType == "basicLSTM":
        model = BasicLSTM(args)
    elif modelType == "stackedLSTM":
        model = stackedLSTM(args)
    elif modelType == "mixBasicLSTM":
        model = mixBasicLSTM(args)
    else:
        print("there is no such type as ", modelType)
        return

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

    prefix = modelType + "_mix"
    for l in lane:
        prefix = prefix + str(l)
    csvPath = conf.csvName(prefix)
    df = [result, target]
    df = pd.DataFrame(df, index=["result", "target"])
    df.to_csv(csvPath)
    print("result saved as ", csvPath)

    return csvPath