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

from utils import batchGenerator
from model import BasicLSTM
import conf

def train(args=conf.args):

    dataFilePrefix = args["prefix"]
    datagenerator = batchGenerator(dataFilePrefix, simTimeStep=args["trainSimStep"])

    model = BasicLSTM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if args["useCuda"]:
        model.cuda()
        criterion.cuda()
    lossMeter = meter.AverageValueMeter()

    for epoch in range(args["epoch"]):
        lossMeter.reset()
        for i in range(args["batchNum"]):
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
                print("loss ", lossMeter.value()[0])

    torch.save(model.state_dict(), conf.modelName(dataFilePrefix))

def test(args=conf.args):

    dataFilePrefix = args["prefix"]
    testFilePrefix = args["testFilePrefix"]
    model = BasicLSTM(args)
    state_dict = torch.load(conf.modelName(dataFilePrefix)) 
    model.load_state_dict(state_dict)
    model.eval()
    testData = batchGenerator(testFilePrefix, simTimeStep=args["testSimStep"])
    target = np.array([])
    result = np.array([])

    for i in range(args["testBatch"]):
        
        testData.generateBatch()
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

    title = "total"
    scatter = pe.Scatter(title=title)
    scatter.add("baseline", target, target)
    scatter.add("model", target, result)
    picturePath = conf.picslName("total")
    scatter.render(path=picturePath)
    print("picture saved as ",picturePath)

    return

def testLane(lane, model=None, testData=None, args = conf.args):

    if model == None:
        dataFilePrefix = args["prefix"]
        testFilePrefix = args["testFilePrefix"]
        model = BasicLSTM(args)
        state_dict = torch.load(conf.modelName(dataFilePrefix)) 
        model.load_state_dict(state_dict)
        model.eval()
        testData = batchGenerator(testFilePrefix, simTimeStep=args["testSimStep"])
    
    target = np.array([])
    result = np.array([])

    for i in range(args["testBatch"]):
        
        testData.generateBatchLane(lane)
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
    
    title = "lane " + str(lane)
    scatter = pe.Scatter(title=title)
    scatter.add("baseline", target, target)
    scatter.add("model", target, result)
    picturePath = conf.picsName(str(lane))
    scatter.render(path=picturePath)
    print("picture saved as ",picturePath)

    return model, testData