import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import conf


class FCNet(nn.Module):

    def __init__(self, addLayer=False, layerSize=[3, conf.args["embeddingLayer"], conf.args["embeddingSize"]]):

        super().__init__()

        self.inputSize = layerSize[0]
        self.hiddenSize = layerSize[1]
        self.outputSize = layerSize[2]
        self.addLayer = addLayer

        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.fc2 = nn.Linear(self.hiddenSize, self.outputSize)
        
        self.relu = nn.ReLU()

        if addLayer:
            self.outputSize2 = layerSize[3]
            self.fc3 = nn.Linear(self.outputSize, self.outputSize2)

    def forward(self, inputs):

        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        if self.addLayer :
            outputs = self.relu(outputs)
            outputs = self.fc3(outputs)

        return outputs


class mdLSTM(nn.Module):

    def __init__(self, args, test_mod=False):

        super().__init__()

        self.args = args
        self.test_mod = test_mod
        
        #相关参数
        self.seqLength = args["seqLength"]
        self.seqPredict = args["seqPredict"]
        self.predictLength = self.seqLength - self.seqPredict
        self.batchSize = args["batchSize"]
        self.hiddenSize = args["hiddenSize"]
        self.embeddingHiddenSize = args["embeddingLayer"]
        self.embeddingSize = args["embeddingSize"]
        self.inputSize = args["inputSize"]
        self.outputSize = args["outputSize"]
        self.gru = args["gru"]
        self.laneGateFCSize = args["laneGateFC"]
        self.outputFC1Size = args["outputFC1"]
        self.outputFC2Size = args["outputFC2"]

        #编码层
        self.embedding = FCNet(layerSize=[self.inputSize, self.embeddingHiddenSize, self.embeddingSize])
        
        #车道控制层
        self.laneGate = FCNet(layerSize=[1, self.laneGateFCSize, 1])

        self.cell = torch.nn.LSTMCell(self.inputSize, self.hiddenSize)

        self.outputLayer = nn.Linear(self.hiddenSize, self.outputSize)

        #这一阶段的输出
        self.outputs = FCNet(addLayer=True, layerSize=[self.outputSize, self.outputFC1Size, self.outputFC2Size, 1])

        #不同bucket间的影响
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.convpool = nn.Conv1d(in_channels=self.hiddenSize, 
                    out_channels=self.hiddenSize, kernel_size=3, stride=1, padding=1)

        #激活函数
        self.relu = nn.ReLU()
        self.sigma = nn.Sigmoid()
        

    def forward(self, inputData, lane, hidden=None):

        [SpatialLength, temporalLength, inputSize] = inputData.size()
        if self.test_mod:
            origin_pred = inputData[:,self.seqPredict:,:]
        if hidden == None:
            h_0 = inputData.data.new(SpatialLength, self.hiddenSize).fill_(0).float()
            c_0 = inputData.data.new(SpatialLength, self.hiddenSize).fill_(0).float()
            h_0 = Variable(h_0)
            c_0 = Variable(c_0)
        else:
            h_0 = hidden[0]
            c_0 = hidden[1]

        predict_h = inputData.data.new(SpatialLength, self.predictLength, self.hiddenSize).fill_(0).float()
        output = inputData.data.new(SpatialLength, self.predictLength).fill_(0).float()
        H = inputData.data.new(SpatialLength, self.hiddenSize, temporalLength).fill_(0).float()
        C = inputData.data.new(SpatialLength, self.hiddenSize, temporalLength).fill_(0).float()

        laneControler = self.laneGate(lane)
        laneControler = self.sigma(laneControler)
        laneControler = laneControler.view(-1, 1, 1)

        inputData = inputData * laneControler
        laneControler = laneControler.view(-1, 1)
        inputData = self.embedding(inputData)
        inputData = self.relu(inputData)

        for time in range(self.seqPredict):
            h_0, c_0 = self.cell(inputData[:, time, :], (h_0, c_0))
            #h_0 = self.maxpool(h_0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            h_0 = self.convpool(h_0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            H[:, :, time] = h_0
            C[:, :, time] = c_0

        for time in range(self.seqPredict, self.seqLength):

            h_0, c_0 = self.cell(inputData[:, time, :], (h_0, c_0))
            #h_0 = self.maxpool(h_0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            h_0 = self.convpool(h_0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            H[:, :, time] = h_0
            C[:, :, time] = c_0
            predict_h[:, time-self.seqPredict, :] = h_0

        predict_h = predict_h.view(SpatialLength*self.predictLength, self.hiddenSize)
        output = self.outputLayer(predict_h)
        output = self.outputs(output)
        
        output = output.view(SpatialLength, self.predictLength)
        output = output / laneControler

        return output, [H, C]

    def infer(self, lane, hidden=None):

        [SpatialLength, temporalLength, inputSize] = inputData.size()
        if self.test_mod:
            origin_pred = inputData[:,self.seqPredict:,:]
        if hidden == None:
            h_0 = inputData.data.new(SpatialLength, self.hiddenSize).fill_(0).float()
            c_0 = inputData.data.new(SpatialLength, self.hiddenSize).fill_(0).float()
            h_0 = Variable(h_0)
            c_0 = Variable(c_0)
        else:
            h_0 = hidden[0]
            c_0 = hidden[1]

        predict_h = inputData.data.new(SpatialLength, self.predictLength, self.hiddenSize).fill_(0).float()
        output = inputData.data.new(SpatialLength, self.predictLength).fill_(0).float()
        H = inputData.data.new(SpatialLength, self.hiddenSize, temporalLength).fill_(0).float()
        C = inputData.data.new(SpatialLength, self.hiddenSize, temporalLength).fill_(0).float()

        laneControler = self.laneGate(lane)
        laneControler = self.sigma(laneControler)
        laneControler = laneControler.view(-1, 1, 1)

        inputData = inputData * laneControler
        laneControler = laneControler.view(-1, 1)
        inputData = self.embedding(inputData)
        inputData = self.relu(inputData)

        for time in range(self.seqPredict):
            h_0, c_0 = self.cell(inputData[:, time, :], (h_0, c_0))
            #h_0 = self.maxpool(h_0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            h_0 = self.convpool(h_0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            H[:, :, time] = h_0
            C[:, :, time] = c_0

        for time in range(self.predictLength):
            pred_input = H[:, :, self.seqPredict+time-1]
            pred_input = self.outputLayer(pred_input)
            pred_input = self.outputs(pred_input)
            pred_input = pred_input / laneControler
            if time > 0:
                output[:, time-1] = pred_input.squeeze(1)
            origin_pred[:, time, 0] = pred_input.squeeze(1)
            pred_input = origin_pred[:, time, :]
            pred_input = pred_input * laneControler
            pred_input = self.embedding(pred_input)
            pred_input = self.relu(pred_input)
            h_0, c_0 = self.cell(pred_input, (h_0, c_0))
            h_0 = self.convpool(h_0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            H[:, :, self.seqPredict+time] = h_0
            C[:, :, self.seqPredict+time] = c_0
        pred_input = h_0
        pred_input = self.outputLayer(pred_input)
        pred_input = self.outputss(pred_input)
        pred_input = pred_input / laneControler
        output[:, time] = pred_input.squeeze(1)
        
        return output, [H, C]