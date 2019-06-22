import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BasicLSTM(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args

        self.seqLength = args["seqLength"]
        self.batchSize = args["batchSize"]

        self.hiddenSize = args["hiddenSize"]
        self.embeddingSize = args["embeddingSize"]
        self.inputSize = args["inputSize"]
        self.outputSize = args["outputSize"]
        self.gru = args["gru"]
        self.laneGateFC = args["laneGateFC"]
        self.fc1Size = args["fc1"]
        self.fc2Size = args["fc2"]

        self.embeddingFC1 = nn.Linear(self.inputSize, args["inputFC1"])
        self.embeddingFC2 = nn.Linear(args["inputFC1"], self.embeddingSize)

        self.RNNlayer = nn.LSTM(self.embeddingSize, self.hiddenSize, batch_first=True)
        if self.gru:
            self.RNNlayer = nn.GRU(self.embeddingSize, self.hiddenSize)

        self.outputLayer = nn.Linear(self.hiddenSize, self.outputSize)
        self.fc1 = nn.Linear(self.outputSize, self.fc1Size)
        self.fc2 = nn.Linear(self.fc1Size, self.fc2Size)
        self.fc3 = nn.Linear(self.fc2Size, 1)

        self.laneGate1 = nn.Linear(1, self.laneGateFC)
        self.laneGate2 = nn.Linear(self.laneGateFC, 1)

        self.relu = nn.ReLU()
        self.sigma = nn.Sigmoid()
        #self.dropOut = nn.Dropout(args["dropOut"])

    def forward(self, lane, inputData, hidden=None):
            
        [batchSize, seqLength, embeddingSize] = inputData.size()
        if hidden == None:
            h_0 = inputData.data.new(1, batchSize, self.hiddenSize).fill_(0).float()
            c_0 = inputData.data.new(1, batchSize, self.hiddenSize).fill_(0).float()
            h_0 = Variable(h_0)
            c_0 = Variable(c_0)
        else:
            h_0 = hidden[0]
            c_0 = hidden[1]

        laneControler = self.laneGate1(lane)
        laneControler = self.relu(laneControler)
        laneControler = self.laneGate2(laneControler)
        laneControler = self.sigma(laneControler)
        laneControler = laneControler.view(-1, 1, 1)

        inputData = inputData * laneControler
        inputData = self.embeddingFC1(inputData)
        inputData = self.relu(inputData)
        inputData = self.embeddingFC2(inputData)
        inputData = self.relu(inputData)

        output, hidden = self.RNNlayer(inputData, (h_0, c_0))

        output = self.outputLayer(hidden[0].view(batchSize, -1))
        output = self.relu(output)

        output = self.fc1(output)
        output = self.relu(output)
        
        output = self.fc2(output)
        output = self.relu(output)


        output = self.fc3(output)
        laneControler = laneControler.view(-1, 1)
        output = output / laneControler

        return output,hidden[0]


class mixBasicLSTM(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args

        self.seqLength = args["seqLength"]

        self.hiddenSize = args["hiddenSize"]
        self.embeddingSize = args["embeddingSize"]
        self.inputSize = args["inputSize"]
        self.outputSize = args["outputSize"]
        self.gru = args["gru"]
        self.laneGateFC = args["laneGateFC"]
        self.fc1Size = args["fc1"]
        self.fc2Size = args["fc2"]

        self.embeddingFC1 = nn.Linear(self.inputSize, args["inputFC1"])
        self.embeddingFC2 = nn.Linear(args["inputFC1"], self.embeddingSize)

        self.RNNlayer = nn.LSTM(self.embeddingSize, self.hiddenSize, batch_first=True)
        if self.gru:
            self.RNNlayer = nn.GRU(self.embeddingSize, self.hiddenSize)

        self.outputLayer = nn.Linear(self.hiddenSize, self.outputSize)
        self.fc1 = nn.Linear(self.outputSize, self.fc1Size)
        self.fc2 = nn.Linear(self.fc1Size, self.fc2Size)
        self.fc3 = nn.Linear(self.fc2Size, 1)

        self.relu = nn.ReLU()
        self.sigma = nn.Sigmoid()
        #self.dropOut = nn.Dropout(args["dropOut"])

    def forward(self, lane, inputData, hidden=None):
            
        [batchSize, seqLength, embeddingSize] = inputData.size()
        
        if hidden == None:
            h_0 = inputData.data.new(1, batchSize, self.hiddenSize).fill_(0).float()
            c_0 = inputData.data.new(1, batchSize, self.hiddenSize).fill_(0).float()
            h_0 = Variable(h_0)
            c_0 = Variable(c_0)
        else:
            h_0 = hidden[0]
            c_0 = hidden[1]

        inputData = self.embeddingFC1(inputData)
        inputData = self.relu(inputData)
        inputData = self.embeddingFC2(inputData)
        inputData = self.relu(inputData)

        #input_lengths = [seqLength for i in range(batchSize)]
        #inputData = pack_padded_sequence(inputData, input_lengths, batch_first=True)
        output, hidden = self.RNNlayer(inputData, (h_0, c_0))

        output = self.outputLayer(hidden[0].view(batchSize, -1))
        output = self.relu(output)

        output = self.fc1(output)
        output = self.relu(output)
        
        output = self.fc2(output)
        output = self.relu(output)


        output = self.fc3(output)

        return output,hidden[0]


class stackedLSTM(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.sHiddenSize = args["sHiddenSize"]
        self.seqLength = args["seqLength"]
        self.inputSize = args["inputSize"]
        self.sEmbeddingSize = args["sEmbeddingSize"]
        self.EmbeddingSize = args["embeddingSize"]
        self.hiddenSize = args["hiddenSize"]
        self.laneGateFC = args["laneGateFC"]
        self.outputSize = args["outputSize"]
        self.gru = args["gru"]
        self.fc1Size = args["fc1"]
        self.fc2Size = args["fc2"]

        self.args = args

        self.laneGate1 = nn.Linear(1, 3)
        self.laneGate2 = nn.Linear(3, 1)

        self.carInFC = nn.Linear(1, self.sEmbeddingSize)
        self.carOutFC = nn.Linear(1, self.sEmbeddingSize)
        self.numberFC = nn.Linear(1, self.sEmbeddingSize)

        self.carInRNNlayer = nn.LSTM(self.sEmbeddingSize, self.sHiddenSize, batch_first=True)
        self.carOutRNNlayer = nn.LSTM(self.sEmbeddingSize, self.sHiddenSize, batch_first=True)
        self.numberRNNlayer = nn.LSTM(self.sEmbeddingSize, self.sHiddenSize, batch_first=True)
        self.gatherRNNlayer = nn.LSTM(self.EmbeddingSize, self.hiddenSize, batch_first=True)
        if self.gru:
            self.carInRNNlayer = nn.GRU(self.sEmbeddingSize, self.sHiddenSize, batch_first=True)
            self.carOutRNNlayer = nn.LSTM(self.sEmbeddingSize, self.sHiddenSize, batch_first=True)
            self.numberRNNlayer = nn.LSTM(self.sEmbeddingSize, self.sHiddenSize, batch_first=True)
            self.gatherRNNlayer = nn.LSTM(self.EmbeddingSize, self.hiddenSize, batch_first=True)

        self.embeddingFC = nn.Linear(3*self.sHiddenSize, self.EmbeddingSize)
        self.outputLayer = nn.Linear(self.hiddenSize, self.outputSize)
        self.fc1 = nn.Linear(self.outputSize, self.fc1Size)
        self.fc2 = nn.Linear(self.fc1Size, self.fc2Size)
        self.fc3 = nn.Linear(self.fc2Size, 1)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, lane, inputData, hidden=None):

        [batchSize, seqLength, embeddingSize] = inputData.size()
        if hidden == None:
            h_0 = inputData.data.new(1, batchSize, self.hiddenSize).fill_(0).float()
            c_0 = inputData.data.new(1, batchSize, self.hiddenSize).fill_(0).float()
            sh_0 = inputData.data.new(1, batchSize, self.sHiddenSize).fill_(0).float()
            sc_0 = inputData.data.new(1, batchSize, self.sHiddenSize).fill_(0).float()
            h_0 = Variable(h_0)
            c_0 = Variable(c_0)
            sh_0 = Variable(sh_0)
            sc_0 = Variable(sc_0)
        else:
            h_0 = hidden[0]
            c_0 = hidden[1]
            sh_0 = hidden[2]
            sc_o = hidden[3]

        carOut = inputData[:, :, 0]
        carIn = inputData[:, :, 1]
        number = inputData[:, :, 2]

        laneControler = self.laneGate1(lane)
        laneControler = self.relu(laneControler)
        laneControler = self.laneGate2(laneControler)
        laneControler = self.sigm(laneControler)

        carOut = carOut * laneControler
        carIn = carIn * laneControler
        number = number * laneControler
        carOut.unsqueeze_(2)
        carIn.unsqueeze_(2)
        number.unsqueeze_(2)

        carOut = self.carOutFC(carOut)
        carIn = self.carInFC(carIn)
        number = self.numberFC(number)

        carOutOutput, carOutHidden = self.carOutRNNlayer(carOut, (sh_0, sc_0))
        carInOutput, carInHidden = self.carInRNNlayer(carIn, (sh_0, sc_0))
        numberOutput, numberHidden = self.numberRNNlayer(number, (sh_0, sc_0))

        data = torch.cat((carOutOutput, carInOutput, numberOutput), 2)

        data = self.embeddingFC(data)
        data = self.relu(data)
        _, hidden = self.gatherRNNlayer(data, (h_0, c_0))
        hidden = hidden[0].view(batchSize, -1)

        output = self.outputLayer(hidden)
        output = self.relu(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.fc3(output)

        output = output / laneControler

        return output, hidden[0]


class mdLSTM(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args
        
        #相关参数
        self.seqLength = args["seqLength"]
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
        self.embeddingFC1 = nn.Linear(self.inputSize, self.embeddingHiddenSize)
        self.embeddingFC2 = nn.Linear(self.embeddingHiddenSize, self.embeddingSize)
        
        #车道控制层
        self.laneGateFC1 = nn.Linear(1, self.laneGateFCSize)
        self.laneGateFC2 = nn.Linear(self.laneGateFCSize, 1)

        self.cell = torch.nn.LSTMCell(self.embeddingSize, self.hiddenSize)
        if self.gru:
            self.cell = torch.nn.GRUCell(self.embeddingSize, self.hiddenSize)

        self.outputLayer = nn.Linear(self.hiddenSize, self.outputSize)

        #这一阶段的输出
        self.fc1 = nn.Linear(self.outputSize, self.outputFC1Size)
        self.fc2 = nn.Linear(self.outputFC1Size, self.outputFC2Size)
        self.fc3 = nn.Linear(self.outputFC2Size, 1)

        #激活函数
        self.relu = nn.ReLU()
        self.sigma = nn.Sigmoid()
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)

    def forward(self, inputData, lane, hidden=None):

        [SpatialLength, temporalLength, inputSize] = inputData.size()
        if hidden == None:
            h_0 = inputData.data.new(SpatialLength, self.hiddenSize).fill_(0).float()
            c_0 = inputData.data.new(SpatialLength, self.hiddenSize).fill_(0).float()
            h_0 = Variable(h_0)
            c_0 = Variable(c_0)
        else:
            h_0 = hidden[0]
            c_0 = hidden[1]

        H = torch.zeros([SpatialLength, self.hiddenSize, temporalLength])
        C = torch.zeros([SpatialLength, self.hiddenSize, temporalLength])

        laneControler = self.laneGateFC1(lane)
        laneControler = self.relu(laneControler)
        laneControler = self.laneGateFC2(laneControler)
        laneControler = self.sigma(laneControler)
        laneControler = laneControler.view(1, 1, 1)

        inputData = inputData * laneControler
        inputData = self.embeddingFC1(inputData)
        inputData = self.relu(inputData)
        inputData = self.embeddingFC2(inputData)
        inputData = self.relu(inputData)

        for time in range(temporalLength):
            h_0, c_0 = self.cell(inputData[:, time, :], (h_0, c_0))
            h_0 = self.maxpool(h_0.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            H[:, :, time] = h_0
            C[:, :, time] = c_0

        output = self.outputLayer(h_0)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        laneControler = laneControler.view(-1, 1)
        output = output / laneControler
        

        return output, [h_0, c_0]