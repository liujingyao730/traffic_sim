import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class BasicLSTM(nn.Module):

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
        self.args = args

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

        inputData = laneControler * inputData 
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
        output = output / laneControler

        return output,hidden[0]


    def embeddings(self, input):

        return input

class stackedLSTM(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args

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