import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class BasicLSTM(nn.Module):

    def __init__(self, args, infer=False):

        super().__init__()

        self.args = args
        self.infer = infer

        self.seqLength = args["seqLength"]

        self.hiddenSize = args["hiddenSize"]
        self.embeddingSize = args["embeddingSize"]
        self.inputSize = args["inputSize"]
        self.outputSize = args["outputSize"]
        self.gru = args["gru"]
        self.args = args

        self.RNNlayer = nn.LSTM(self.embeddingSize, self.hiddenSize, batch_first=True)
        if self.gru:
            self.RNNlayer = nn.GRU(self.embeddingSize, self.hiddenSize)

        self.outputLayer = nn.Linear(self.hiddenSize, self.outputSize)
        self.fc1 = nn.Linear(self.outputSize, args['fc1'])
        self.fc2 = nn.Linear(args["fc1"], args["fc2"])
        self.fc3 = nn.Linear(args["fc2"], 1)

        self.laneGate1 = nn.Linear(1, 3)
        self.laneGate2 = nn.Linear(3, 1)

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

        inputData = laneControler * inputData 
        embedds = self.embeddings(inputData)
        output, hidden = self.RNNlayer(embedds, (h_0, c_0))

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

