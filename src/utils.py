import os
import pickle
import numpy as np 
import pandas as pd 
import torch
import math
import random
from torch.autograd import Variable

import conf 

class batchGenerator(object):

    def __init__(self, prefix="defualt", simTimeStep = conf.simTimeStep, deltaT=conf.deltaT, batchSize=conf.batchSize, seqLength=conf.sequenceLength, infer=False, generate=False):

        self.deltaT = deltaT 

        self.prefix = prefix
        self.batchSize = batchSize
        self.simTimeStep = simTimeStep
        self.seqLength = seqLength
        self.infer = infer
        self.generate = generate

        self.CarInFile = self.CarInFile()
        self.CarOutFile = self.CarOutFile()
        self.NumberFile = self.NumberFile()
        self.LaneNumberFile = self.LaneNumberFile()

        self.CarIn = pd.read_csv(self.CarInFile, index_col=0)
        self.CarOut = pd.read_csv(self.CarOutFile, index_col=0)
        self.Number = pd.read_csv(self.NumberFile, index_col=0)
        self.LaneNumber = pd.read_csv(self.LaneNumberFile, index_col=0)
                
        self.CurrentEdgePoint = 0
        self.CurrentTime = 0
        self.TimeBoundary = self.CarIn.index[-1]
        self.indexNumber = len(self.CarIn.index) - 1
        self.columnNumber = len(conf.edges) - 1
    
        self.CurrentSequences = []
        self.CurrentLane = 0
        self.CurrentOutputs = []


    def generateNewSequence(self):
        
        if self.CurrentTime + (self.seqLength + 1) * self.deltaT > self.TimeBoundary:
            return False

        edge = conf.edges[self.CurrentEdgePoint]
        seq = []
        time = self.CurrentTime

        for i in range(self.seqLength):
            timeSlice = self.deltaTAccumulate(edge, time)
            timeSlice.append(self.Number.loc[time, edge])
            seq.append(timeSlice)
            time += self.deltaT

        self.CurrentSequences.append(seq)
        self.CurrentOutputs.append(self.deltaTAccumulate(edge, time)[0])

        return True


    def generateBatch(self, isRandom=True):

        self.CurrentSequences.clear()
        self.CurrentOutputs.clear()

        if isRandom:
            self.CurrentEdgePoint = random.randint(0, self.columnNumber)
            edge = conf.edges[self.CurrentEdgePoint]
            self.CurrentLane = self.LaneNumber.loc["laneNumber", edge]
            for i in range(self.batchSize):
                while True:
                    self.CurrentTime = self.CarIn.index[random.randint(0, self.indexNumber)]
                    if self.isTimeIdeal():
                        break
                self.generateNewSequence()
        else:
            edge = conf.edges[self.CurrentEdgePoint]
            self.CurrentLane = self.LaneNumber.loc["laneNumber", edge]
            for i in range(self.batchSize):
                while not self.isTimePassable():
                    self.CurrentTime += self.simTimeStep
                
                if self.isTimeOutBoundary():
                    self.CurrentTime = 0

                self.generateNewSequence()
                self.CurrentTime += self.simTimeStep
            self.CurrentEdgePoint = (self.CurrentEdgePoint + 1) % self.columnNumber


    def generateBatchLane(self, lane):

        self.CurrentOutputs.clear()
        self.CurrentSequences.clear()
        self.CurrentLane = lane
   
        for i in range(len(self.LaneNumber.columns)):
            edge = self.LaneNumber.columns[i]
            if self.LaneNumber.loc["laneNumber", edge] == lane:
                self.CurrentEdgePoint = i
                break

        for i in range(self.batchSize):
            while True:
                self.CurrentTime = self.CarIn.index[random.randint(0, self.indexNumber)]
                if self.isTimeIdeal():
                        break
            self.generateNewSequence()


    def isTimeIdeal(self):

        outStartTime = self.CurrentTime + self.seqLength * self.deltaT
        outEndTime = self.CurrentTime + (self.seqLength + 1) * self.deltaT

        if outEndTime > self.TimeBoundary:
            return False

        if outStartTime % conf.cycle > conf.greenPass:
            return False

        if outEndTime % conf.cycle > conf.greenPass:
            return False

        return True

    def isTimeOutBoundary(self):
        
        return self.CurrentTime + (self.seqLength + 1) * self.deltaT > self.TimeBoundary

    def isTimePassable(self):

        outStartTime = self.CurrentTime + self.seqLength * self.deltaT
        outEndTime = self.CurrentTime + (self.seqLength + 1) * self.deltaT
        return outStartTime % conf.cycle <= conf.greenPass and outEndTime % conf.cycle <= conf.greenPass

    def deltaTAccumulate(self, edge, time):

        if self.CurrentTime + self.deltaT > self.TimeBoundary:
            return False

        Out = 0
        In = 0
        time = round(time + self.simTimeStep, 3)

        for i in range(int(self.deltaT / self.simTimeStep)):
            Out += self.CarOut.loc[time, edge]
            In += self.CarIn.loc[time, edge]
            time = round(time + self.simTimeStep, 3)

        return [Out, In]
        

    def CarInFile(self):
        return conf.midDataPath + "/" + self.prefix + "CarIn.csv"
    
    def CarOutFile(self):
        return conf.midDataPath + "/" + self.prefix + "CarOut.csv"

    def NumberFile(self):
        return conf.midDataPath + "/" + self.prefix + "Number.csv"

    def LaneNumberFile(self):
        return conf.midDataPath + "/" + self.prefix + "LaneNumber.csv"


    