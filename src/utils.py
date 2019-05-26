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

    def __init__(self, prefix=["defualt"], laneNumberPrefix="", simTimeStep = conf.simTimeStep, 
                deltaT=conf.deltaT, batchSize=conf.batchSize, seqLength=conf.sequenceLength, 
                cycle = conf.cycle, Pass=conf.greenPass):

        self.deltaT = deltaT 
        self.filePrefixList = prefix
        self.prefixPoint = 0
        self.prefix = self.filePrefixList[0]
        self.fileNumber = len(prefix) - 1
        self.batchSize = batchSize
        self.simTimeStep = simTimeStep
        self.seqLength = seqLength
        self.cycle = cycle
        self.Pass = Pass
        self.laneNumberPrefix = laneNumberPrefix

        self.CarInFileName = self.CarInFile()
        self.CarOutFileName = self.CarOutFile()
        self.NumberFileName = self.NumberFile()
        self.LaneNumberFileName = self.LaneNumberFile()

        self.CarIn = pd.read_csv(self.CarInFileName, index_col=0)
        self.CarOut = pd.read_csv(self.CarOutFileName, index_col=0)
        self.Number = pd.read_csv(self.NumberFileName, index_col=0)
        self.LaneNumber = pd.read_csv(self.LaneNumberFileName, index_col=0)
                
        self.CurrentEdgePoint = 0
        self.CurrentTime = 0
        self.TimeBoundary = self.CarIn.index[-1]
        self.indexNumber = len(self.CarIn.index) - 1
        self.cycleNumber = int(self.indexNumber * self.simTimeStep / conf.cycle) 
        self.columnNumber = len(conf.edges) - 1
        while not self.isTimePassable():
            self.CurrentTime += self.simTimeStep
        self.CurrentTime = round(self.CurrentTime, 3)
        self.CurrentSequences = []
        self.CurrentLane = []
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
        self.CurrentLane.append([self.LaneNumber.loc["laneNumber", edge]])

        return True


    def generateBatch(self, laneNumber=[1,2,3,4,5,6]):

        self.CurrentSequences.clear()
        self.CurrentOutputs.clear()
        self.CurrentLane.clear()
        
        edges = []
        l = 0
        numberEdge = len(laneNumber)
        for lane in laneNumber:
            for i in range(len(self.LaneNumber.columns)):
                edge = self.LaneNumber.columns[i]
                if self.LaneNumber.loc["laneNumber", edge] == lane:
                    edges.append(i)
                    break
            
        for i in range(self.batchSize):
            self.CurrentEdgePoint = edges[l]
            l += 1
            if l >= numberEdge:
                l = 0
                self.CurrentTime += self.simTimeStep
                self.CurrentTime = round(self.CurrentTime, 3)
            
            if self.isTimeOutBoundary():
                if self.prefixPoint == self.fileNumber:
                    return False
                else:
                    self.setFilePoint(self.prefixPoint+1)
            if not self.isTimePassable():
                self.CurrentTime += self.cycle - self.Pass - self.simTimeStep + self.deltaT
                self.CurrentTime = round(self.CurrentTime, 3)
            self.generateNewSequence()
            
        return True
            

    def setFilePoint(self, point):

        self.prefixPoint = point
        self.prefix = self.filePrefixList[self.prefixPoint]
        self.CarInFileName = self.CarInFile()
        self.CarOutFileName = self.CarOutFile()
        self.NumberFileName = self.NumberFile()
        self.CarIn = pd.read_csv(self.CarInFileName, index_col=0)
        self.CarOut = pd.read_csv(self.CarOutFileName, index_col=0)
        self.Number = pd.read_csv(self.NumberFileName, index_col=0)
        self.CurrentEdgePoint = 0
        self.CurrentTime = 0
        while not self.isTimePassable():
            self.CurrentTime += self.simTimeStep
        self.CurrentTime = round(self.CurrentTime, 3)
        self.TimeBoundary = self.CarIn.index[-1]
        self.indexNumber = len(self.CarIn.index) - 1
        self.cycleNumber = int(self.indexNumber * self.simTimeStep / conf.cycle)
 

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
                self.generateRandomTime()
                if self.isTimeIdeal():
                        break
            self.generateNewSequence()


    def isTimeIdeal(self):

        outStartTime = self.CurrentTime + self.seqLength * self.deltaT
        outEndTime = self.CurrentTime + (self.seqLength + 1) * self.deltaT

        if outEndTime > self.TimeBoundary:
            return False

        if outStartTime % self.cycle > self.Pass:
            return False

        if outEndTime % self.cycle > self.Pass:
            return False

        return True

    def generateRandomTime(self):

        cycleIndex = random.randint(2, self.cycleNumber - 1)
        timeIndex = random.randint(0, (conf.greenPass - self.deltaT) / self.simTimeStep)
        self.CurrentTime = cycleIndex * conf.cycle + timeIndex * self.simTimeStep - self.deltaT * self.seqLength
        self.CurrentTime = round(self.CurrentTime, 3)
        return 


    def isTimeOutBoundary(self):
        
        return self.CurrentTime + (self.seqLength + 1) * self.deltaT > self.TimeBoundary

    def isTimePassable(self):

        outStartTime = self.CurrentTime + self.seqLength * self.deltaT
        outEndTime = self.CurrentTime + (self.seqLength + 1) * self.deltaT
        return outStartTime % self.cycle <= self.Pass and outEndTime % self.cycle <= self.Pass

    def isTimeStartBeforeGreen(self):

        return (self.CurrentTime + self.seqLength * self.deltaT) % self.cycle > self.Pass

    def isTimeEndAfterGreen(self):

        return (self.CurrentTime + (self.seqLength + 1) * self.deltaT) % self.cycle > self.Pass

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
        return conf.midDataPath + "/" + self.laneNumberPrefix + "LaneNumber.csv"


    