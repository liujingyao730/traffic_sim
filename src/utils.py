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

    def __init__(self, prefix=["defualt"], laneNumberPrefix="", simTimeStep = conf.args["trainSimStep"], 
                deltaT=conf.args["deltaT"], batchSize=conf.args["batchSize"], seqLength=conf.args["seqLength"], 
                cycle = conf.args["cycle"], Pass=conf.args["greenPass"], seqPredict=conf.args["seqPredict"]):

        self.deltaT = deltaT 
        self.filePrefixList = prefix
        self.prefixPoint = 0
        self.prefix = self.filePrefixList[0]
        self.fileNumber = len(prefix) - 1
        self.batchSize = batchSize
        self.simTimeStep = simTimeStep
        self.seqLength = seqLength
        self.seqPredict = seqPredict
        self.cycle = cycle
        self.Pass = Pass
        self.CurrentEdgePoint = 0
        self.laneNumberPrefix = laneNumberPrefix
        self.LaneNumberFileName = self.LaneNumberFile()
        self.LaneNumber = pd.read_csv(self.LaneNumberFileName, index_col=0)
        self.CarIn = {}
        self.CarOut = {}
        self.Number = {}
        
        for i in range(len(self.filePrefixList)):
            self.prefix = self.filePrefixList[i]
            self.CarInFileName = self.CarInFile()
            self.CarOutFileName = self.CarOutFile()
            self.NumberFileName = self.NumberFile()
            self.CarIn[i] = pd.read_csv(self.CarInFileName, index_col=0)
            self.CarOut[i] = pd.read_csv(self.CarOutFileName, index_col=0)
            self.Number[i] = pd.read_csv(self.NumberFileName, index_col=0)
             
        self.TimeBoundary = self.CarIn[self.prefixPoint].index[-1]
        self.indexNumber = len(self.CarIn[self.prefixPoint].index) - 1
        self.cycleNumber = int(self.indexNumber * self.simTimeStep / self.cycle)
        self.disableCycle = int(self.deltaT * self.seqLength / self.cycle) + 1 
        self.columnNumber = len(conf.edges) - 1
        self.CurrentTime = 0
        while not self.isTimePassable():
            self.CurrentTime += self.simTimeStep
        self.CurrentTime = round(self.CurrentTime, 3)

        self.CurrentSequences = []
        self.CurrentLane = []
        self.CurrentOutputs = []


    def generateNewSequence(self, column=None):
        
        edge = conf.edges[self.CurrentEdgePoint]
        if column == None:
            column = edge
        seq = []
        out = []
        time = self.CurrentTime

        for i in range(self.seqPredict):
            timeSlice = self.deltaTAccumulate(column, time)
            timeSlice.append(self.Number[self.prefixPoint].loc[time, column])
            seq.append(timeSlice)
            time += self.deltaT
        
        for i in range(self.seqPredict, self.seqLength):
            timeSlice = self.deltaTAccumulate(column, time)
            timeSlice.append(self.Number[self.prefixPoint].loc[time, column])
            seq.append(timeSlice)
            out.append(timeSlice[0])
            time += self.deltaT

        self.CurrentOutputs.append(out)
        self.CurrentSequences.append(seq)
        self.CurrentLane.append([self.LaneNumber.loc["laneNumber", edge]])

        return True

    def generateNewMatrix(self):

        self.CurrentSequences.clear()
        self.CurrentOutputs.clear()
        self.CurrentLane.clear()

        edge = self.CurrentEdgePoint + 1
        bucketL = edge * 100
        bucketH = bucketL + 100
        bucketList = []
        

        for bucket in self.CarIn[self.prefixPoint].columns:
            b = float(bucket)
            if b >= bucketL and  b < bucketH:
                bucketList.append(bucket)

        for bucket in bucketList:
            self.generateNewSequence(bucket)
            


    def generateBatch(self, laneNumber=[1,2,3,4,5,6]):

        #这里edge的车道数就是edge名，不高复杂的转换了……
        self.CurrentSequences.clear()
        self.CurrentOutputs.clear()
        self.CurrentLane.clear()
        numberEdge = len(laneNumber)

        for i in range(self.batchSize):

            if self.CurrentEdgePoint >= numberEdge:
                self.CurrentEdgePoint = 0
                self.CurrentTime += self.simTimeStep
                self.CurrentTime = round(self.CurrentTime, 3)
            
            if not self.isTimePassable():
                self.CurrentTime += self.cycle - self.Pass - self.simTimeStep + self.deltaT
                self.CurrentTime = round(self.CurrentTime, 3)
            if self.isTimeOutBoundary():
                if self.prefixPoint == self.fileNumber:
                    return False
                else:
                    self.setFilePoint(self.prefixPoint+1)
            self.generateNewSequence()
            
        return True
 

    def generateBatchRandom(self, laneNumber=[1,2,3,4,5,6]):

        self.CurrentOutputs.clear()
        self.CurrentSequences.clear()
        self.CurrentLane.clear()
        number = len(laneNumber) - 1

        for i in range(self.batchSize):
            self.setFilePoint(random.randint(0, self.fileNumber))
            self.generateRandomTime()
            self.CurrentEdgePoint = random.randint(0, number)
            self.generateNewSequence()


    def generateBatchForBucket(self, laneNumber=[1,2,3,4,5,6]):

        self.CurrentSequences = []
        self.CurrentOutputs = []
        self.CurrentLane = []
        

        numberEdge = len(laneNumber)
        tmplanes = np.array([])
        tmpInput = np.array([])
        tmpOutput = np.array([])
            
        edge = str(laneNumber[self.CurrentEdgePoint]) #这个地方不搞复杂的转换了.....
            
        if not self.isTimePassable():
            self.CurrentTime += self.cycle - self.Pass - self.simTimeStep + self.deltaT * (self.seqLength - self.seqPredict)
            self.CurrentTime = round(self.CurrentTime, 3)
        if self.isTimeOutBoundary():
            if self.prefixPoint == self.fileNumber:
                return False
            else:
                self.setFilePoint(self.prefixPoint+1)

        self.generateNewMatrix()
        self.CurrentOutputs = np.array(self.CurrentOutputs)
        self.CurrentSequences = np.array(self.CurrentSequences)
        self.CurrentLane = np.array(self.CurrentLane)

        self.CurrentEdgePoint += 1
        if self.CurrentEdgePoint >= numberEdge:
            self.CurrentEdgePoint = 0
            self.CurrentTime += self.simTimeStep
            self.CurrentTime = round(self.CurrentTime, 3)

        return True

    
    def generateBatchRandomForBucket(self, lane=[1,2,3,4,5,6]):
        self.CurrentOutputs = []
        self.CurrentSequences = []
        self.CurrentLane = []
        number = len(lane) - 1
        tmplanes = np.array([])
        tmpInput = np.array([])
        tmpOutput = np.array([])

        self.setFilePoint(random.randint(0, self.fileNumber))
        self.generateRandomTime()
        self.CurrentEdgePoint = random.randint(0, number)
        self.generateNewMatrix()
        


    def setFilePoint(self, point):

        self.prefixPoint = point
        self.prefix = self.filePrefixList[self.prefixPoint]
        self.TimeBoundary = self.CarIn[self.prefixPoint].index[-1]
        self.indexNumber = len(self.CarIn[self.prefixPoint].index) - 1
        self.cycleNumber = int(self.indexNumber * self.simTimeStep / self.cycle)
        
        self.CurrentEdgePoint = 0
        self.CurrentTime = 0
        while not self.isTimePassable():
            self.CurrentTime += self.simTimeStep
        self.CurrentTime = round(self.CurrentTime, 3)


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

        cycleIndex = random.randint(self.disableCycle, self.cycleNumber - 1)
        timeIndex = random.randint(0, (self.Pass - self.deltaT) / self.simTimeStep)
        self.CurrentTime = cycleIndex * self.cycle + timeIndex * self.simTimeStep - self.deltaT * self.seqLength
        self.CurrentTime = round(self.CurrentTime, 3)
        return 


    def isTimeOutBoundary(self):
        
        return self.CurrentTime + (self.seqLength + 1) * self.deltaT > self.TimeBoundary

    def isTimePassable(self):

        outStartTime = self.CurrentTime + self.seqPredict * self.deltaT
        outEndTime = self.CurrentTime + self.seqLength * self.deltaT
        return outStartTime % self.cycle <= self.Pass and outEndTime % self.cycle <= self.Pass

    def isTimeStartBeforeGreen(self):

        return (self.CurrentTime + self.seqPredict * self.deltaT) % self.cycle > self.Pass

    def isTimeEndAfterGreen(self):

        return (self.CurrentTime + self.seqLength * self.deltaT) % self.cycle > self.Pass

    def deltaTAccumulate(self, column, time):

        if self.CurrentTime + self.deltaT > self.TimeBoundary:
            return False

        Out = 0
        In = 0
        time = round(time + self.simTimeStep, 3)

        for i in range(int(self.deltaT / self.simTimeStep)):
            Out += self.CarOut[self.prefixPoint].loc[time, column]
            In += self.CarIn[self.prefixPoint].loc[time, column]
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


    