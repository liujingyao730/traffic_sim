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
        self.fileNumber = len(prefix) - 1 #因为后面转换文件的时候的索引是从0开始的
        self.batchSize = batchSize
        self.simTimeStep = simTimeStep
        self.seqLength = seqLength
        self.seqPredict = seqPredict
        self.predLength = self.seqLength - self.seqPredict + 1
        self.cycle = cycle
        self.Pass = Pass
        self.timeindexBoundary = (self.Pass - self.predLength * self.deltaT) / self.simTimeStep
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

        self.CurrentSequences = np.array([])
        self.CurrentLane = []
        self.CurrentOutputs = np.array([])

    def generateNewMatrix(self):

        self.CurrentSequences = np.array([])
        self.CurrentLane = []
        self.CurrentOutputs = np.array([])

        edge = self.CurrentEdgePoint + 1
        bucketL = edge * 100
        bucketH = bucketL + 100
        bucketList = []
        timeList = []

        for bucket in self.CarIn[self.prefixPoint].columns:
            b = float(bucket)
            if b >= bucketL and  b < bucketH:
                bucketList.append(bucket)

        for i in range(self.seqLength):
            timeList.append(self.CurrentTime + i * self.deltaT)
        timeList.append(self.CurrentTime + self.seqLength * self.deltaT)

        spatial = len(bucketList)
        temporal = len(timeList)

        In = np.array(self.CarIn[self.prefixPoint].loc[timeList, bucketList].T).reshape(spatial, temporal, 1)
        out = np.array(self.CarOut[self.prefixPoint].loc[timeList, bucketList].T).reshape(spatial, temporal, 1)
        number = np.array(self.Number[self.prefixPoint].loc[timeList, bucketList].T).reshape(spatial, temporal, 1)

        self.CurrentSequences = np.concatenate((In[:, :-1, :], out[:, :-1, :], number[:, :-1, :]), axis=2)
        self.CurrentOutputs = np.concatenate((In[:, self.seqPredict+1:, :], out[:, self.seqPredict+1:, :], number[:, self.seqPredict+1:, :]), axis=2)        

        edge = str(edge)
        self.CurrentLane.append(self.LaneNumber.loc["laneNumber", edge])


    def generateBatchForBucket(self, laneNumber=conf.laneNumber):
        
        numberEdge = len(laneNumber)
        self.CurrentSequences = np.array([])
        self.CurrentLane = []
        self.CurrentOutputs = np.array([])
        batch_data = []
        tmpOutput = []
        tmpLane = []
            
        for i in range(self.batchSize):
            edge = str(laneNumber[self.CurrentEdgePoint]) #这个地方不搞复杂的转换了.....

            if not self.isTimePassable():
                self.CurrentTime += self.cycle - self.Pass - self.simTimeStep + self.deltaT * (self.seqLength - self.seqPredict + 1)
                self.CurrentTime = round(self.CurrentTime, 3)
            if self.isTimeOutBoundary():
                if self.prefixPoint == self.fileNumber:
                    return False # 重置文件的动作放在外面进行
                else:
                    self.setFilePoint(self.prefixPoint+1)
                    break # 保证每个batch都是相同长度的路段

            self.generateNewMatrix()
            tmpOutput.append(self.CurrentOutputs)
            batch_data.append(self.CurrentSequences)
            tmpLane.append(self.CurrentLane)

            self.CurrentEdgePoint += 1
            if self.CurrentEdgePoint >= numberEdge:
                self.CurrentEdgePoint = 0
                self.CurrentTime += self.simTimeStep
                self.CurrentTime = round(self.CurrentTime, 3)

        self.CurrentOutputs = np.array(tmpOutput)
        self.CurrentSequences = np.array(batch_data)
        self.CurrentLane = np.array(tmpLane)

        return True

    
    def generateBatchRandomForBucket(self, lane=conf.laneNumber):
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
        timeIndex = random.randint(0, self.timeindexBoundary)
        self.CurrentTime = cycleIndex * self.cycle + timeIndex * self.simTimeStep + \
                            self.predLength * self.deltaT - self.deltaT * (self.seqLength + 1)
        self.CurrentTime = round(self.CurrentTime, 3)
        return 


    def isTimeOutBoundary(self):
        
        return self.CurrentTime + (self.seqLength + 1) * self.deltaT > self.TimeBoundary

    def isTimePassable(self):

        outStartTime = self.CurrentTime + self.seqPredict * self.deltaT
        outEndTime = self.CurrentTime + (self.seqLength + 1) * self.deltaT
        return outStartTime % self.cycle <= self.Pass and outEndTime % self.cycle <= self.Pass

    def isTimeStartBeforeGreen(self):

        return (self.CurrentTime + self.seqPredict * self.deltaT) % self.cycle > self.Pass

    def isTimeEndAfterGreen(self):

        return (self.CurrentTime + (self.seqLength + 1) * self.deltaT) % self.cycle > self.Pass
        

    def CarInFile(self):
        return conf.midDataPath + "/" + self.prefix + "CarIn.csv"
    
    def CarOutFile(self):
        return conf.midDataPath + "/" + self.prefix + "CarOut.csv"

    def NumberFile(self):
        return conf.midDataPath + "/" + self.prefix + "Number.csv"

    def LaneNumberFile(self):
        return conf.midDataPath + "/" + self.laneNumberPrefix + "LaneNumber.csv"    