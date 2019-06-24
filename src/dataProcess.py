import pandas as pd 
import xml.etree.cElementTree as ET 
import pyecharts as pe
import numpy as np
from sklearn import metrics

import conf
'''

这部分是用来对SUMO的路网以及仿真文件进行处理，carIn, carOut, number三个csv文件的代码
carIn表示这一时刻与前一时刻相比，路段流入了多少辆车
carOut表示这一时刻与前一时刻相比，路段流出的多少辆车
number表示这一时刻，路段内有多少辆车
以上每一个数据以一个dataframe的形式单独存放在一个csv文件中，columns是路段名，index是对应的时刻

'''


def laneEdge(netXml):
    '''
    路网处理，从路网文件中获得lane对应的edge，并以字典形式返回
    '''
    netFile = ET.parse(netXml)
    root = netFile.getroot()
    result = {}
    
    for element in root:
        if element.tag == "edge":
            edge = element.attrib["id"]
            for laneEle in element:
                lane = laneEle.attrib['id']
                result[lane] = edge

    return result


def laneNumber(netXml, laneNumberFile=conf.laneNumberDefualt):
    '''
    路网处理，提取出我们关心的路段的车道数并存入文件
    '''
    netFile = ET.parse(netXml)
    root = netFile.getroot()
    result = {}

    for element in root:
        if element.tag == "edge" and element.attrib['id'] in conf.edges:
            edge = element.attrib['id']
            result[edge] = 0
            for lane in element:
                result[edge] += 1

    result = pd.DataFrame(result, index=["laneNumber"])
    result.to_csv(laneNumberFile)
    print("laneNumber information have been saved as ", laneNumberFile)

    return result


def edgeRecord(netXml, fcdXml, carInFile=conf.carInDefualt, carOutFile=conf.carOutDefualt, numberFile=conf.numberDefualt):
    '''
    从仿真文件中读取出来我们关注的路段每个时刻的流入、流出以及瞬时车辆存量，并把结果存放在相应的文件中
    '''
    LaneEdge = laneEdge(netXml)
    fcdFile = ET.parse(fcdXml)
    fcdRoot = fcdFile.getroot()
    formStep = {}
    nowStep = {}
    columns = conf.edges
    carIn = pd.DataFrame(columns=columns)
    carOut = pd.DataFrame(columns=columns)
    number = pd.DataFrame(columns=columns)

    for timeStep in fcdRoot:
        time = float(timeStep.attrib['time'])
        carIn.loc[time] = 0
        carOut.loc[time] = 0
        number.loc[time] = 0

        for vehicleEle in timeStep:
            vehicle = vehicleEle.attrib['id']
            lane = vehicleEle.attrib['lane']
            edge = LaneEdge[lane]
            nowStep[vehicle] = edge
            if edge in carIn.columns:
                number.loc[time, edge] += 1
        
        for vehicle in nowStep.keys():
            if vehicle not in formStep.keys():
                edge = nowStep[vehicle]
                if edge in carIn.columns:
                    carIn.loc[time, edge] += 1
            elif nowStep[vehicle] != formStep[vehicle]:
                nowEdge = nowStep[vehicle]
                formEdge = formStep[vehicle]
                if nowEdge in carIn.columns:
                    carIn.loc[time, nowEdge] += 1
                if formEdge in carOut.columns:
                    carOut.loc[time, formEdge] += 1
                formStep.pop(vehicle)
            else:
                formStep.pop(vehicle)
        for vehicle in formStep.keys():
            edge = formStep[vehicle]
            if edge in carIn.columns:
                carOut.loc[time, edge] += 1

        formStep = nowStep.copy()
        nowStep.clear()

    carIn.to_csv(carInFile)
    carOut.to_csv(carOutFile)
    number.to_csv(numberFile)
    print("carIn information have been saved as ", carInFile)
    print("carOut information have been saved as ", carOutFile)
    print("number information have been saved as ", numberFile)

    return 


def bucketId(pos, length):
    

    pos -= conf.cut
    length -= conf.cut
    bucketNumber = int(length / 50)

    if pos >= bucketNumber * 50:
        return False

    for i in range(bucketNumber):
        if pos < (i+1)*50:
            return i+1


def bucketRecord(netXml, fcdXml, length, carInFile=conf.carInDefualt, carOutFile=conf.carOutDefualt, numberFile=conf.numberDefualt):
    '''这里还是采用和基于路段的中间数据的格式
    但不同的是这里是基于每个小bucket的流入流出，所以需要按照bucket进行存储
    每个bucket的命名规则是三位数字，百位代表车道数，十位与各位代表编号，从小到大
    每个bucket长50米
    '''
    LaneEdge = laneEdge(netXml)
    laneNumbers = laneNumber(netXml)
    fcdFile = ET.parse(fcdXml)
    fcdRoot = fcdFile.getroot()
    formStep = {}
    nowStep = {}
    carIn = pd.DataFrame(columns=["label"])
    carOut = pd.DataFrame(columns=["label"])
    number = pd.DataFrame(columns=["label"])

    for timeStep in fcdRoot:
        time = float(timeStep.attrib['time'])
        carIn.loc[time] = 0
        carOut.loc[time] = 0
        number.loc[time] = 0

        for vehicleEle in timeStep:
            vehicle = vehicleEle.attrib['id']
            lane = vehicleEle.attrib['lane']
            edge = LaneEdge[lane]
            if edge in conf.edges:
                pos = float(vehicleEle.attrib["pos"])
                lanes = laneNumbers.loc["laneNumber", edge]
                bucket = bucketId(pos, length)
                if time == 24 and vehicle == '3to-gneE30.0':
                    a = 1
                if bucket is False:
                    continue
                nowStep[vehicle] = lanes*100 + bucket
                if nowStep[vehicle] not in number.columns:
                    number[nowStep[vehicle]] = 0
                    carIn[nowStep[vehicle]] = 0
                    carOut[nowStep[vehicle]] = 0
                number.loc[time, nowStep[vehicle]] += 1
            else:
                a = 1

        for vehicle in nowStep.keys():
            if vehicle not in formStep.keys():
                carIn.loc[time, nowStep[vehicle]] += 1
            else:
                if formStep[vehicle] != nowStep[vehicle]:
                    for bucket in range(formStep[vehicle], nowStep[vehicle]+1, 1):
                        carOut.loc[time, bucket] += 1
                        carIn.loc[time, bucket] += 1
                    carOut.loc[time, nowStep[vehicle]] -= 1
                    carIn.loc[time, formStep[vehicle]] -= 1
                formStep.pop(vehicle)

        for vehicle in formStep.keys():
            carOut.loc[time, formStep[vehicle]] += 1

        formStep = nowStep.copy()
        nowStep.clear()

    carIn.drop("label", axis=1, inplace=True)
    carOut.drop("label", axis=1, inplace=True)
    number.drop("label", axis=1, inplace=True)
    carIn.to_csv(carInFile)
    carOut.to_csv(carOutFile)
    number.to_csv(numberFile)
    print("carIn information have been saved as ", carInFile)
    print("carOut information have been saved as ", carOutFile)
    print("number information have been saved as ", numberFile)

    return 


def dataCheck(carInFile, carOutFile, numberFile):
    '''
    检查我们提取的数据满不满足 carIn_t + number_t-1 - carOut_t = number_t
    如果存在不平衡的状态，则输出两者之间的差值
    '''
    carIn = pd.read_csv(carInFile, index_col=0)
    carOut = pd.read_csv(carOutFile, index_col=0)
    number = pd.read_csv(numberFile, index_col=0)

    for bucket in carIn.columns:
        for time in carIn.index:
            if time-1 not in carIn.index:
                continue
            OK = carIn.loc[time, bucket]-carOut.loc[time, bucket]+number.loc[time-1, bucket] - number.loc[time, bucket] == 0
            if not OK:
                print(time, bucket, carIn.loc[time, bucket]-carOut.loc[time, bucket]+number.loc[time-1, bucket], number.loc[time, bucket])



def resultBoxplot(prefix):

    fileName = conf.csvName(prefix)
    try:
        data = pd.read_csv(fileName, index_col=0)
        data = data.T
    except:
        print("there is no such file as ", fileName)
        return

    target = np.array(data["target"])
    result = np.array(data["result"])
    r2 = "r2_sroce : " + str(metrics.r2_score(target, result))
    absv = "mean_absolute_error : " + str(metrics.mean_absolute_error(target, result))

    target = set(list(target))
    x = list(target)
    y = []

    for number in  target:
        l = list(data["result"][data["target"]==number])
        y.append(l)

    boxplot = pe.Boxplot(prefix+"\n"+r2+"\n"+absv)
    boxplot.add("", x, boxplot.prepare_data(y))
    baseline = pe.Scatter("baseline")
    baseline.add("", x, x)
    overlap = pe.Overlap()
    overlap.add(boxplot)
    overlap.add(baseline)

    overlap.render(conf.picsName(prefix))
    return 


# edgeRecord(conf.netDebug, conf.fcdDebug, conf.carInDebug, conf.carOutDebug, conf.numberDebug)
# dataCheck(conf.carInDebug, conf.carOutDebug, conf.numberDebug)
# laneNumber(conf.netDebug, conf.laneNumberDebug)

'''
[carIn, carOut, number] = edgeRecord(conf.netDebug, conf.fcdDebug)
carIn.to_csv(conf.carInDebug)
carOut.to_csv(conf.carOutDebug)
number.to_csv(conf.numberDebug)

'''