import os

# 相关路径
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.dirname(curPath)

dataPath = rootPath + "/dataFile"
fcdOutputPath = dataPath + "/fcdOutput"
simConfigurePath = dataPath + "/simConfigure"
midDataPath = dataPath + "/midData"
modelPath = dataPath + "/model"
picsPath = dataPath + "/pics"
resultPath = dataPath + "/result"

firstStageSim = simConfigurePath + "/firstStageSim"
firstStageFcd = fcdOutputPath + "/firstStage"

# debug文件
fcdDebug = fcdOutputPath + "/debug.xml"
netDebug = firstStageSim + "/basic.net.xml"
carInDebug = midDataPath + "/debugCarIn.csv"
carOutDebug = midDataPath + "/debugCarOut.csv"
numberDebug = midDataPath + "/debugNumber.csv"
laneNumberDebug = midDataPath + "/debugLaneNumber.csv"
modelDebug = modelPath + "/debugModel.pth"

# 一些固定输出的文件，会覆盖，注意及时保存
fcdDefualt = firstStageFcd + "/0.5_0.5_7200.xml"
carInDefualt = midDataPath + "/defualtCarIn.csv"
carOutDefualt = midDataPath + "/defualtCarOut.csv"
numberDefualt = midDataPath + "/defualtNumber.csv"
laneNumberDefualt = midDataPath + "/defualtLaneNumeber.csv"
modelDefualt = modelPath + "/defualtModel.pth"

# 关心的路段
edges = ["1", "2", "3", "4", "5", "6"]

# 数据相关变量
deltaT = 10
sequenceLength = 40
batchSize = 50
simTimeStep = 1
cycle = 90
greenPass = 42
yellowPass = 45


# 网络参数
'''    不区分进入lstm层的     '''
args = {}
args["useCuda"] = True
args["seqLength"] = 20
args["hiddenSize"] = 64
args["batchSize"] = batchSize
args["embeddingSize"] = 32
args["inputSize"] = 3
args["outputSize"] = 16
args["fc1"] = 32
args["fc2"] = 16
args["inputFC1"] = 8
args["laneGateFC"] = 4
'''     分别进入三个lstm的     '''
args["sHiddenSize"] = 8
args["sEmbeddingSize"] = 4
args["modelFilePrefix"] = "800"

# 训练参数
args["gru"] = False
args["dropOut"] = 0.4
args["batchNum"] = 300
args["epoch"] = 10
args["plotEvery"] = 5
args["prefix"] = ["800_1", "800_2", "800_3"]
args["trainSimStep"] = 0.1
args["testFilePrefix"] = "defualt"
args["testSimStep"] = 1
args["testBatch"] = 1000
args["gpu_id"] = [0]

args["version"] = "0427"

# 文件名的生成
def modelName(prefix):
    return modelPath + "/" + prefix + "_" + args["version"] + ".pth"

def picsName(prefix):
    return picsPath + "/" + prefix + "_" + args["version"] + ".png"

def csvName(prefix):
    return resultPath + "/" + prefix + "_" + args["version"] + ".csv"

def fcd(prefix, fold="fristStage"):
    return fcdOutputPath + "/" + fold + "/" + prefix + ".xml"

def midDataName(prefix, filetype="carIn"):
    return midDataPath + "/" + prefix + filetype + ".csv"