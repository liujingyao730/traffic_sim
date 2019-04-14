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
deltaT = 5
sequenceLength = 20
batchSize = 50
simTimeStep = 1
cycle = 120
greenPass = 57
yellowPass = 60


# 网络参数
'''    不区分进入lstm层的     '''
args = {}
args["useCuda"] = True
args["seqLength"] = 20
args["hiddenSize"] = 64
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

# 训练参数
args["gru"] = False
args["dropOut"] = 0.4
args["batchNum"] = 300
args["epoch"] = 50
args["plotEvery"] = 5
args["prefix"] = "data"
args["trainSimStep"] = 0.1
args["testFilePrefix"] = "defualt"
args["testSimStep"] = 1
args["testBatch"] = 1000

args["version"] = "0410"

# 文件名的生成
def modelName(prefix):
    return modelPath + "/" + prefix + "_" + args["version"] + ".pth"

def picsName(prefix):
    return picsPath + "/" + prefix + "_" + args["version"] + ".png"