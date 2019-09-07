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
logPath = dataPath + "/log"

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
speedDefualt = midDataPath + "/defualtSpeed.csv"
laneNumberDefualt = midDataPath + "/defualtLaneNumeber.csv"
modelDefualt = modelPath + "/defualtModel.pth"

# 

# 关心的路段
# edges = ["1", "2", "3", "4", "5", "6"] # 8-20改 之后的数据是单双车道
# laneNumber = [1, 2, 3, 4, 5, 6]

edges = ['1', '2']
laneNumber = [1, 2]

# 路段裁剪量
cut = 45

args = {}
args["modelFilePrefix"] = "7-26"
args["prefix"] = ["300", "400", "500", "600", "700", "800", "900", "1000", "1100", "1200", "1300", 
                "1400", "1500"]
args["test_prefix"] = ['1000_1']
args["testFilePrefix"] = ["300_1", "400_1", "500_1", "600_1", "700_1", "800_1", "900_1", "1000_1",
                 "1100_1", "1200_1", "1300_1", "1400_1", "1500_1"]

# 数据处理中用到的 dataprocess.py中
args["deltaT"] = 10
args["trainSimStep"] = 0.1

# 网络中的结构


# 文件名的生成
def modelName(prefix):
    return modelPath + "/" + prefix + "_" + args["version"] + ".pth"

def picsName(prefix):
    return picsPath + "/" + prefix + "_" + args["version"] + ".png"

def csvName(prefix):
    return resultPath + "/" + prefix + "_" + args["version"] + ".csv"

def fcd(prefix, fold="fristStage"):
    return fcdOutputPath + "/" + fold + "/" + prefix + ".xml"

def midDataName(prefix, filetype="CarIn"):
    return midDataPath + "/" + prefix + filetype + ".csv"