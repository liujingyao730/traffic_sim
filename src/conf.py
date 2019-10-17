import os

# 相关路径
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.dirname(curPath)

dataPath = os.path.join(rootPath, 'dataFile')
fcdOutputPath = os.path.join(dataPath, 'fcdOutput')
simConfigurePath = os.path.join(dataPath, 'simConfigure')
midDataPath = os.path.join(dataPath, 'midData')
modelPath = os.path.join(dataPath, 'model')
picsPath = os.path.join(dataPath, 'pics')
logPath = os.path.join(dataPath, 'log')
configPath = os.path.join(dataPath, 'config')

laneNumberDefualt = os.path.join(midDataPath, 'defualtLaneNumber.csv')
carInDefualt = os.path.join(midDataPath, 'defualtCarIn.csv')
carOutDefualt = os.path.join(midDataPath, 'defualtCarOut.csv')
numberDefualt = os.path.join(midDataPath, 'defualtNumber.csv')

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
args["eva_prefix"] = ['2000_nosignal']

# 数据处理中用到的 dataprocess.py中
args["deltaT"] = 10
args["trainSimStep"] = 0.1

# mask
args["mask"] = [0.6, 0.85, 0.9, 0.58, 0.54, 0.91, 1.15, 1.25, 1.36, 6.45, 13.35, 48.75, 890]
#args['mask'] = [0.55, 0.8, 0.88, 0.78, 0.92, 1.66, 2.23, 2.39, 3.04, 2.86, 2.36, 1.43, 1]

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