import numpy as np
import xml.etree.cElementTree as etree
import pandas as pd
import os 

import conf

merge_fcd_path = os.path.join(conf.fcdOutputPath, 'merge', 'fcd.xml')
#路网的结构是固定的
length = {1:600, 2:600, 3:200, 4:200, 7:200, 8:200}
lane_number = {1:1, 2:2, 3:2, 4:1, 5:2, 6:3, 8:1, 7:2}
special_edge = [5, 6]
lane_edge = {'1_0':1, '2_0':2, '2_1':2, '3_0':3, '3_1':3, '4_0':4, '7_0':7, '7_1':7,
            '8_0':8, ':gneJ1_0_0':5, ':gneJ1_4_0':5, ':gneJ2_6_0':6, ':gneJ2_10_0':6,
            ':gneJ2_10_1':6}

def bucketid(pos, length):
    
    bucketNumber = int(length / 50)

    if pos >= bucketNumber * 50 or pos < 0:
        print("there must be something wrong !!!!!!!")
        return False

    for i in range(bucketNumber):
        if pos < (i+1)*50:
            return i+1

    return bucketNumber + 1

def data_record(file, lane_edge=lane_edge, length=length, prefix='default', 
                special_edge=special_edge, fold=conf.midDataPath):

    fcdxml = etree.parse(file)
    root = fcdxml.getroot()

    formstep = {}
    nowstep = {}

    car_in = pd.DataFrame(columns=["label"])
    car_out = pd.DataFrame(columns=["label"])
    number = pd.DataFrame(columns=["label"])

    for timeStep in root:
        time = float(timeStep.attrib["time"])
        car_in.loc[time] = 0
        car_out.loc[time] = 0
        number.loc[time] = 0

        for vehicle in timeStep:
            vehicle_id = vehicle.attrib["id"]
            lane = vehicle.attrib['lane']
            edge = lane_edge[lane]

            if edge not in special_edge:
                pos = float(vehicle.attrib["pos"])
                bucket = bucketid(pos, length[edge]) + edge*100
            else:
                bucket = edge*100 + 1

            nowstep[vehicle_id] = bucket
            if bucket not in number.columns:
                number[bucket] = 0
                car_in[bucket] = 0
                car_out[bucket] = 0

            number.loc[time, bucket] += 1

        for vehicle in nowstep.keys():
            if vehicle not in formstep.keys():
                car_in.loc[time, nowstep[vehicle]] += 1
            else:
                if formstep[vehicle] != nowstep[vehicle]:
                    car_in.loc[time, nowstep[vehicle]] += 1
                    car_out.loc[time, formstep[vehicle]] += 1
                formstep.pop(vehicle)

        for vehicle in formstep.keys():
            car_out.loc[time, formstep[vehicle]] += 1

        formstep = nowstep.copy()
        nowstep.clear()

    car_in_file = os.path.join(fold, prefix+'CarIn.csv')
    car_out_file = os.path.join(fold, prefix+'CarOut.csv')
    number_file = os.path.join(fold, prefix+'Number.csv')

    car_in.drop("label", axis=1, inplace=True)
    car_out.drop("label", axis=1, inplace=True)
    number.drop("label", axis=1, inplace=True)

    car_in.to_csv(car_in_file)
    car_out.to_csv(car_out_file)
    number.to_csv(number_file)
    print("carIn information have been saved as ", car_in_file)
    print("carOut information have been saved as ", car_out_file)
    print("number information have been saved as ", number_file)

    return True

if __name__ == "__main__":
    data_record(merge_fcd_path)