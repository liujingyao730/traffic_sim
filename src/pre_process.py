import numpy as np
import xml.etree.cElementTree as etree
import pandas as pd
import os 
import time as ttt

import conf

#merge_fcd_path = os.path.join(conf.fcdOutputPath, 'merge', 'big_test.xml')
merge_fcd_path = os.path.join(conf.fcdOutputPath, 'merge', "fcd.xml")
#路网的结构是固定的
length = {1:4950, 2:4950, 3:4950, 4:4950, 7:4950, 8:4950}
lane_number = {1:1, 2:2, 3:2, 4:1, 5:2, 6:3, 8:1, 7:2}
connections = {1:1, 2:2, 3:1, 4:1, 5:0, 6:0, 8:2, 7:2}
special_edge = {5:8, 6:7}
lane_edge = {'1_0':1, '2_0':2, '2_1':2, '3_0':3, '3_1':3, '4_0':4, '7_0':7, '7_1':7,
            '8_0':8, ':gneJ1_0_0':5, ':gneJ1_4_0':5, ':gneJ2_6_0':6, ':gneJ2_10_0':6,
            ':gneJ2_10_1':6}

def bucketid(pos, length):
    
    bucketNumber = int(length / 50)

    if pos >= bucketNumber * 50 or pos < 0:
        print("there must be something wrong !!!!!!! pos ", pos)
        return False

    for i in range(bucketNumber):
        if pos < (i+1)*50:
            return i+1

    return bucketNumber + 1

def data_record(file, lane_edge=lane_edge, length=length, prefix='default', 
                special_edge=special_edge, fold=conf.midDataPath):

    root = etree.iterparse(file, events=["start"])

    formstep = {}
    nowstep = {}
    now_speed = {}
    before_speed = {}

    car_in = pd.DataFrame(columns=["Nan"])
    car_out = pd.DataFrame(columns=["Nan"])
    number = pd.DataFrame(columns=["Nan"])
    speed = pd.DataFrame(columns=["Nan"])
    speed_out = pd.DataFrame(columns=["Nan"])
    speed_in = pd.DataFrame(columns=["Nan"])

    time = -1

    for event, elem in root:
        if elem.tag == "timestep":

            #处理之前的数据
            for vehicle in nowstep.keys():
                if vehicle not in formstep.keys():
                    car_in.loc[time, nowstep[vehicle]] += 1
                    speed_in.loc[time, nowstep[vehicle]] += now_speed[vehicle]
                else:
                    if formstep[vehicle] != nowstep[vehicle]:
                        car_in.loc[time, nowstep[vehicle]] += 1
                        speed_in.loc[time, nowstep[vehicle]] += now_speed[vehicle]
                        car_out.loc[time, formstep[vehicle]] += 1
                        speed_out.loc[time, formstep[vehicle]] += before_speed[vehicle]
                    formstep.pop(vehicle)

            for vehicle in formstep.keys():
                car_out.loc[time, formstep[vehicle]] += 1
                speed_out.loc[time, formstep[vehicle]] += before_speed[vehicle]
            
            if time >= 0:
                speed.loc[time] /= number.loc[time].replace(0, 1)
                speed_in.loc[time] /= car_in.loc[time].replace(0, 1)
                speed_out.loc[time] /= car_out.loc[time].replace(0, 1)

            formstep = nowstep.copy()
            before_speed = now_speed.copy()
            now_speed.clear()
            nowstep.clear()

            #创建现在的数据
            time = float(elem.attrib["time"])
            car_in.loc[time] = 0
            car_out.loc[time] = 0
            number.loc[time] = 0
            speed.loc[time] = 0
            speed_in.loc[time] = 0
            speed_out.loc[time] = 0
        elif elem.tag == "vehicle":
            vehicle_id = elem.attrib["id"]
            lane = elem.attrib['lane']
            edge = lane_edge[lane]
            vehicle_speed = float(elem.attrib['speed'])

            if edge not in special_edge.keys():
                pos = float(elem.attrib["pos"])
                bucket = bucketid(pos, length[edge]) + edge*100
            else:
                edge = special_edge[edge]
                bucket = edge*100 + 1

            nowstep[vehicle_id] = bucket
            if bucket not in number.columns:
                speed[bucket] = 0
                speed_in[bucket] = 0
                speed_out[bucket] = 0
                number[bucket] = 0
                car_in[bucket] = 0
                car_out[bucket] = 0
            now_speed[vehicle_id] = vehicle_speed
            speed.loc[time, bucket] += vehicle_speed
            number.loc[time, bucket] += 1
        
        elem.clear()

    car_in_file = os.path.join(fold, prefix+'CarIn.csv')
    car_out_file = os.path.join(fold, prefix+'CarOut.csv')
    number_file = os.path.join(fold, prefix+'Number.csv')
    speed_file = os.path.join(fold, prefix+'Speed.csv')
    speed_in_file = os.path.join(fold, prefix+'SpeedIn.csv')
    speed_out_file = os.path.join(fold, prefix+'SpeedOut.csv')

    car_in.drop("Nan", axis=1, inplace=True)
    car_out.drop("Nan", axis=1, inplace=True)
    number.drop("Nan", axis=1, inplace=True)
    speed.drop("Nan", axis=1, inplace=True)
    speed_out.drop("Nan", axis=1, inplace=True)
    speed_in.drop("Nan", axis=1, inplace=True)

    car_in.to_csv(car_in_file)
    car_out.to_csv(car_out_file)
    number.to_csv(number_file)
    speed.to_csv(speed_file)
    speed_in.to_csv(speed_in_file)
    speed_out.to_csv(speed_out_file)
    
    print("finish process fcd file "+file)

    return True


def reset_data(prefix, fold=conf.midDataPath, deltaT=conf.args["deltaT"], 
                sim_step=conf.args["trainSimStep"]):

    car_in_file = os.path.join(fold, prefix+'CarIn.csv')
    car_out_file = os.path.join(fold, prefix+'CarOut.csv')
    speed_in_file = os.path.join(fold, prefix+'SpeedIn.csv')
    speed_out_file = os.path.join(fold, prefix+'SpeedOut.csv')

    car_in = pd.read_csv(car_in_file, index_col=0)
    car_out = pd.read_csv(car_out_file, index_col=0)
    speed_in = pd.read_csv(speed_in_file, index_col=0)
    speed_out = pd.read_csv(speed_out_file, index_col=0)
    max_time = len(car_in.index)
    max_time -= int(deltaT / sim_step)

    reset_car_in = pd.DataFrame(columns=car_in.columns, index=car_in.index)
    reset_car_out = pd.DataFrame(columns=car_out.columns, index=car_in.index)
    reset_speed_in = pd.DataFrame(columns=speed_in.columns, index=speed_in.index)
    reset_speed_out = pd.DataFrame(columns=speed_out.columns, index=speed_out.index)

    reset_car_in.loc[0] = 0
    reset_car_out.loc[0] = 0
    reset_speed_in.loc[0] = 0
    reset_speed_out.loc[0] = 0
    t = 0

    for i in range(int(deltaT / sim_step)):
        reset_car_in.loc[0] += car_in.loc[t]
        reset_car_out.loc[0] += car_out.loc[t]
        reset_speed_in.loc[0] += car_in.loc[t] * speed_in.loc[t]
        reset_speed_out.loc[0] += car_out.loc[t] * speed_out.loc[t]
        t = round(t+sim_step, 1)
    
    for i in range(1, max_time+1):
        time = round(i*sim_step, 1)
        minus_time = round(time-sim_step, 1)
        plus_time = round(time+deltaT-sim_step, 1)
        reset_car_in.loc[time] = reset_car_in.loc[minus_time] - car_in.loc[minus_time] + car_in.loc[plus_time]
        reset_car_out.loc[time] = reset_car_out.loc[minus_time] - car_out.loc[minus_time] + car_out.loc[plus_time]
        reset_speed_in.loc[time] = reset_speed_in.loc[minus_time] - car_in.loc[minus_time]*speed_in.loc[minus_time] + car_in.loc[plus_time]*speed_in.loc[plus_time]
        reset_speed_out.loc[time] = reset_speed_out.loc[minus_time] - car_out.loc[minus_time]*speed_out.loc[minus_time] + car_out.loc[plus_time]*speed_out.loc[plus_time]

    reset_speed_in /= reset_car_in.replace(0, 1)
    reset_speed_out /= reset_car_out.replace(0, 1)

    reset_speed_in = reset_speed_in.round(decimals=2)
    reset_speed_out = reset_speed_in.round(decimals=2)

    reset_car_in.to_csv(car_in_file)
    reset_car_out.to_csv(car_out_file)
    reset_speed_in.to_csv(speed_in_file)
    reset_speed_out.to_csv(speed_out_file)
    
    print(prefix + "car in and out file have been reset")


def add_lane(prefix, fold=conf.midDataPath):

    car_in_file = os.path.join(fold, prefix+'CarIn.csv')
    car_in = pd.read_csv(car_in_file, index_col=0)
    lanes = pd.DataFrame(index=car_in.index, columns=car_in.columns)

    for block in car_in.columns:
        edge = int(int(block) / 100)
        lanes[block] = lane_number[edge]

    lanes_file = os.path.join(fold, prefix+'LaneNumber.csv')
    lanes.to_csv(lanes_file)

    print(prefix + " lane number file have been saved")

def add_connection(prefix, fold=conf.midDataPath):

    car_in_file = os.path.join(fold, prefix+'CarIn.csv')
    car_in = pd.read_csv(car_in_file, index_col=0)
    connection = pd.DataFrame(index=car_in.index, columns=car_in.columns)

    for block in car_in.columns:
        edge = int(int(block) / 100)
        connection[block] = connections[edge]

    connection_file = os.path.join(fold, prefix+'Connection.csv')
    connection.to_csv(connection_file)

    print(prefix +" connection number file have been saved")


if __name__ == "__main__":
    length[1] = 1400
    length[2] = 1400
    length[3] = 1000
    length[4] = 1000
    length[7] = 1000
    length[8] = 1000
    t = ttt.time()
    data_record(merge_fcd_path)
    #reset_data('default')
    #add_lane("default")
    #add_connection("default")
    t2 = ttt.time()
    print(t2 - t)
    