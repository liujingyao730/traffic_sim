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

def generate_lane_edge(edges, lane_number):

    lane_edge = {}
    length = {}
    for i in range(len(edges)):
        edge = edges[i]
        number = lane_number[i]
        length[i] = 4950
        for j in range(number):
            lane_edge[edge+'_'+str(j)] = i+1

    return lane_edge, length

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

    car_in = pd.DataFrame(columns=["Nan"])
    car_out = pd.DataFrame(columns=["Nan"])
    number = pd.DataFrame(columns=["Nan"])

    time = -1

    for event, elem in root:
        if elem.tag == "timestep":

            #处理之前的数据
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

            #创建现在的数据
            time = float(elem.attrib["time"])
            if time % 10 == 0:
                print(time)
            car_in.loc[time] = 0
            car_out.loc[time] = 0
            number.loc[time] = 0
        elif elem.tag == "vehicle":
            vehicle_id = elem.attrib["id"]
            lane = elem.attrib['lane']
            edge = lane_edge[lane]

            if edge not in special_edge.keys():
                pos = float(elem.attrib["pos"])
                bucket = bucketid(pos, length[edge]) + edge*100
            else:
                edge = special_edge[edge]
                bucket = edge*100 + 1

            nowstep[vehicle_id] = bucket
            if bucket not in number.columns:
                number[bucket] = 0
                car_in[bucket] = 0
                car_out[bucket] = 0

            number.loc[time, bucket] += 1
        
        elem.clear()

    car_in_file = os.path.join(fold, prefix+'CarIn.csv')
    car_out_file = os.path.join(fold, prefix+'CarOut.csv')
    number_file = os.path.join(fold, prefix+'Number.csv')

    car_in.drop("Nan", axis=1, inplace=True)
    car_out.drop("Nan", axis=1, inplace=True)
    number.drop("Nan", axis=1, inplace=True)

    car_in.to_csv(car_in_file)
    car_out.to_csv(car_out_file)
    number.to_csv(number_file)
    
    print("finish process fcd file "+file)

    return True

def two_type_data_record(file, lane_edge=lane_edge, length=length, prefix='default', 
                special_edge=special_edge, fold=conf.midDataPath):

    root = etree.iterparse(file, events=["start"])

    pv_formstep = {}
    pv_nowstep = {}
    hov_formstep = {}
    hov_nowstep = {}

    pv_car_in = pd.DataFrame(columns=["Nan"])
    pv_car_out = pd.DataFrame(columns=["Nan"])
    pv_number = pd.DataFrame(columns=["Nan"])
    hov_car_in = pd.DataFrame(columns=["Nan"])
    hov_car_out = pd.DataFrame(columns=["Nan"])
    hov_number = pd.DataFrame(columns=["Nan"])

    time = -1

    for event, elem in root:
        if elem.tag == "timestep":

            #处理pv之前的数据
            for vehicle in pv_nowstep.keys():
                if vehicle not in pv_formstep.keys():
                    pv_car_in.loc[time, pv_nowstep[vehicle]] += 1
                else:
                    if pv_formstep[vehicle] != pv_nowstep[vehicle]:
                        pv_car_in.loc[time, pv_nowstep[vehicle]] += 1
                        pv_car_out.loc[time, pv_formstep[vehicle]] += 1
                    pv_formstep.pop(vehicle)

            for vehicle in pv_formstep.keys():
                pv_car_out.loc[time, pv_formstep[vehicle]] += 1    
            #处理hov之前的数据
            for vehicle in hov_nowstep.keys():
                if vehicle not in hov_formstep.keys():
                    hov_car_in.loc[time, hov_nowstep[vehicle]] += 1
                else:
                    if hov_formstep[vehicle] != hov_nowstep[vehicle]:
                        hov_car_in.loc[time, hov_nowstep[vehicle]] += 1
                        hov_car_out.loc[time, hov_formstep[vehicle]] += 1
                    hov_formstep.pop(vehicle)

            for vehicle in hov_formstep.keys():
                hov_car_out.loc[time, hov_formstep[vehicle]] += 1       

            pv_formstep = pv_nowstep.copy()
            pv_nowstep.clear()
            hov_formstep = hov_nowstep.copy()
            hov_nowstep.clear()

            #创建现在的数据
            time = float(elem.attrib["time"])
            if time % 10 == 0:
                print(time)
            pv_car_in.loc[time] = 0
            pv_car_out.loc[time] = 0
            pv_number.loc[time] = 0
            hov_car_in.loc[time] = 0
            hov_car_out.loc[time] = 0
            hov_number.loc[time] = 0
        elif elem.tag == "vehicle":
            vehicle_id = elem.attrib["id"]
            lane = elem.attrib['lane']
            edge = lane_edge[lane]
            vehicle_speed = float(elem.attrib['speed'])
            vehicle_type = elem.attrib['type']

            if edge not in special_edge:
                pos = float(elem.attrib["pos"])
                bucket = bucketid(pos, length[edge]) + edge*100
            else:
                bucket = edge*100 + 1
            
            if vehicle_type == "PV": 

                pv_nowstep[vehicle_id] = bucket
                if bucket not in pv_number.columns:
                    pv_number[bucket] = 0
                    pv_car_in[bucket] = 0
                    pv_car_out[bucket] = 0
                pv_number.loc[time, bucket] += 1
            
            elif vehicle_type == "HOV":

                hov_nowstep[vehicle_id] = bucket
                if bucket not in hov_number.columns:
                    hov_car_in[bucket] = 0
                    hov_car_out[bucket] = 0
                    hov_number[bucket] = 0
                hov_number.loc[time, bucket] += 1
        
        elem.clear()

    pv_car_in_file = os.path.join(fold, prefix+'PVCarIn.csv')
    pv_car_out_file = os.path.join(fold, prefix+'PVCarOut.csv')
    pv_number_file = os.path.join(fold, prefix+'PVNumber.csv')
    hov_car_in_file = os.path.join(fold, prefix+'PVCarIn.csv')
    hov_car_out_file = os.path.join(fold, prefix+'HOVCarOut.csv')
    hov_number_file = os.path.join(fold, prefix+'HOVNumber.csv')

    pv_car_in.drop("Nan", axis=1, inplace=True)
    pv_car_out.drop("Nan", axis=1, inplace=True)
    pv_number.drop("Nan", axis=1, inplace=True)
    hov_car_in.drop("Nan", axis=1, inplace=True)
    hov_car_out.drop("Nan", axis=1, inplace=True)
    hov_number.drop("Nan", axis=1, inplace=True)

    pv_car_in.to_csv(pv_car_in_file)
    pv_car_out.to_csv(pv_car_out_file)
    pv_number.to_csv(pv_number_file)
    hov_car_in.to_csv(hov_car_in_file)
    hov_car_out.to_csv(hov_car_out_file)
    hov_number.to_csv(hov_number_file)
    
    print("finish process fcd file "+file)

    return True

def reset_data(prefix, fold=conf.midDataPath, deltaT=conf.args["deltaT"], 
                sim_step=conf.args["trainSimStep"]):

    car_in_file = os.path.join(fold, prefix+'CarIn.csv')
    car_out_file = os.path.join(fold, prefix+'CarOut.csv')

    car_in = pd.read_csv(car_in_file, index_col=0)
    car_out = pd.read_csv(car_out_file, index_col=0)
    max_time = len(car_in.index)
    max_time -= int(deltaT / sim_step)

    reset_car_in = pd.DataFrame(columns=car_in.columns, index=car_in.index)
    reset_car_out = pd.DataFrame(columns=car_out.columns, index=car_in.index)

    reset_car_in.loc[0] = 0
    reset_car_out.loc[0] = 0
    t = 0

    for i in range(int(deltaT / sim_step)):
        reset_car_in.loc[0] += car_in.loc[t]
        reset_car_out.loc[0] += car_out.loc[t]
        t = round(t+sim_step, 1)
    
    for i in range(1, max_time+1):
        time = round(i*sim_step, 1)
        minus_time = round(time-sim_step, 1)
        plus_time = round(time+deltaT-sim_step, 1)
        reset_car_in.loc[time] = reset_car_in.loc[minus_time] - car_in.loc[minus_time] + car_in.loc[plus_time]
        reset_car_out.loc[time] = reset_car_out.loc[minus_time] - car_out.loc[minus_time] + car_out.loc[plus_time]

    reset_car_in.to_csv(car_in_file)
    reset_car_out.to_csv(car_out_file)
    
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

    edges = ['1', '2', '3', '4', '5', '6', 
            '1_t', '2_t', '3_t', '4_t', '5_t', '6_t',
            ':1_j_7', ':1_j_21', ':2_j_7', ':2_j_21', ':3_j_7', ':3_j_21',
            ':4_j_7', ':4_j_21', ':5_j_7', ':5_j_21', ':6_j_0', ':6_j_6']
    lane_number = [6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6]

    special_edge_real = list(range(13, 25))
    lane_edge_real, length_real = generate_lane_edge(edges, lane_number)
    file = os.path.join(conf.fcdOutputPath, 'two_type.xml')
    t = ttt.time()
    two_type_data_record(file, lane_edge=lane_edge_real, length=length_real, 
                        prefix="two_type", special_edge=special_edge_real)
    t2 = ttt.time()
    print(t2 - t)
    