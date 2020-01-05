import os
import pickle
import numpy as np 
import pandas as pd 
import torch
import math
import random
from torch.autograd import Variable
from torch.utils.data import Dataset

import time as ttt
import argparse

import conf 

seg_topology = [1, 2]
inter_topology = {"major":2, "minor":4, "end":7, "inter":6}
co_topology =[{"major":2, "minor":4, "end":7, "inter":6}, {"major":3, "minor":1, "end":8, "inter":5}]
seg = [1, 2, 3, 4, 7, 8]
inter_node = [5, 6]
fold = conf.midDataPath

class traffic_data(Dataset):

    '''每次读取进来一个场景下的文件，通过输入的参数指定是读取路段的数据样本还是读取路口数据样本
    '''

    def __init__(self, args, data_prefix='default', fold=fold, mod='seg', topology=seg_topology):

        super().__init__()

        self.sim_step = args["sim_step"]
        self.delta_T = args["delta_T"]
        self.temporal_length = args["temporal_length"]
        self.seqPredict = args["t_predict"]

        self.mod = mod
        self.topology = topology

        self.car_in_file = os.path.join(fold, data_prefix+'CarIn.csv')
        self.car_out_file = os.path.join(fold, data_prefix+'CarOut.csv')
        self.number_file = os.path.join(fold, data_prefix+'Number.csv')

        self.all_car_in = pd.read_csv(self.car_in_file, index_col=0).dropna(axis=0)
        self.all_car_out = pd.read_csv(self.car_out_file, index_col=0).dropna(axis=0)
        self.all_number = pd.read_csv(self.number_file, index_col=0)
        self.time_number = len(self.all_car_in.index) - (self.temporal_length + 1)* self.delta_T / self.sim_step

        self.car_in = pd.DataFrame()
        self.car_out = pd.DataFrame()
        self.number = pd.DataFrame()
        self.edge_number = 0
        self.bucketlist = []
        self.use_speed = args["use_speed"]

        self.mean = -1
        self.std = -1

        if self.use_speed:
            self.speed_file = os.path.join(fold, data_prefix+'Speed.csv')
            self.speed_in_file = os.path.join(fold, data_prefix+'SpeedIn.csv')
            self.speed_out_file = os.path.join(fold, data_prefix+'SpeedOut.csv')

            self.all_speed = pd.read_csv(self.speed_file, index_col=0).dropna(axis=0)
            self.all_speed_in = pd.read_csv(self.speed_in_file, index_col=0).dropna(axis=0)
            self.all_speed_out = pd.read_csv(self.speed_out_file, index_col=0).dropna(axis=0)

            self.speed = pd.DataFrame()
            self.speed_in = pd.DataFrame()
            self.speed_out = pd.DataFrame()

        self.filter_data()

    def filter_data(self):

        if self.mod == 'seg':
            if not isinstance(self.topology, list):
                print("wrong type of topology for mod segment!")
                raise RuntimeError("TOPOLOGY TYPE ERROR")
            self.edge_number = len(self.topology)
            bucketlist = [item for item in self.all_car_in.columns if int(int(item)/100) in self.topology]
            self.car_in = self.all_car_in[bucketlist].values
            self.car_out = self.all_car_out[bucketlist].values
            self.number = self.all_number[bucketlist].values
            if self.use_speed:
                self.speed = self.all_speed[bucketlist].values
                self.speed_in = self.all_speed_in[bucketlist].values
                self.speed_out = self.all_speed_out[bucketlist].values

        elif self.mod == 'inter':
            if not isinstance(self.topology, dict):
                print("wrong type of topology for mod intersection!")
                raise RuntimeError("TOPOLOGY TYPE ERROR")

            major_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['major']]
            minor_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['minor']]
            end_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['end']]
            inter_bucket = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['inter']]
            
            major_buckets.sort()
            minor_buckets.sort()
            end_buckets.sort()

            bucketlist = major_buckets[-2:] + minor_buckets[-2:] + inter_bucket + end_buckets[:2]
            self.bucketlist = bucketlist
            self.car_in = self.all_car_in[bucketlist].values
            self.car_out = self.all_car_out[bucketlist].values
            self.number = self.all_number[bucketlist].values
        elif self.mod == 'cooperate':

            major_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['major']]
            minor_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['minor']]
            end_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['end']]
            inter_bucket = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['inter']]
            
            major_buckets.sort()
            minor_buckets.sort()
            end_buckets.sort()

            bucketlist = major_buckets + minor_buckets + end_buckets + inter_bucket
            self.car_in = self.all_car_in[bucketlist].values
            self.car_out = self.all_car_out[bucketlist].values
            self.number = self.all_number[bucketlist].values
            self.bucket_number = [len(major_buckets)-2, len(major_buckets)-1,
                                len(major_buckets)+len(minor_buckets)-2, len(major_buckets)+len(minor_buckets)-1,
                                len(major_buckets)+len(minor_buckets), len(major_buckets)+len(minor_buckets)+1,
                                len(major_buckets)+len(minor_buckets)+len(end_buckets)]

        else:
            print("wrong mod to generate data !")
            raise RuntimeError('MOD ERROR')

    def reload(self, data_prefix=None, fold=fold, mod='seg', topology=seg_topology):

        self.mod = mod
        self.topology = topology

        if data_prefix is not None:
            self.car_in_file = os.path.join(fold, data_prefix+'CarIn.csv')
            self.car_out_file = os.path.join(fold, data_prefix+'CarOut.csv')
            self.number_file = os.path.join(fold, data_prefix+'Number.csv')

            self.all_car_in = pd.read_csv(self.car_in_file, index_col=0).dropna(axis=0)
            self.all_car_out = pd.read_csv(self.car_out_file, index_col=0).dropna(axis=0)
            self.all_number = pd.read_csv(self.number_file, index_col=0)

            if self.use_speed:
                self.speed_file = os.path.join(fold, data_prefix+'Speed.csv')
                self.speed_in_file = os.path.join(fold, data_prefix+'SpeedIn.csv')
                self.speed_out_file = os.path.join(fold, data_prefix+'SpeedOut.csv')

                self.all_speed = pd.read_csv(self.speed_file, index_col=0).dropna(axis=0)
                self.all_speed_in = pd.read_csv(self.speed_in_file, index_col=0).dropna(axis=0)
                self.all_speed_out = pd.read_csv(self.speed_out_file, index_col=0).dropna(axis=0)

        self.filter_data()

    def __getitem__(self, index):
        # seg mod下返回的数据是(spatial, temporal, feature)
        # inter mod下返回的数据是(temporal, n_unit, feature)
        
        if self.mod == "seg":
            
            time = int(index / self.edge_number)
            
            timelist = [int(i*self.delta_T/self.sim_step + time) for i in range(self.temporal_length+1)]

            In = self.car_in[timelist][:, :, np.newaxis]
            out = self.car_out[timelist][:, :, np.newaxis]
            number = self.number[timelist][:, :, np.newaxis]
            if self.use_speed:
                Speed = self.speed[timelist][:, :, np.newaxis]
                Speed_in = self.speed_in[timelist][:, :, np.newaxis]
                Speed_out = self.speed_out[timelist][:, :, np.newaxis]
            
                data = np.concatenate((out, In, number, Speed, Speed_in, Speed_out), axis=2)
            else:
                data = np.concatenate((out, In, number), axis=2)

            self.mean = np.mean(data, axis=(0, 1))
            data = data - self.mean
            self.std = np.std(data, axis=(0, 1))
            data = data / self.std

            data = torch.Tensor(data)

            return data
            
        elif self.mod == "inter":
            
            raise NotImplementedError
            time = index 
            timelist = [int(i*self.delta_T/self.sim_step + time) for i in range(self.temporal_length+1)]

            In = np.array(self.car_in[timelist])[:, :, np.newaxis]
            out = np.array(self.car_out[timelist])[:, :, np.newaxis]
            number = np.array(self.number[timelist])[:, :, np.newaxis]
            
            data = torch.Tensor(np.concatenate((out, In, number), axis=2)).float()

            return data

        elif self.mod == "cooperate":

            raise NotImplementedError
            time = index
            time_list = [int(i*self.delta_T/self.sim_step) + time for i in range(self.temporal_length+1)]

            #t2 = ttt.time()
            In = self.car_in[time_list]
            out = self.car_out[time_list]
            number = self.number[time_list]
            Speed = self.speed[time_list]
            Speed_in = self.speed_in[time_list]
            Speed_out = self.speed_out[time_list]
   
            #t3 = ttt.time()
            In = In[:, :, np.newaxis]
            out = out[:, :, np.newaxis]
            number = number[:, :, np.newaxis]
            Speed = Speed[:, :, np.newaxis]
            Speed_in = Speed_in[:, :, np.newaxis]
            Speed_out = Speed_out[:, :, np.newaxis]

            #t4 = ttt.time()
            data = np.concatenate((out, In, number, Speed, Speed_in, Speed_out), axis=2)
            #t5 = ttt.time()
            data = torch.Tensor(data).float()

            #t6 = ttt.time()
            return data

        else:
            print("wrong mod to generate data !")
            raise RuntimeError('MOD ERROR')

    def recover(self, data):

        data = data * self.std
        data = data + self.mean

        return data

    def __len__(self):

        if self.mod == 'seg':
            length = self.edge_number * self.time_number
        elif self.mod == 'inter' :
            length = self.time_number
        elif self.mod == 'cooperate':
            length = self.time_number
        else:
            print("wrong mod to generate data !")
            raise RuntimeError('MOD ERROR')

        return int(length)


if __name__ == "__main__":
    
    args = {}
    args["t_predict"] = 4
    args["temporal_length"] = 8
    args["sim_step"] = 0.1
    args["delta_T"] = 10
    args["use_speed"] = False

    dataset = traffic_data(mod='seg', topology=[1], args=args)
    t = ttt.time()
    a = dataset[0]
    a = dataset[0]
    a = dataset[0]
    a = dataset[0]
    a = dataset[0]
    a = dataset[0]
    a = dataset[0]
    a = dataset[0]
    a = dataset[0]
    t2 = ttt.time()
    print(len(dataset))