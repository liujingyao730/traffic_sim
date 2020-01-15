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

            self.mean = np.array([np.mean(self.car_out), np.mean(self.car_in), np.mean(self.number)])
            self.car_out = self.car_out - self.mean[0]
            self.car_in = self.car_in - self.mean[1]
            self.number = self.number - self.mean[2]

            self.std = np.array([np.std(self.car_out), np.std(self.car_in), np.std(self.number)])
            self.car_out = self.car_out / self.std[0]
            self.car_in = self.car_in / self.std[1]
            self.number = self.number /self.std[2]
            
            if self.use_speed:
                raise NotImplementedError
                self.speed = self.all_speed[bucketlist].values
                self.speed_in = self.all_speed_in[bucketlist].values
                self.speed_out = self.all_speed_out[bucketlist].values

        elif self.mod == 'inter':
            raise NotImplementedError
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
            raise NotImplementedError
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

            #data = torch.Tensor(data)

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


class two_type_data(Dataset):

    '''每次读取进来一个场景下的文件，通过输入的参数指定是读取路段的数据样本还是读取路口数据样本
    '''

    def __init__(self, args, data_prefix="defualt", fold=fold, topology=seg_topology):

        super().__init__()

        self.sim_step = args["sim_step"]
        self.delta_T = args["delta_T"]
        self.temporal_length = args["temporal_length"]
        self.seqPredict = args["t_predict"]

        self.topology = topology

        self.pv_car_in_file = os.path.join(fold, data_prefix+'PVCarIn.csv')
        self.pv_car_out_file = os.path.join(fold, data_prefix+'PVCarOut.csv')
        self.pv_number_file = os.path.join(fold, data_prefix+'PVNumber.csv')
        self.hov_car_in_file = os.path.join(fold, data_prefix+'HOVCarIn.csv')
        self.hov_car_out_file = os.path.join(fold, data_prefix+'HOVCarOut.csv')
        self.hov_number_file = os.path.join(fold, data_prefix+'HOVNumber.csv')

        self.pv_all_car_in = pd.read_csv(self.pv_car_in_file, index_col=0).dropna(axis=0)
        self.pv_all_car_out = pd.read_csv(self.pv_car_out_file, index_col=0).dropna(axis=0)
        self.pv_all_number = pd.read_csv(self.pv_number_file, index_col=0)
        self.hov_all_car_in = pd.read_csv(self.hov_car_in_file, index_col=0).dropna(axis=0)
        self.hov_all_car_out = pd.read_csv(self.hov_car_out_file, index_col=0).dropna(axis=0)
        self.hov_all_number = pd.read_csv(self.hov_number_file, index_col=0)
        self.index = min(len(self.pv_all_car_in.index), len(self.hov_all_car_in.index))
        self.time_number = int(self.index - (self.temporal_length + 1)* self.delta_T / self.sim_step)

        self.car_in = pd.DataFrame()
        self.car_out = pd.DataFrame()
        self.number = pd.DataFrame()
        self.car_in = pd.DataFrame()
        self.car_out = pd.DataFrame()
        self.number = pd.DataFrame()
        self.bucketlist = []
        
        self.mean = -1
        self.std = -1

        self.filter_data()

    def filter_data(self):

        
        bucketlist = [item for item in self.pv_all_car_in.columns if int(int(item)/100) == self.topology]
            
        self.pv_car_in = self.pv_all_car_in[bucketlist].values
        self.pv_car_out = self.pv_all_car_out[bucketlist].values
        self.pv_number = self.pv_all_number[bucketlist].values
        self.hov_car_in = self.hov_all_car_in[bucketlist].values
        self.hov_car_out = self.hov_all_car_out[bucketlist].values
        self.hov_number = self.hov_all_number[bucketlist].values

        self.mean = np.array([np.mean(self.pv_car_out), np.mean(self.pv_car_in), np.mean(self.pv_number), 
                            np.mean(self.hov_car_out), np.mean(self.hov_car_in), np.mean(self.hov_number)])
        self.pv_car_out = self.pv_car_out - self.mean[0]
        self.pv_car_in = self.pv_car_in - self.mean[1]
        self.pv_number = self.pv_number - self.mean[2]
        self.hov_car_out = self.hov_car_out - self.mean[3]
        self.hov_car_in = self.hov_car_in - self.mean[4]
        self.hov_number = self.hov_number - self.mean[5]

        self.std = np.array([np.std(self.pv_car_out), np.std(self.pv_car_in), np.std(self.pv_number),
                            np.std(self.hov_car_out), np.std(self.hov_car_in), np.std(self.hov_number)])
        self.pv_car_out = self.pv_car_out / self.std[0]
        self.pv_car_in = self.pv_car_in / self.std[1]
        self.pv_number = self.pv_number /self.std[2]
        self.hov_car_out = self.hov_car_out / self.std[3]
        self.hov_car_in = self.hov_car_in / self.std[4]
        self.hov_number = self.hov_number /self.std[5]


    def reload(self, data_prefix=None, fold=fold, topology=seg_topology):

        self.topology = topology

        if data_prefix is not None:
            self.pv_car_in_file = os.path.join(fold, data_prefix[0]+'CarIn.csv')
            self.pv_car_out_file = os.path.join(fold, data_prefix[0]+'CarOut.csv')
            self.pv_number_file = os.path.join(fold, data_prefix[0]+'Number.csv')
            self.hov_car_in_file = os.path.join(fold, data_prefix[1]+'CarIn.csv')
            self.hov_car_out_file = os.path.join(fold, data_prefix[1]+'CarOut.csv')
            self.hov_number_file = os.path.join(fold, data_prefix[1]+'Number.csv')

            self.pv_all_car_in = pd.read_csv(self.pv_car_in_file, index_col=0).dropna(axis=0)
            self.pv_all_car_out = pd.read_csv(self.pv_car_out_file, index_col=0).dropna(axis=0)
            self.pv_all_number = pd.read_csv(self.pv_number_file, index_col=0)
            self.hov_all_car_in = pd.read_csv(self.hov_car_in_file, index_col=0).dropna(axis=0)
            self.hov_all_car_out = pd.read_csv(self.hov_car_out_file, index_col=0).dropna(axis=0)
            self.hov_all_number = pd.read_csv(self.hov_number_file, index_col=0)
            self.time_number = len(self.pv_all_car_in.index) - (self.temporal_length + 1)* self.delta_T / self.sim_step

        self.filter_data()

    def __getitem__(self, index):
            
        time = index
            
        timelist = [int(i*self.delta_T/self.sim_step + time) for i in range(self.temporal_length+1)]

        pv_In = self.pv_car_in[timelist][:, :, np.newaxis]
        pv_out = self.pv_car_out[timelist][:, :, np.newaxis]
        pv_number = self.pv_number[timelist][:, :, np.newaxis]
        hov_In = self.hov_car_in[timelist][:, :, np.newaxis]
        hov_out = self.hov_car_out[timelist][:, :, np.newaxis]
        hov_number = self.hov_number[timelist][:, :, np.newaxis]
       
        data = np.concatenate((pv_out, pv_In, pv_number, hov_out, hov_In, hov_number), axis=2)

        return data

    def __len__(self):

        return self.time_number

if __name__ == "__main__":
    
    args = {}
    args["t_predict"] = 4
    args["temporal_length"] = 8
    args["sim_step"] = 0.1
    args["delta_T"] = 10
    args["use_speed"] = False

    dataset = two_type_data(args, data_prefix=["basePV", "baseHOV"], topology=1)
    a = dataset[37889]
    print(a.shape)
    print(len(dataset))