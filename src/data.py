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

        self.filter_data()

    def filter_data(self):

        if self.mod == 'seg':
            if not isinstance(self.topology, list):
                print("wrong type of topology for mod segment!")
                raise RuntimeError("TOPOLOGY TYPE ERROR")
            self.edge_number = len(self.topology)
            bucketlist = [item for item in self.all_car_in.columns if int(int(item)/100) in self.topology]
            self.car_in = self.all_car_in[bucketlist]
            self.car_out = self.all_car_out[bucketlist]
            self.number = self.all_number[bucketlist]
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
            self.car_in = self.all_car_in[bucketlist]
            self.car_out = self.all_car_out[bucketlist]
            self.number = self.all_number[bucketlist]
        elif self.mod == 'cooperate':

            major_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['major']]
            minor_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['minor']]
            end_buckets = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['end']]
            inter_bucket = [item for item in self.all_car_in.columns if int(int(item)/100)==self.topology['inter']]
            
            major_buckets.sort()
            minor_buckets.sort()
            end_buckets.sort()

            bucketlist = major_buckets + minor_buckets + end_buckets + inter_bucket
            self.car_in = self.all_car_in[bucketlist]
            self.car_out = self.all_car_out[bucketlist]
            self.number = self.all_number[bucketlist]
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

        self.filter_data()

    def __getitem__(self, index):
        # seg mod下返回的数据是(spatial, temporal, feature)
        # inter mod下返回的数据是(temporal, n_unit, feature)
        
        if self.mod == "seg":
            
            edge = self.topology[index % self.edge_number]
            time = int(index / self.edge_number) * self.sim_step
            
            timelist = [round(i*self.delta_T+time, 1) for i in range(self.temporal_length+1)]

            In = np.array(self.car_in.loc[timelist].T)[:, :, np.newaxis]
            out = np.array(self.car_out.loc[timelist].T)[:, :, np.newaxis]
            number = np.array(self.number.loc[timelist].T)[:, :, np.newaxis]
            
            data = torch.Tensor(np.concatenate((out, In, number), axis=2)).float()

            return data
            
        elif self.mod == "inter":
        
            time = index * self.sim_step
            timelist = [round(i*self.delta_T+time, 1) for i in range(self.temporal_length+1)]

            In = np.array(self.car_in.loc[timelist])[:, :, np.newaxis]
            out = np.array(self.car_out.loc[timelist])[:, :, np.newaxis]
            number = np.array(self.number.loc[timelist])[:, :, np.newaxis]
            
            data = torch.Tensor(np.concatenate((out, In, number), axis=2)).float()

            return data

        elif self.mod == "cooperate":
            
            seg_data = []
            time = round(index * self.sim_step)
            time_list = [round(i*self.delta_T+time, 1)for i in range(self.temporal_length+1)]

            In = np.array(self.car_in.loc[time_list])[:, :, np.newaxis]
            out = np.array(self.car_out.loc[time_list])[:, :, np.newaxis]
            number = np.array(self.number.loc[time_list])[:, :, np.newaxis]

            data = torch.Tensor(np.concatenate((out, In, number), axis=2)).float()

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


if __name__ == "__main__":
    
    args = {}
    args["t_predict"] = 4
    args["temporal_length"] = 8
    args["sim_step"] = 0.1
    args["delta_T"] = 10

    dataset = traffic_data(mod='cooperate', topology=co_topology[0], args=args)
    a = dataset[0]
    print(len(dataset))