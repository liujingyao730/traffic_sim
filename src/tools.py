import os
import pickle
import numpy as np 
import pandas as pd 
import torch
import math
import random
from torch.autograd import Variable
from torch.utils.data import Dataset

import argparse

import conf 

topology = [1, 2]
mod = 'seg'
#topology = {"major":2, "minor":4, "end":7, "inter":6}
#mod = 'inter'
fold = conf.midDataPath

class data_generator(Dataset):

    '''每次读取进来一个场景下的文件，通过输入的参数指定是读取路段的数据样本还是读取路口数据样本
    '''

    def __init__(self, args, data_prefix='default', fold=fold, mod='seg', topology=topology):

        super().__init__()

        self.sim_step = args.sim_step
        self.delta_T = args.delta_T
        self.temporal_length = args.temporal_length
        self.seqPredict = args.t_predict

        self.mod = mod
        self.topology = topology

        self.car_in_file = os.path.join(fold, data_prefix+'CarIn.csv')
        self.car_out_file = os.path.join(fold, data_prefix+'CarOut.csv')
        self.number_file = os.path.join(fold, data_prefix+'Number.csv')

        self.all_car_in = pd.read_csv(self.car_in_file, index_col=0).dropna(axis=0)
        self.all_car_out = pd.read_csv(self.car_out_file, index_col=0).dropna(axis=0)
        self.all_number = pd.read_csv(self.number_file, index_col=0)

        self.car_in = pd.DataFrame()
        self.car_out = pd.DataFrame()
        self.number = pd.DataFrame()
        self.edge_number = 0

        self.filter_data()

    def filter_data(self):

        if mod == 'seg':
            if not isinstance(self.topology, list):
                print("wrong type of topology for mod segment!")
                raise RuntimeError("TOPOLOGY TYPE ERROR")
            self.edge_number = len(self.topology)
            bucketlist = [item for item in self.all_car_in.columns if int(int(item)/100) in self.topology]
            self.car_in = self.all_car_in[bucketlist]
            self.car_out = self.all_car_out[bucketlist]
            self.number = self.all_number[bucketlist]
        elif mod == 'inter':
            if not isinstance(self.topology, dict):
                print("wrong type of topology for mod intersection!")
                raise RuntimeError("TOPOLOGY TYPE ERROR")
            edgelist = self.topology.values()
            bucketlist = [item for item in self.all_car_in.columns if int(int(item)/100) in edgelist]
            self.car_in = self.all_car_in[bucketlist]
            self.car_out = self.all_car_out[bucketlist]
            self.number = self.all_number[bucketlist]
        else:
            print("wrong mod to generate data !")
            raise RuntimeError('MOD ERROR')

    def __getitem__(self, index):
        
        if mod == "seg":
            
            edge = self.topology[index % self.edge_number]
            time = int(index / self.edge_number) * self.sim_step
            
            bucketlist = [item for item in self.car_in.columns if int(int(item)/100)==edge]
            timelist = [i*self.delta_T+time for i in range(self.temporal_length+1)]
            
        elif mod == "inter":
        
            time = index * self.sim_step
            timelist = [i*self.delta_T+time for i in range(self.temporal_length+1)]

            major_buckets = [item for item in self.car_in.columns if int(int(item)/100)==self.topology['major']]
            minor_buckets = [item for item in self.car_in.columns if int(int(item)/100)==self.topology['minor']]
            end_buckets = [item for item in self.car_in.columns if int(int(item)/100)==self.topology['end']]
            inter_bucket = [item for item in self.car_in.columns if int(int(item)/100)==self.topology['inter']]
            
            major_buckets.sort()
            minor_buckets.sort()
            end_buckets.sort()

            bucketlist = major_buckets[-2:] + minor_buckets[-2:] + inter_bucket + end_buckets[:2]

        else:
            print("wrong mod to generate data !")
            raise RuntimeError('MOD ERROR')

        In = np.array(self.car_in.loc[timelist, bucketlist].T)[:, :, np.newaxis]
        out = np.array(self.car_out.loc[timelist, bucketlist].T)[:, :, np.newaxis]
        number = np.array(self.number.loc[timelist, bucketlist].T)[:, :, np.newaxis]
            
        inputs = torch.Tensor(np.concatenate((out[:, :-1, :], In[:, :-1, :], number[:, :-1, :]), axis=2)).float()
        outputs = torch.Tensor(np.concatenate((out[:, self.seqPredict+1:, :], In[:, self.seqPredict+1:, :], number[:, self.seqPredict+1:, :]), axis=2)).float()

        return [inputs, outputs]

    def __len__(self):

        if self.mod == 'seg':
            length = self.edge_number * (len(self.car_in.index) - self.temporal_length * self.delta_T * self.sim_step)
        elif self.mod == 'inter':
            length = len(self.car_in.index) - self.temporal_length * self.delta_T * self.sim_step
        else:
            print("wrong mod to generate data !")
            raise RuntimeError('MOD ERROR')

        return int(length)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--t_predict', type=int, default=4)
    parser.add_argument('--temporal_length', type=int, default=8)
    parser.add_argument('--sim_step', type=float, default=0.1)
    parser.add_argument('--delta_T', type=int, default=10)
    
    args = parser.parse_args()

    dataset = data_generator(mod=mod, args=args)
    print(len(dataset))