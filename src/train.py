import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from torchnet import meter
import numpy as np 
import sklearn.metrics as metrics
from tqdm import tqdm
import torchvision.models as models
import pyecharts as pe
import pandas as pd

import argparse
import os
import time
import pickle

from utils import batchGenerator
from model import TP_lstm
from model import loss_function
import conf

def main():
    
    parser = argparse.ArgumentParser()
    
    #网络结构
    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lane_gate_size', type=int, default=4)
    parser.add_argument('--output_hidden_size', type=int, default=16)
    parser.add_argument('--t_predict', type=int, default=7)
    parser.add_argument('--temporal_length', type=int, default=11)
    parser.add_argument('--spatial_length', type=int, default=5)

    #训练参数
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learing_rate', type=float, default=0.003)
    parser.add_argument('--decay_rate', type=float, default=0.95)
    parser.add_argument('--lambda_param', type=float, default=0.0005)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--spatial_loss_weight', type=float, default=0.5)
    parser.add_argument('--temporal_loss_weight', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=10.)

    #数据参数
    parser.add_argument('--sim_step', type=float, default=0.1)
    parser.add_argument('--delta_T', type=int, default=7)

    args = parser.parse_args()
    train(args)


def train(args):

        data_prefix = conf.args["prefix"]
        model_prefix = conf.args["modelFilePrefix"]
        test_prefix = conf.args["testFilePrefix"] 
        data_generator = batchGenerator(
            data_prefix, 
            batchSize=args.batch_size, 
            simTimeStep=args.sim_step,
            seqLength=args.temporal_length,
            seqPredict=args.t_predict,
            deltaT=args.delta_T
            )
        test_generator = batchGenerator(
            test_prefix, 
            batchSize=args.batch_size, 
            simTimeStep=args.sim_step,
            seqLength=args.temporal_length,
            seqPredict=args.t_predict,
            deltaT=args.delta_T
            )

        log_directory = os.path.join(conf.logPath, model_prefix+"/")
        plot_directory = os.path.join(conf.picsPath, model_prefix+'_plot/')

        log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w+')
        log_file = open(os.path.join(log_directory + 'val.txt'), 'w+')

        save_directory = os.path.join(conf.logPath, model_prefix)

        with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
            pickle.dump(args, f)

        def checkpoint_path(x):
            return os.path.join(save_directory, str(x)+'.tar')

        net = TP_lstm(args)
        optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
        criterion = loss_function(args.spatial_loss_weight, args.temporal_loss_weight)
        learing_rate = args.learing_rate
        if args.use_cuda:
            net = net.cuda()
            criterion = criterion.cuda()
        loss_meter = meter.AverageValueMeter()

        print("********training epoch beginning***********")
        for epoch in range(args.num_epochs):
            
            loss_meter.reset()
            i = 0
            start = time.time()

            while data_generator.generateBatchForBucket():
        
                net.zero_grad()
                optimizer.zero_grad()
                
                data = Variable(torch.Tensor(data_generator.CurrentSequences))
                laneT = Variable(torch.Tensor(data_generator.CurrentLane))
                target = Variable(torch.Tensor(data_generator.CurrentOutputs))

                if args.use_cuda:
                    data = data.cuda()
                    laneT = laneT.cuda()
                    target = target.cuda()

                output = net(data, laneT)
                loss = criterion(target, output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()

                loss_meter.add(loss.item())

            print("epoch{}, train_loss = {:.3f}, time{}".format(epoch, loss_meter.value()[0], time.time()-start))
            log_file_curve.write("training epoch: " + str(epoch) + " loss: " + str(loss_meter.value()[0]) + '\n')
            loss_meter.reset()
            number_batch = 0

            while test_generator.generateBatchForBucket():

                number_batch += 1
                init_data = torch.tensor(test_generator.CurrentSequences[:, 0, :]).float()
                temporal_data = torch.tensor(test_generator.CurrentSequences[0, 1:, :]).float()
                laneT = torch.tensor(test_generator.CurrentLane).float()
                target = torch.tensor(test_generator.CurrentOutputs).float()

                if args.use_cuda:
                    init_data = init_data.cuda()
                    temporal_data = temporal_data.cuda()
                    laneT = laneT.cuda()
                    target = target.cuda()

                output = net.infer(temporal_data, init_data, laneT)
                test_loss = criterion(target, output)
                loss_meter.add(test_loss.item())
                error += metrics.mean_absolute_error(target, output)

            error = error / number_batch
            loss = loss_meter.value()[0]
            print("epoch{}, test_loss={:.3f}, test_err={:.3f}".format(epoch, loss, error))
            log_file_curve.write("Validation epoch: "+str(epoch)+" loss: "+str(loss)+" err: "+str(error)+'\n')

            print('saving model')
            torch.save({
                'epoch':epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path(epoch))   

        return 

if __name__ == '__main__':
    main()