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
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--learing_rate', type=float, default=0.003)
    parser.add_argument('--decay_rate', type=float, default=0.95)
    parser.add_argument('--lambda_param', type=float, default=0.0005)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--flow_loss_weight', type=float, default=1)
    parser.add_argument('--grad_clip', type=float, default=10.)

    #数据参数
    parser.add_argument('--sim_step', type=float, default=0.1)
    parser.add_argument('--delta_T', type=int, default=7)

    args = parser.parse_args()
    train(args)


def train(args):

        # 初始化一些变量， 数据文件不太好通过命令行输入，所以在conf文件中提取
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

        # 记录文件
        log_directory = os.path.join(conf.logPath, model_prefix+"/")

        log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w+')
        log_file = open(os.path.join(log_directory + 'val.txt'), 'w+')

        save_directory = os.path.join(conf.logPath, model_prefix)

        # 保存参数设置
        with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
            pickle.dump(args, f)

        # 保存checkpoint的位置
        def checkpoint_path(x):
            return os.path.join(save_directory, str(x)+'.tar')

        # 初始化模型对象
        net = TP_lstm(args)
        
        # 初始化不同的损失指标
        criterion = loss_function()
        mes_criterion = torch.nn.MSELoss()
        
        # 学习率的设置
        learing_rate = args.learing_rate
        
        if args.use_cuda:
            net = net.cuda()
            criterion = criterion.cuda()
            mes_criterion = mes_criterion.cuda()
        
        # 初始化优化器
        optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
        
        # 训练过程中衡量的损失
        loss_meter = meter.AverageValueMeter()
        
        # 测试过程中的损失指标
        flow_loss_meter = meter.AverageValueMeter()
        last_loss_meter = meter.AverageValueMeter()
        flow_last_loss_meter = meter.AverageValueMeter()

        # 预测的时段长
        predict_preiod = args.temporal_length - args.t_predict
        
        for epoch in range(args.num_epochs):
            
            loss_meter.reset()
            i = 0
            start = time.time()
            flag = True

            print("********training epoch beginning***********")
            while True:

                flag = data_generator.generateBatchForBucket()
                if not flag and data_generator.CurrentOutputs == []:
                    data_generator.setFilePoint(0)
                    break

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

                number_before = data[:, :, args.t_predict:, 2]
                number_current = target[:, :, :, 2]
                In = data[:, 0, args.t_predict:, 1].view(-1, 1, predict_preiod)
                flow_loss = criterion(number_current, number_before, In, output)
                mes_loss = mes_criterion(target[:, :, :, 0], output)
                loss = args.flow_loss_weight * flow_loss + mes_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()

                loss_meter.add(loss.item())
                
                if i % args.save_every == 0:
                    print("batch{}, train_loss = {:.3f}".format(i, loss_meter.value()[0]))
                    log_file_curve.write("batch{}, train_loss = {:.3f}\n".format(i, loss_meter.value()[0]))
                #if i > 5:
                #break
                i += 1
            
            t = time.time()
            print("epoch{}, train_loss = {:.3f}, time{}".format(epoch, loss_meter.value()[0], t-start))
            log_file_curve.write("epoch{}, train_loss = {:.3f}, time{}\n".format(epoch, loss_meter.value()[0], t-start))
            loss_meter.reset()
            flow_loss_meter.reset()
            last_loss_meter.reset()
            flow_last_loss_meter.reset()
            flag = True
            i = 0

            print("********validation epoch beginning***********")
            while True:
                
                flag = test_generator.generateBatchForBucket()
                if not flag and test_generator.CurrentOutputs == []:
                    test_generator.setFilePoint(0)
                    break

                data = torch.tensor(test_generator.CurrentSequences).float()
                init_data = data[:, :, :args.t_predict, :]
                temporal_data = data[:, 0, args.t_predict:, :]
                laneT = torch.tensor(test_generator.CurrentLane).float()
                target = torch.tensor(test_generator.CurrentOutputs).float()              

                if args.use_cuda:
                    data = data.cuda()
                    init_data = init_data.cuda()
                    temporal_data = temporal_data.cuda()
                    laneT = laneT.cuda()
                    target = target.cuda()

                output = net.infer(temporal_data, init_data, laneT)
                number_current = target[:, :, :, 2]
                number_before = data[:, :, args.t_predict:, 2]
                In = data[:, 0, args.t_predict:, 1].view(-1, 1, predict_preiod)
                flow_loss = criterion(number_current, number_before, In, output)
                mes_loss = mes_criterion(target[:, :, :, 0], output)
                last_frame_loss = mes_criterion(target[:, :, -1, 0], output[:, :, -1])
                last_frame_flow_loss = criterion(number_before[:, :, -1].unsqueeze(2), number_current[:, :, -1].unsqueeze(2), In[:, :, -1].unsqueeze(1), output[:, :, -1].unsqueeze(2))
                loss_meter.add(mes_loss.item())
                flow_loss_meter.add(flow_loss.item())
                last_loss_meter.add(last_frame_loss.item())
                flow_last_loss_meter.add(last_frame_flow_loss.item())
                
                if i % args.save_every == 0:
                    print("batch{}, flow_loss={:.3f}, mes_loss={:.3f}, last_frame_loss={:.3f}".format(i, loss_meter.value()[0], flow_loss_meter.value()[0], last_loss_meter.value()[0]))
                    log_file_curve.write("batch{}, flow_loss={:.3f}, mes_loss={:.3f}, last_frame_loss={:.3f}, last_frame_flow_loss={:.3f}\n".format(i, loss_meter.value()[0], flow_loss_meter.value()[0], last_loss_meter.value()[0], flow_last_loss_meter.value()[0]))
                #if i > 5:
                #break
                i += 1

            print('saving model')
            torch.save({
                'epoch':epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path(epoch))   

        return 

if __name__ == '__main__':
    main()