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
import random

import argparse
import os
import time
import pickle
import yaml
from tqdm import tqdm

from utils import batchGenerator
from model import TP_lstm
from model import loss_function
from net_model import network_model
from data import traffic_data
import data as d
import conf

'''路口与路段之间需要联合训练
'''


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="co_train")

    args = parser.parse_args()
    train(args)

def train(args):

    with open(os.path.join(conf.configPath, args.config+'.yaml'), encoding='UTF-8') as config:
        args = yaml.load(config)

    log_directory = os.path.join(conf.logPath, args["model_prefix"])
    log_curve_file = open(os.path.join(log_directory, "log_curve.txt"), "w+")

    def checkpoint_path(x):
        return os.path.join(log_directory, str(x)+'.tar')

    model = network_model(args)

    criterion = loss_function()
    sim_error_criterion = torch.nn.ReLU()
    pool = torch.nn.AvgPool1d(3, stride=1, padding=1)

    if args["use_cuda"]:
        model = model.cuda()
        criterion = criterion.cuda()
        sim_error_criterion = sim_error_criterion.cuda()
        pool = pool.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=args["lambda_param"])

    acc_meter = meter.AverageValueMeter()
    flow_loss_meter = meter.AverageValueMeter()
    last_loss_meter = meter.AverageValueMeter()
    last_flow_loss_meter = meter.AverageValueMeter()

    best_acc_epoch = 0
    best_acc = float('inf')
    best_flow_epoch = 0
    best_flow_loss = float('inf')

    topology = d.co_topology
    seg = d.seg
    inter = d.inter_node

    for epoch in range(args["num_epochs"]):
        print("****************training epoch beginning**********************")
        model.train()
        acc_meter.reset()
        start = time.time()
        i = 0
        for prefix in args["prefix"]:
            
            for topology_index in range(len(d.co_topology)):

                data_set = traffic_data(args, data_prefix=prefix,
                        mod='cooperate', topology=d.co_topology[topology_index])
            
                dataloader = torch.utils.data.DataLoader(data_set, 
                                                    batch_size=args["batch_size"], 
                                                    num_workers=args["num_workers"])

                [temporal, spatial, input_size] = data_set[0].shape
                bucket_number = data_set.bucket_number
                init_bucket = [0, bucket_number[1]+1]
                non_flow_from = [bucket_number[1], bucket_number[3], 
                                bucket_number[5], bucket_number[6]]
                non_flow_in = [0, bucket_number[1]+1, 
                                bucket_number[3], bucket_number[6]]
                flow_from = [i for i in range(spatial) if i not in non_flow_from]
                flow_in = [i for i in range(spatial) if i not in non_flow_in]

                for ii, co_data in tqdm(enumerate(dataloader)):

                    #t1 = time.time()
                    model.zero_grad()
                    optimizer.zero_grad()
                
                    #t2 = time.time()
                    co_data = Variable(co_data)
                    batch_size = co_data.shape[0]

                    if args["use_cuda"]:
                        co_data = co_data.cuda()

                    #t3 = time.time()
                    inputs = co_data[:, :-1, :, :]
                    target = co_data[:, args["t_predict"]+1:, :]
                    In = Variable(inputs.data.new(batch_size, temporal-args["t_predict"]-1, spatial).fill_(0).float())
                    In[:, :, init_bucket] += target[:, :, init_bucket, 1]

                    #t4 = time.time()
                    if random.random() < args["sample_rate"]:
                    
                        outputs = model(inputs, bucket_number)
                        outputs = outputs.squeeze(3)
                        number_before = inputs[:, args["t_predict"]:, :, 2]
                        number_current = target[:, :, :, 2]
                        In[:, :, flow_in] += outputs[:, :, flow_from]
                        In[:, :, bucket_number[6]] += outputs[:, :, bucket_number[1]] + outputs[:, :, bucket_number[3]]
                        In[:, :, bucket_number[4]] += outputs[:, :, bucket_number[6]]
                        number_caculate = number_before + In - outputs

                        flow_loss = criterion(number_current, number_caculate)
                    
                        if args["use_mask"]:
                            mask = get_mask(target[:, :, :, 0], args["mask"])
                            acc_loss = criterion(target[:, :, :, 0], outputs, mask)
                        else:
                            acc_loss = criterion(target[:, :, :, 0], outputs)

                        loss = args["flow_loss_weight"] * flow_loss + (2 - args["flow_loss_weight"]) * acc_loss

                    else:
                        
                        h_all = None
                        c_all = None
                        topology_struct = None

                        for t in range(args["t_predict"]):
                            input_data = inputs[:, t, :, :]
                            output, [h_all, c_all], topology_struct = model.infer(input_data, 
                                                                bucket_number, [h_all, c_all], topology_struct)

                        input_data = inputs[:, args["t_predict"], :, :]
                        last_error = Variable(input_data.data.new(batch_size, spatial, 1).fill_(0).float())
                        loss = 0

                        for t in range(args["temporal_length"]-args["t_predict"]):

                            output, [h_all, c_all],topology_struct = model.infer(input_data, bucket_number, 
                                                                    [h_all, c_all], topology_struct)

                            number_before = inputs[:, args["t_predict"]+t, :, 2].view(batch_size, spatial, 1)
                            number_current = target[:, t, :, 2].view(batch_size, spatial, 1)
                            In[:, t, flow_in] += output[:, flow_from, 0]
                            In[:, t, bucket_number[6]] += output[:, bucket_number[1], 0] + output[:, bucket_number[3], 0]
                            In[:, t, bucket_number[4]] += output[:, bucket_number[6], 0]
                            number_caculate = number_before + In[:, t, :].unsqueeze(2) - output
                            input_data = torch.cat((output, In[:, t, :].unsqueeze(2), number_caculate), 2)

                            if args["use_simerror"]:
                                current_error = target[:, t, :, 0].unsqueeze(2) - output
                                sim_error = sim_error_criterion(last_error * current_error)
                                last_error = last_error * 0 + number_caculate - number_current
                            else:
                                sim_error = Variable(inputs.data.new(1).fill_(0).float())

                            if args["use_mask"]:
                                mask = get_mask(target[:, t, :, 0].view(batch_size, spatial, 1), args["mask"]).view(batch_size, spatial, 1)
                                acc_loss = criterion(target[:, t, :, 0].view(batch_size, spatial, 1), output, mask)
                                sim_error = sim_error * mask
                            else:
                                acc_loss = criterion(target[:, t, :, 0].view(batch_size, spatial, 1), output)
                            
                            sim_error = torch.mean(sim_error)
                            flow_loss = criterion(number_current, number_caculate)
                            loss += acc_loss + args["flow_loss_weight"] * flow_loss + args["gamma"] * sim_error
                        
                        loss = loss / (args["temporal_length"] - args["t_predict"])

                    #t5 = time.time()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                    optimizer.step()

                    #t6 = time.time()
                    acc_meter.add(loss.item())

                    i += 1
                    if i % args["show_every"] == 0:
                        print("batch{}, train_loss = {:.3f}".format(i, acc_meter.value()[0]))
                        log_curve_file.write("batch{}, train_loss = {:.3f}".format(i, acc_meter.value()[0]))
                    
                    #print(t6-t1)
                print("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(ii, acc_meter.value()[0], topology_index, prefix))
                log_curve_file.write("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(ii, acc_meter.value()[0], topology_index, prefix))
        
        t = time.time()
        args["sample_rate"] -= args["sample_decay"]
        print("epoch{}, train_loss = {:.3f}, time{:.2f}".format(epoch, acc_meter.value()[0], t-start))
        log_curve_file.write("epoch{}, train_loss = {:.3f}, time{:.2f}\n".format(epoch, acc_meter.value()[0], t-start))
        acc_meter.reset()
        flow_loss_meter.reset()
        last_loss_meter.reset()
        last_flow_loss_meter.reset()
        i = 0

        print("********validation epoch beginning***********")
        model.eval()
        for prefix in args["test_prefix"]:

            for topology_index in range(len(d.co_topology)):

                data_set = traffic_data(args, data_prefix=prefix,
                        mod='cooperate', topology=d.co_topology[topology_index])
            
                dataloader = torch.utils.data.DataLoader(data_set, 
                                                    batch_size=args["batch_size"], 
                                                    num_workers=args["num_workers"])

                [temporal, spatial, input_size] = data_set[0].shape
                bucket_number = data_set.bucket_number
                init_bucket = [0, bucket_number[1]+1]
                non_flow_from = [bucket_number[1], bucket_number[3], 
                                bucket_number[5], bucket_number[6]]
                non_flow_in = [0, bucket_number[1]+1, 
                                bucket_number[3], bucket_number[6]]
                flow_from = [i for i in range(spatial) if i not in non_flow_from]
                flow_in = [i for i in range(spatial) if i not in non_flow_in]

                for ii, co_data in tqdm(enumerate(dataloader)):
                    #t1 = time.time()
                    batch_size = co_data.shape[0]

                    if args["use_cuda"]:
                        co_data = co_data.cuda()
                    
                    #t2 = time.time()
                    inputs = co_data[:, :-1, :, :]
                    target = co_data[:, args["t_predict"]+1:, :]
                    In = inputs.data.new(batch_size, args["temporal_length"]-args["t_predict"], spatial).fill_(0).float()
                    numbers_caculate = inputs.data.new(batch_size, args["temporal_length"]-args["t_predict"], spatial).fill_(0).float()
                    numbers_current = target[:, :, :, 2]
                    In[:, :, init_bucket] += target[:, :, init_bucket, 1]
                    
                    #t3 = time.time()
                    h_all = None
                    c_all = None
                    topology_struct = None
                    input_data = inputs[:, 0, :, :]
                    outputs = inputs.data.new(batch_size, args["temporal_length"]-args["t_predict"], spatial).fill_(0).float()

                    #t4 = time.time()
                    for t in range(args["temporal_length"]):

                        output, [h_all, c_all], topology_struct = model.infer(input_data, 
                                                        bucket_number, [h_all, c_all], topology_struct)
                        
                        if t < args["t_predict"]:
                            input_data = inputs[:, t+1, :, :]
                        else:
                            number_before = inputs[:, t, :, 2].view(batch_size, spatial, 1)
                            number_current = target[:, t-args["t_predict"], :, 2].view(batch_size, spatial, 1)
                            In[:, t-args["t_predict"], flow_in] += output[:, flow_from, 0]
                            In[:, t-args["t_predict"], bucket_number[6]] += output[:, bucket_number[1], 0] + output[:, bucket_number[3], 0]
                            In[:, t-args["t_predict"], bucket_number[4]] += output[:, bucket_number[6], 0]
                            number_caculate = number_before + In[:, t-args["t_predict"], :].unsqueeze(2) - output
                            numbers_caculate[:, t-args["t_predict"], :] += number_caculate.squeeze(2)
                            input_data = torch.cat((output, In[:, t-args["t_predict"], :].unsqueeze(2), number_caculate), 2)
                            outputs[:, t-args["t_predict"], :] += output.squeeze(2)

                    #t5 = time.time()    
                    acc_loss = criterion(target[:, :, :, 0], outputs)
                    flow_loss = criterion(numbers_current, numbers_caculate)
                    last_acc_loss = criterion(target[:, -1, :, 0], outputs[:, -1, :])
                    last_flow_loss = criterion(numbers_current[:, -1, :], numbers_caculate[:, -1, :])

                    #t6 = time.time()
                    acc_meter.add(acc_loss.item())
                    flow_loss_meter.add(flow_loss.item())
                    last_loss_meter.add(last_acc_loss.item())
                    last_flow_loss_meter.add(last_flow_loss.item())

                    if i % args["show_every"] == 0:
                        print("batch{}, acc_loss={:.3f}, flow_loss={:.3f}, last_frame_loss={:.3f}, last_frame_flow_loss={:.3f}".format(i, acc_meter.value()[0], flow_loss_meter.value()[0], last_loss_meter.value()[0], last_flow_loss_meter.value()[0]))
                        log_curve_file.write("batch{}, acc_loss={:.3f}, flow_loss={:.3f}, last_frame_loss={:.3f}, last_frame_flow_loss={:.3f}\n".format(i, acc_meter.value()[0], flow_loss_meter.value()[0], last_loss_meter.value()[0], last_flow_loss_meter.value()[0]))
                            #if i > 5:
                            #    break
                    i += 1

        if acc_meter.value()[0] < best_acc:
            best_acc_epoch = epoch
            best_acc = acc_meter.value()[0]

        if flow_loss_meter.value()[0] < best_flow_loss:
            best_flow_epoch = epoch
            best_flow_loss = flow_loss_meter.value()[0]

        print("epoch{}, acc_loss={:.3f}, flow_loss={:.3f}, last_frame_loss={:.3f}, last_frame_flow_loss={:.3f}".format(epoch, acc_meter.value()[0], flow_loss_meter.value()[0], last_loss_meter.value()[0], last_flow_loss_meter.value()[0]))
        log_curve_file.write("epoch{}, acc_loss={:.3f}, flow_loss={:.3f}, last_frame_loss={:.3f}, last_frame_flow_loss={:.3f}\n".format(epoch, acc_meter.value()[0], flow_loss_meter.value()[0], last_loss_meter.value()[0], last_flow_loss_meter.value()[0]))
        print('saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))                


    log_curve_file.write("best mes epoch {}, best mes loss {:.3f}, best flow epoch{}, best flow loss{:.3f}".format(best_acc_epoch, best_acc, best_flow_epoch, best_flow_loss))
    log_curve_file.close()

    return True

def get_mask(target, mask):

    shape = target.shape
    weight = target.data.new(shape).fill_(0).float()
    all_ones = target.data.new(shape).fill_(1).float()

    weight = torch.where(weight==0, mask[0]*all_ones, weight)
    weight = torch.where(weight==1, mask[1]*all_ones, weight)
    weight = torch.where(weight==2, mask[2]*all_ones, weight)
    weight = torch.where(weight==3, mask[3]*all_ones, weight)
    weight = torch.where(weight==4, mask[4]*all_ones, weight)
    weight = torch.where(weight==5, mask[5]*all_ones, weight)
    weight = torch.where(weight==6, mask[6]*all_ones, weight)
    weight = torch.where(weight==7, mask[7]*all_ones, weight)
    weight = torch.where(weight==8, mask[8]*all_ones, weight)
    weight = torch.where(weight==9, mask[9]*all_ones, weight)
    weight = torch.where(weight==10, mask[10]*all_ones, weight)
    weight = torch.where(weight==11, mask[11]*all_ones, weight)
    weight = torch.where(weight==12, mask[12]*all_ones, weight)
    
    return weight

if __name__ == "__main__":
    main()