import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from torchnet import meter
import numpy as np 
import sklearn.metrics as metrics
from tqdm import tqdm
import torchvision.models as models
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
from unit_net_model import uni_network_model as unlstm
from data import traffic_data
import data as d
import conf


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
    
    for key in args.keys():
        print(key, "   ", args[key])
        log_curve_file.write(key+"  "+str(args[key])+'\n')

    def checkpoint_path(x):
        return os.path.join(log_directory, str(x)+'.tar')

    model = unlstm(args)

    criterion = loss_function()
    sim_error_criterion = torch.nn.ReLU()
    
    if args["use_cuda"]:
        model = model.cuda()
        criterion = criterion.cuda()
        sim_error_criterion = sim_error_criterion.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=args["lambda_param"])

    acc_meter = meter.AverageValueMeter()
    flow_loss_meter = meter.AverageValueMeter()
    last_frame_acc_meter = meter.AverageValueMeter()
    last_frame_flow_meter = meter.AverageValueMeter()

    best_acc_epoch = 0
    best_acc = float('inf')
    best_flow_epoch = 0
    best_flow_loss = float('inf')
    
    for epoch in range(args["num_epochs"]):

        print("*********************train epoch beginning****************************")

        model.train()
        acc_meter.reset()
        start = time.time()
        i = 0 

        for prefix in args["prefix"]:
            #break
            for topology_index in range(len(d.co_topology)):

                data_set = traffic_data(args, data_prefix=prefix,
                                mod='cooperate', topology=d.co_topology[topology_index])
                bucket_number = data_set.bucket_number
                
                dataloader = torch.utils.data.DataLoader(data_set,
                                                        batch_size=args["batch_size"],
                                                        num_workers=args["num_workers"])
                
                for ii, co_data in tqdm(enumerate(dataloader)):

                    model.zero_grad()
                    optimizer.zero_grad()

                    co_data = Variable(co_data)

                    if args["use_cuda"]:
                        co_data = co_data.cuda()

                    inputs = co_data[:, :-1, :, :]
                    target = co_data[:, args["t_predict"]+1:, :, :]

                    if random.random() < args["sample_rate"]:

                        outputs, _ = model(co_data, bucket_number)
                        output_pred = outputs[:, :, :, 0]
                        number_pred = outputs[:, :, :, 2]

                        flow_loss = criterion(target[:, :, :, 2], number_pred)

                        if args["use_mask"]:
                            mask = get_mask(target[:, :, :, 0], args["mask"])
                            acc_loss = criterion(target[:, :, :, 0], output_pred, mask)
                        else:
                            acc_loss = criterion(target[:, :, :, 0], output_pred)

                        loss = args["flow_loss_weight"] * flow_loss + (2 - args["flow_loss_weight"]) * acc_loss
                    
                    else:

                        outputs, _ = model.infer(co_data, bucket_number)

                        if args["use_simerror"]:
                            sim_error = outputs[:, :-1, :, 2] - target[:, :-1, :, 2]
                            sim_error = sim_error * (target[:, 1:, :, 0] - outputs[:, 1:, :, 0])
                            sim_error = sim_error_criterion(sim_error)
                        else:
                            sim_error = Variable(co_data.data.new(1).fill_(0).float())
                        
                        if args["use_mask"] :
                            mask = get_mask(target[:, :, :, 0], args["mask"])
                            acc_loss = criterion(target[:, :, :, 0], outputs[:, :, :, 0], mask)
                            sim_error = sim_error * mask[:, 1:, :]
                        else:
                            acc_loss = criterion(target[:, :, :, 0], outputs[:, :, :, 0])

                        sim_error = torch.mean(sim_error)
                        flow_loss = criterion(target[:, :, :, 2], outputs[:, :, :, 2])
                        loss = (2 - args["flow_loss_weight"]) * acc_loss + args["flow_loss_weight"] * flow_loss + args["gamma"] * sim_error

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                    optimizer.step()

                    acc_meter.add(loss.item())

                    #break
                    i += 1
                    if i % args["show_every"] == 0:
                        print("batch{}, train_loss = {:.3f}".format(i, acc_meter.value()[0]))
                        log_curve_file.write("batch{}, train_loss = {:.3f}\n".format(i, acc_meter.value()[0]))
                    
                    #print(t6-t1)
                #break
                print("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(ii, acc_meter.value()[0], topology_index, prefix))
                log_curve_file.write("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(ii, acc_meter.value()[0], topology_index, prefix))
        
        t = time.time()
        args["sample_rate"] -= args["sample_decay"]
        print("epoch{}, train_loss = {:.3f}, time{:.2f}".format(epoch, acc_meter.value()[0], t-start))
        log_curve_file.write("epoch{}, train_loss = {:.3f}, time{:.2f}\n".format(epoch, acc_meter.value()[0], t-start))

        acc_meter.reset()
        flow_loss_meter.reset()
        last_frame_acc_meter.reset()
        last_frame_flow_meter.reset()
        i = 0

        print("********validation epoch beginning***********")
        model.eval()
        for prefix in args["test_prefix"]:

            for topology_index in range(len(d.co_topology)):

                data_set = traffic_data(args, data_prefix=prefix,
                                        mod="cooperate", topology=d.co_topology[topology_index])
                
                dataloader = torch.utils.data.DataLoader(data_set,
                                                        batch_size=args["batch_size"],
                                                        num_workers=args["num_workers"])
                
                bucket_number = data_set.bucket_number

                for ii, co_data in tqdm(enumerate(dataloader)):

                    batch_size = co_data.shape[0]
                    
                    if args["use_cuda"]:
                        co_data = co_data.cuda()

                    inputs = co_data[:, :-1, :, :]
                    target = co_data[:, args["t_predict"]+1:, :, :]
                    
                    outputs, _ = model.infer(co_data, bucket_number)

                    acc_loss = criterion(target[:, :, :, 0], outputs[:, :, :, 0])
                    flow_loss = criterion(target[:, :, :, 2], outputs[:, :, :, 2])
                    last_frame_acc_loss = criterion(target[:, -1, :, 0], outputs[:, -1, :, 0])
                    last_frame_flow_loss = criterion(target[:, -1, :, 2], outputs[:, -1, :, 2])

                    acc_meter.add(acc_loss.item())
                    flow_loss_meter.add(flow_loss.item())
                    last_frame_acc_meter.add(last_frame_acc_loss.item())
                    last_frame_flow_meter.add(last_frame_flow_loss.item())

                    if i % args["show_every"] == 0:
                        print("batch{}, acc_loss={:.3f}, flow_loss={:.3f}, last_frame_loss={:.3f}, last_frame_flow_loss={:.3f}".format(i, acc_meter.value()[0], flow_loss_meter.value()[0], last_frame_acc_meter.value()[0], last_frame_flow_meter.value()[0]))
                        log_curve_file.write("batch{}, acc_loss={:.3f}, flow_loss={:.3f}, last_frame_loss={:.3f}, last_frame_flow_loss={:.3f}\n".format(i, acc_meter.value()[0], flow_loss_meter.value()[0], last_frame_acc_meter.value()[0], last_frame_flow_meter.value()[0]))

                    i += 1

        if acc_meter.value()[0] < best_acc :
            best_acc_epoch = epoch
            best_acc = acc_meter.value()[0]

        if flow_loss_meter.value()[0] < best_flow_loss:
            best_flow_epoch = epoch
            best_flow_loss = flow_loss_meter.value()[0]

        print("epoch{}, acc_loss={:.3f}, flow_loss={:.3f}, last_frame_acc_loss={:.3f}, last_frame_flow_loss={:.3f}".format(epoch, acc_meter.value()[0], flow_loss_meter.value()[0], last_frame_acc_meter.value()[0], last_frame_flow_meter.value()[0]))
        log_curve_file.write("epoch{}, acc_loss={:.3f}, flow_loss={:.3f}, last_frame_loss={:.3f}, last_frame_flow_loss={:.3f}\n".format(epoch, acc_meter.value()[0], flow_loss_meter.value()[0], last_frame_acc_meter.value()[0], last_frame_flow_meter.value()[0]))
        print('saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))
    
    print("best mes epoch {}, best mes loss {:.3f}, best flow epoch{}, best flow loss{:.3f}".format(best_acc_epoch, best_acc, best_flow_epoch, best_flow_loss))
    log_curve_file.write("best mes epoch {}, best mes loss {:.3f}, best flow epoch{}, best flow loss{:.3f}".format(best_acc_epoch, best_acc, best_flow_epoch, best_flow_loss))
    log_curve_file.close()

def get_mask(target, mask):

    shape = target.shape
    weight = Variable(target.data.new(shape).fill_(0).float())
    all_ones = Variable(target.data.new(shape).fill_(1).float())

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