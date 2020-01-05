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

from model import loss_function
from net_model import network_model
from seg_model import continuous_seg
from seg_model import continuous_seg_nonspeed
from sn_model import sn_lstm
from data import traffic_data
import conf


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="single_train")

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

    #model = continuous_seg(args)
    model = continuous_seg_nonspeed(args)
    #model = sn_lstm(args)

    criterion = loss_function()
    sim_error_criterion = torch.nn.ReLU()
    
    if args["use_cuda"]:
        model = model.cuda()
        criterion = criterion.cuda()
        sim_error_criterion = sim_error_criterion.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=args["lambda_param"])
    #optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

    acc_meter = meter.AverageValueMeter()
    flow_loss_meter = meter.AverageValueMeter()
    last_frame_acc_meter = meter.AverageValueMeter()
    last_frame_flow_meter = meter.AverageValueMeter()

    best_acc_epoch = 0
    best_acc = float('inf')
    best_flow_epoch = 0
    best_flow_loss = float('inf')

    for epoch in range(args["num_epochs"]):

        print("**********************train epoch beginning**********************")
        model.train()
        acc_meter.reset()
        start = time.time()
        i = 0

        for prefix in args["prefix"]:
            #break
            data_set = traffic_data(args, data_prefix=prefix, mod="seg", topology=[args["seg"][0]])
            dataloader = torch.utils.data.DataLoader(data_set,
                                                    batch_size=args["batch_size"],
                                                    num_workers=args["num_workers"])
            for seg in args["seg"]:
                data_set.reload(topology=[seg], mod='seg')
                for ii, data in tqdm(enumerate(dataloader)):

                    model.zero_grad()
                    optimizer.zero_grad()

                    data = Variable(data)

                    if args["use_cuda"]:
                        data = data.cuda()

                    target = data[:, args["t_predict"]+1:, :, :]

                    if random.random() < args["sample_rate"]:
                    
                        outputs = model(data)
                        output_pred = outputs[:, :, :, 0]
                        number_pred = outputs[:, :, :, 2]
                        #speed_pred = outputs[:, :, :, 3]

                        seg_flow_loss = criterion(target[:, :, :, 2].sum(dim=2), number_pred.sum(dim=2))
                        #speed_loss = criterion(target[:, :, :, 3], speed_pred)

                        if args["use_mask"]:
                            mask = get_mask(target[:, :, :, 0], args["mask"])
                            acc_loss = criterion(target[:, :, :, 0], output_pred, mask)
                            flow_loss = criterion(target[:, :, :, 2], number_pred, mask)
                        else:
                            acc_loss = criterion(target[:, :, :, 0], output_pred)
                            flow_loss = criterion(target[:, :, :, 2], number_pred)
                    
                        loss = args["flow_loss_weight"] * flow_loss + args["output_loss_weight"] * acc_loss+ args["seg_loss_weight"]*seg_flow_loss #+ args['speed_loss_weight'] * speed_loss

                    else:
                    
                        outputs = model.infer(data)

                        if args["use_simerror"]:
                            sim_error = outputs[:, :-1, :, 2] - target[:, :-1, :, 2]
                            sim_error = sim_error * (target[:, 1:, :, 0] - outputs[:, 1:, :, 0])
                            sim_error = sim_error_criterion(sim_error)
                        else:
                            sim_error = Variable(data.data.new(1).fill_(0).float())
                    
                        if args["use_mask"]:
                            mask = get_mask(target[:, :, :, 0], args["mask"])
                            acc_loss = criterion(target[:, :, :, 0], outputs[:, :, :, 0], mask)
                            sim_error = sim_error * mask[:, 1:, :]
                            flow_loss = criterion(target[:, :, :, 2], outputs[:, :, :, 2], mask)
                        else:
                            acc_loss = criterion(target[:, :, :, 0], outputs[:, :, :, 0])
                            flow_loss = criterion(target[:, :, :, 2], outputs[:, :, :, 2])
                    
                        sim_error = torch.mean(sim_error)
                        seg_flow_loss = criterion(target[:, :, :, 2].sum(dim=2), outputs[:, :, :, 2].sum(dim=2))
                        #speed_loss = criterion(target[:, :, :, 3], outputs[:, :, :, 3])
                        loss = args["output_loss_weight"] * acc_loss + args["flow_loss_weight"]*flow_loss + args["gamma"] * sim_error + args["seg_loss_weight"]*seg_flow_loss #+ args["speed_loss_weight"]*speed_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                    optimizer.step()

                    acc_meter.add(loss.item())

                    if i % args["show_every"] == 0:
                        print("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(i, acc_meter.value()[0], str(seg), prefix))
                        log_curve_file.write("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(i, acc_meter.value()[0], str(seg), prefix))
                    i += 1
            
            print("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(i, acc_meter.value()[0], str(seg), prefix))
            log_curve_file.write("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(ii, acc_meter.value()[0], str(seg), prefix))

        t = time.time()
        args["sample_rate"] -= args["sample_decay"]

        print("epoch{}, train_loss = {:.3f}, time{:.2f}".format(epoch, acc_meter.value()[0], t-start))
        log_curve_file.write("epoch{}, train_loss = {:.3f}, time{:.2f}\n".format(epoch, acc_meter.value()[0], t-start))

        acc_meter.reset()
        flow_loss_meter.reset()
        last_frame_acc_meter.reset()
        last_frame_flow_meter.reset()
        i = 0

        print("*****************validation epoch beginning******************")
        model.eval()

        for prefix in args["test_prefix"]:

            data_set = traffic_data(args, data_prefix=prefix, mod="seg", topology=[args["seg"][0]])
            dataloader = torch.utils.data.DataLoader(data_set,
                                                    batch_size=args["batch_size"],
                                                    num_workers=args["num_workers"])
            
            for seg in args['seg']:
                data_set.reload(topology=[seg], mod='seg')

                for ii, data in tqdm(enumerate(dataloader)):

                    model.zero_grad()
                    optimizer.zero_grad()

                    data = Variable(data)

                    if args["use_cuda"]:
                        data = data.cuda()

                    target = data[:, args["t_predict"]+1:, :, :]

                    outputs = model.infer(data)

                    target = data_set.recover(target)
                    outputs = data_set.recover(outputs)

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

    return

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