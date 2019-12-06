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
from separate_seg_model import sp_network_model as splstm
from Discrete_model import discrete_net_model
from data import traffic_data
import data as d
import conf

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="discrete_train")

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

    model = discrete_net_model(args)

    output_criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(args["mask"]).float())
    flow_criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    if args["use_cuda"]:
        model = model.cuda()
        flow_criterion = flow_criterion.cuda()
        output_criterion = output_criterion.cuda()
        criterion = criterion.cuda()

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
            data_set = traffic_data(args, data_prefix=prefix, mod="seg", topology=[args["seg"]])
            dataloader = torch.utils.data.DataLoader(data_set,
                                                    batch_size=args["batch_size"],
                                                    num_workers=args["num_workers"])

            for ii, data in tqdm(enumerate(dataloader)):

                model.zero_grad()
                optimizer.zero_grad()

                data = Variable(data)

                if args["use_cuda"]:
                    data = data.cuda()

                target = data[:, args["t_predict"]+1:, :, :]

                output_proba, delta_N_proba = model(data)
                
                output_target = target[:, :, :, 0].long()
                N_target = target[:, :, :, 2].long()
                N_former = data[:, args["t_predict"]:-1, :, :].long()
                delta_N = N_target - N_target
                delta_N += args["output_size"]-1

                [batch_size, temporal, spatial, output_size] = output_proba.shape
                output_proba = output_proba.view(batch_size*temporal*spatial, output_size)
                output_target = output_target.view(batch_size*temporal*spatial)
                delta_N_proba = delta_N_proba.view(batch_size*temporal*spatial, -1)
                delta_N = delta_N.view(batch_size*temporal*spatial)

                output_loss = output_criterion(output_proba, output_target)
                flow_loss = flow_criterion(delta_N_proba, delta_N)

                loss = args["output_loss_weight"] * output_loss + args["flow_loss_weight"] * flow_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                optimizer.step()

                acc_meter.add(loss.item())

                i += 1

            print("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(i, acc_meter.value()[0], topology_index, prefix))
            log_curve_file.write("batch{}, train_loss = {:.3f} for topology {}, preifx {}".format(ii, acc_meter.value()[0], topology_index, prefix))
        
        t = time.time()

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

            data_set = traffic_data(args, data_prefix=prefix, mod="seg", topology=[args["seg"]])
            dataloader = torch.utils.data.DataLoader(data_set, batch_size=args["batch_size"],
                                                    num_workers=args["num_workers"])

            for ii, data in tqdm(enumerate(dataloader)):

                data = Variable(data)
                if args["use_cuda"]:
                    data = data.cuda()

                target = data[:, args["t_predict"]+1:, :, :]

                outputs = model.infer(data)

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

if __name__ == "__main__":
    main()