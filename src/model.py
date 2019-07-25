import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import conf


class FCNet(nn.Module):

    def __init__(self, addLayer=False, layerSize=[3, conf.args["embeddingLayer"], conf.args["embeddingSize"]]):

        super().__init__()

        self.inputSize = layerSize[0]
        self.hiddenSize = layerSize[1]
        self.outputSize = layerSize[2]
        self.addLayer = addLayer

        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.fc2 = nn.Linear(self.hiddenSize, self.outputSize)
        
        self.relu = nn.ReLU()

        if addLayer:
            self.outputSize2 = layerSize[3]
            self.fc3 = nn.Linear(self.outputSize, self.outputSize2)

    def forward(self, inputs):

        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        if self.addLayer :
            outputs = self.relu(outputs)
            outputs = self.fc3(outputs)

        return outputs

class MD_lstm_cell(nn.Module):

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        
        self.sigma = torch.nn.Sigmoid()

        self.spatial_embedding = torch.nn.Linear(2*hidden_size, hidden_size)
         

    def forward(self, inputs, h_s_t, c_s_t, h_after_t, h_before_t):

        [batch_size, spatial_size, hidden_size] = h_s_t.shape

        spatial_gate = torch.cat((h_after_t, h_before_t), dim=2)
        
        #处理batch
        spatial_gate = spatial_gate.view(-1, 2*hidden_size)
        h_s_t = h_s_t.view(-1, hidden_size)
        c_s_t = c_s_t.view(-1, hidden_size)
        h_after_t = h_after_t.view(-1, hidden_size)
        h_before_t = h_before_t.view(-1, hidden_size)
        inputs = inputs.view(-1, self.input_size)

        spatial_gate = self.spatial_embedding(spatial_gate)
        spatial_gate = self.sigma(spatial_gate)

        h_hat, c_s_tp = self.cell(inputs, (h_s_t, c_s_t))

        #还原
        spatial_gate = spatial_gate.view(batch_size, spatial_size, hidden_size)
        c_s_tp = c_s_tp.view(batch_size, spatial_size, hidden_size)
        h_hat = h_hat.view(batch_size, spatial_size, hidden_size)
        
        h_s_tp = h_hat * spatial_gate

        return h_s_tp, c_s_tp


class TP_lstm(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.lane_gate_size = args.lane_gate_size
        self.output_hidden_size = args.output_hidden_size
        self.output_size = 1#这里先写死
        self.t_predict = args.t_predict
        self.temporal_length = args.temporal_length
        self.spatial_length = args.spatial_length

        self.cell = MD_lstm_cell(self.input_size, self.hidden_size)
        self.lane_gate = FCNet(layerSize=[1, self.lane_gate_size, 1])
        self.output_layer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])

    def infer(self, temporal_data, init_input, lane):

        predict_input_length = temporal_data.shape[1]
        [batch_size, spatial_length, input_temporal, input_size]= init_input.shape
        
        # lane_gate
        lane_controller = self.lane_gate(lane)

        # 处理batch
        temporal_data = temporal_data.view(batch_size, -1)
        init_input = init_input.view(batch_size, -1)

        temporal_data = temporal_data * lane_controller
        init_input = init_input * lane_controller

        #还原
        temporal_data = temporal_data.view(batch_size, predict_input_length, input_size)
        init_input = init_input.view(batch_size, spatial_length, input_temporal, input_size)

        cell_state = temporal_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state = temporal_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_after = temporal_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_before = temporal_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        zero_hidden = temporal_data.data.new(batch_size, 1, self.hidden_size).fill_(0).float()
        output = temporal_data.data.new(batch_size, spatial_length, predict_input_length)

        for time in range(input_temporal):

            hidden_state, cell_state = self.cell(init_input[:, :, time, :], hidden_state, cell_state, hidden_state_after, hidden_state_before)
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)

        predict_input = init_input[:, :, -1, :]

        for time in range(predict_input_length):
            
            outflow = self.output_layer(hidden_state)
            if time > 0:
                output[:, :, time] = outflow.view(batch_size, spatial_length)
            inflow = torch.cat((temporal_data[:, time, 1].view(batch_size,1,1), outflow[:, :spatial_length-1]), 1)
            number = predict_input[:, :, 2].view(batch_size, spatial_length, 1) - outflow + inflow
            predict_input = torch.cat((outflow, inflow, number), 2)
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)
            hidden_state, cell_state = self.cell(predict_input, hidden_state, cell_state, hidden_state_after, hidden_state_before)

        return output    

    def forward(self, input_data, lane):

        [batch_size, spatial_length, temporal_length, input_size] = input_data.shape

        cell_state = input_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state = input_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_after = input_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_before = input_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        zero_hidden = input_data.data.new(batch_size, 1, self.hidden_size).fill_(0).float()
        output = input_data.data.new(batch_size, spatial_length, temporal_length-self.t_predict)

        # 处理batch
        input_data = input_data.view(batch_size, -1)

        lane_controller = self.lane_gate(lane)
        input_data = input_data * lane_controller

        #还原
        input_data = input_data.view(batch_size, spatial_length, temporal_length, input_size)

        for time in range(temporal_length):
                
            hidden_state, cell_state = self.cell(input_data[:, :, time, :], 
                    hidden_state, cell_state, hidden_state_after, hidden_state_before)
            if time >= self.t_predict:
                output[:, :, time-self.t_predict] = self.output_layer(hidden_state).view(batch_size, spatial_length)
                
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)

        return output

class loss_function(nn.Module):

    def __init__(self):

        super(loss_function, self).__init__()

        self.mes_criterion = nn.MSELoss()

    def forward(self, number_current, number_before, In, outflow):

        [spatial_size, temporal_size] = number_current.shape


        inflow = torch.cat((In, outflow[:spatial_size-1,:]))
        number_caculate = number_before + inflow - outflow

        loss = self.mes_criterion(number_current, number_caculate)

        return loss


class network(nn.Module):

    def __init__(self):
        a = 1