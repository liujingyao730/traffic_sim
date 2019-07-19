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
         

    def forward(self, inputs, h_s_t, c_s_t, h_sp_t, h_sm_t):

        spatial_gate = torch.cat((h_sp_t, h_sm_t), dim=1)
        spatial_gate = self.spatial_embedding(spatial_gate)
        spatial_gate = self.sigma(spatial_gate)

        h_hat, c_s_tp = self.cell(inputs, (h_s_t, c_s_t))
        
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
        self.output_size = 3#这里先写死
        self.t_predict = args.t_predict
        self.temporal_length = args.temporal_length
        self.spatial_length = args.spatial_length

        self.cell = MD_lstm_cell(self.input_size, self.hidden_size)
        self.lane_gate = FCNet(layerSize=[1, self.lane_gate_size, 1])
        self.output_layer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])

    def infer(self, temporal_data, init_input, lane):

        temporal_length = self.temporal_length - 1
        spatial_length = self.spatial_length
        
        lane_controller = self.lane_gate(lane)
        temporal_data = temporal_data * lane_controller
        init_input = init_input * lane_controller

        cell_state = temporal_data.data.new(spatial_length, self.hidden_size).fill_(0).float()
        hidden_state = temporal_data.data.new(spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_sp = temporal_data.data.new(spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_mp = temporal_data.data.new(spatial_length, self.hidden_size).fill_(0).float()
        zero_hidden = temporal_data.data.new(1, self.hidden_size).fill_(0).float()
        output = temporal_data.data.new(spatial_length, temporal_length-self.t_predict+1, self.output_size)

        hidden_state, cell_state = self.cell(init_input, hidden_state, cell_state, hidden_state_sp, hidden_state_sm)

        for time in range(temporal_length):
            
            init_input = self.output_layer(hidden_state)
            if time >= self.t_predict-1:
                output[:, time-self.t_predict, :] = init_input
            input_of_first_block = temporal_data[time, :].unsqueeze(0)
            init_input = torch.cat((input_of_first_block, init_input[1:, :]))
            hidden_state_sp = torch.cat((hidden_state[:spatial_length-1, :], zero_hidden))
            hidden_state_sm = torch.cat((zero_hidden, hidden_state[1:, :]))
            hidden_state, cell_state = self.cell(init_input, hidden_state, cell_state, hidden_state_sp, hidden_state_sm)

        return output    

    def forward(self, input_data, lane):

        [spatial_length, temporal_length, input_size] = input_data.shape

        cell_state = input_data.data.new(spatial_length, self.hidden_size).fill_(0).float()
        hidden_state = input_data.data.new(spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_sp = input_data.data.new(spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_sm = input_data.data.new(spatial_length, self.hidden_size).fill_(0).float()
        zero_hidden = input_data.data.new(1, self.hidden_size).fill_(0).float()
        output = input_data.data.new(spatial_length, temporal_length-self.t_predict, self.output_size)

        lane_controller = self.lane_gate(lane)
        input_data = input_data * lane_controller

        for time in range(temporal_length):
                
            hidden_state, cell_state = self.cell(input_data[:, time, :], 
                    hidden_state, cell_state, hidden_state_sp, hidden_state_sm)
            if time >= self.t_predict:
                output[:, time-self.t_predict, :] = self.output_layer(hidden_state)
                
            hidden_state_sp = torch.cat((hidden_state[:spatial_length-1, :], zero_hidden))
            hidden_state_sm = torch.cat((zero_hidden, hidden_state[1:, :]))

        return output

class loss_function(nn.Module):

    def __init__(self, alpha, beta):

        super(loss_function, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.mes_criterion = nn.MSELoss()

    def forward(self, target, result):

        [spatial_size, temporal_size, input_size] = target.shape

        temporal_number_ahead = result[:, 1:, 2]
        temporal_number_before = result[:, :temporal_size-1, 2]
        temporal_in_before = result[:, :temporal_size-1, 1]
        temporal_out_before = result[:, :temporal_size-1, 0]
        caculate_temport_reslut = temporal_number_before + temporal_in_before - temporal_out_before

        spatial_in = result[1:, :, 1]
        spatial_out = result[:spatial_size-1, :, 0]

        spatial_match_loss = self.mes_criterion(spatial_in, spatial_out)
        temporal_match_loss = self.mes_criterion(temporal_number_ahead, caculate_temport_reslut)
        mes_loss = self.mes_criterion(target, result)

        loss = mes_loss + self.alpha * spatial_match_loss + self.beta * temporal_match_loss

        return loss
