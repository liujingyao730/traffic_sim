import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import conf
from model import FCNet
from net_model import inter_model
from net_model import inter_LSTM
from net_model import seg_model

class sn_lstm(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.number_input_size = args["number_input_size"]
        self.speed_input_size = args["speed_input_size"]
        self.number_output_size = args["number_output_size"]
        self.speed_output_size = args["speed_output_size"]
        self.number_encoder_hidden = args["number_encoder_hidden"]
        self.speed_encoder_hidden = args["speed_encoder_hidden"]
        self.decoder_hidden = args["decoder_hidden"]

        self.t_predict = args["t_predict"]
        self.rand_range = args["rand_range"]
        self.use_cuda = args["use_cuda"]

        number_encoder_args = {"input_size": self.number_input_size, 
                            "hidden_size": self.number_encoder_hidden}
        speed_encoder_args = {"input_size": self.speed_input_size,
                            "hidden_size": self.speed_encoder_hidden}
        decoder_args = {"input_size": self.number_encoder_hidden+self.speed_encoder_hidden,
                        "hidden_size": self.decoder_hidden}
        
        self.number_encoder = seg_model(number_encoder_args)
        self.speed_encoder = seg_model(speed_encoder_args)

        self.decoder = seg_model(decoder_args)

        self.number_output_layer = nn.Linear(self.decoder_hidden, self.number_output_size)
        self.speed_output_layer = nn.Linear(self.decoder_hidden, self.speed_output_size)

    def get_spatial_hidden(self, h):

        [batch_size, spatial, hidden_size] = h.shape

        h_after = Variable(h.data.new(batch_size, spatial, hidden_size).fill_(0).float())
        h_before = Variable(h.data.new(batch_size, spatial, hidden_size).fill_(0).float())

        h_after[:, :-1, :] += h[:, 1:, :]
        h_before[:, 1:, :] += h[:, :-1, :]

        return h_after, h_before

    def caculate_delta_N(self, output_proba, init_input):

        [batch_size, temporal, spatial, output_size] = output_proba.shape
        [batch_size_i, temporal_i, _] = init_input.shape

        assert batch_size == batch_size_i and temporal == temporal_i

        input_proba = Variable(output_proba.data.new(output_proba.shape).fill_(0).float())

        input_proba[:, :, 0, :].scatter_(2, init_input, 1)
        input_proba[:, :, 1:, :] += output_proba[:, :, :-1, :]

        output_proba = output_proba.view(batch_size*temporal, 1, spatial, output_size)
        input_proba = input_proba.view(batch_size*temporal, spatial, output_size)

        output_proba = torch.nn.functional.unfold(output_proba, (spatial, output_size), padding=(0, output_size-1))
        output_proba = output_proba.view(batch_size*temporal, spatial, output_size, -1)

        delta_N = torch.einsum("bist,bis->bit", [output_proba, input_proba])
        delta_N = delta_N.view(batch_size, temporal, spatial, 2*output_size-1)

        return delta_N
    
    def sample(self, proba_list):

        sample_result = torch.max(proba_list, dim=2)[1]

        return sample_result.unsqueeze(2).float()

    def random_sample(self, proba_list):

        [batch_size, spatial, output_size] = proba_list

        proba_list = torch.nn.functional.softmax(proba_list, dim=2)

        mask = torch.rand(proba_list.shape)
        if self.use_cuda:
            mask = mask.cuda()
        mask = mask * self.rand_range

        proba_list += mask

        sample_result = torch.max(proba_list, dim=2)[1]

        return sample_result.unsqueeze(2).float()

    def caculate_next_input(self, former_input, next_input, output):

        speed_In = next_input[:, [0], 3:]
        speed_next = torch.cat((speed_In, output[:, 1:, 1:]), dim=1)

        In = torch.cat((next_input[:, 0:1, 1:2], output[:, :-1, [0]]), dim=1)
        former_number = former_input[:, :, [2]]
        number_caculate = former_number + In - output[:, :, [0]]

        next_data = torch.cat((output[:, :, [0]], In, number_caculate, speed_next), dim=2)

        return next_data

    def forward(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape

        assert self.number_input_size+self.speed_input_size == input_size

        h_speed = Variable(input_data.data.new(batch_size, spatial, self.speed_encoder_hidden).fill_(0).float())
        h_number = Variable(input_data.data.new(batch_size, spatial, self.number_encoder_hidden).fill_(0).float())
        h_decoder = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())
        c_speed = Variable(input_data.data.new(batch_size, spatial, self.speed_encoder_hidden).fill_(0).float())
        c_number = Variable(input_data.data.new(batch_size, spatial, self.number_encoder_hidden).fill_(0).float())
        c_decoder = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())
        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, self.number_input_size+self.speed_input_size).fill_(0).float())

        init_input = input_data[:, self.t_predict-1, [0], 0].long()

        for time in range(temporal - 1):

            h_speed_after, h_speed_before = self.get_spatial_hidden(h_speed)
            h_number_after, h_number_before = self.get_spatial_hidden(h_number)
            h_decoder_after, h_decoder_before = self.get_spatial_hidden(h_decoder)

            number_data = input_data[:, time, :, :3].contiguous()
            speed_data = input_data[:, time, :, 3:].contiguous()

            h_number, c_number = self.number_encoder(number_data, h_number, c_number, h_number_after, h_number_before)
            h_speed, c_speed = self.speed_encoder(speed_data, h_speed, c_speed, h_speed_after, h_speed_before)

            decoder_input = torch.cat((h_number, h_speed), dim=2)
            
            h_decoder, c_decoder = self.decoder(decoder_input, h_decoder, c_decoder, h_decoder_after, h_decoder_before)

            if time >= self.t_predict:
                number_output = self.number_output_layer(h_decoder)
                speed_output = self.speed_output_layer(h_decoder)
                output = torch.cat((number_output, speed_output), dim=2)
                data = self.caculate_next_input(
                                                number_data,
                                                input_data[:, time+1, :, :],
                                                output
                                                )
                outputs[:, time-self.t_predict, :, :] += data
        
        return outputs

    def infer(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape

        h_speed = Variable(input_data.data.new(batch_size, spatial, self.speed_encoder_hidden).fill_(0).float())
        h_number = Variable(input_data.data.new(batch_size, spatial, self.number_encoder_hidden).fill_(0).float())
        h_decoder = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())
        c_speed = Variable(input_data.data.new(batch_size, spatial, self.speed_encoder_hidden).fill_(0).float())
        c_number = Variable(input_data.data.new(batch_size, spatial, self.number_encoder_hidden).fill_(0).float())
        c_decoder = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())
        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, self.number_input_size+self.speed_input_size).fill_(0).float())

        init_input = input_data[:, self.t_predict-1, [0], 0].long()

        for time in range(temporal):

            if time <= self.t_predict:
                number_data = input_data[:, time, :, :3].contiguous()
                speed_data = input_data[:, time, :, 3:].contiguous()
            else:
                number_output = self.number_output_layer(h_decoder)
                speed_output = self.speed_output_layer(h_decoder)
                output = torch.cat((number_output, speed_output), dim=2)
                data = self.caculate_next_input(
                                                number_data,
                                                input_data[:, time, :, :],
                                                output
                                                )
                outputs[:, time-self.t_predict-1, :, :] += data
                number_data = data[:, :, :3]
                speed_data = data[:, :, 3:]

            h_speed_after, h_speed_before = self.get_spatial_hidden(h_speed)
            h_number_after, h_number_before = self.get_spatial_hidden(h_number)
            h_decoder_after, h_decoder_before = self.get_spatial_hidden(h_decoder)

            h_number, c_number = self.number_encoder(number_data, h_number, c_number, h_number_after, h_number_before)
            h_speed, c_speed = self.speed_encoder(speed_data, h_speed, c_speed, h_speed_after, h_speed_before)

            decoder_input = torch.cat((h_number, h_speed), dim=2)
            
            h_decoder, c_decoder = self.decoder(decoder_input, h_decoder, c_decoder, h_decoder_after, h_decoder_before)
        
        return outputs

if __name__ == "__main__":
    
    args = {}

    args["speed_input_size"] = 3
    args["speed_output_size"] = 3
    args["speed_encoder_hidden"] = 64
    args["number_input_size"] = 3
    args["number_encoder_hidden"] = 64
    args["number_output_size"] = 1
    args["decoder_hidden"] = 64

    args["use_cuda"] = False
    args["t_predict"] = 4
    args["rand_range"] = 0.1

    model = sn_lstm(args)

    inputs = torch.rand(5, 17, 8, 6)
    outputs = model.infer(inputs)
    fake_loss = torch.mean(outputs)
    fake_loss.backward()