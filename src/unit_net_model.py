import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import conf
from model import FCNet

class spatial_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(self.input_size, self.hidden_size)

        self.spatial_forget = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
        self.spatial_input = torch.nn.Linear(2*self.hidden_size, self.hidden_size)

        self.sigma = torch.nn.Sigmoid()

    def forward(self, inputs, hidden):

        [batch_size, input_size] = inputs.shape
        [h_current, c_current, h_after, h_before] = hidden

        spatial_hidden = torch.cat((h_after, h_before), dim=1)
        spatial_i = self.spatial_input(spatial_hidden)
        spatial_f = self.spatial_forget(spatial_hidden)
        spatial_i = self.sigma(spatial_i)
        spatial_f = self.sigma(spatial_f)

        c_current = c_current * spatial_f
        h_current = h_current * spatial_i

        h_next, c_next = self.cell(inputs, (h_current,c_current))
        
        return h_next, c_next

class uni_network_model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.output_hidden_size = args["output_hidden_size"]
        self.t_predict = args["t_predict"]
        self.output_size = 1

        self.uni_lstm = spatial_LSTM(self.input_size, self.hidden_size)
        self.output_layer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])

        self.end_resize = nn.Linear(3*self.hidden_size, self.hidden_size)
        self.inter_resize = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.inter_place = [1, 3, 4, 6]

    def get_topology(self, bucket_number, spatial):

        self.not_after_block = [0, bucket_number[1]+1, bucket_number[4], bucket_number[6]]
        self.not_before_block = [bucket_number[1], bucket_number[3], 
                            bucket_number[6]-1, bucket_number[6]]
        self.init_bucket = [0, bucket_number[1]+1]
        
        self.after_block = [i for i in range(spatial) if i not in self.not_after_block]
        self.before_block = [i for i in range(spatial) if i not in self.not_before_block]

        self.major_after_input = [bucket_number[3], bucket_number[4], bucket_number[6]]
        self.minor_after_input = [bucket_number[1], bucket_number[4], bucket_number[6]]
        self.end_before_input = [bucket_number[1], bucket_number[3], bucket_number[6]]
        self.inter_before_input = [bucket_number[1], bucket_number[3]]

    def generate_spatial_hidden(self, h, h_after, h_before):

        [batch_size, spatial, hidden_size] = h.shape
            
        #添加路段上的after和before
        h_after[:, self.before_block, :] += h[:, self.after_block, :]
        h_before[:, self.after_block, :] += h[:, self.before_block, :]

        #添加主路段的最后一个节点的后续hidden
        h_after[:, bucket_number[1], :] = self.end_resize(h[:, self.major_after_input, :].view(batch_size, -1))

        #添加次路段的最后一个节点的后续hidden
        h_after[:, bucket_number[3], :] = self.end_resize(h[:, self.minor_after_input, :].view(batch_size, -1))

        #添加后续路段的第一个节点的之前的hidden
        h_before[:, bucket_number[4], :] = self.end_resize(h[:, self.end_before_input, :].view(batch_size, -1))

        #添加inter node 的前后hidden
        h_after[:, bucket_number[6], :] = h[:, bucket_number[4], :]
        h_before[:, bucket_number[6], :] = self.inter_resize(h[:, self.inter_before_input, :].view(batch_size, -1))

        return h_after, h_before

    def caculate_next_input(self, former_input, next_input, output):

        [batch_size, spatial, _] = former_input.shape

        In = Variable(former_input.data.new(batch_size, spatial, 1).fill_(0).float())

        number_former = former_input[:, :, 2].unsqueeze(2)
        In[:, self.after_block, 0] += output[:, self.before_block, 0]
        In[:, self.init_bucket, 0] += next_input[:, self.init_bucket, 1]
        In[:, bucket_number[6], 0] += output[:, bucket_number[1], 0] + output[:, bucket_number[3], 0]
        In[:, bucket_number[4], 0] += output[:, bucket_number[6], 0]
        number_caculate = number_former + In - output

        input_caculate = torch.cat((output, In, number_caculate), dim=2)

        return input_caculate


    def forward(self, inputs, bucket_number):
        
        [batch_size, temporal, spatial, input_size] = inputs.shape

        self.get_topology(bucket_number, spatial)

        h = Variable(inputs.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(inputs.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_after = Variable(inputs.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_before = Variable(inputs.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_output = Variable(inputs.data.new(batch_size, temporal-self.t_predict, spatial, self.hidden_size).fill_(0).float())

        for time in range(temporal):

            h = h.view(batch_size*spatial, self.hidden_size)
            c = c.view(batch_size*spatial, self.hidden_size)
            h_after = h_after.view(batch_size*spatial, self.hidden_size)
            h_before = h_before.view(batch_size*spatial, self.hidden_size)

            h, c = self.uni_lstm(inputs[:, time, :, :].contiguous().view(-1, self.input_size),
                                             (h, c, h_after, h_before))

            h = h.view(batch_size, spatial, self.hidden_size)
            c = c.view(batch_size, spatial, self.hidden_size)
            h_after = h_after.view(batch_size, spatial, self.hidden_size)
            h_before = h_before.view(batch_size, spatial, self.hidden_size)
            
            h_after, h_before = self.generate_spatial_hidden(h, h_after, h_before)

            if time > self.t_predict:
                h_output[:, time - self.t_predict, :, :] += h
        
        output = self.output_layer(h_output)

        return output

    def infer(self, inputs, bucket_number):

        [batch_size, temporal, spatial, input_size] = inputs.shape
        temporal -= 1

        self.get_topology(bucket_number, spatial)

        h = Variable(inputs.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(inputs.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_after = Variable(inputs.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_before = Variable(inputs.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_output = Variable(inputs.data.new(batch_size, temporal-self.t_predict, spatial, self.hidden_size).fill_(0).float())
        outputs = Variable(inputs.data.new(batch_size, temporal-self.t_predict, spatial, 3).fill_(0).float())

        for time in range(self.t_predict):
            
            h = h.view(batch_size*spatial, self.hidden_size)
            c = c.view(batch_size*spatial, self.hidden_size)
            h_after = h_after.view(batch_size*spatial, self.hidden_size)
            h_before = h_before.view(batch_size*spatial, self.hidden_size)

            h, c = self.uni_lstm(inputs[:, time, :, :].contiguous().view(-1, self.input_size),
                                             (h, c, h_after, h_before))

            h = h.view(batch_size, spatial, self.hidden_size)
            c = c.view(batch_size, spatial, self.hidden_size)
            h_after = h_after.view(batch_size, spatial, self.hidden_size)
            h_before = h_before.view(batch_size, spatial, self.hidden_size)
            
            h_after, h_before = self.generate_spatial_hidden(h, h_after, h_before)

        input_data = inputs[:, self.t_predict, :, :].contiguous()

        for time in range(temporal - self.t_predict):

            h = h.view(batch_size*spatial, self.hidden_size)
            c = c.view(batch_size*spatial, self.hidden_size)
            h_after = h_after.view(batch_size*spatial, self.hidden_size)
            h_before = h_before.view(batch_size*spatial, self.hidden_size)
            input_data = input_data.view(batch_size*spatial, self.input_size)

            h, c = self.uni_lstm(input_data, (h, c, h_after, h_before))

            h = h.view(batch_size, spatial, self.hidden_size)
            c = c.view(batch_size, spatial, self.hidden_size)
            h_after = h_after.view(batch_size, spatial, self.hidden_size)
            h_before = h_before.view(batch_size, spatial, self.hidden_size)
            input_data = input_data.view(batch_size, spatial, self.input_size)
            
            output = self.output_layer(h)
            h_after, h_before = self.generate_spatial_hidden(h, h_after, h_before)

            input_data = self.caculate_next_input(input_data, 
                                                inputs[:, time+self.t_predict+1, :, :], output)
            
            h_output[:, time, :, :] += h
            outputs[:, time, :, :] += input_data

        return outputs, h_output


if __name__ == "__main__":
    args = {}
    args["input_size"] = 3
    args["hidden_size"] = 64
    args["t_predict"] = 4
    args["output_hidden_size"] = 16
    inputs = Variable(torch.rand(10, 9, 21, 3))
    bucket_number = [10, 11, 14, 15, 16, 17, 20]
    model = uni_network_model(args)

    output, _ = model.infer(inputs, bucket_number)
    loss = torch.mean(output)
    loss.backward()