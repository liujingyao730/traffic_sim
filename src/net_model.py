import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import conf
from model import FCNet

major_order = [1, 0, 3, 4, 6]
minor_order = [3, 2, 1, 4, 6]
end_order = [4, 5, 1, 3, 6]
inter_order = [6, 6, 1, 3, 4]

class inter_cell(nn.Module):

    def __init__(self, input_size, hidden_size):
        
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(self.input_size, self.hidden_size)

        self.sigma = torch.nn.Sigmoid()

        self.sptail_forget = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
        self.sptail_input = torch.nn.Linear(2*self.hidden_size, self.hidden_size)

        self.inter_layer = torch.nn.Linear(3*self.hidden_size, self.hidden_size)

    def forward(self, inputs, hidden, cell_state):

        batch_size = hidden.shape[0]
        
        self_hidden = hidden[:, 0, :]
        seg_hidden = hidden[:, 1, :]
        inter_hidden = hidden[:, 2:, :].view(batch_size, self.hidden_size*3)

        inter_hidden = self.inter_layer(inter_hidden)
        
        spatial_hidden = torch.cat((seg_hidden, inter_hidden), 1)

        spatial_f = self.sptail_forget(spatial_hidden)
        spatial_f = self.sigma(spatial_f)
        spatial_i = self.sptail_input(spatial_hidden)
        spatial_i = self.sigma(spatial_i)

        cell_state = cell_state * spatial_f
        self_hidden = self_hidden * spatial_i

        h, c = self.cell(inputs, (self_hidden, cell_state))

        return h, c

class inter_LSTM(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.n_units = args["n_units"]

        self.major_cell = inter_cell(self.input_size, self.hidden_size)
        self.minor_cell = inter_cell(self.input_size, self.hidden_size)
        self.end_cell = inter_cell(self.input_size, self.hidden_size)
        self.inter_node = inter_cell(self.input_size, self.hidden_size)

    def forward(self, inputs, hidden):

        [batch_size, n_units, input_feature] = inputs.shape

        assert n_units == self.n_units

        [h, c] = hidden

        major_input = inputs[:, 1, :]
        major_h = h[:, major_order, :]
        major_c = c[:, 1, :]
        major_h, major_c = self.major_cell(major_input, major_h, major_c)

        minor_input = inputs[:, 3, :]
        minor_h = h[:, minor_order, :]
        minor_c = c[:, 3, :]
        minor_h, minor_c = self.minor_cell(minor_input, minor_h, minor_c)

        end_input = inputs[:, 4, :]
        end_h = h[:, end_order, :]
        end_c = c[:, 4, :]
        end_h, end_c = self.end_cell(end_input, end_h, end_c)

        inter_input = inputs[:, 6, :]
        inter_h = h[:, inter_order, :]
        inter_c = c[:, 6, :]
        inter_h, inter_c = self.inter_node(inter_input, inter_h, inter_c)

        h = h * 0
        c = c * 0

        h[:, 1, :] += major_h
        h[:, 3, :] += minor_h
        h[:, 4, :] += end_h
        h[:, 6, :] += inter_h

        c[:, 1, :] += major_c
        c[:, 3, :] += minor_c
        c[:, 4, :] += end_c
        c[:, 6, :] += inter_c

        return h, c

class inter_model(nn.Module):
    
    #直接拼接式的路口模型
    
    def __init__(self, args):
        super().__init__()

        self.n_unit = args["n_unit"]
        self.input_feature = args["input_size"]
        self.hidden_size = args["hidden_size"]

        self.cell = nn.LSTMCell(self.input_feature*self.n_unit, self.hidden_size*self.n_unit)        

    def forward(self, inputs, hidden):
        
        [batch_size, n_unit, input_feature] = inputs.shape

        assert n_unit == self.n_unit and input_feature == self.input_feature

        [h_0, c_0] = hidden

        inputs = inputs.view(batch_size, n_unit*input_feature)
        h_0 = h_0.view(batch_size, n_unit*self.hidden_size)
        c_0 = c_0.view(batch_size, n_unit*self.hidden_size)

        h, c = self.cell(inputs, (h_0, c_0))

        c = c.view(batch_size, n_unit, self.hidden_size)
        h = h.view(batch_size, n_unit, self.hidden_size)
        
        return h, c


class seg_model(nn.Module):
    '''使用到的路段模型
    '''

    def __init__(self, args):
        
        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]

        self.cell = torch.nn.LSTMCell(self.input_size, self.hidden_size)
        
        self.sigma = torch.nn.Sigmoid()

        self.spatial_forget = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
        self.spatial_input = torch.nn.Linear(2*self.hidden_size, self.hidden_size)

    def forward(self, inputs, h_s_t, c_s_t, h_after_t, h_before_t):
        
        [batch_size, input_size] = inputs.shape
        [batch_size, hidden_size] = h_s_t.shape

        assert input_size == self.input_size

        spatial_hidden = torch.cat((h_after_t, h_before_t), dim=1)
        spatial_i = self.spatial_input(spatial_hidden)
        spatial_f = self.spatial_forget(spatial_hidden)
        spatial_i = self.sigma(spatial_i)
        spatial_f = self.sigma(spatial_f)

        c_s_t = c_s_t * spatial_f
        h_s_t = h_s_t * spatial_i

        h_s_tp, c_s_tp = self.cell(inputs, (h_s_t, c_s_t))

        return h_s_tp, c_s_tp


class network_model(nn.Module):

    def __init__(self, args):
        
        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.output_hidden_size = args["output_hidden_size"]
        self.t_predict = args["t_predict"]
        self.output_size = 1

        self.segment_model = seg_model(args)
        self.intersection_model = inter_model(args)
        self.outputlayer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])

    def forward(self, co_data, bucket_number):

        [batch_size, temporal, spatial, input_size] = co_data.shape

        inter_place = [1, 3, 4, 6]
        inter_bucket = [bucket_number[1], bucket_number[3], bucket_number[4], bucket_number[6]]
        seg_bucket = [i for i in range(spatial) if i not in inter_bucket]
        seg_bucket_number = len(seg_bucket) * batch_size
        non_after_bucket = [0, bucket_number[1]+1, bucket_number[4], bucket_number[5], bucket_number[6]]
        non_before_bucket = [bucket_number[0], bucket_number[1], 
                            bucket_number[2], bucket_number[3], 
                            bucket_number[6]-1, bucket_number[6]]
        h_after_in_all = [i for i in range(spatial) if i not in non_after_bucket]
        h_before_in_all = [i for i in range(spatial) if i not in non_before_bucket]
        h_after_in_seg = list(range(len(seg_bucket) - 1))
        h_before_in_seg = [i for i in range(1, len(seg_bucket)) if not i == bucket_number[1]]

        seg_data = co_data[:, :, seg_bucket, :]
        inter_data = co_data[:, :, bucket_number, :]
        
        h_inter = Variable(co_data.data.new(batch_size, len(bucket_number), self.hidden_size).fill_(0).float())
        c_inter = Variable(co_data.data.new(batch_size, len(bucket_number), self.hidden_size).fill_(0).float())
        h_seg = Variable(co_data.data.new(batch_size, len(seg_bucket), self.hidden_size).fill_(0).float())
        c_seg = Variable(co_data.data.new(batch_size, len(seg_bucket), self.hidden_size).fill_(0).float())
        h_s_t = Variable(co_data.data.new(seg_bucket_number, self.hidden_size).fill_(0).float())
        c_s_t = Variable(co_data.data.new(seg_bucket_number, self.hidden_size).fill_(0).float())
        h_after_t = Variable(co_data.data.new(seg_bucket_number, self.hidden_size).fill_(0).float())
        h_before_t = Variable(co_data.data.new(seg_bucket_number, self.hidden_size).fill_(0).float())

        h_tmp = Variable(co_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c_tmp = Variable(co_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_output = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, self.hidden_size).fill_(0).float())

        for time in range(temporal):

            h_s_t = h_s_t.view(seg_bucket_number, self.hidden_size)
            c_s_t = c_s_t.view(seg_bucket_number, self.hidden_size)
            h_seg = h_seg.view(seg_bucket_number, self.hidden_size)
            c_seg = c_seg.view(seg_bucket_number, self.hidden_size)
            h_after_t = h_after_t.view(seg_bucket_number, self.hidden_size)
            h_before_t = h_before_t.view(seg_bucket_number, self.hidden_size)

            h_inter, c_inter = self.intersection_model(inter_data[:, time, :, :], (h_inter, c_inter))
            h_seg, c_seg = self.segment_model(seg_data[:, time, :, :].contiguous().view(seg_bucket_number, input_size), h_s_t, c_s_t, h_after_t, h_before_t)

            h_s_t = h_s_t.view(batch_size, len(seg_bucket), self.hidden_size)
            c_s_t = c_s_t.view(batch_size, len(seg_bucket), self.hidden_size)
            h_seg = h_seg.view(batch_size, len(seg_bucket), self.hidden_size)
            c_seg = c_seg.view(batch_size, len(seg_bucket), self.hidden_size)
            h_after_t = h_after_t.view(batch_size, len(seg_bucket), self.hidden_size)
            h_before_t = h_before_t.view(batch_size, len(seg_bucket), self.hidden_size)

            h_tmp[:, inter_bucket, :] += h_inter[:, inter_place, :]
            h_tmp[:, seg_bucket, :] += h_seg
            c_tmp[:, inter_bucket, :] += c_inter[:, inter_place, :]
            c_tmp[:, seg_bucket, :] += c_seg

            h_inter = h_inter * 0
            c_inter = c_inter * 0
            h_inter += h_tmp[:, bucket_number, :]
            c_inter += c_tmp[:, bucket_number, :]

            h_s_t = h_s_t * 0
            h_s_t += h_tmp[:, seg_bucket, :]
            c_s_t = c_s_t * 0
            c_s_t += c_tmp[:, seg_bucket, :]

            h_after_t = h_after_t * 0
            h_before_t = h_before_t * 0
            h_after_t[:, h_after_in_seg, :] += h_tmp[:, h_after_in_all, :]
            h_before_t[:, h_before_in_seg, :] += h_tmp[:, h_before_in_all, :]

            if time >= self.t_predict:
                h_output[:, time-self.t_predict, :, :] += h_tmp

            h_tmp = h_tmp * 0
            c_tmp = c_tmp * 0

        output = self.outputlayer(h_output)

        return output

    def infer(self, co_data, bucket_number, hidden=None, topology_struct=None):

        [batch_size, spatial, input_size] = co_data.shape

        if topology_struct is None:
            inter_place = [1, 3, 4, 6]
            inter_bucket = [bucket_number[1], bucket_number[3], bucket_number[4], bucket_number[6]]
            seg_bucket = [i for i in range(spatial) if i not in inter_bucket]
            seg_bucket_number = len(seg_bucket) * batch_size
            non_after_bucket = [0, bucket_number[1]+1, bucket_number[4], bucket_number[5], bucket_number[6]]
            non_before_bucket = [bucket_number[0], bucket_number[1], 
                                bucket_number[2], bucket_number[3], 
                                bucket_number[6]-1, bucket_number[6]]
            h_after_in_all = [i for i in range(spatial) if i not in non_after_bucket]
            h_before_in_all = [i for i in range(spatial) if i not in non_before_bucket]
            h_after_in_seg = list(range(len(seg_bucket) - 1))
            h_before_in_seg = [i for i in range(1, len(seg_bucket)) if not i == bucket_number[1]]

            topology_struct = [inter_bucket, inter_place, seg_bucket, seg_bucket_number, 
                                h_after_in_all, h_after_in_seg, h_before_in_all, h_before_in_seg]
        else:

            inter_bucket = topology_struct[0]
            inter_place = topology_struct[1]
            seg_bucket = topology_struct[2]
            seg_bucket_number = topology_struct[3]
            h_after_in_all = topology_struct[4]
            h_after_in_seg = topology_struct[5]
            h_before_in_all = topology_struct[6]
            h_before_in_seg = topology_struct[7]

        inter_data = co_data[:, bucket_number, :]
        seg_data = co_data[:, seg_bucket, :]

        h_inter = Variable(co_data.data.new(batch_size, len(bucket_number), self.hidden_size).fill_(0).float())
        c_inter = Variable(co_data.data.new(batch_size, len(bucket_number), self.hidden_size).fill_(0).float())
        h_seg = Variable(co_data.data.new(batch_size, len(seg_bucket), self.hidden_size).fill_(0).float())
        c_seg = Variable(co_data.data.new(batch_size, len(seg_bucket), self.hidden_size).fill_(0).float())
        h_s_t = Variable(co_data.data.new(batch_size, len(seg_bucket), self.hidden_size).fill_(0).float())
        c_s_t = Variable(co_data.data.new(batch_size, len(seg_bucket), self.hidden_size).fill_(0).float())
        h_after_t = Variable(co_data.data.new(batch_size, len(seg_bucket), self.hidden_size).fill_(0).float())
        h_before_t = Variable(co_data.data.new(batch_size, len(seg_bucket), self.hidden_size).fill_(0).float())

        if hidden[0] is None:
            h_all = Variable(co_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
            c_all = Variable(co_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        else:
            h_all = hidden[0]
            c_all = hidden[1]

        h_inter = h_inter * 0
        c_inter = c_inter * 0
        h_inter += h_all[:, bucket_number, :]
        c_inter += c_all[:, bucket_number, :]

        h_s_t = h_s_t * 0
        h_s_t += h_all[:, seg_bucket, :]
        c_s_t = c_s_t * 0
        c_s_t += c_all[:, seg_bucket, :]

        h_after_t = h_after_t * 0
        h_before_t = h_before_t * 0
        h_after_t[:, h_after_in_seg, :] += h_all[:, h_after_in_all, :]
        h_before_t[:, h_before_in_seg, :] += h_all[:, h_before_in_all, :]   

        h_s_t = h_s_t.view(seg_bucket_number, self.hidden_size)
        c_s_t = c_s_t.view(seg_bucket_number, self.hidden_size)
        h_seg = h_seg.view(seg_bucket_number, self.hidden_size)
        c_seg = c_seg.view(seg_bucket_number, self.hidden_size)
        h_after_t = h_after_t.view(seg_bucket_number, self.hidden_size)
        h_before_t = h_before_t.view(seg_bucket_number, self.hidden_size)

        h_inter, c_inter = self.intersection_model(inter_data, (h_inter, c_inter))
        h_seg, c_seg = self.segment_model(seg_data.contiguous().view(seg_bucket_number, input_size), h_s_t, c_s_t, h_after_t, h_before_t)

        h_s_t = h_s_t.view(batch_size, len(seg_bucket), self.hidden_size)
        c_s_t = c_s_t.view(batch_size, len(seg_bucket), self.hidden_size)
        h_seg = h_seg.view(batch_size, len(seg_bucket), self.hidden_size)
        c_seg = c_seg.view(batch_size, len(seg_bucket), self.hidden_size)
        h_after_t = h_after_t.view(batch_size, len(seg_bucket), self.hidden_size)
        h_before_t = h_before_t.view(batch_size, len(seg_bucket), self.hidden_size)

        h_all = h_all * 0
        c_all = c_all * 0
        h_all[:, inter_bucket, :] += h_inter[:, inter_place, :]
        h_all[:, seg_bucket, :] += h_seg
        c_all[:, inter_bucket, :] += c_inter[:, inter_place, :]
        c_all[:, seg_bucket, :] += c_seg

        outputs = self.outputlayer(h_all)
        hidden = [h_all, c_all]

        return outputs, hidden, topology_struct


if __name__ == "__main__":
    
    args = {}
    
    args["n_unit"] = 7
    args["input_size"] = 3
    args["hidden_size"] = 64
    args["output_hidden_size"] = 16
    args["t_predict"] = 4
    model = network_model(args)

    