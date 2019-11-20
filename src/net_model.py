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

        self.n_unit = args["n_units"]
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
        self.inter_place = [1, 3, 4, 6]

        self.segment_model = seg_model(args)
        self.intersection_model = inter_model(args)
        #self.interoutputlayer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])
        #self.segoutputlayer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])
        self.outputlayer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])

    def get_topoloy(self, bucket_number, spatial):

        self.bucket_number = bucket_number
        self.spatial = spatial
        self.seg_input = list(range(spatial - 1))

        self.not_after_block = [0, bucket_number[1]+1, bucket_number[4], 
                                    bucket_number[5], bucket_number[6]]
        self.not_before_block = [bucket_number[0], bucket_number[1],
                                bucket_number[2], bucket_number[3],
                                bucket_number[6]-1, bucket_number[6]]
        self.not_after_seg_block = [0, bucket_number[1]+1, bucket_number[4], bucket_number[6]]
        self.not_before_seg_block = [bucket_number[1], bucket_number[3],
                                bucket_number[6]-1, bucket_number[6]]
        self.init_block = [0, bucket_number[1]+1]

        self.inter_block = [bucket_number[i] for i in self.inter_place]
        self.seg_block = [i for i in range(spatial) if i not in self.inter_block]
        self.len_inter_block = len(self.inter_block)
        self.seg_block_number = len(self.seg_block)
        self.inter_block_number = len(self.bucket_number)

        self.after_block = [i for i in range(spatial) if i not in self.not_after_seg_block]
        self.before_block = [i for i in range(spatial) if i not in self.not_before_seg_block]

        self.before_in_seg = [i for i in range(1, self.seg_block_number) if not i == bucket_number[1]]
        self.after_in_seg = list(range(self.seg_block_number - 1))
        self.after_in_all = [i for i in range(spatial) if i not in self.not_after_block]
        self.before_in_all = [i for i in range(spatial) if i not in self.not_before_block]

    def collect_spatial_hidden(self, h_inter, c_inter, h_seg, c_seg):

        batch_size = h_inter.shape[0]

        h_tmp = Variable(h_inter.data.new(batch_size, self.spatial, self.hidden_size).fill_(0).float())
        c_tmp = Variable(h_inter.data.new(batch_size, self.spatial, self.hidden_size).fill_(0).float())

        h_tmp[:, self.seg_block, :] += h_seg
        c_tmp[:, self.seg_block, :] += c_seg
        h_tmp[:, self.inter_block, :] += h_inter[:, self.inter_place, :]
        c_tmp[:, self.inter_block, :] += c_inter[:, self.inter_place, :]

        return h_tmp, c_tmp

    def distribute_spatial_hidden(self, h_tmp, c_tmp):

        [batch_size, spatial, _] = h_tmp.shape

        h_seg = Variable(h_tmp.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        c_seg = Variable(h_tmp.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        h_after = Variable(h_tmp.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        h_before = Variable(h_tmp.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        h_inter = Variable(h_tmp.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        c_inter = Variable(h_tmp.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())

        h_seg += h_tmp[:, self.seg_block, :]
        c_seg += c_tmp[:, self.seg_block, :]
        h_inter += h_tmp[:, self.bucket_number, :]
        c_inter += c_tmp[:, self.bucket_number, :]

        h_after[:, self.after_in_seg, :] += h_tmp[:, self.after_in_all, :]
        h_before[:, self.before_in_seg, :] += h_tmp[:, self.before_in_all, :]

        return h_seg, c_seg, h_after, h_before, h_inter, c_inter

    def caculate_next_input(self, former_input, next_input, output):

        [batch_size, spatial, _] = former_input.shape

        In = Variable(former_input.data.new(batch_size, spatial, 1).fill_(0).float())
        number_former = former_input[:, :, 2].unsqueeze(2)

        In[:, self.after_block, 0] += output[:, self.before_block, 0]
        In[:, self.init_block, 0] += next_input[:, self.init_block, 1]
        In[:, self.bucket_number[6], 0] += output[:, self.bucket_number[1], 0] + output[:, self.bucket_number[3], 0]
        In[:, self.bucket_number[4], 0] += output[:, self.bucket_number[6], 0]
        number_caculate = number_former + In - output

        input_caculate = torch.cat((output, In, number_caculate), dim=2)

        return input_caculate

    def forward(self, co_data, bucket_number):

        [batch_size, temporal, spatial, input_size] = co_data.shape
        temporal -= 1

        self.get_topoloy(bucket_number, spatial)

        seg_data = co_data[:, :, self.seg_block, :]
        inter_data = co_data[:, :, bucket_number, :]
        
        h_inter = Variable(co_data.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        c_inter = Variable(co_data.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        h_seg = Variable(co_data.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        c_seg = Variable(co_data.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        h_after = Variable(co_data.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        h_before = Variable(co_data.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())

        outputs = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, 3).fill_(0).float())
        h_outputs = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, self.hidden_size).fill_(0).float())

        for time in range(temporal):

            h_seg = h_seg.view(batch_size*self.seg_block_number, self.hidden_size)
            c_seg = c_seg.view(batch_size*self.seg_block_number, self.hidden_size)
            h_after = h_after.view(batch_size*self.seg_block_number, self.hidden_size)
            h_before = h_before.view(batch_size*self.seg_block_number, self.hidden_size)

            h_seg, c_seg = self.segment_model(seg_data[:, time, :, :].contiguous().view(-1, self.input_size),
                                            h_seg, c_seg, h_after, h_before)
            h_inter, c_inter = self.intersection_model(inter_data[:, time, :, :], (h_inter, c_inter))

            h_seg = h_seg.view(batch_size, self.seg_block_number, self.hidden_size)
            c_seg = c_seg.view(batch_size, self.seg_block_number, self.hidden_size)
            h_after = h_after.view(batch_size, self.seg_block_number, self.hidden_size)
            h_before = h_before.view(batch_size, self.seg_block_number, self.hidden_size)

            h_tmp, c_tmp = self.collect_spatial_hidden(h_inter, c_inter, h_seg, c_seg)
            h_seg, c_seg, h_after, h_before, h_inter, c_inter = self.distribute_spatial_hidden(
                                                                                h_tmp, c_tmp,
                                                                                )

            if time > self.t_predict:
                h_outputs[:, time-self.t_predict, :, :] += h_tmp
                output = self.outputlayer(h_tmp)
                outputs[:, time-self.t_predict, :, :] += self.caculate_next_input(
                                                            co_data[:, time, :, :],
                                                            co_data[:, time+1, :, :],
                                                            output,                  
                                                            )

        return outputs, h_outputs
    
    def infer(self, co_data, bucket_number):

        [batch_size, temporal, spatial, input_size] = co_data.shape
        temporal -= 1

        self.get_topoloy(bucket_number, spatial)

        seg_data = co_data[:, :, self.seg_block, :]
        inter_data = co_data[:, :, bucket_number, :]
        
        h_inter = Variable(co_data.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        c_inter = Variable(co_data.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        h_seg = Variable(co_data.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        c_seg = Variable(co_data.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        h_after = Variable(co_data.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())
        h_before = Variable(co_data.data.new(batch_size, self.seg_block_number, self.hidden_size).fill_(0).float())

        outputs = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, 3).fill_(0).float())
        h_outputs = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, self.hidden_size).fill_(0).float())

        for time in range(self.t_predict):

            h_seg = h_seg.view(batch_size*self.seg_block_number, self.hidden_size)
            c_seg = c_seg.view(batch_size*self.seg_block_number, self.hidden_size)
            h_after = h_after.view(batch_size*self.seg_block_number, self.hidden_size)
            h_before = h_before.view(batch_size*self.seg_block_number, self.hidden_size)

            h_seg, c_seg = self.segment_model(seg_data[:, time, :, :].contiguous().view(-1, self.input_size),
                                            h_seg, c_seg, h_after, h_before)
            h_inter, c_inter = self.intersection_model(inter_data[:, time, :, :], (h_inter, c_inter))

            h_seg = h_seg.view(batch_size, self.seg_block_number, self.hidden_size)
            c_seg = c_seg.view(batch_size, self.seg_block_number, self.hidden_size)
            h_after = h_after.view(batch_size, self.seg_block_number, self.hidden_size)
            h_before = h_before.view(batch_size, self.seg_block_number, self.hidden_size)

            h_tmp, c_tmp = self.collect_spatial_hidden(h_inter, c_inter, h_seg, c_seg)
            h_seg, c_seg, h_after, h_before, h_inter, c_inter = self.distribute_spatial_hidden(
                                                                                        h_tmp, c_tmp,
                                                                                        )

        input_data = co_data[:, self.t_predict, :, :]

        for time in range(temporal - self.t_predict):

            seg_input_data = input_data[:, self.seg_block, :].contiguous()
            inter_input_data = input_data[:, bucket_number, :]

            h_seg = h_seg.view(batch_size*self.seg_block_number, self.hidden_size)
            c_seg = c_seg.view(batch_size*self.seg_block_number, self.hidden_size)
            h_after = h_after.view(batch_size*self.seg_block_number, self.hidden_size)
            h_before = h_before.view(batch_size*self.seg_block_number, self.hidden_size)

            h_seg, c_seg = self.segment_model(seg_input_data.view(-1, self.input_size),
                                            h_seg, c_seg, h_after, h_before)
            h_inter, c_inter = self.intersection_model(inter_input_data, (h_inter, c_inter))

            h_seg = h_seg.view(batch_size, self.seg_block_number, self.hidden_size)
            c_seg = c_seg.view(batch_size, self.seg_block_number, self.hidden_size)
            h_after = h_after.view(batch_size, self.seg_block_number, self.hidden_size)
            h_before = h_before.view(batch_size, self.seg_block_number, self.hidden_size)

            h_tmp, c_tmp = self.collect_spatial_hidden(h_inter, c_inter, h_seg, c_seg)
            output = self.outputlayer(h_tmp)
            h_seg, c_seg, h_after, h_before, h_inter, c_inter = self.distribute_spatial_hidden(h_tmp, c_tmp)

            input_data = self.caculate_next_input(
                                    input_data,
                                    co_data[:, time+1+self.t_predict, :, :],
                                    output
                                    )

            h_outputs[:, time, :, :] += h_tmp
            outputs[:, time, :, :] += input_data

        return outputs, h_outputs


if __name__ == "__main__":
    
    args = {}
    
    args["n_unit"] = 7
    args["input_size"] = 3
    args["hidden_size"] = 64
    args["output_hidden_size"] = 16
    args["t_predict"] = 4
    args["n_units"] = 7
    model = network_model(args)

    co_data = Variable(torch.randn(11, 9, 21, 3))
    bucket_number = [10, 11, 14, 15, 16, 17, 20]

    output, _ = model.infer(co_data, bucket_number) 
    fake_loss = torch.mean(output)
    fake_loss.backward()
    