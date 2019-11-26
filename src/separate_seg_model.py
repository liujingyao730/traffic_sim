import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import conf
from model import FCNet
from net_model import inter_model
from net_model import inter_LSTM
from net_model import seg_model

class sp_network_model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.output_hidden_size = args["output_hidden_size"]
        self.t_predict = args["t_predict"]
        self.output_size = 1
        self.inter_place = [1, 3, 4, 6]

        self.major_seg_model = seg_model(args)
        self.minor_seg_model = seg_model(args)
        self.end_seg_model = seg_model(args)
        self.intersectio_model = inter_LSTM(args)

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

        self.major_seg_block = list(range(bucket_number[1]))
        self.minor_seg_block = list(range(bucket_number[1]+1, bucket_number[3]))
        self.end_seg_block = list(range(bucket_number[5], bucket_number[6]))
        self.major_block_number = len(self.major_seg_block)
        self.minor_block_number = len(self.minor_seg_block)
        self.end_block_number = len(self.end_seg_block)

        self.major_after_all = list(range(1, self.bucket_number[1]+1))
        self.major_before_all = list(range(self.bucket_number[0]))
        self.minor_after_all = list(range(bucket_number[1]+2, bucket_number[3]+1))
        self.minor_before_all = list(range(bucket_number[1]+1, bucket_number[2]))
        self.end_after_all = list(range(bucket_number[5]+1, bucket_number[6]))
        self.end_before_all = list(range(bucket_number[4], bucket_number[6]-1))
        
        self.major_before_seg = list(range(1, self.major_block_number))
        self.minor_before_seg = list(range(1, self.minor_block_number))
        self.end_after_seg = list(range(self.end_block_number-1))

    def caculate_next_input(self, former_input, next_input, output):

        [batch_size, spatial, _] = former_input.shape

        In = Variable(former_input.data.new(batch_size, spatial, 1).fill_(0).float())
        number_former = former_input[:, :, 2].unsqueeze(2)

        In[:, self.after_block, 0] += output[:, self.before_block, 0]
        In[:, self.init_block, 0] += next_input[:, self.init_block, 1]
        In[:, self.bucket_number[6], 0] += output[:, self.bucket_number[1], 0] + output[:, self.bucket_number[3], 0]
        In[:, self.bucket_number[4], 0] += output[:, self.bucket_number[1], 0] + output[:, self.bucket_number[3], 0]
        number_caculate = number_former + In - output

        input_caculate = torch.cat((output, In, number_caculate), dim=2)

        return input_caculate

    def forward(self, co_data, bucket_number):

        [batch_size, temporal, spatial, input_size] = co_data.shape
        temporal -= 1

        self.get_topoloy(bucket_number, spatial)
        major_data = co_data[:, :, self.major_seg_block, :]
        minor_data = co_data[:, :, self.minor_seg_block, :]
        end_data = co_data[:, :, self.end_seg_block, :]
        inter_data = co_data[:, :, bucket_number, :]

        h_inter = Variable(co_data.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        c_inter = Variable(co_data.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        h_major = Variable(co_data.data.new(batch_size, self.major_block_number, self.hidden_size).fill_(0).float())
        c_major = Variable(co_data.data.new(batch_size, self.major_block_number, self.hidden_size).fill_(0).float())
        h_minor = Variable(co_data.data.new(batch_size, self.minor_block_number, self.hidden_size).fill_(0).float())
        c_minor = Variable(co_data.data.new(batch_size, self.minor_block_number, self.hidden_size).fill_(0).float())
        h_end = Variable(co_data.data.new(batch_size, self.end_block_number, self.hidden_size).fill_(0).float())
        c_end = Variable(co_data.data.new(batch_size, self.end_block_number, self.hidden_size).fill_(0).float())
        h_major_after = Variable(co_data.data.new(batch_size, self.major_block_number, self.hidden_size).fill_(0).float())
        h_major_before = Variable(co_data.data.new(batch_size, self.major_block_number, self.hidden_size).fill_(0).float())
        h_minor_after = Variable(co_data.data.new(batch_size, self.minor_block_number, self.hidden_size).fill_(0).float())
        h_minor_before = Variable(co_data.data.new(batch_size, self.minor_block_number, self.hidden_size).fill_(0).float())
        h_end_after = Variable(co_data.data.new(batch_size, self.end_block_number, self.hidden_size).fill_(0).float())
        h_end_before = Variable(co_data.data.new(batch_size, self.end_block_number, self.hidden_size).fill_(0).float())

        outputs = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, 3).fill_(0).float())
        h_outputs = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, self.hidden_size).fill_(0).float())
        h_tmp = Variable(h_inter.data.new(batch_size, self.spatial, self.hidden_size).fill_(0).float())
        c_tmp = Variable(h_inter.data.new(batch_size, self.spatial, self.hidden_size).fill_(0).float())

        for time in range(temporal):

            h_major = h_major.view(batch_size*self.major_block_number, self.hidden_size)
            c_major = c_major.view(batch_size*self.major_block_number, self.hidden_size)
            h_minor = h_minor.view(batch_size*self.minor_block_number, self.hidden_size)
            c_minor = c_minor.view(batch_size*self.minor_block_number, self.hidden_size)
            h_end = h_end.view(batch_size*self.end_block_number, self.hidden_size)
            c_end = c_end.view(batch_size*self.end_block_number, self.hidden_size)
            h_major_after = h_major_after.view(batch_size*self.major_block_number, self.hidden_size)
            h_major_before = h_major_before.view(batch_size*self.major_block_number, self.hidden_size)
            h_minor_after = h_minor_after.view(batch_size*self.minor_block_number, self.hidden_size)
            h_minor_before = h_minor_before.view(batch_size*self.minor_block_number, self.hidden_size)
            h_end_after = h_end_after.view(batch_size*self.end_block_number, self.hidden_size)
            h_end_before = h_end_before.view(batch_size*self.end_block_number, self.hidden_size)

            h_major, c_major = self.major_seg_model(major_data[:, time, :, :].contiguous().view(-1, self.input_size),
                                                    h_major, c_major, h_major_after, h_major_before)
            h_minor, c_minor = self.minor_seg_model(minor_data[:, time, :, :].contiguous().view(-1, self.input_size),
                                                    h_minor, c_minor, h_minor_after, h_minor_before)
            h_end, c_end = self.end_seg_model(end_data[:, time, :, :].contiguous().view(-1, self.input_size),
                                                    h_end, c_end, h_end_after, h_end_before)
            h_inter, c_inter = self.intersectio_model(inter_data[:, time, :, :], (h_inter, c_inter))

            h_major = h_major.view(batch_size, self.major_block_number, self.hidden_size)
            c_major = c_major.view(batch_size, self.major_block_number, self.hidden_size)
            h_minor = h_minor.view(batch_size, self.minor_block_number, self.hidden_size)
            c_minor = c_minor.view(batch_size, self.minor_block_number, self.hidden_size)
            h_end = h_end.view(batch_size, self.end_block_number, self.hidden_size)
            c_end = c_end.view(batch_size, self.end_block_number, self.hidden_size)
            h_major_after = h_major_after.view(batch_size, self.major_block_number, self.hidden_size)
            h_major_before = h_major_before.view(batch_size, self.major_block_number, self.hidden_size)
            h_minor_after = h_minor_after.view(batch_size, self.minor_block_number, self.hidden_size)
            h_minor_before = h_minor_before.view(batch_size, self.minor_block_number, self.hidden_size)
            h_end_after = h_end_after.view(batch_size, self.end_block_number, self.hidden_size)
            h_end_before = h_end_before.view(batch_size, self.end_block_number, self.hidden_size)

            h_tmp = h_tmp * 0
            c_tmp = c_tmp * 0
            h_tmp[:, self.major_seg_block, :] += h_major
            h_tmp[:, self.minor_seg_block, :] += h_minor
            h_tmp[:, self.end_seg_block, :] += h_end
            h_tmp[:, self.inter_block, :] += h_inter[:, self.inter_place, :]
            c_tmp[:, self.major_seg_block, :] += c_major
            c_tmp[:, self.minor_seg_block, :] += c_minor
            c_tmp[:, self.end_seg_block, :] += c_end
            c_tmp[:, self.inter_block, :] += c_inter[:, self.inter_place, :]

            if time > self.t_predict:
                h_outputs[:, time-self.t_predict-1, :, :] += h_tmp
                output = self.outputlayer(h_tmp)
                outputs[:, time-self.t_predict, :, :] += self.caculate_next_input(
                                                        co_data[:, time, :, :],
                                                        co_data[:, time+1, :, :],
                                                        output
                                                    )
            
            h_major_after = h_major_after * 0
            h_major_before = h_major_before * 0
            h_minor_after = h_minor_after * 0
            h_minor_before = h_minor_before * 0
            h_end_after = h_end_after * 0
            h_end_before = h_end_before * 0

            h_major_after += h_tmp[:, self.major_after_all]
            h_major_before[:, self.major_before_seg, :] += h_tmp[:, self.major_before_all, :]
            h_minor_after += h_tmp[:, self.minor_after_all]
            h_minor_before[:, self.minor_before_seg, :] += h_tmp[:, self.minor_before_all, :]
            h_end_after[:, self.end_after_seg, :] += h_tmp[:, self.end_after_all, :]
            h_end_before += h_tmp[:, self.end_before_all, :]

        return outputs, h_outputs

    def infer(self, co_data, bucket_number):

        [batch_size, temporal, spatial, input_size] = co_data.shape
        temporal -= 1

        self.get_topoloy(bucket_number, spatial)

        h_inter = Variable(co_data.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        c_inter = Variable(co_data.data.new(batch_size, self.inter_block_number, self.hidden_size).fill_(0).float())
        h_major = Variable(co_data.data.new(batch_size, self.major_block_number, self.hidden_size).fill_(0).float())
        c_major = Variable(co_data.data.new(batch_size, self.major_block_number, self.hidden_size).fill_(0).float())
        h_minor = Variable(co_data.data.new(batch_size, self.minor_block_number, self.hidden_size).fill_(0).float())
        c_minor = Variable(co_data.data.new(batch_size, self.minor_block_number, self.hidden_size).fill_(0).float())
        h_end = Variable(co_data.data.new(batch_size, self.end_block_number, self.hidden_size).fill_(0).float())
        c_end = Variable(co_data.data.new(batch_size, self.end_block_number, self.hidden_size).fill_(0).float())
        h_major_after = Variable(co_data.data.new(batch_size, self.major_block_number, self.hidden_size).fill_(0).float())
        h_major_before = Variable(co_data.data.new(batch_size, self.major_block_number, self.hidden_size).fill_(0).float())
        h_minor_after = Variable(co_data.data.new(batch_size, self.minor_block_number, self.hidden_size).fill_(0).float())
        h_minor_before = Variable(co_data.data.new(batch_size, self.minor_block_number, self.hidden_size).fill_(0).float())
        h_end_after = Variable(co_data.data.new(batch_size, self.end_block_number, self.hidden_size).fill_(0).float())
        h_end_before = Variable(co_data.data.new(batch_size, self.end_block_number, self.hidden_size).fill_(0).float())

        outputs = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, 3).fill_(0).float())
        h_outputs = Variable(co_data.data.new(batch_size, temporal-self.t_predict, spatial, self.hidden_size).fill_(0).float())
        h_tmp = Variable(h_inter.data.new(batch_size, self.spatial, self.hidden_size).fill_(0).float())
        c_tmp = Variable(h_inter.data.new(batch_size, self.spatial, self.hidden_size).fill_(0).float())

        for time in range(temporal+1):

            if time <= self.t_predict:
                input_data = co_data[:, time, :, :]
            else:
                output = self.outputlayer(h_tmp)
                input_data = self.caculate_next_input(
                                    input_data,
                                    co_data[:, time, :, :],
                                    output
                                )
                outputs[:, time - self.t_predict - 1, :, :] += input_data
                h_outputs[:, time - self.t_predict - 1, :, :] += h_tmp
            
            major_data = input_data[:, self.major_seg_block, :]
            minor_data = input_data[:, self.minor_seg_block, :]
            end_data = input_data[:, self.end_seg_block, :]
            inter_data = input_data[:, bucket_number, :]

            h_major = h_major.view(batch_size*self.major_block_number, self.hidden_size)
            c_major = c_major.view(batch_size*self.major_block_number, self.hidden_size)
            h_minor = h_minor.view(batch_size*self.minor_block_number, self.hidden_size)
            c_minor = c_minor.view(batch_size*self.minor_block_number, self.hidden_size)
            h_end = h_end.view(batch_size*self.end_block_number, self.hidden_size)
            c_end = c_end.view(batch_size*self.end_block_number, self.hidden_size)
            h_major_after = h_major_after.view(batch_size*self.major_block_number, self.hidden_size)
            h_major_before = h_major_before.view(batch_size*self.major_block_number, self.hidden_size)
            h_minor_after = h_minor_after.view(batch_size*self.minor_block_number, self.hidden_size)
            h_minor_before = h_minor_before.view(batch_size*self.minor_block_number, self.hidden_size)
            h_end_after = h_end_after.view(batch_size*self.end_block_number, self.hidden_size)
            h_end_before = h_end_before.view(batch_size*self.end_block_number, self.hidden_size)

            h_major, c_major = self.major_seg_model(major_data.contiguous().view(-1, self.input_size),
                                                    h_major, c_major, h_major_after, h_major_before)
            h_minor, c_minor = self.minor_seg_model(minor_data.contiguous().view(-1, self.input_size),
                                                    h_minor, c_minor, h_minor_after, h_minor_before)
            h_end, c_end = self.end_seg_model(end_data.contiguous().view(-1, self.input_size),
                                                    h_end, c_end, h_end_after, h_end_before)
            h_inter, c_inter = self.intersectio_model(inter_data, (h_inter, c_inter))

            h_major = h_major.view(batch_size, self.major_block_number, self.hidden_size)
            c_major = c_major.view(batch_size, self.major_block_number, self.hidden_size)
            h_minor = h_minor.view(batch_size, self.minor_block_number, self.hidden_size)
            c_minor = c_minor.view(batch_size, self.minor_block_number, self.hidden_size)
            h_end = h_end.view(batch_size, self.end_block_number, self.hidden_size)
            c_end = c_end.view(batch_size, self.end_block_number, self.hidden_size)
            h_major_after = h_major_after.view(batch_size, self.major_block_number, self.hidden_size)
            h_major_before = h_major_before.view(batch_size, self.major_block_number, self.hidden_size)
            h_minor_after = h_minor_after.view(batch_size, self.minor_block_number, self.hidden_size)
            h_minor_before = h_minor_before.view(batch_size, self.minor_block_number, self.hidden_size)
            h_end_after = h_end_after.view(batch_size, self.end_block_number, self.hidden_size)
            c_end_before = h_end_before.view(batch_size, self.end_block_number, self.hidden_size)

            h_tmp = h_tmp * 0
            c_tmp = c_tmp * 0
            h_tmp[:, self.major_seg_block, :] += h_major
            h_tmp[:, self.minor_seg_block, :] += h_minor
            h_tmp[:, self.end_seg_block, :] += h_end
            h_tmp[:, self.inter_block, :] += h_inter[:, self.inter_place, :]
            c_tmp[:, self.major_seg_block, :] += c_major
            c_tmp[:, self.minor_seg_block, :] += c_minor
            c_tmp[:, self.end_seg_block, :] += c_end
            c_tmp[:, self.inter_block, :] += c_inter[:, self.inter_place, :]

        return outputs, h_outputs
            

if __name__ == "__main__":
    
    args = {}
    
    args["n_unit"] = 7
    args["input_size"] = 3
    args["hidden_size"] = 64
    args["output_hidden_size"] = 16
    args["t_predict"] = 4
    args["n_units"] = 7
    model = sp_network_model(args)

    co_data = Variable(torch.randn(11, 9, 21, 3))
    bucket_number = [10, 11, 14, 15, 16, 17, 20]

    output, _ = model(co_data, bucket_number) 
    fake_loss = torch.mean(output)
    fake_loss.backward()
    
            




        