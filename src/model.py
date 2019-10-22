import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import conf


class FCNet(nn.Module):
    '''
    为了简化结构写的一个封装好的三层或者四层的全连接网络
    '''
    def __init__(self, addLayer=False, layerSize=[3, 8, 32]):
        '''
        初始化函数
        addLayer: bool 表示是否是4层的全连接网络，如果是的话后面layerSize需要4个变量表示
        layerSize： list like 表示每一层的单元数量
        '''
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

    #cell3
    #处理时间和路段长度两个维度的lstm变体cell

    def __init__(self, input_size, hidden_size):

        #input_size: int 输入的维度
        #hidden_size: int 隐层状态的维度

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        
        self.sigma = torch.nn.Sigmoid()

        self.spatial_forget = torch.nn.Linear(2*hidden_size, hidden_size)
        self.spatial_input = torch.nn.Linear(2*hidden_size, hidden_size)
         

    def forward(self, inputs, h_s_t, c_s_t, h_after_t, h_before_t):

        #inputs: tensor [batch_size, spatial_size, input_size] 当前节点此时刻的输入
        #h_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的隐层状态
        #c_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的细胞状态
        #h_after_t: tensor [batch_size, spatial_size, hidden_size] 下一个节点前一个时刻的隐层状态
        #h_before_t: tensor [batch_size, spatial_size, hidden_size] 前一个节点前一个时刻的隐层状态

        [batch_size, spatial_size, hidden_size] = h_s_t.shape

        spatial_gate = torch.cat((h_after_t, h_before_t), dim=2)
        
        #处理batch 因为batch内部的不同路段的不同节点在这里都是独立的，所以可以分开来
        spatial_gate = spatial_gate.view(-1, 2*hidden_size)
        h_s_t = h_s_t.view(-1, hidden_size)
        c_s_t = c_s_t.view(-1, hidden_size)
        inputs = inputs.view(-1, self.input_size)

        spatial_f = self.spatial_forget(spatial_gate)
        spatial_f = self.sigma(spatial_f)
        spatial_i = self.spatial_input(spatial_gate)
        spatial_i = self.sigma(spatial_i)

        c_s_t = c_s_t * spatial_f
        h_s_t = h_s_t * spatial_i

        h_s_tp, c_s_tp = self.cell(inputs, (h_s_t, c_s_t))

        c_s_tp = c_s_tp.view(batch_size, spatial_size, hidden_size)
        h_s_tp = h_s_tp.view(batch_size, spatial_size, hidden_size)

        return h_s_tp, c_s_tp


class TP_lstm(nn.Module):
    '''
    代表一个路段的模型
    '''
    def __init__(self, args):

        super().__init__()

        self.args = args
        #网络相关参数
        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.lane_gate_size = args["lane_gate_size"]
        self.output_hidden_size = args["output_hidden_size"]
        self.output_size = 1 # 这里先写死
        self.t_predict = args["t_predict"] # 开始预测的时间
        self.temporal_length = args["temporal_length"] # 整个输入的时间长
        #网络结构
        self.cell = MD_lstm_cell(self.input_size, self.hidden_size)
        self.sigma = torch.nn.Sigmoid()
        self.lane_gate = FCNet(layerSize=[1, self.lane_gate_size, 1])
        self.output_layer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])

    def infer(self, temporal_data, init_input, lane):
        '''
        测试过程中的函数，在进行一部分输入之后，将模型当前时刻的输出重组为下一时刻的输入进入网络中
        temporal_data: tensor[batch_size, predict_input_length, input_size] 开始预测之后第一个节点每一个时刻的输入
        init_data: tensor[batch_size, spatial_size, t_predict, input_size] 初始的输入
        lane： tensor[batch_size, 1] 车道数
        '''
        # 获取长度信息
        predict_input_length = temporal_data.shape[1]
        [batch_size, spatial_length, input_temporal, input_size] = init_input.shape

        # 开始时序的推演 创建一些初始变量
        
        # 初始化细胞状态变量
        cell_state = temporal_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        
        # 初始化隐层状态变量
        hidden_state = temporal_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        
        # 初始化前向后向节点的隐层状态
        hidden_state_after = temporal_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_before = temporal_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        
        # 初始化填充值
        zero_hidden = temporal_data.data.new(batch_size, 1, self.hidden_size).fill_(0).float()
        
        #初始化输出变量
        output = temporal_data.data.new(batch_size, spatial_length, predict_input_length).fill_(0).float()

        # 在非预测时序空间计算
        for time in range(input_temporal):

            hidden_state, cell_state = self.cell(init_input[:, :, time, :], hidden_state, cell_state, hidden_state_after, hidden_state_before)
            
            # 后一个节点的隐层状态是从第二个节点的隐层状态开始到最后，再添加一个填充的0值tensor
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            
            # 前一个节点的隐层状态是一开始填充一个0值的tensor，然后再拼上从第一个节点开始到倒数第二个节点
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)

        predict_input = init_input[:, :, -1, :]

        for time in range(predict_input_length):
            
            outflow = self.output_layer(hidden_state)

            # output是下一时刻的输出，所以与当前时刻要错后一位
            if time > 0:
                output[:, :, time-1] = outflow.view(batch_size, spatial_length)
            
            # 每个节点的输入是由第一个节点的输入与预测得到的从第一个节点到倒数第二个节点的流出值
            inflow = torch.cat((temporal_data[:, time, 1].view(batch_size,1,1), outflow[:, :spatial_length-1]), 1)
            
            # 每个节点的车辆数是由每个节点前一个时刻的车辆数以及输入和输出车辆数计算得到的
            number = predict_input[:, :, 2].view(batch_size, spatial_length, 1) - outflow + inflow
            
            # 拼接得到下一时刻的输入
            predict_input = torch.cat((outflow, inflow, number), 2)
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)
            hidden_state, cell_state = self.cell(predict_input, hidden_state, cell_state, hidden_state_after, hidden_state_before)

        # output是下一时段的输出，所以与当前时刻要错后一位
        outflow = self.output_layer(hidden_state)
        output[:, :, predict_input_length-1] = outflow.view(batch_size, spatial_length)

        return output    

    def train_infer(self, init_data, hidden_state=None, cell_state=None):
        
        [batch_size, spatial_length, _] = init_data.shape

        if hidden_state is None:
            cell_state = init_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
            hidden_state = init_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
            hidden_state_after = init_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
            hidden_state_before = init_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        else:
            zero_hidden = init_data.data.new(batch_size, 1, self.hidden_size).fill_(0).float()
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)

        hidden_state, cell_state = self.cell(init_data, hidden_state, cell_state, hidden_state_after, hidden_state_before)
        output = self.output_layer(hidden_state)

        return output, [hidden_state, cell_state]



    def forward(self, input_data, lane):
        '''
        训练过程中的前向函数
        input_data: Varible or tensor [batch_size, spatial_length, temporal_length, input_size] 输入数据
        lane: Varible or tensor [batch_size, 1] 车道数
        '''
        [batch_size, spatial_length, temporal_length, input_size] = input_data.shape

        # 初始化
        cell_state = input_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state = input_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_after = input_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        hidden_state_before = input_data.data.new(batch_size, spatial_length, self.hidden_size).fill_(0).float()
        zero_hidden = input_data.data.new(batch_size, 1, self.hidden_size).fill_(0).float()
        output = input_data.data.new(batch_size, spatial_length, temporal_length-self.t_predict)

        for time in range(temporal_length):

            # 计算每个时刻的输出，在预测时刻之后开始保存下来每个时刻的输出    
            hidden_state, cell_state = self.cell(input_data[:, :, time, :], 
                    hidden_state, cell_state, hidden_state_after, hidden_state_before)
            if time >= self.t_predict:
                output[:, :, time-self.t_predict] = self.output_layer(hidden_state).view(batch_size, spatial_length)

            # 调整前后节点的隐层状态    
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)

        return output

class loss_function(nn.Module):

    def __init__(self, epsilon=1):

        super().__init__()

        self.epsilon = epsilon

        self.criterion = nn.MSELoss()
        #self.criterion = nn.SmoothL1Loss()
        
    def forward(self, target, output, mask=None):

        if mask is None:
            loss = self.criterion(target, output)
        else :
            loss = target - output
            loss = loss * mask
            loss = torch.pow(loss, 2)
            loss = torch.mean(loss)

        return loss

class inter_model(nn.Module):
    '''
    全连接式的路口形式
    '''
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

    seg_inputs = torch.randn(32, 3)
    inter_inputs = torch.randn(3, 7, 3)
    [[seg_output, inter_output], [h_seg, h_inter]] = model(seg_inputs, inter_inputs)
    print(seg_output.shape)
    print(inter_output.shape)
    print(h_seg)
    print(h_inter)