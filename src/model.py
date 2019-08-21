import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    '''
    处理时间和路段长度两个维度的lstm变体cell
    '''
    def __init__(self, input_size, hidden_size):
        '''
        目前的结构是在传统LSTM cell之外，对隐层状态施加一个门控制，这里我们叫做空间门，后续可能会要改
        input_size: int 输入的维度
        hidden_size: int 隐层状态的维度
        '''
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        
        self.sigma = torch.nn.Sigmoid()

        self.spatial_embedding = torch.nn.Linear(2*hidden_size, hidden_size)
         

    def forward(self, inputs, h_s_t, c_s_t, h_after_t, h_before_t):
        '''
        inputs: tensor [batch_size, spatial_size, input_size] 当前节点此时刻的输入
        h_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的隐层状态
        c_s_t: tensor [batch_size, spatial_size, hidden_size] 当前节点前一个时刻的细胞状态
        h_after_t: tensor [batch_size, spatial_size, hidden_size] 下一个节点前一个时刻的隐层状态
        h_before_t: tensor [batch_size, spatial_size, hidden_size] 前一个节点前一个时刻的隐层状态
        '''
        [batch_size, spatial_size, hidden_size] = h_s_t.shape

        spatial_gate = torch.cat((h_after_t, h_before_t), dim=2)
        
        #处理batch 因为batch内部的不同路段的不同节点在这里都是独立的，所以可以分开来
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
        
        #空间门
        h_s_tp = h_hat * spatial_gate

        return h_s_tp, c_s_tp


class TP_lstm(nn.Module):
    '''
    代表一个路段的模型
    '''
    def __init__(self, args):

        super().__init__()

        self.args = args
        #网络相关参数
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.lane_gate_size = args.lane_gate_size
        self.output_hidden_size = args.output_hidden_size
        self.output_size = 1 # 这里先写死
        self.t_predict = args.t_predict # 开始预测的时间
        self.temporal_length = args.temporal_length # 整个输入的时间长
        self.spatial_length = args.spatial_length # 路段长……好像没什么用可能需要以后改改
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
        [batch_size, spatial_length, input_temporal, input_size]= init_input.shape
        

        # 处理batch
        temporal_data = temporal_data.view(batch_size, -1)
        init_input = init_input.view(batch_size, -1)

        # lane_gate
        lane_controller = self.lane_gate(lane)
        lane_controller = self.sigma(lane_controller)
        temporal_data = temporal_data * lane_controller
        init_input = init_input * lane_controller

        # 还原
        temporal_data = temporal_data.view(batch_size, predict_input_length, input_size)
        init_input = init_input.view(batch_size, spatial_length, input_temporal, input_size)

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

        output = output.view(batch_size, -1) / lane_controller
        output = output.view(batch_size, spatial_length, predict_input_length)

        return output    

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

        # 处理batch
        input_data = input_data.view(batch_size, -1)

        # lane_gate
        lane_controller = self.lane_gate(lane)
        lane_controller = self.sigma(lane_controller)
        input_data = input_data * lane_controller

        # 还原
        input_data = input_data.view(batch_size, spatial_length, temporal_length, input_size)

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

    def __init__(self):

        super(loss_function, self).__init__()

        self.mes_criterion = nn.MSELoss()

    def forward(self, number_current, number_before, In, outflow):

        [batch_size, spatial_size, temporal_size] = number_current.shape

        inflow = torch.cat((In, outflow[:, :spatial_size-1,:]), 1)
        number_caculate = number_before + inflow - outflow

        loss = self.mes_criterion(number_current, number_caculate)

        return loss
        
class embedding_TP_lstm(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args.input_size
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size

        self.embedding = torch.nn.Linear(self.input_size, self.embedding_size)

        args.input_size = args.embedding_size

        self.tp_lstm = TP_lstm(args)

    def forward(self, input_data, lane):

        [batch_size, spatial, temporal, input_size] = input_data.shape

        input_data = self.embedding(input_data.view(-1, input_size)).view(batch_size, spatial, temporal, -1)
        output = self.tp_lstm(input_data, lane)

        return output

    def infer(self, temporal_data, init_input, lane):

        predict_input_length = temporal_data.shape[1]
        [batch_size, spatial_length, input_temporal, input_size]= init_input.shape

        temporal_data = self.embedding(temporal_data.contiguous().view(-1, input_size)).view(batch_size, predict_input_length, -1)
        init_input = self.embedding(init_input.contiguous().view(-1, input_size)).view(batch_size, spatial_length, input_temporal, -1)

         # 处理batch
        temporal_data = temporal_data.view(batch_size, -1)
        init_input = init_input.view(batch_size, -1)

        # lane_gate
        lane_controller = self.tp_lstm.lane_gate(lane)
        lane_controller = self.tp_lstm.sigma(lane_controller)
        temporal_data = temporal_data * lane_controller
        init_input = init_input * lane_controller

        # 还原
        temporal_data = temporal_data.view(batch_size, predict_input_length, self.embedding_size)
        init_input = init_input.view(batch_size, spatial_length, input_temporal, self.embedding_size)

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

            hidden_state, cell_state = self.tp_lstm.cell(init_input[:, :, time, :], hidden_state, cell_state, hidden_state_after, hidden_state_before)
            
            # 后一个节点的隐层状态是从第二个节点的隐层状态开始到最后，再添加一个填充的0值tensor
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            
            # 前一个节点的隐层状态是一开始填充一个0值的tensor，然后再拼上从第一个节点开始到倒数第二个节点
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)

        predict_input = init_input[:, :, -1, :]

        for time in range(predict_input_length):
            
            outflow = self.tp_lstm.output_layer(hidden_state)

            # output是下一时刻的输出，所以与当前时刻要错后一位
            if time > 0:
                output[:, :, time-1] = outflow.view(batch_size, spatial_length)
            
            # 每个节点的输入是由第一个节点的输入与预测得到的从第一个节点到倒数第二个节点的流出值
            inflow = torch.cat((temporal_data[:, time, 1].view(batch_size,1,1), outflow[:, :spatial_length-1]), 1)
            
            # 每个节点的车辆数是由每个节点前一个时刻的车辆数以及输入和输出车辆数计算得到的
            number = predict_input[:, :, 2].view(batch_size, spatial_length, 1) - outflow + inflow
            
            # 拼接得到下一时刻的输入
            predict_input = torch.cat((outflow, inflow, number), 2)
            predict_input = self.embedding(predict_input.view(-1, self.input_size)).view(batch_size, spatial_length, self.embedding_size)
            hidden_state_after = torch.cat((hidden_state[:, 1:, :], zero_hidden), 1)
            hidden_state_before = torch.cat((zero_hidden, hidden_state[:, :spatial_length-1, :]), 1)
            hidden_state, cell_state = self.tp_lstm.cell(predict_input, hidden_state, cell_state, hidden_state_after, hidden_state_before)

        # output是下一时段的输出，所以与当前时刻要错后一位
        outflow = self.tp_lstm.output_layer(hidden_state)
        output[:, :, predict_input_length-1] = outflow.view(batch_size, spatial_length)

        return output