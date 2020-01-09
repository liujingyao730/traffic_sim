import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.init as init

import conf
from model import FCNet

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
        
        [batch_size, spatial, input_size] = inputs.shape
        [batch_size, spatial, hidden_size] = h_s_t.shape

        assert input_size == self.input_size

        inputs = inputs.view(batch_size*spatial, input_size)
        h_s_t = h_s_t.view(batch_size*spatial, hidden_size)
        c_s_t = c_s_t.view(batch_size*spatial, hidden_size)
        h_after_t = h_after_t.view(batch_size*spatial, hidden_size)
        h_before_t = h_before_t.view(batch_size*spatial, hidden_size)

        spatial_hidden_input = torch.cat((h_after_t, h_before_t), dim=1)

        spatial_i = self.spatial_input(spatial_hidden_input)
        spatial_i = self.sigma(spatial_i)
        spatial_f = self.spatial_forget(spatial_hidden_input)
        spatial_f = self.sigma(spatial_f)

        h = h_s_t * spatial_i
        c = c_s_t * spatial_f

        h_s_tp, c_s_tp = self.cell(inputs, (h, c))

        h_s_tp = h_s_tp.view(batch_size, spatial, hidden_size)
        c_s_tp = c_s_tp.view(batch_size, spatial, hidden_size)
        
        return h_s_tp, c_s_tp



class att_cell(nn.Module):

    def __init__(self, args):

        super(att_cell, self).__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.n_head = args["n_head"]

        self.cell = nn.LSTMCell(self.input_size, self.hidden_size)

        self.a_src = Parameter(torch.Tensor(2*self.hidden_size, self.n_head))

        self.spatial_forget_gate = nn.Linear(self.n_head*self.hidden_size, self.hidden_size)
        self.spatial_input_gate = nn.Linear(self.n_head*self.hidden_size, self.hidden_size)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.sigma = nn.Sigmoid()

        init.xavier_uniform_(self.a_src)

    def forward(self, inputs, h, c, h_spatial):

        [batch_size, spatial, _] = inputs.shape
        spatial_n = h_spatial.shape[2]

        inputs = inputs.view(batch_size*spatial, self.input_size)
        h = h.view(batch_size*spatial, self.hidden_size)
        c = c.view(batch_size*spatial, self.hidden_size)
        h_spatial = h_spatial.view(batch_size*spatial, spatial_n, self.hidden_size)

        h_prime = h.unsqueeze(1).expand(-1, spatial_n, -1)
        h_prime = torch.cat((h_prime, h_spatial), dim=2)
        
        attn = torch.matmul(h_prime, self.a_src)
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = attn.permute(0, 2, 1)

        h_spatial = torch.matmul(attn, h_spatial).view(batch_size*spatial, self.n_head*self.hidden_size)
        spatial_forget = self.spatial_forget_gate(h_spatial)
        spatial_f = self.sigma(spatial_forget)
        spatial_input = self.spatial_input_gate(h_spatial)
        spatial_i = self.sigma(spatial_input)

        h = h * spatial_i
        c = c * spatial_f

        h, c = self.cell(inputs, (h, c))

        h = h.view(batch_size, spatial, self.hidden_size)
        c = c.view(batch_size, spatial, self.hidden_size)

        return h, c 





if __name__ == "__main__":
    
    args = {}
    
    args["n_unit"] = 7
    args["input_size"] = 3
    args["hidden_size"] = 64
    args["output_hidden_size"] = 16
    args["t_predict"] = 4
    args["n_units"] = 7
    args["n_head"] = 2
    model = att_cell(args)

    inputs = torch.randn(7, 11, 3)
    h = torch.randn(7, 11, 64)
    c = torch.randn(7, 11, 64)
    h_spatial = torch.randn(7, 11, 4, 64)

    h, c = model(inputs, h, c, h_spatial)

    loss = torch.mean(h)
    loss.backward()
    