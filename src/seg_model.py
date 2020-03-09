import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import conf
from model import FCNet
from net_model import seg_model
#from net_model import att_cell as cell_model
from net_model import non_att_cell as cell_model

class basic_model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.encoder_hidden = args["encoder_hidden"]
        self.decoder_input = args["decoder_input"]
        self.decoder_hidden = args["decoder_hidden"]
        self.t_predict = args["t_predict"]
        self.output_size = args["output_size"]

        self.encoder = seg_model({"input_size":self.input_size, "hidden_size":self.encoder_hidden})
        self.decoder = seg_model({"input_size":self.decoder_input, "hidden_size":self.decoder_hidden})

        self.outputLayer = nn.Linear(self.decoder_hidden, self.output_size)
        self.embedding_layer = nn.Linear(self.encoder_hidden, self.decoder_input)

        self.sigma = nn.Sigmoid()

    def get_spatial_hidden(self, h):

        h_after = Variable(h.data.new(h.shape).fill_(0).float())
        h_before = Variable(h.data.new(h.shape).fill_(0).float())

        h_after[:, :-1, :] += h[:, 1:, :]
        h_before[:, 1:, :] += h[:, :-1, :]

        return h_after, h_before

    def caculate_next_input(self, former_input, next_input, output):

        In = torch.cat((next_input[:, 0:1, 1:2], output[:, :-1, :]), dim=1)
        former_number = former_input[:, :, [2]]
        number_caculate = former_number + In - output

        next_data = torch.cat((output, In, number_caculate), dim=2)

        return next_data

    def forward(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape

        encoder_h = Variable(input_data.data.new(batch_size, spatial, self.encoder_hidden).fill_(0).float())
        encoder_c = Variable(input_data.data.new(batch_size, spatial, self.encoder_hidden).fill_(0).float())
        decoder_h = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())
        decoder_c = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())

        for time in range(temporal - 1):

            encoder_h_after, encoder_h_before = self.get_spatial_hidden(encoder_h)
            decoder_h_after, decoder_h_before = self.get_spatial_hidden(decoder_h)

            data = input_data[:, time, :, :].contiguous()

            encoder_h, encoder_c = self.encoder(data, encoder_h, encoder_c, encoder_h_after, encoder_h_before)

            code = self.sigma(self.embedding_layer(encoder_h))

            decoder_h, decoder_c = self.decoder(code, decoder_h, decoder_h, decoder_h_after, decoder_h_before)

            if time >= self.t_predict:
                output = self.outputLayer(decoder_h)
                next_input = self.caculate_next_input(
                                                    input_data[:, time, :, :],
                                                    input_data[:, time+1, :, :],
                                                    output
                                                    )
                outputs[:, time-self.t_predict, :, :] += next_input

        return outputs

    def infer(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape
        self.spatial = spatial

        encoder_h = Variable(input_data.data.new(batch_size, spatial, self.encoder_hidden).fill_(0).float())
        encoder_c = Variable(input_data.data.new(batch_size, spatial, self.encoder_hidden).fill_(0).float())
        decoder_h = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())
        decoder_c = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())
            
        for time in range(temporal):

            encoder_h_after, encoder_h_before = self.get_spatial_hidden(encoder_h)
            decoder_h_after, decoder_h_before = self.get_spatial_hidden(decoder_h)

            if time <= self.t_predict:
                data = input_data[:, time, :, :]
            else:
                output = self.outputLayer(decoder_h)
                data = self.caculate_next_input(
                                                data,
                                                input_data[:, time, :, :],
                                                output
                                            )
                outputs[:, time-self.t_predict-1, :, :] += data

            data = data.contiguous()

            encoder_h, encoder_c = self.encoder(data, encoder_h, encoder_c, encoder_h_after, encoder_h_before)

            code = self.sigma(self.embedding_layer(encoder_h))

            decoder_h, decoder_c = self.decoder(code, decoder_h, decoder_h, decoder_h_after, decoder_h_before)

        return outputs

class attn_model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.n_head = args["n_head"]
        self.t_predict = args["t_predict"]
        self.output_size = args["output_size"]

        self.cell = cell_model(args)

        self.outputLayer = nn.Linear(self.hidden_size, self.output_size)

        self.sigma = nn.Sigmoid()

    def get_spatial_hidden(self, h):

        [batch_size, spatial, hidden] = h.shape

        h_after1 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before1 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_after2 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before2 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())

        h_after1[:, :-1, 0, :] += h[:, 1:, :]
        h_before1[:, 1:, 0, :] += h[:, :-1, :]
        h_after2[:, :-2, 0, :] += h[:, 2:, :]
        h_before2[:, 2:, 0, :] += h[:, :-2, :]

        h_spatial = torch.cat((h_after2, h_after1, h_before1, h_before2), dim=2)

        return h_spatial

    def caculate_next_input(self, former_input, next_input, output):

        In = torch.cat((next_input[:, 0:1, 1:2], output[:, :-1, :]), dim=1)
        former_number = former_input[:, :, [2]]
        number_caculate = former_number + In - output

        next_data = torch.cat((output, In, number_caculate), dim=2)

        return next_data

    def forward(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape

        h = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())

        for time in range(temporal - 1):

            spatial_h = self.get_spatial_hidden(h)

            data = input_data[:, time, :, :].contiguous()

            h, c = self.cell(data, h, c, spatial_h)

            if time >= self.t_predict:
                output = self.outputLayer(h)
                next_input = self.caculate_next_input(
                                                    input_data[:, time, :, :],
                                                    input_data[:, time+1, :, :],
                                                    output
                                                    )
                outputs[:, time-self.t_predict, :, :] += next_input

        return outputs

    def infer(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape
        self.spatial = spatial

        h = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())
            
        for time in range(temporal):

            spatial_h = self.get_spatial_hidden(h)

            if time <= self.t_predict:
                data = input_data[:, time, :, :]
            else:
                output = self.outputLayer(h)
                data = self.caculate_next_input(
                                                data,
                                                input_data[:, time, :, :],
                                                output
                                            )
                outputs[:, time-self.t_predict-1, :, :] += data

            data = data.contiguous()

            h, c = self.cell(data, h, c, spatial_h)

        return outputs


class attn_model_ad(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.encoder_hidden = args["encoder_hidden"]
        self.decoder_input = args["decoder_input"]
        self.decoder_hidden = args["decoder_hidden"]
        self.n_head = args["n_head"]
        self.t_predict = args["t_predict"]
        self.output_size = args["output_size"]

        self.encoder = cell_model({"input_size":self.input_size, "hidden_size":self.encoder_hidden, "n_head":self.n_head})
        self.decoder = cell_model({"input_size":self.decoder_input, "hidden_size":self.decoder_hidden, "n_head":self.n_head})

        self.outputLayer = nn.Linear(self.decoder_hidden, self.output_size)
        self.embedding_layer = nn.Linear(self.encoder_hidden, self.decoder_input)

        self.sigma = nn.Sigmoid()

    def get_spatial_hidden(self, h):

        [batch_size, spatial, hidden] = h.shape

        h_after1 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before1 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_after2 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before2 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())

        h_after1[:, :-1, 0, :] += h[:, 1:, :]
        h_before1[:, 1:, 0, :] += h[:, :-1, :]
        h_after2[:, :-2, 0, :] += h[:, 2:, :]
        h_before2[:, 2:, 0, :] += h[:, :-2, :]

        h_spatial = torch.cat((h_after2, h_after1, h_before1, h_before2), dim=2)

        return h_spatial

    def caculate_next_input(self, former_input, next_input, output):

        In = torch.cat((next_input[:, 0:1, 1:2], output[:, :-1, :]), dim=1)
        former_number = former_input[:, :, [2]]
        number_caculate = former_number + In - output

        next_data = torch.cat((output, In, number_caculate), dim=2)

        return next_data

    def forward(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape

        encoder_h = Variable(input_data.data.new(batch_size, spatial, self.encoder_hidden).fill_(0).float())
        encoder_c = Variable(input_data.data.new(batch_size, spatial, self.encoder_hidden).fill_(0).float())
        decoder_h = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())
        decoder_c = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())

        for time in range(temporal - 1):

            encoder_spatial = self.get_spatial_hidden(encoder_h)
            decoder_spatial = self.get_spatial_hidden(decoder_h)

            data = input_data[:, time, :, :].contiguous()

            encoder_h, encoder_c = self.encoder(data, encoder_h, encoder_c, encoder_spatial)

            code = self.sigma(self.embedding_layer(encoder_h))

            decoder_h, decoder_c = self.decoder(code, decoder_h, decoder_h, decoder_spatial)

            if time >= self.t_predict:
                output = self.outputLayer(decoder_h)
                next_input = self.caculate_next_input(
                                                    input_data[:, time, :, :],
                                                    input_data[:, time+1, :, :],
                                                    output
                                                    )
                outputs[:, time-self.t_predict, :, :] += next_input

        return outputs

    def infer(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape
        self.spatial = spatial

        encoder_h = Variable(input_data.data.new(batch_size, spatial, self.encoder_hidden).fill_(0).float())
        encoder_c = Variable(input_data.data.new(batch_size, spatial, self.encoder_hidden).fill_(0).float())
        decoder_h = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())
        decoder_c = Variable(input_data.data.new(batch_size, spatial, self.decoder_hidden).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())
            
        for time in range(temporal):

            encoder_spatial = self.get_spatial_hidden(encoder_h)
            decoder_spatial = self.get_spatial_hidden(decoder_h)

            if time <= self.t_predict:
                data = input_data[:, time, :, :]
            else:
                output = self.outputLayer(decoder_h)
                data = self.caculate_next_input(
                                                data,
                                                input_data[:, time, :, :],
                                                output
                                            )
                outputs[:, time-self.t_predict-1, :, :] += data

            data = data.contiguous()

            encoder_h, encoder_c = self.encoder(data, encoder_h, encoder_c, encoder_spatial)

            code = self.sigma(self.embedding_layer(encoder_h))

            decoder_h, decoder_c = self.decoder(code, decoder_h, decoder_h, decoder_spatial)

        return outputs


class two_type_attn_model_1(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.n_head = args["n_head"]
        self.t_predict = args["t_predict"]
        self.output_size = args["output_size"]

        self.cell = cell_model(args)

        self.outputLayer = nn.Linear(self.hidden_size, self.output_size)

        self.sigma = nn.Sigmoid()

    def get_spatial_hidden(self, h):

        [batch_size, spatial, hidden] = h.shape

        h_after1 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before1 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_after2 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before2 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_after3 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before3 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())

        h_after1[:, :-1, 0, :] += h[:, 1:, :]
        h_before1[:, 1:, 0, :] += h[:, :-1, :]
        h_after2[:, :-2, 0, :] += h[:, 2:, :]
        h_before2[:, 2:, 0, :] += h[:, :-2, :]
        h_after3[:, :-3, 0, :] += h[:, 3:, :]
        h_before3[:, 3:, 0, :] += h[:, :-3, :]

        h_spatial = torch.cat((h_after3, h_after2, h_after1, h_before1, h_before2, h_after3), dim=2)

        return h_spatial

    def caculate_next_input(self, former_input, next_input, output):

        pv_In = torch.cat((next_input[:, 0:1, 1:2], output[:, :-1, [0]]), dim=1)
        pv_former_number = former_input[:, :, [2]]
        pv_number_caculate = pv_former_number + pv_In - output[:, :, [0]]

        hov_In = torch.cat((next_input[:, 0:1, 4:5], output[:, :-1, [1]]), dim=1)
        hov_former_number = former_input[:, :, [5]]
        hov_number_caculate = hov_former_number + hov_In - output[:, :, [1]]

        next_data = torch.cat((output[:, :, [0]], pv_In, pv_number_caculate, 
                            output[:, :, [1]], hov_In, hov_number_caculate), dim=2)

        return next_data

    def forward(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape

        h = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())

        for time in range(temporal - 1):

            spatial_h = self.get_spatial_hidden(h)

            data = input_data[:, time, :, :].contiguous()

            h, c = self.cell(data, h, c, spatial_h)

            if time >= self.t_predict:
                output = self.outputLayer(h)
                next_input = self.caculate_next_input(
                                                    input_data[:, time, :, :],
                                                    input_data[:, time+1, :, :],
                                                    output
                                                    )
                outputs[:, time-self.t_predict, :, :] += next_input

        return outputs

    def infer(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape
        self.spatial = spatial

        h = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())
            
        for time in range(temporal):

            spatial_h = self.get_spatial_hidden(h)

            if time <= self.t_predict:
                data = input_data[:, time, :, :]
            else:
                output = self.outputLayer(h)
                data = self.caculate_next_input(
                                                data,
                                                input_data[:, time, :, :],
                                                output
                                            )
                outputs[:, time-self.t_predict-1, :, :] += data

            data = data.contiguous()

            h, c = self.cell(data, h, c, spatial_h)

        return outputs

class two_type_attn_model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.n_head = args["n_head"]
        self.t_predict = args["t_predict"]
        self.output_size = args["output_size"]
        self.gain = args["gain"]

        self.after_para = nn.Parameter(torch.Tensor(3, self.hidden_size))
        self.before_para = nn.Parameter(torch.Tensor(3, self.hidden_size))

        nn.init.xavier_uniform_(self.after_para, gain=self.gain)
        nn.init.xavier_uniform_(self.before_para, gain=self.gain)

        self.cell = cell_model(args)

        self.outputLayer = nn.Linear(self.hidden_size, self.output_size)

        self.sigma = nn.Sigmoid()

    def get_spatial_hidden(self, h):

        [batch_size, spatial, hidden] = h.shape

        h_after1 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before1 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_after2 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before2 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_after3 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())
        h_before3 = Variable(h.data.new(batch_size, spatial, 1, hidden).fill_(0).float())

        h_after1[:, :-1, 0, :] += h[:, 1:, :]
        h_after1[:, -1, 0, :] += self.after_para[0, :]

        h_before1[:, 1:, 0, :] += h[:, :-1, :]
        h_before1[:, 0, 0, :] += self.before_para[0, :]

        h_after2[:, :-2, 0, :] += h[:, 2:, :]
        h_after2[:, -2:, 0, :] += self.after_para[:2, :]

        h_before2[:, 2:, 0, :] += h[:, :-2, :]
        h_before2[:, :2, 0, :] += self.before_para[:2, :]

        h_after3[:, :-3, 0, :] += h[:, 3:, :]
        h_after3[:, -3:, 0, :] += self.after_para

        h_before3[:, 3:, 0, :] += h[:, :-3, :]
        h_before3[:, :3, 0, :] += self.before_para

        h_spatial = torch.cat((h_after3, h_after2, h_after1, h_before1, h_before2, h_before3), dim=2)

        return h_spatial

    def caculate_next_input(self, former_input, next_input, output):

        pv_In = torch.cat((next_input[:, 0:1, 1:2], output[:, :-1, [0]]), dim=1)
        pv_former_number = former_input[:, :, [2]]
        pv_number_caculate = pv_former_number + pv_In - output[:, :, [0]]

        hov_In = torch.cat((next_input[:, 0:1, 4:5], output[:, :-1, [1]]), dim=1)
        hov_former_number = former_input[:, :, [5]]
        hov_number_caculate = hov_former_number + hov_In - output[:, :, [1]]

        next_data = torch.cat((output[:, :, [0]], pv_In, pv_number_caculate, 
                            output[:, :, [1]], hov_In, hov_number_caculate), dim=2)

        return next_data

    def forward(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape

        h = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())

        for time in range(temporal - 1):

            spatial_h = self.get_spatial_hidden(h)

            data = input_data[:, time, :, :].contiguous()

            h, c = self.cell(data, h, c, spatial_h)

            if time >= self.t_predict:
                output = self.outputLayer(h)
                next_input = self.caculate_next_input(
                                                    input_data[:, time, :, :],
                                                    input_data[:, time+1, :, :],
                                                    output
                                                    )
                outputs[:, time-self.t_predict, :, :] += next_input

        return outputs

    def infer(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape
        self.spatial = spatial

        h = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())
            
        for time in range(temporal):

            spatial_h = self.get_spatial_hidden(h)

            if time <= self.t_predict:
                data = input_data[:, time, :, :]
            else:
                output = self.outputLayer(h)
                data = self.caculate_next_input(
                                                data,
                                                input_data[:, time, :, :],
                                                output
                                            )
                outputs[:, time-self.t_predict-1, :, :] += data

            data = data.contiguous()

            h, c = self.cell(data, h, c, spatial_h)

        return outputs

if __name__ == "__main__":
    
    args = {}
    
    args["input_size"] = 6
    args["hidden_size"] = 64
    args["output_size"] = 2
    args["t_predict"] = 4
    args["n_head"] = 4
    args["gain"] = 1
    model = two_type_attn_model(args)

    inputs = torch.randn(7, 11, 5, 6)
    output = model(inputs)
    fake_loss = torch.mean(output)
    fake_loss.backward()
    