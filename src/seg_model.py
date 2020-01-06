import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import conf
from model import FCNet
from net_model import seg_model
from net_model import att_cell


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
        self.encoder_hidden = args["encoder_hidden"]
        self.decoder_input = args["decoder_input"]
        self.decoder_hidden = args["decoder_hidden"]
        self.n_head = args["n_head"]
        self.t_predict = args["t_predict"]
        self.output_size = args["output_size"]

        self.encoder = att_cell({"input_size":self.input_size, "hidden_size":self.encoder_hidden, "n_head":self.n_head})
        self.decoder = att_cell({"input_size":self.decoder_input, "hidden_size":self.decoder_hidden, "n_head":self.n_head})

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


if __name__ == "__main__":
    
    args = {}
    
    args["n_unit"] = 7
    args["input_size"] = 3
    args["hidden_size"] = 64
    args["output_size"] = 10
    args["t_predict"] = 4
    args["n_units"] = 7
    model = discrete_net_model(args)

    co_data = Variable(torch.randint(0, 10,(2, 9, 5, 3)))
    init = torch.randint(0, 10, (2, 4, 1)).long()
    bucket_number = [10, 11, 14, 15, 16, 17, 20]

    output, delta_N = model(co_data)
    infer_result = model.infer(co_data)
    sample_result = model.sample(output.view(2*4, 5, 10))
    fake_loss = torch.mean(output)
    fake_loss.backward()
    