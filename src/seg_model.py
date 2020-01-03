import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import conf
from model import FCNet
from net_model import inter_model
from net_model import inter_LSTM
from net_model import seg_model

class discrete_net_model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.number_output_size = args["number_output_size"]
        self.speed_output_size = args["speed_output_size"]
        self.t_predict = args["t_predict"]
        self.inter_place = [1, 3, 4, 6]

        self.rand_range = args["rand_range"]
        self.use_cuda = args["use_cuda"]

        self.cell = seg_model(args)

        self.number_output_layer = torch.nn.Linear(self.hidden_size, self.number_output_size)
        self.speed_output_layer = torch.nn.Linear(self.hidden_size, self.speed_output_size)

    def get_spatial_hidden(self, h):

        [batch_size, spatial, _] = h.shape

        h_after = Variable(h.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_before = Variable(h.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        h_after[:, :-1, :] += h[:, 1:, :]
        h_before[:, 1:, :] += h[:, :-1, :]

        return h_after, h_before

    def caculate_delta_N(self, output_proba, init_input):

        [batch_size, temporal, spatial, output_size] = output_proba.shape
        [batch_size, temporal, _] = init_input.shape
        
        assert output_size == self.number_output_size

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

        [batch_size, spatial, output_size] = proba_list.shape

        proba_list = torch.nn.functional.softmax(proba_list, dim=2)

        mask = torch.rand(proba_list.shape)
        if self.use_cuda :
            mask = mask.cuda()
        mask = mask * self.rand_range

        proba_list += mask

        sample_result = torch.max(proba_list, dim=2)[1]

        return sample_result.unsqueeze(2).float()       

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
        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, self.number_output_size+self.speed_output_size).fill_(0).float())

        init_input = input_data[:, self.t_predict:-1, [0], 0].long()

        for time in range(temporal - 1):

            h_after, h_before = self.get_spatial_hidden(h)

            data = input_data[:, time, :, :].contiguous()

            h, c = self.cell(data, h, c, h_after, h_before)

            if time >= self.t_predict:
                outputs[:, time-self.t_predict, :, :self.number_output_size] += self.number_output_layer(h)
                outputs[:, time-self.t_predict, :, self.number_output_size:] += self.speed_output_layer(h)

        delta_N = self.caculate_delta_N(outputs[:, :, :, :self.number_output_size], init_input)

        return outputs, delta_N

    def infer(self, input_data, mod="uni"):

        [batch_size, temporal, spatial, input_size] = input_data.shape

        h = input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float()
        c = input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float()
        outputs = input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, 3+self.speed_output_size).fill_(0).float()

        init_input = input_data[:, self.t_predict:-1, [0], 0].long()

        for time in range(temporal):

            h_after, h_before = self.get_spatial_hidden(h)

            if time <= self.t_predict:
                data = input_data[:, time, :, :]
            else:
                proba = self.number_output_layer(h)
                speed = self.speed_output_layer(h)

                if mod == "uni":
                    output = self.sample(proba)
                elif mod == "random":
                    output = self.random_sample(proba)
                else :
                    raise RuntimeError("wrong mod !!!")
                number_data = data[:, :, :3]
                number_data = self.caculate_next_input(
                                                number_data,
                                                input_data[:, time, :, :3],
                                                output
                                            )

                outputs[:, time-self.t_predict-1, :, :3] += number_data
                outputs[:, time-self.t_predict-1, :, 3:] += speed
                data = outputs[:, time-self.t_predict-1, :, :]
            
            data = data.contiguous()

            h, c = self.cell(data, h, c, h_after, h_before)

        return outputs

class continuous_seg(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_size = args["input_size"]
        self.hidden_size = args["hidden_size"]
        self.output_hidden_size = args["output_hidden_size"]
        self.t_predict = args["t_predict"]
        self.output_size = args["output_size"]

        self.cell = seg_model(args)

        self.outputLayer = FCNet(layerSize=[self.hidden_size, self.output_hidden_size, self.output_size])

    def get_spatial_hidden(self, h):

        [batch_size, spatial, _] = h.shape

        h_after = Variable(h.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        h_before = Variable(h.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        h_after[:, :-1, :] += h[:, 1:, :]
        h_before[:, 1:, :] += h[:, :-1, :]

        return h_after, h_before

    def caculate_next_input(self, former_input, next_input, output):

        speed_output = output[:, 1:, 1:]
        number_output = output[:, :, [0]]
        speed_input = next_input[:, 0:1, 3:]

        In = torch.cat((next_input[:, 0:1, 1:2], number_output[:, :-1, :]), dim=1)
        former_number = former_input[:, :, [2]]
        number_caculate = former_number + In - number_output
        speed_next = torch.cat((speed_input, speed_output), dim=1)

        next_data = torch.cat((number_output, In, number_caculate, speed_next), dim=2)

        return next_data

    def forward(self, input_data):

        [batch_size, temporal, spatial, input_size] = input_data.shape
        self.spatial = spatial

        h = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())
        c = Variable(input_data.data.new(batch_size, spatial, self.hidden_size).fill_(0).float())

        outputs = Variable(input_data.data.new(batch_size, temporal-self.t_predict-1, spatial, input_size).fill_(0).float())

        for time in range(temporal - 1):

            h_after, h_before = self.get_spatial_hidden(h)

            data = input_data[:, time, :, :].contiguous()

            h, c = self.cell(data, h, c, h_after, h_before)

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

            h_after, h_before = self.get_spatial_hidden(h)

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

            h, c = self.cell(data, h, c, h_after, h_before)

        return outputs


class continuous_seg_nonspeed(nn.Module):

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
    