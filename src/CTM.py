import numpy as np
import os
import yaml
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class f_mctm_cell(object):

    def __init__(self, args):

        super().__init__()

        self.number_vclass = args["number_vclass"]
        self.cell_length = args["cell_length"]
        self.time_slot = args["time_slot"]
        self.lane_number = args["lane_number"]

        self.vlength = np.array(args["vlength"])
        self.vspeed = np.array(args["vspeed"])
        self.v_max_index = np.argmax(self.vspeed)
        self.v_max = self.vspeed[self.v_max_index]
        self.l_base = self.vlength[self.v_max_index]
        self.vspeed = self.vspeed / self.v_max
        self.vlength = self.vlength / self.l_base

        self.c = self.cell_length * self.lane_number / self.l_base

        self.phi = np.array(args["phi"])
        self.sigma = np.array(args["sigma"])
        self.over_take_factor = np.array(args["over_take_factor"])
        self.congest_factor = np.array(args["congest_factor"])
        self.q = np.array(args["q"])
        self.number = np.zeros(self.number_vclass)
        self.a = np.zeros(self.number_vclass)
        self.b = np.zeros(self.number_vclass)

    def receiving_capability(self):
    
        return min(self.q, self.sigma*(self.c-np.sum(self.vlength*self.phi*self.number))) 

    def head_of_cell(self, r):

        tot_output_flow = np.sum(self.a * self.vlength)

        if tot_output_flow < r:
            output_flow = self.a
        else:
            over_take_flow = self.over_take_factor * self.a
            factor = r / np.sum(over_take_flow * self.vlength)
            output_flow = over_take_flow * factor

        return output_flow

    def end_of_cell(self):

        ka = (2*self.vspeed - 1) / self.vspeed
        send_cap = self.a + ka * self.b
        s = np.sum(send_cap)

        if s > self.q and s < np.sum(self.c * self.congest_factor):

            v_bar = np.sum(self.vspeed * send_cap) / s
            v = np.where(self.vspeed < v_bar, self.vspeed, v_bar)
            ka = (2*v - 1) / v

        elif s > self.q:

            v = np.ones(self.vspeed.shape) * np.min(self.vspeed)
            ka = (2*v - 1) / v

        next_cell_reci = self.receiving_capability()
        trans_head_cap = next_cell_reci - np.sum(self.vlength * self.a)

        if trans_head_cap < 0 :

            output_flow = np.zeros(self.b.shape)

        elif trans_head_cap < np.sum(self.b * ka):

            base = ka * self.b
            output_flow = (base * trans_head_cap) / np.sum(base * self.vlength)

        else:

            output_flow = ka * self.b

        return output_flow

    def caculation(self, in_flow, recive):

        output_head = self.head_of_cell(recive)
        output_end = self.end_of_cell()

        output_flow = output_head + output_end
        self.a = self.number - output_flow
        self.b = in_flow
        self.number = self.a + self.b

        return output_flow


class seg_f_ctm(object):

    def __init__(self, args):

        self.length = args["cell_number"]
        self.cell_list = []
        self.revice_cap = []
        for i in range(args["cell_number"]):
            self.cell_list.append(f_mctm_cell(args))

    def single_iter(self, input_flow):

        for i in range(self.length-1):
            self.revice_cap.append(self.cell_list[i+1].receiving_capability())
        
        self.revice_cap.append(np.array(100))
        
        for i in range(self.length):
            input_flow = self.cell_list[i].caculation(input_flow, self.revice_cap[i])

    def eva(self, input_flows):

        outputs = np.zeros([len(input_flows), self.length])

        for i in range(len(input_flows)):
            self.single_iter(input_flows[i])
            output = self.show()
            outputs[i, :] += output

        return outputs

    def show(self):

        result = np.array([])
        for i in range(self.length):
            result = np.append(result, np.sum(self.cell_list[i].number))

        return result 

if __name__ == "__main__":
    args = {}
    args["number_vclass"] = 2
    args["cell_length"] = 150
    args["time_slot"] = 5
    args["lane_number"] = 6
    args["vlength"] = [4.5, 13]
    args["vspeed"] = [13.8, 10]
    args["cell_number"] = 6

    args["phi"] = [1, 0.5]
    args["sigma"] = 0.7
    args["over_take_factor"] = [1, 0.8]
    args["congest_factor"] = [1, 0.8]
    args["q"] = 30

    seg = seg_f_ctm(args)
    inputs = [[8, 7], [6, 5], [6, 7], [6, 7]]
    output = seg.eva(inputs)
    print(output)
