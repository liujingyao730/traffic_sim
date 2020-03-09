import os
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib
import sklearn.metrics as metrics
import torch

import conf
from CTM import seg_f_ctm
from data import two_type_data


def Flow_denstiy_diagram(prefix, length=50, lane=6):

    flow_file = os.path.join(conf.midDataPath, prefix+'CarOut.csv')
    number_file = os.path.join(conf.midDataPath, prefix+'Number.csv')

    flow = pd.read_csv(flow_file, index_col=0).values
    number = pd.read_csv(number_file, index_col=0).values
    
    y = flow.flatten()
    x = number.flatten()

    plt.plot(x, y, 'o', color='g')
    plt.xlabel("density (vehicle per cell)")
    plt.ylabel("flow (vehicle per timestep)")
    plt.show()


def caclutation_error(args, data_set):

    model = seg_f_ctm(args)

    test_data = data_set[0]
    test_data = test_data * data_set.std
    test_data = test_data + data_set.mean
    test_input_flow = test_data[:, 0, [1,4]]
    target = test_data[:, :, [2, 5]].sum(axis=2)
    outputs = model.eva(test_input_flow)

    print("seg")
    print("MAE ", metrics.mean_absolute_error(target, outputs))
    print("R2 ", metrics.r2_score(target, outputs))
    print("EVR ", metrics.explained_variance_score(target, outputs))

    predict_flow = outputs.sum(axis=1)
    real_flow = target.sum(axis=1)
    x = range(len(real_flow))

    plt.figure(13, figsize=(6, 4))
    plt.plot(x, real_flow, 's-', color='r', label='real')
    plt.plot(x, predict_flow, 'o-', color='g', label='predict')
    plt.xlabel('time')
    plt.ylabel('num_vehicle')
    plt.legend(loc='best')
    plt.title('flow with time')
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    heat = fig.add_subplot(311)
    im = heat.imshow(target.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("ground truth")
    heat = fig.add_subplot(312)
    im = heat.imshow(outputs.T, cmap=plt.cm.hot_r, vmin=0, vmax=20)
    plt.colorbar(im)
    plt.title("simulation result")
    heat = fig.add_subplot(313)
    im = heat.imshow(outputs.T-target.T, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("error")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()
    print("seg")
    print("MAE ", metrics.mean_absolute_error(real_flow, predict_flow))
    print("R2 ", metrics.r2_score(real_flow, predict_flow))
    print("EVR ", metrics.explained_variance_score(real_flow, predict_flow))

    return metrics.mean_absolute_error(target, outputs)

def parameter_calibration(prefix, seg, epoch, data_args):

    i = 0
    data_set = two_type_data(data_args, prefix, topology=seg)
    error_dict = {}

    args = {}
    args["number_vclass"] = 2
    args["cell_length"] = 50
    args["time_slot"] = 5
    args["lane_number"] = 6
    args["vlength"] = [4.5, 13]
    args["vspeed"] = [13.8, 10]
    args["cell_number"] = 10

    args["phi"] = [1, 0.5]
    args["sigma"] = 0.8
    args["over_take_factor"] = [1, 0.8]
    args["congest_factor"] = [1, 0.8]
    args["q"] = 30
    best_q = 30
    best_sigma = 0.8
    best_phi = [1, 0.5]
    best_over_take_factor = [1, 0.8]
    best_congest_factor = [1, 0.8]
    min_error = np.float('inf')

    while i < epoch:
        i += 1
        
        for q in range(10, 31, 1):
            args["q"] = q
            error = caclutation_error(args, data_set)
            if error < min_error:
                print("best error changed by q ", q, " from ", min_error, " to ", error)
                best_q = q
                min_error = error
        args["q"] = best_q
        
        for sigma in [i for i in range(1, 16, 1)]:
            args["sigma"] = sigma
            error = caclutation_error(args, data_set) 
            if error < min_error:
                print("best error changed by sigma ", sigma, " from ", min_error, " to ", error)
                best_sigma = sigma
                min_error = error
        args["sigma"] = best_sigma

        for phi in [i/10 for i in range(1, 11, 1)]:
            args["phi"][0] = phi
            error = caclutation_error(args, data_set)
            if error < min_error:
                print("best error changed by phi 1 ", phi, " from ", min_error, " to ", error)
                best_phi[0] = phi
                min_error = error
        args["phi"] = best_phi.copy()

        for phi in [i/10 for i in range(1, 11, 1)]:
            args["phi"][1] = phi
            error = caclutation_error(args, data_set)
            if error < min_error:
                print("best error changed by phi 2 ", phi, " from ", min_error, " to ", error)
                best_phi[1] = phi
                min_error = error
        args["phi"] = best_phi.copy()

        for over_take in [i/10 for i in range(1, 11, 1)]:
            args["over_take_factor"][0] = over_take
            error = caclutation_error(args, data_set)
            if error < min_error:
                print("best error changed by over_take_factor 1 ", over_take, " from ", min_error, " to ", error)
                best_over_take_factor[0] = over_take
                min_error = error
        args["over_take_factor"] = best_over_take_factor.copy()

        for over_take in [i/10 for i in range(1, 11, 1)]:
            args["over_take_factor"][1] = over_take
            error = caclutation_error(args, data_set)
            if error < min_error:
                print("best error changed by over_take_factor s ", over_take, " from ", min_error, " to ", error)
                best_over_take_factor[1] = over_take
                min_error = error
        args["over_take_factor"] = best_over_take_factor.copy()

        for congest in [i/10 for i in range(1, 11, 1)]:
            args["congest_factor"][0] = congest
            error = caclutation_error(args, data_set)
            if error < min_error:
                print("best error changed by over_take_factor 1 ", congest, " from ", min_error, " to ", error)
                best_congest_factor[0] = congest
                min_error = error
        args["congest_factor"] = best_congest_factor.copy()

        for congest in [i/10 for i in range(1, 11, 1)]:
            args["congest_factor"][1] = congest
            error = caclutation_error(args, data_set)
            if error < min_error:
                print("best error changed by over_take_factor 1 ", congest, " from ", min_error, " to ", error)
                best_congest_factor[1] = congest
                min_error = error
        args["congest_factor"] = best_congest_factor.copy()

        print("====================================")
        print("best error is ", min_error)
        print("best q ", best_q)
        print("best sigma ", best_sigma)
        print("best over take factor ", best_over_take_factor)
        print("best phi ", best_phi)
        print("best congest factor", best_congest_factor)
        print("====================================")
    

if __name__ == "__main__":
    prefix = "_US101_2"
    seg = 0
    args = {}
    args["number_vclass"] = 2
    args["cell_length"] = 50
    args["time_slot"] = 5
    args["lane_number"] = 6
    args["vlength"] = [4.5, 13]
    args["vspeed"] = [13.8, 10]
    args["cell_number"] = 10

    args["phi"] = [1, 1]
    args["sigma"] = 0.7
    args["over_take_factor"] = [1, 0.8]
    args["congest_factor"] = [1, 0.8]
    args["q"] = 18

    data_args = {"sim_step":0.1, "delta_T":5, "temporal_length":100, "t_predict":4}

    data_set = two_type_data(data_args, prefix, topology=seg)
    print(caclutation_error(args, data_set))
    #parameter_calibration(prefix="I-80_1", seg=0, epoch=1, data_args=data_args)