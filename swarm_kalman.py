import pyswarm
import numpy
from rgbd_mocap.tracking.kalman import Kalman
from scipy import signal
from utils import get_next_frame_from_kalman, reorder_markers
from rgbd_mocap.camera.camera_converter import CameraConverter
from utils import refine_synchro
import biorbd

"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""
import utils, os
from post_process_data import ProcessData
import numpy as np
from biosiglive import OfflineProcessing
from utils import _convert_cluster_to_anato
import json, os
from scapula_cluster.from_cluster_to_anato import ScapulaCluster
from utils import load_data

part = "P10"
markers_from_source_tmp, names_from_source_tmp, forces, f_ext, emg, vicon_to_depth, peaks, rt = load_data(
    "/mnt/shared" + "/Projet_hand_bike_markerless/process_data", part, f"gear_20",
    True
)
dlc_data_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/P10/gear_20_15-01-2024_10_30_30{os.sep}marker_pos_multi_proc_3_crops_normal_500_down_b1_ribs_and_cluster_1_with_model_pp.bio"
dlc_data_meter, dlc_data_pixel, names, frame_idx = utils.load_data_from_dlc(None, dlc_data_path, part)
n_final = 1000
in_pixel = False
dlc_data_pixel = dlc_data_pixel[..., :n_final]
dlc_data_meter = dlc_data_meter[..., :n_final]
frame_idx = frame_idx[:n_final]
measurements_dir_path = "data_collection_mesurement"
calibration_matrix_dir = "../scapula_cluster/calibration_matrix"
measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_P9.json"))
measurements = measurement_data[f"with_depth"]["measure"]
calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"][
    "calibration_matrix_name"]
new_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                             measurements[4], measurements[5], calibration_matrix)
anato_from_cluster = _convert_cluster_to_anato(new_cluster, dlc_data_meter[:, -3:, :] * 1000) * 0.001
anato_tmp = anato_from_cluster.copy()
prefix = "/mnt/shared"
model_dir = prefix + "/Projet_hand_bike_markerless/RGBD"
model_path = f"{model_dir}/{part}/model_scaled_dlc_ribs_new_seth.bioMod"
model = biorbd.Model(model_path)
camera_converter = CameraConverter()
camera_converter.set_intrinsics("config_camera_files/config_camera_P10.json")
anato_in_pixel = anato_tmp[:3, ...].copy()
idx_cluster = names.index("clavac")
marker_names = names[:idx_cluster + 1] + ["scapaa", "scapia", "scapts"] + names[idx_cluster + 1:]
if in_pixel:
    for i in range(anato_tmp.shape[2]):
        anato_in_pixel[:2, :, i] = camera_converter.get_marker_pos_in_pixel(anato_from_cluster[:3, :, i].T).T
    dlc_data_tmp = np.concatenate((dlc_data_pixel[:, :idx_cluster + 1, :], anato_in_pixel[:3, ...],
                               dlc_data_pixel[:, idx_cluster + 1:, :]), axis=1)
else:
    dlc_data_tmp = np.concatenate((dlc_data_meter[:, :idx_cluster + 1, :], anato_in_pixel[:3, ...],
                                   dlc_data_meter[:, idx_cluster + 1:, :]), axis=1)

new_markers_dlc = ProcessData()._fill_and_interpolate(data=dlc_data_tmp,
                                                      idx=frame_idx,
                                                      shape=frame_idx[-1] - frame_idx[0],
                                                      fill=True)
new_markers_dlc, reorder_names = reorder_markers(new_markers_dlc[:, :-3, :], model,
                                                                 marker_names[:-3])

markers_dlc_hom = np.ones((4, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
markers_dlc_hom[:3, :, :] = new_markers_dlc

for k in range(new_markers_dlc.shape[2]):
    new_markers_dlc[..., k] = np.dot(np.array(rt), markers_dlc_hom[:, :, k])[:3, :]

new_markers_dlc_filtered = np.zeros((3, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
for i in range(3):
    new_markers_dlc_filtered[i, :8, :] = OfflineProcessing().butter_lowpass_filter(
        new_markers_dlc[i, :8, :],
        2, 60, 2)
    new_markers_dlc_filtered[i, 8:, :] = OfflineProcessing().butter_lowpass_filter(
        new_markers_dlc[i, 8:, :],
        10, 60, 2)

if in_pixel:
    new_markers_dlc_filtered[2, ...] *= 500

# plt.show()


# process_noise = [1] * dlc_data.shape[1]
# measurement_noise = [1] * dlc_data.shape[1]
measurement_noise_factor = [5] * dlc_data_tmp.shape[1]
process_noise_factor = [5] * dlc_data_tmp.shape[1]
measurement_noise = [30] * 17
proc_noise = [3] * 17
measurement_noise[8] = 10
proc_noise[8] = 3
measurement_noise[3] = 50
proc_noise[3] = 1
measurement_noise[5:8] = [150] * 3
proc_noise[5:8] = [3] * 3
measurement_noise[4] = 50
proc_noise[4] = 1

function_inputs = measurement_noise + proc_noise
# error_cov_post_factor = [0] * dlc_data.shape[1]
# error_cov_pre_factor = [0] * dlc_data.shape[1]
# function_inputs = measurement_noise_factor + process_noise_factor # + error_cov_post_factor + error_cov_pre_factor
# function_inputs = process_noise + measurement_noise

lb = [0] * len(function_inputs)
ub = [800] * len(function_inputs)

param_names = [f"P_{i}" for i in range(len(function_inputs))]
param_dic = {param_names[i]: function_inputs[i] for i in range(len(function_inputs))}
dlc_kalman = np.zeros((new_markers_dlc.shape[0], new_markers_dlc.shape[1], new_markers_dlc.shape[2]))

def get_all_frame(function_inputs):
    import json
    measurements_dir_path = "data_collection_mesurement"
    calibration_matrix_dir = "../scapula_cluster/calibration_matrix"
    measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{part}.json"))
    measurements = measurement_data[f"with_depth"]["measure"]
    calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"][
        "calibration_matrix_name"]
    new_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                                 measurements[4], measurements[5], calibration_matrix)
    count = 0
    count_bis = 0
    all_kalman = None
    data_dlc_tmp = dlc_data_pixel if in_pixel else dlc_data_meter
    for i in range(frame_idx[0], frame_idx[-1]):
        if count_bis == dlc_data_pixel.shape[2]:
            break
        if i in frame_idx:
            dlc_data_tmp = data_dlc_tmp[:, :, count:count + 1]
            count += 1
        else:
            dlc_data_tmp = None
        if count_bis == 50:
            pass
        markers_tmp, all_kalman = get_next_frame_from_kalman(
            kalman_instance=all_kalman, markers_data=dlc_data_tmp, params=function_inputs, scapula_cluster=new_cluster,
            idx_cluster_markers=idx_cluster, camera_converter=camera_converter, in_pixel=in_pixel,
            convert_cluster_before_kalman=True, return_in_meter=True, forward=0.5, fps=60, rt_matrix=rt)
        # dlc_kalman[:, :, count_bis:count_bis + 1] = markers_tmp[:, :-3, 0]
        dlc_kalman[:, :, count_bis:count_bis + 1], reorder_names = reorder_markers(markers_tmp[:, :-3, None], model,
                                                     marker_names[:-3])
        if in_pixel:
            dlc_kalman[2, :, count_bis:count_bis + 1] *= 500

        count_bis += 1
    return dlc_kalman


import matplotlib.pyplot as plt
optim = [np.float64(800.0), np.float64(527.584372499839), np.float64(382.03048432114883), np.float64(224.3608489209641), np.float64(784.3371161043293), np.float64(380.79432020007215), np.float64(623.5944564826665), np.float64(631.9786227664722), np.float64(442.83958008720055), np.float64(14.85835554874223), np.float64(30.608323096273658), np.float64(7.4105360001634235), np.float64(42.2225846432281), np.float64(0.0), np.float64(219.6481543256065), np.float64(540.9727185955664), np.float64(800.0), np.float64(644.4104678245333), np.float64(93.98007907558751), np.float64(41.52665336376876), np.float64(243.89726481748224), np.float64(450.79324904593665), np.float64(745.0435023831777), np.float64(476.09849164603236), np.float64(96.06973857571838), np.float64(800.0), np.float64(206.4458409605946), np.float64(706.4080053617395), np.float64(526.3138860132597), np.float64(800.0), np.float64(720.9021673508166), np.float64(639.8780637584136), np.float64(33.21978867871258), np.float64(395.70483616964316)]

measurement_noise = [30] * 17
proc_noise = [3] * 17
measurement_noise[8] = 10
proc_noise[8] = 3
measurement_noise[3] = 50
proc_noise[3] = 1
measurement_noise[5:8] = [150] * 3
proc_noise[5:8] = [3] * 3
measurement_noise[4] = 50
proc_noise[4] = 1

function_inputs = measurement_noise + proc_noise
measurement_noise = [2] * 17
proc_noise = [1] * 17
measurement_noise[:8] = [5] * 8
proc_noise[:8] = [1e-1] * 8

# measurement_noise = [800] * 17
# proc_noise = [1e-1] * 17
function_inputs = measurement_noise + proc_noise

# kalman_params[0:8] = [0.4] * 8
# kalman_params[0 + 17:8 + 17] = [1e-3] * 8
# kalman_params[8] = 1e-2
# kalman_params[8 + 17] = 1e-3
# function_inputs = [float(optim) for optim in optim]
# print([float(optim) for optim in optim])
dlc_mark = get_all_frame(function_inputs)
dlc_mark, idx = refine_synchro(new_markers_dlc_filtered,
                               dlc_mark,
                               plot_fig=False)
# optim = [1] * len(optim)
# function_inputs = [float(optim) for optim in optim]
# dlc_mark_bis = get_all_frame(function_inputs)

for i in range(new_markers_dlc.shape[1]):
    plt.subplot(4, 5, i + 1)
    for j in range(3):
        plt.plot(dlc_mark[j, i, :], "g")
        # plt.plot(dlc_mark_bis[j, i, :], "b")
        plt.plot(new_markers_dlc_filtered[j, i, :], "--", c="r")
if in_pixel:
    dlc_mark[2, ...] /= 500
    new_markers_dlc_filtered[2, ...] /= 500
    dlc_in_meter = dlc_mark.copy()
    new_markers_dlc_filtered_in_meter = new_markers_dlc_filtered.copy()
    for k in range(dlc_mark.shape[2]):
        dlc_in_meter[..., k] = camera_converter.get_markers_pos_in_meter(dlc_mark[:, :, k].T)
        new_markers_dlc_filtered_in_meter[..., k] = camera_converter.get_markers_pos_in_meter(new_markers_dlc_filtered[:, :, k].T)
    plt.figure("in_meter")
    for i in range(new_markers_dlc.shape[1]):
        plt.subplot(4, 5, i + 1)
        for j in range(3):
            plt.plot(dlc_in_meter[j, i, :], "g")
            # plt.plot(dlc_mark_bis[j, i, :], "b")
            plt.plot(new_markers_dlc_filtered_in_meter[j, i, :], "--", c="r")
plt.show()

def objective_function(function_inputs):
    def compute_error(markers_depth, markers_vicon):
        return np.sqrt(((markers_depth - markers_vicon) ** 2).mean(axis=2)).mean(axis=0).mean()

    dlc_mark = get_all_frame(function_inputs)
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    #squre root of two curves
    # fitness = np.linalg.norm(dlc_kalman[..., 2:-2] - new_markers_dlc_filtered[..., 3:])

    # def meanfreq(x, fs):
    #     f, Pxx_den = signal.periodogram(x, fs, detrend=False)
    #     Pxx_den = np.reshape(Pxx_den, (1, -1))
    #     width = np.tile(f[1] - f[0], (1, Pxx_den.shape[1]))
    #     f = np.reshape(f, (1, -1))
    #     P = Pxx_den * width
    #     pwr = np.sum(P)
    #     mnfreq = np.dot(P, f.T) / pwr
    #     return mnfreq

    # dlc_mark, idx = refine_synchro(new_markers_dlc_filtered,
    #                                dlc_mark,
    #                                plot_fig=True)
    # mean_freq_tot = 0
    # lag = 0
    # for i in range(3):
    #     for j in range(dlc_mark.shape[1]):
    #         x = dlc_mark[i, j, :]
    #         y = new_markers_dlc_filtered[i, j, :]
    #         mean_freq = meanfreq(y, 60)
    #         mean_freq_2 = meanfreq(x, 60)
    #         mean_freq_tot += float(abs(mean_freq - mean_freq_2) * 100)
    #         correlation = signal.correlate(x, y, mode="full")
    #         lags = signal.correlation_lags(x.size, y.size, mode="full")
    #         lag += lags[np.argmax(correlation)] * 1000

    # for i in range(new_markers_dlc.shape[1]):
    #     plt.subplot(4, 5, i + 1)
    #     for j in range(3):
    #         plt.plot(dlc_mark[j, i, :])
    #         plt.plot(new_markers_dlc_filtered[j, i, :], "--")
    # plt.show()
    return compute_error(dlc_mark[..., 50:], new_markers_dlc_filtered[..., 50:])


# options = {'popsize': 600, 'tolfun': 1e-5, 'tolx': 1e-5,
#            'maxfevals': 500000, "bounds": [lb, ub],
#            "maxiter": 100000}

# opts = cma.CMAOptions()
# for key in options.keys():
#     opts.set(key, options[key])
optim, cost = pyswarm.pso(objective_function, lb, ub, swarmsize=50, omega=0.5, phip=0.5, phig=0.5, maxiter=500, debug=True)
print([optim for optim in optim])
function_inputs = [float(optim) for optim in optim]
dlc_mark = get_all_frame(function_inputs)
dlc_mark, idx = refine_synchro(new_markers_dlc_filtered,
                               dlc_mark,
                               plot_fig=True)
plt.show()