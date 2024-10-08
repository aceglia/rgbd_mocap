import os
from scipy.signal import find_peaks

import biorbd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import ceil
import biorbd
from utils import load_data
try:
    from msk_utils import process_all_frames, get_tracking_idx
except:
    pass

from biosiglive import (
    OfflineProcessing, OfflineProcessingMethod, load, MskFunctions,
    InverseKinematicsMethods, load, save, RealTimeProcessing)
from biosiglive.processing.msk_utils import ExternalLoads

import glob


def _convert_string(string):
    return string.lower().replace("_", "")


def rmse(data, data_ref):
    return np.sqrt(np.mean(((data - data_ref) ** 2), axis=-1))


def compute_error(biomech_results_list, vicon_to_depth_idx):
    n_markers_depth = markers_depth.shape[1]
    err_markers = np.zeros((n_markers_depth, 1))
    # new_markers_depth_int = OfflineProcessing().butter_lowpass_filter(new_markers_depth_int, 6, 120, 4)
    for i in range(len(vicon_to_depth_idx)):
        # ignore NaN values
        # if i not in [4, 5, 6]:
        nan_index = np.argwhere(np.isnan(markers_vicon[:, vicon_to_depth_idx[i], :]))
        new_markers_depth_tmp = np.delete(markers_depth[:, i, :], nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(markers_vicon[:, vicon_to_depth_idx[i], :], nan_index, axis=1)
        nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
        new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
        err_markers[i, 0] = np.median(np.sqrt(
            np.mean(((new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000) ** 2), axis=0)))

    err_q = []
    for i in range(state_depth.shape[0]):
        err_q.append(np.mean(np.sqrt(np.mean(((state_depth[i, :] - state_vicon[i, :]) ** 2), axis=0))))

    err_q_dot = []
    for i in range(state_depth.shape[0]):
        err_q_dot.append(np.mean(np.sqrt(np.mean(((state_depth[i, :] - state_vicon[i, :]) ** 2), axis=0))))

    return err_markers, err_q


def _get_vicon_to_depth_idx(names_depth=None, names_vicon=None):
    vicon_markers_names = [_convert_string(name) for name in names_vicon]
    depth_markers_names = [_convert_string(name) for name in names_depth]
    vicon_to_depth_idx = []
    for name in vicon_markers_names:
        if name in depth_markers_names:
            vicon_to_depth_idx.append(vicon_markers_names.index(name))
    return vicon_to_depth_idx


def _interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(markers_depth.shape[0]):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def _interpolate_data_2d(data, shape):
    new_data = np.zeros((data.shape[0], shape))
    x = np.linspace(0, 100, data.shape[1])
    f_mark = interp1d(x, data)
    x_new = np.linspace(0, 100, int(new_data.shape[1]))
    new_data = f_mark(x_new)
    return new_data


def refine_synchro(depth_markers, vicon_markers, vicon_to_depth_idx):
    error = np.inf
    depth_markers_tmp = np.zeros((3, depth_markers.shape[1], depth_markers.shape[2]))
    vicon_markers_tmp = np.zeros((3, vicon_markers.shape[1], vicon_markers.shape[2]))
    idx = 0
    for i in range(30):
        vicon_markers_tmp = vicon_markers[:, :, i:] if i != 0 else vicon_markers
        depth_markers_tmp = _interpolate_data(depth_markers, vicon_markers_tmp.shape[2])
        error_markers, _ = compute_error(
            depth_markers_tmp, vicon_markers_tmp, vicon_to_depth_idx, np.zeros((1,1)), np.zeros((1,1))
        )
        error_tmp = np.mean(error_markers)
        # print(error_tmp, i)
        if error_tmp < error:
            error = error_tmp
            idx = i
        else:
            break
    return depth_markers_tmp, vicon_markers_tmp, idx


def reorder_markers(markers, model, names):
    model_marker_names = [_convert_string(model.markerNames()[i].to_string()) for i in range(model.nbMarkers())]
    assert len(model_marker_names) == len(names)
    assert len(model_marker_names) == markers.shape[1]
    count = 0
    reordered_markers = np.zeros((markers.shape[0], len(model_marker_names), markers.shape[2]))
    for i in range(len(names)):
        if names[i] == "elb":
            names[i] = "elbow"
        if _convert_string(names[i]) in model_marker_names:
            reordered_markers[:, model_marker_names.index(_convert_string(names[i])),
            :] = markers[:, count, :]
            count += 1
    return reordered_markers


def plot_results(all_results, markers, track_idx, vicon_to_depth, sources=("depth", "vicon", "minimal_vicon"),
                 stop_frame=None, cycle=False, f_ext=None, models=None):

    results_from_sources = []
    count = 0
    q_new, qdot_new, qddot_new, tau_new = [], [], [], []
    for key in all_results.keys():
        q, qdot, qddot, tau = compute_id(all_results[key]["q_raw"], models[count], f_ext)
        q_new.append(q)
        qdot_new.append(qdot)
        qddot_new.append(qddot)
        tau_new.append(tau)
        count +=1
        results_from_sources.append(all_results[key]) if not cycle else results_from_sources.append(all_results[key]["cycles"])
        print(f"mean time for source: {key} ", np.mean(all_results[key]["time"]["tot"]))

    if cycle:
        results_from_sources_tmp = []
        for result in results_from_sources:
            dic_tmp = {}
            for key in result.keys():
                if isinstance(result[key], np.ndarray):
                    dic_tmp[key] = np.median(result[key], axis=0)
                else:
                    dic_tmp[key] = result[key]
            results_from_sources_tmp.append(dic_tmp)
        results_from_sources = results_from_sources_tmp

    # plot markers
    plt.figure("markers")
    stop_frame = markers[0].shape[2] if stop_frame is None else stop_frame
    color = ["b", "orange", "g"]
    # line = [".-", "-", "--"]
    line = ["-", "-", "-"]

    for i in range(markers[0].shape[1]):
        plt.subplot(4, ceil(markers[0].shape[1] / 4), i+1)
        for j in range(3):
            for k in range(len(results_from_sources)):
                idx = vicon_to_depth[i] if sources[k] == "vicon" else i
                plt.plot(results_from_sources[k]["markers"][j, idx, :stop_frame], line[k], color=color[k])

        plt.legend(sources)

    key = "q" if isinstance(results_from_sources[0]["q"], np.ndarray) else "q_raw"
    # plot q

    plt.figure("q")
    for i in range(results_from_sources[0][key].shape[0]):
        plt.subplot(4, ceil(results_from_sources[0][key].shape[0] / 4), i+1)
        for k in range(len(results_from_sources)):

            plt.plot(results_from_sources[k][key][i, :] * 180/np.pi, line[k], color=color[k])
            #plt.plot(q_new[k][i, :]* 180/np.pi, color='r')
        plt.legend(sources)

        # plot qdot
    plt.figure("qdot")
    for i in range(results_from_sources[0]["q_dot"].shape[0]):
        plt.subplot(4, ceil(results_from_sources[0]["q_dot"].shape[0] / 4), i + 1)
        for k in range(len(results_from_sources)):
            plt.plot(results_from_sources[k]["q_dot"][i, :], line[k], color=color[k])
            #plt.plot(qdot_new[k][i, :], color='r')

        plt.legend(sources)

    if not isinstance(results_from_sources[0]["q_ddot"], list):
        # plot qddot
        plt.figure("qddot")
        for i in range(results_from_sources[0]["q_ddot"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["q_ddot"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["q_ddot"][i, :], line[k], color=color[k])
            plt.legend(sources)

        # plot tau
        plt.figure("tau")
        for i in range(results_from_sources[0]["tau"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["tau"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["tau"][i, :], line[k], color=color[k])
                #plt.plot(tau_new[k][i, :] , color='r')
            plt.legend(sources)

    # plot muscle activations
    if not isinstance(results_from_sources[0]["mus_act"], list):
        plt.figure("muscle activations")
        for i in range(results_from_sources[0]["mus_act"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["mus_act"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["mus_act"][i, :], line[k], color=color[k])
            if not isinstance(results_from_sources[0]["emg_proc"], list):
                if i in track_idx:
                    plt.plot(results_from_sources[0]["emg_proc"][track_idx.index(i), :stop_frame])
            plt.legend(sources)

    if not isinstance(results_from_sources[0]["res_tau"], list):
        # plot residual tau
        plt.figure("residual tau")
        for i in range(results_from_sources[0]["res_tau"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["res_tau"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["res_tau"][i, :], line[k], color=color[k])
            plt.legend(sources)

    if not isinstance(results_from_sources[0]["jrf"], list):
    # plot jrf
        plt.figure("jrf")
        for i in range(results_from_sources[0]["jrf"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["jrf"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["jrf"][i, :], line[k], color=color[k])
            plt.legend(sources)
    plt.show()


def process_cycles(all_results, peaks, n_peaks=None):
    for key in all_results.keys():
        data_size = all_results[key]["q_raw"].shape[1]
        dic_tmp = {}
        for key2 in all_results[key].keys():
            if key2 == "cycle":
                continue
            array_tmp = None
            if not isinstance(all_results[key][key2], np.ndarray):
                dic_tmp[key2] = []
                continue
            if n_peaks and n_peaks > len(peaks) - 1:
                raise ValueError("n_peaks should be less than the number of peaks")
            for k in range(len(peaks) - 1):
                if peaks[k + 1] > data_size:
                    break
                interp_function = _interpolate_data_2d if len(all_results[key][key2].shape) == 2 else _interpolate_data
                if array_tmp is None:
                    array_tmp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], 120)
                    array_tmp = array_tmp[None, ...]
                else:
                    data_interp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], 120)
                    array_tmp = np.concatenate((array_tmp, data_interp[None, ...]), axis=0)
            dic_tmp[key2] = array_tmp
        all_results[key]["cycles"] = dic_tmp
    return all_results

def compute_id(q_init, model, f_ext):
    f_ext = f_ext.copy()
    q_filtered = OfflineProcessing().butter_lowpass_filter(q_init,
                                                           6, 120, 2)
    # import matplotlib.pyplot as plt
    # plt.plot(q_init[0, :])
    # plt.plot(q_filtered[0, :])
    # plt.show()
    qdot_new = np.zeros_like(q_init)
    qdot_new[:, 1:-1] = (q_filtered[:, 2:] - q_filtered[:, :-2]) / (2 / 120)
    qdot_new[:, 0] = q_filtered[:, 1] - q_filtered[:, 0]
    qdot_new[:, -1] = q_filtered[:, -1] - q_filtered[:, -2]

    # for i in range(1, q_filtered.shape[1] - 2):
    #     qdot_new[:, i] = (q_filtered[:, i + 1] - q_filtered[:, i - 1]) / (2 / 120)
    qddot_new = np.zeros_like(qdot_new)
    qddot_new[:, 1:-1] = (qdot_new[:, 2:] - qdot_new[:, :-2]) / (2 / 120)
    qddot_new[:, 0] = qdot_new[:, 1] - qdot_new[:, 0]
    qddot_new[:, -1] = qdot_new[:, -1] - qdot_new[:, -2]


    # for i in range(1, qdot_new.shape[1] - 2):
    #     qddot_new[:, i] = (qdot_new[:, i + 1] - qdot_new[:, i - 1]) / (2 / 120)
    tau = np.zeros_like(q_init)
    for i in range(q_init.shape[1]):
        if f_ext is not None:
            B = [0, 0, 0, 1]
            all_jcs = model.allGlobalJCS(q_filtered[:, i])
            RT = all_jcs[-1].to_array()
            # A = RT @ A
            B = RT @ B
            vecteur_OB = B[:3]
            f_ext[:3, i] = f_ext[:3, i] + np.cross(vecteur_OB, f_ext[3:6, i])
            # force_global = change_ref_for_global(ind_1, q, model, force_locale)
            # ddq = nlp.model.forward_dynamics(q, qdot, tau, force_global)
            ext = model.externalForceSet()
            ext.add("hand_left", f_ext[:, i])
            tau[:, i] = model.InverseDynamics(q_filtered[:, i], qdot_new[:, i], qddot_new[:, i], ext).to_array()
        else:
            tau[:, i] = model.InverseDynamics(q_filtered[:, i], qdot_new[:, i], qddot_new[:, i]).to_array()
        #tau[:, i] -= model.passiveJointTorque(q_filtered[:, i], qdot_new[:, i]).to_array()
        #tau[3, i] += 15 * np.exp(-40*q_filtered[3, i] + 18) + 1
    return q_filtered, qdot_new, qddot_new, tau

def main(model_dir, participants, processed_data_path, save_data=False, plot=True, results_from_file=False, stop_frame=None):
    emg_names = ["PectoralisMajorThorax_M",
                 "BIC",
                 "TRI_lat",
                 "LatissimusDorsi_S",
                 'TrapeziusScapula_S',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                 'DeltoideusScapula_P']
    # emg_names = ["PECM",
    #              "bic",
    #              "tri",
    #              "LAT",
    #              'TRP1',
    #              "DELT1",
    #              'DELT2',
    #              'DELT3']
    source = ["depth", "vicon", "minimal_vicon"]
    model_source = ["depth", "vicon", "depth"]
    processed_source = []
    models=[]
    for part in participants:
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file and "result_biomech" not in file and "3_crops" in file]
        for file in all_files:
            # if "gear_15" not in file:
            #     continue
            print(f"Processing participant {part}, trial : {file}")
            markers_from_source, names_from_source, forces, f_ext, emg, vicon_to_depth, peaks = load_data(
                processed_data_path, part, file, False
            )
            # model_path = f"{model_dir}/{part}/model_scaled_{source[0]}_seth.bioMod"
            if not results_from_file:
                all_results = {}
                for s in range(0, len(markers_from_source)):
                    model_path = f"{model_dir}/{part}/model_scaled_{model_source[s]}_new_seth.bioMod"
                    track_idx = get_tracking_idx(biorbd.Model(model_path), emg_names)
                    processed_source.append(source[s])

                    reorder_marker_from_source = reorder_markers(markers_from_source[s][:, :-3, :],
                                                 biorbd.Model(model_path),
                                                 names_from_source[s][:-3])
                    model = biorbd.Model(model_path)


                    msk_function = MskFunctions(model=model, data_buffer_size=6, system_rate=120)
                    result_biomech = process_all_frames(reorder_marker_from_source, msk_function,
                                                        source[s],
                                                        forces, (1000, 10), emg,
                                                        f_ext,
                                                        compute_id=True, compute_so=False, compute_jrf=False,
                                                        stop_frame=stop_frame,
                                                        file=f"{processed_data_path}/{part}" + "/" + file,
                                                        print_optimization_status=False, filter_depth=False,
                                                        emg_names=emg_names
                                                        )
                    result_biomech["markers"] = markers_from_source[s][..., :stop_frame]
                    result_biomech["track_idx"] = track_idx
                    result_biomech["vicon_to_depth"] = vicon_to_depth

                    all_results[source[s]] = result_biomech
                    all_results = process_cycles(all_results, peaks, n_peaks=None)

                    print("Done for source ", source[s])
                    models.append(msk_function.model)
                    # import bioviz
                    # b = bioviz.Viz(loaded_model=msk_function.model, show_floor=False)
                    # b.load_movement(result_biomech["q_raw"])
                    # b.load_experimental_markers(reorder_marker_from_source)
                    # b.exec()
                if save_data:
                    save(all_results, f"{processed_data_path}/{part}/result_biomech_{Path(file).stem}_seth_new_model.bio",
                         safe=False)
            else:
                all_results = load(f"{processed_data_path}/{part}/result_biomech_{Path(file).stem}_wt_filter.bio")
                all_results = process_cycles(all_results, peaks, n_peaks=None)
                processed_source = list(all_results.keys())
            if plot:
                plot_results(all_results, markers_from_source, track_idx, vicon_to_depth,
                             stop_frame=stop_frame, sources=processed_source, cycle=True, models=models, f_ext=f_ext)


if __name__ == '__main__':
    model_dir = "/mnt/shared/Projet_hand_bike_markerless/RGBD"
    participants = ["P10", "P15", "P16"]#, "P14", "P15", "P16"]#, "P16"]  # ,"P9", "P10",
    processed_data_path = "/mnt/shared/Projet_hand_bike_markerless/process_data"
    main(model_dir, participants, processed_data_path, save_data=False, results_from_file=False, stop_frame=1000,
         plot=True)