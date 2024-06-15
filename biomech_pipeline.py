import os
from scipy.signal import find_peaks

import biorbd
from post_process_data import ProcessData

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import ceil
import biorbd
from utils import load_data, load_data_from_dlc
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
                 stop_frame=None, cycle=False):

    results_from_sources = []

    for key in all_results.keys():
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
        plt.legend(sources)

        # plot qdot
    plt.figure("qdot")
    for i in range(results_from_sources[0]["q_dot"].shape[0]):
        plt.subplot(4, ceil(results_from_sources[0]["q_dot"].shape[0] / 4), i + 1)
        for k in range(len(results_from_sources)):
            plt.plot(results_from_sources[k]["q_dot"][i, :], line[k], color=color[k])
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
            if key2 == "cycle" or key2 == "rt_matrix":
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

def _get_dlc_data(model, filt, part, file, path, labeled_data_path, rt, shape):
    dlc_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_{model}_{filt}_pp.bio"
    data_dlc, data_labeling = load_data_from_dlc(labeled_data_path, dlc_data_path, part, file)
    new_markers_dlc = np.zeros((3,
                                data_dlc["markers_in_meters"].shape[1],
                                data_dlc["markers_in_meters"].shape[2]
                                ))
    markers_dlc_hom = np.ones((4, data_dlc["markers_in_meters"].shape[1], data_dlc["markers_in_meters"].shape[2]))
    markers_dlc_hom[:3, ...] = data_dlc["markers_in_meters"][:3, ...]
    frame_idx = data_dlc["frame_idx"]

    # markers_from_source = [data_dlc["markers_in_meters"], data_init["markers_depth_initial"],
    #                        markers_vicon]

    # if "gear_15" not in file:
    #     continue

    for k in range(new_markers_dlc.shape[2]):
        new_markers_dlc[:, :, k] = np.dot(np.array(rt), markers_dlc_hom[:, :, k])[:3, :]
    new_markers_dlc = ProcessData()._fill_and_interpolate(data=new_markers_dlc,
                                                          idx=frame_idx,
                                                          shape=shape,
                                                          fill=True)
    new_markers_dlc_filtered = np.zeros((3, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
    for i in range(3):
        new_markers_dlc_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(
            new_markers_dlc[i, :, :],
            2, 120, 2)
    return new_markers_dlc_filtered, frame_idx


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
    source = ["depth", "dlc", "dlc_filtered", "minimal_vicon"]#, "depth"]#, "minimal_vicon"]
    # model_source = ["depth", "vicon", "depth"]
    model_source = ["depth", "dlc", "dlc", "minimal_vicon"]
    processed_source = []
    models = ["normal"] #"non_augmented", "hist_eq",
    filtered = ["alone", "filtered"]
    for part in participants:
        all_files = os.listdir(f"{processed_data_path}/{part}")
        # all_files = [file for file in all_files if "gear" in file and "result_biomech" not in file and "3_crops" in file]
        all_files = [file for file in all_files if "gear" in file and "less" not in file and "more" not in file]
        for file in all_files:
            # if part == "P14" and "gear_5" in file:
            #     continue
            path = f"{processed_data_path}{os.sep}{part}{os.sep}{file}"
            labeled_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_pp.bio"
            print(f"Processing participant {part}, trial : {file}")
            markers_from_source, names_from_source, forces, f_ext, emg, vicon_to_depth, peaks, rt = load_data(
                "Q://Projet_hand_bike_markerless/process_data", part, f"{file.split('_')[0]}_{file.split('_')[1]}",
                True
            )
            for model in models:
                for f, filt in enumerate(filtered):
                    shape = markers_from_source[0].shape[2]
                    marker_dlc_filtered, frame_idx = _get_dlc_data(model, filt, part, file, path, labeled_data_path, rt, shape)

                    # emg, forces, f_ext, vicon_to_depth, peaks = None, None, None, None, None
                    markers_from_source[f + 1] = marker_dlc_filtered
                    names_from_source[f + 1] = names_from_source[0]

                    if not results_from_file:
                        all_results = {}
                        markers_to_save = []
                        q_to_save = []
                        q_dot_to_save = []
                        dic_to_save_tmp = {}
                        existing_data = False
                        for s in range(0, len(markers_from_source)):
                            if os.path.exists(f"{path}{os.sep}reoriented_dlc_markers.bio"):
                                data = load(f"{path}{os.sep}reoriented_dlc_markers.bio")
                                for key in data.keys():
                                    if "vicon" in key or "depth" in key:
                                        dic_to_save_tmp[key] = data[key]
                                existing_data = True
                            if existing_data and s not in [1, 2]:
                                continue
                            model_path = f"{model_dir}/{part}/model_scaled_{model_source[s]}_new_seth.bioMod"
                            processed_source.append(source[s])
                            reorder_marker_from_source = reorder_markers(markers_from_source[s][:, :-3, :],
                                                                         biorbd.Model(model_path),
                                                                         names_from_source[s][:-3])
                            markers_to_save.append(reorder_marker_from_source)
                            bio_model = biorbd.Model(model_path)

                            msk_function = MskFunctions(model=bio_model, data_buffer_size=6, system_rate=120)
                            result_biomech = process_all_frames(reorder_marker_from_source, msk_function,
                                                                source[s],
                                                                forces, (1000, 10), emg,
                                                                f_ext,
                                                                compute_id=False, compute_so=False, compute_jrf=False,
                                                                stop_frame=stop_frame,
                                                                file=f"{processed_data_path}/{part}" + "/" + file,
                                                                print_optimization_status=False, filter_depth=False,
                                                                emg_names=emg_names,
                                                                measurements=None
                                                                )
                            q_to_save.append(result_biomech["q_raw"])
                            q_dot_to_save.append(result_biomech["q_dot"])

                        dic_to_save_tmp = {
                                           "image_idx": frame_idx,
                                           "emg": emg,
                                           "f_ext": f_ext,
                                           "peaks": peaks,
                                           "rt_matrix": rt,
                                           "markers_names": names_from_source[0],
                                           }
                        for s in source:
                            dic_to_save_tmp[f"markers_{s}"] = markers_to_save[source.index(s)]
                            dic_to_save_tmp[f"{s}_q"] = q_to_save[source.index(s)]
                            dic_to_save_tmp[f"{s}_q_dot"] = q_dot_to_save[source.index(s)]
                        save(dic_to_save_tmp, f"{path}{os.sep}reoriented_dlc_markers.bio", add_data=False, safe=False)
                    #         result_biomech["markers"] = reorder_marker_from_source[..., :stop_frame]
                    #         result_biomech["image_idx"] = data_dlc["frame_idx"]
                    #         result_biomech["rt_matrix"] = rt
                    #         # result_biomech["track_idx"] = track_idx
                    #         # result_biomech["vicon_to_depth"] = vicon_to_depth
                    #         all_results[source[s]] = result_biomech
                    #         print("Done for source ", source[s])
                    #
                    #         # import bioviz
                    #         # b = bioviz.Viz(loaded_model=msk_function.model, show_floor=False)
                    #         # b.load_movement(result_biomech["q_raw"])
                    #         # b.load_experimental_markers(reorder_marker_from_source)
                    #         # b.exec()
                    #     all_results = process_cycles(all_results, peaks, n_peaks=None)
                    #     if save_data:
                    #         save(all_results, f"Q://Projet_hand_bike_markerless/process_data\{part}/result_biomech_{file.split('_')[0]}_{file.split('_')[1]}_{model}_{filt}.bio",
                    #              safe=False)
                    #         print("Saved")
                    #
                    # else:
                    #     all_results = load(f"{processed_data_path}/{part}/result_biomech_{Path(file).stem}_wt_filter.bio")
                    #     all_results = process_cycles(all_results, peaks, n_peaks=None)
                    #     processed_source = list(all_results.keys())
                    # if plot:
                    #     plot_results(all_results, markers_from_source, track_idx, vicon_to_depth,
                    #                  stop_frame=stop_frame, sources=processed_source, cycle=False)


if __name__ == '__main__':
    model_dir = "Q://Projet_hand_bike_markerless/RGBD"
    # model_dir = "F:\markerless_project"
    participants = [f"P{i}" for i in range(9, 17)]
    # if "P12" in participants:
    #     participants.pop(participants.index("P12"))
#, "P10"]#"P14", "P15", "P16"]#, "P14", "P15", "P16"]#, "P16"]  # ,"P9", "P10",
    processed_data_path = "Q://Projet_hand_bike_markerless/RGBD" #"/mnt/shared/Projet_hand_bike_markerless/process_data"
    # processed_data_path = "F://markerless_project"
    main(model_dir, participants, processed_data_path, save_data=True, results_from_file=False, stop_frame=2000,
         plot=False)