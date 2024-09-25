from typing import Union

from biosiglive import load
import os
import numpy as np
import scipy.stats as st
try:
    import matplotlib.pyplot as plt
except:
    pass
from biosiglive.processing.msk_utils import ExternalLoads
from biosiglive import OfflineProcessing
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
try:
    from scapula_cluster.from_cluster_to_anato import ScapulaCluster
except:
    pass
from rgbd_mocap.tracking.kalman import Kalman
import json


def _load_data(data_path, part, file, end_idx=None):
    data = load(f"{data_path}/{part}/{file}")
    names_from_source = [data["depth_markers_names"], data["vicon_markers_names"]]
    depth_markers_names = data["depth_markers_names"]
    idx_ts = depth_markers_names.index("scapts")
    idx_ai = depth_markers_names.index("scapia")
    depth_markers_names[idx_ts] = "scapia"
    depth_markers_names[idx_ai] = "scapts"
    vicon_markers_names = data["vicon_markers_names"]
    idx_ts = vicon_markers_names.index("scapts")
    idx_ai = vicon_markers_names.index("scapia")
    vicon_markers_names[idx_ts] = "scapia"
    vicon_markers_names[idx_ai] = "scapts"
    vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)
    return names_from_source, vicon_to_depth_idx


def _convert_string(string):
    return string.lower().replace("_", "")


def _get_vicon_to_depth_idx(names_depth=None, names_vicon=None):
    vicon_markers_names = [_convert_string(name) for name in names_vicon]
    depth_markers_names = [_convert_string(name) for name in names_depth]
    vicon_to_depth_idx = []
    for name in vicon_markers_names:
        if name in depth_markers_names:
            vicon_to_depth_idx.append(vicon_markers_names.index(name))
    return vicon_to_depth_idx


def load_results_offline(participants, processed_data_path, trials=None, file_name="", to_exclude=None, recompute_cycles=True):
    if trials is None:
        trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    all_data = {}
    for p, part in enumerate(participants):
        all_data[part] = {}
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file
                     # and "result_offline" in file
                     # and file_name in file
                     # and "3_crops" in file and "3_crops_3_crops" not in file
                     ]
        trials_tmp = []
        for file in all_files:
            for trial in trials[p]:
                if trial not in file:
                    continue
                if not os.path.isfile(f"{processed_data_path}/{part}/{file}/result_offline_{file}_normal_alone.bio"):
                    continue
                trial = file.split("_")[0] + "_" + file.split("_")[1]
                print(f"Processing participant {part}, trial : {file}")
                all_data[part][file] = load(f"{processed_data_path}/{part}/{file}/result_offline_{file}_normal_alone.bio")
                if recompute_cycles:
                    peaks = find_peaks(all_data[part][file]["vicon"]["markers"][0, -1, :])[0]
                    all_data[part][file] = process_cycles_offline(all_data[part][file], peaks)
                trials_tmp.append(trial)
        trials[p] = trials_tmp
    return all_data, trials


def load_results(participants, processed_data_path, trials=None, file_name="", to_exclude=(), recompute_cycles=True,
                 trials_to_exclude=()):
    if trials is None:
        trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    all_data = {}
    for p, part in enumerate(participants):
        all_data[part] = {}
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file
                     # and "result_offline" in file
                     and file_name in file
                     and "ik" not in file
                     # and "3_crops" in file and "3_crops_3_crops" not in file
                     ]
        for exc in to_exclude:
            all_files = [file for file in all_files if exc not in file]
        trials_tmp = []

        for file in all_files:
            for trial in trials[p]:
                pass_trial = False
                for trial_to_excl in trials_to_exclude:
                    if part in trial_to_excl[0] and trial_to_excl[1] in trial:
                        pass_trial = True
                        continue
                if trial not in file or pass_trial:
                    continue
                print(f"Processing participant {part}, trial : {file}")
                all_data[part][file] = load(f"{processed_data_path}/{part}/{file}")
                if recompute_cycles:
                    peaks = find_peaks(all_data[part][file]["minimal_vicon"]["markers"][2, -4, :])[0]
                    all_data[part][file] = process_cycles(all_data[part][file], peaks)
                trials_tmp.append(trial)
        trials[p] = trials_tmp
    return all_data, trials


def load_in_markers_ref(data):
    markers_depth = data["markers_depth_initial"]
    markers_vicon = data["markers_vicon_rotated"]
    depth_markers_names = data["depth_markers_names"]
    idx_ts = depth_markers_names.index("scapts")
    idx_ai = depth_markers_names.index("scapia")
    depth_markers_names[idx_ts] = "scapia"
    depth_markers_names[idx_ai] = "scapts"
    vicon_markers_names = data["vicon_markers_names"]
    idx_ts = vicon_markers_names.index("scapts")
    idx_ai = vicon_markers_names.index("scapia")
    vicon_markers_names[idx_ts] = "scapia"
    vicon_markers_names[idx_ai] = "scapts"
    vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)
    return markers_depth, markers_vicon, vicon_to_depth_idx


def load_data(data_path, part, file, filter_depth, end_idx=None, ):
    data = load(f"{data_path}/{part}/{file}_processed_3_crops_rt.bio")
    rt = data["rt_matrix"]

    markers_depth = data["markers_depth_interpolated"]
    is_visible = np.repeat(data["is_visible"][np.newaxis, :, :], 3, axis=0)
    markers_vicon = data["truncated_markers_vicon"]
    names_from_source = [data["depth_markers_names"], data["vicon_markers_names"]]
    sensix_data = data["sensix_data_interpolated"]
    # put nan at idx where the marker is not visible
    depth_markers_names = data["depth_markers_names"]
    idx_ts = depth_markers_names.index("scapts")
    idx_ai = depth_markers_names.index("scapia")
    depth_markers_names[idx_ts] = "scapia"
    depth_markers_names[idx_ai] = "scapts"
    vicon_markers_names = data["vicon_markers_names"]
    idx_ts = vicon_markers_names.index("scapts")
    idx_ai = vicon_markers_names.index("scapia")
    vicon_markers_names[idx_ts] = "scapia"
    vicon_markers_names[idx_ai] = "scapts"
    vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)

    emg = data["emg_proc_interpolated"]
    if not isinstance(emg, np.ndarray):
        emg = None
    # find peak in crank angle
    peaks, _ = find_peaks(sensix_data["crank_angle"][0, :])
    peaks = [peak for peak in peaks if sensix_data["crank_angle"][0, peak] > 6]
    # plt.plot(sensix_data["crank_angle"][0, :])
    # plt.plot(peaks, sensix_data["crank_angle"][0, peaks], "x")
    # plt.show()


    markers_minimal_vicon = markers_vicon[:, vicon_to_depth_idx, :]
    names_from_source.append(list(np.array(vicon_markers_names)[vicon_to_depth_idx]))

    if filter_depth:
        markers_depth_filtered = np.zeros((3, markers_depth.shape[1], markers_depth.shape[2]))
        for i in range(3):
            markers_depth_filtered[i, :7, :] = OfflineProcessing().butter_lowpass_filter(
                markers_depth[i, :7, :],
                2, 120, 2)
            markers_depth_filtered[i, 7:, :] = OfflineProcessing().butter_lowpass_filter(
                markers_depth[i, 7:, :],
                10, 120, 2)
        markers_depth = markers_depth_filtered
        # markers_vicon_filtered = np.zeros((3, markers_vicon.shape[1], markers_vicon.shape[2]))
        # for i in range(3):
        #     markers_vicon_filtered[i, :7, :] = OfflineProcessing().butter_lowpass_filter(
        #         markers_vicon[i, :7, :],
        #         2, 120, 2)
        #     markers_vicon_filtered[i, 7:, :] = OfflineProcessing().butter_lowpass_filter(
        #         markers_vicon[i, 7:, :],
        #         10, 120, 2)
        # markers_vicon = markers_vicon_filtered
        # markers_minimal_vicon = markers_vicon_filtered[:, vicon_to_depth_idx, :]
    markers_from_source = [markers_depth, markers_vicon, markers_minimal_vicon]
    # plt.figure("markers")
    # for i in range(markers_depth_filtered.shape[1]):
    #     plt.subplot(4, ceil(markers_depth_filtered.shape[1] / 4), i + 1)
    #     for j in range(3):
    #         plt.plot(markers_depth_filtered[j, i, :])
    #         plt.plot(markers_vicon[j, vicon_to_depth_idx[i], :])
    #         plt.plot(markers_minimal_vicon[j, i, :])
    #         plt.plot(peaks, markers_minimal_vicon[j, i, peaks], "x")
    #
    # plt.show()
    forces = ExternalLoads()
    forces.add_external_load(
        point_of_application=[0, 0, 0],
        #applied_on_body="radius_left_pro_sup_left",
        applied_on_body="hand_left",
        express_in_coordinate="ground",
        name="hand_pedal",
        load=np.zeros((6, 1)),
    )
    if part in ["P10", "P11", "P12", "P13"]:
        f_ext = np.array([sensix_data["LMY"],
                          sensix_data["LMX"],
                          sensix_data["LMZ"],
                          sensix_data["LFY"],
                          -sensix_data["LFX"],
                          -sensix_data["LFZ"]])
    else:
        f_ext = np.array([sensix_data["LMY"],
                          sensix_data["LMX"],
                          sensix_data["LMZ"],
                          sensix_data["LFY"],
                          sensix_data["LFX"],
                          sensix_data["LFZ"]])
    f_ext = f_ext[:, 0, :]
    return markers_from_source, names_from_source, forces, f_ext, emg, vicon_to_depth_idx, peaks, rt


def _reorder_markers_from_names(markers_data, ordered_markers_names, markers_names):
    # count = 0
    # reordered_markers = np.zeros((markers_data.shape[0], len(ordered_markers_names), markers_data.shape[2]))
    idx = []
    # for i in range(len(markers_names)):
    #     if markers_names[i] == "elb":
    #         markers_names[i] = "elbow"
    #     if _convert_string(markers_names[i]) in ordered_markers_names:
    #         reordered_markers[:, ordered_markers_names.index(_convert_string(markers_names[i])),
    #         :] = markers_data[:, count, :]
    #         # idx.append(ordered_markers_names.index(_convert_string(markers_names[i])))
    #         count += 1
    markers_names = [_convert_string(name) for name in markers_names]
    for i in range(len(ordered_markers_names)):
        if markers_names[i] == "elb":
            markers_names[i] = "elbow"
        if _convert_string(ordered_markers_names[i]) in markers_names:
            idx.append(markers_names.index(_convert_string(ordered_markers_names[i])))
    return markers_data[:, idx], idx


def load_data_from_dlc(labeled_data_path=None, dlc_data_path=None, part=None, file=None, in_pixel=False):
    init_depth_markers_names = ['ster', 'xiph', 'clavsc', 'clavac',
                           'delt', 'arml',  'epicl', 'larml', 'stylu', 'stylr', 'm1', 'm2', 'm3']
    # init_dlc_markers_names = ['ster', 'xiph', 'clavsc', 'clavac',
    #                        'delt', 'epicl', 'arml', 'stylu', 'stylr', 'larml',   'm1', 'm2', 'm3']
    init_dlc_markers_names = ["ribs", 'ster', 'xiph', 'clavsc', 'clavac',
                                    'delt', 'arml', 'epicl', 'larml', 'stylr', 'stylu', 'm1', 'm2', 'm3']

    names = [init_depth_markers_names, init_dlc_markers_names]
    measurements_dir_path = "data_collection_mesurement"
    calibration_matrix_dir = "../scapula_cluster/calibration_matrix"
    markers_names_list = []
    reordered_markers_list = []
    dict_list = [{}, {}]
    plt.figure("markers")
    measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{part}.json"
                                      ))
    measurements = measurement_data[f"with_depth"]["measure"]
    calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"][
        "calibration_matrix_name"]
    for p, path in enumerate([labeled_data_path, dlc_data_path]):
        markers_names_list.append([])
        reordered_markers_list.append([])
        if p == 0 and labeled_data_path is None:
            continue
        data = load(path)
        dict_list[p]["frame_idx"] = data["frame_idx"]
        depth_markers = data["markers_in_meters"]
        markers_in_pixel = data["markers_in_pixel"]
        depth_markers_names = list(data["markers_names"][:, 0])
        reordered_markers_depth, idx = _reorder_markers_from_names(depth_markers, names[p], depth_markers_names,
                                                                   )
        reordered_markers_pixel, _ = _reorder_markers_from_names(markers_in_pixel, names[p],  depth_markers_names,
                                                                   )
        # reordered_markers_depth = depth_markers

        # anato_from_cluster, landmarks_dist = _convert_cluster_to_anato(measurements,
        #                                                                     calibration_matrix,
        #                                                                     reordered_markers_depth[:,
        #                                                                     -3:, :] * 1000)
        # first_idx = names[p].index("clavac")
        # reordered_markers_depth = np.concatenate((reordered_markers_depth[:3, :first_idx + 1, :],
        #                                             anato_from_cluster[:3, :, :] * 0.001,
        #                                             reordered_markers_depth[:3, first_idx + 1:, :]),
        #                                            axis=1)
        # depth_markers_names = names[p][:first_idx + 1] + ["scapaa", "scapts", "scapia"] + names[p][first_idx+1:]
        dict_list[p]["markers_in_meters"] = reordered_markers_depth
        dict_list[p]["markers_in_pixel"] = reordered_markers_pixel
        dict_list[p]["markers_names"] = names[p]
        dict_list[p]["time_to_process"] = data["time_to_process"]
        # styl_u = dict_list[p]["markers_in_meters"][:, -1, :].copy()
        # larm = dict_list[p]["markers_in_meters"][:, -3, :].copy()
        # dict_list[p]["markers_in_meters"][:, -3, :] = styl_u
        # dict_list[p]["markers_in_meters"][:, -1, :] = larm
    if labeled_data_path is None:
        return dict_list[1]["markers_in_meters"], dict_list[1]["markers_in_pixel"], dict_list[1]["markers_names"], dict_list[1]["frame_idx"]
    data_dlc, data_labeling, idx_start, idx_end = check_frames(dict_list[0], dict_list[1])
    return data_dlc, data_labeling, dict_list[1]["markers_names"], dict_list[1]["time_to_process"], idx_start, idx_end

def refine_synchro(marker_full, marker_to_refine, plot_fig=True):
    error_list = []
    for i in range(300):
        marker_to_refine_tmp = marker_to_refine[:, :, :-i] if i != 0 else marker_to_refine
        marker_to_refine_tmp = _interpolate_data(marker_to_refine_tmp, marker_full.shape[2])
        error_markers = compute_error_mark(
            marker_full[:, ...], marker_to_refine_tmp[:, ...])
        error_tmp = np.abs(np.mean(error_markers))
        error_list.append(error_tmp)

    idx = error_list.index(min(error_list))
    marker_to_refine_tmp = marker_to_refine[:, :, :-idx] if idx != 0 else marker_to_refine[:, :, :]
    marker_to_refine_tmp = _interpolate_data(marker_to_refine_tmp, marker_full.shape[2])
    if plot_fig:
        plt.figure("refine synchro")
        for i in range(marker_to_refine_tmp.shape[1]):
            plt.subplot(4, 4, i + 1)
            for j in range(0, 3):
                plt.plot(marker_to_refine_tmp[j, i, :], "b")
                plt.plot(marker_full[j, i, :], 'r')
    print("idx to refine synchro : ", idx, "error",  min(error_list))
    return marker_to_refine_tmp, idx

def check_frames(data_labeling, data_dlc):
    data = list(np.copy(data_labeling["frame_idx"]))
    ref = list(np.copy(data_dlc["frame_idx"]))
    type_1 = 0
    type = 0
    idx_init = 0
    idx = 0
    overall_init_idx = None
    overall_final_idx = None
    datalist = [data_dlc, data_labeling]
    for key in data_dlc.keys():
        if key == "markers_names":
            continue
        if data[0] > ref[0]:
            idx_init = ref.index(data[0])
            ref = ref[idx_init:]
            type_1 = 0
        elif data[0] < ref[0]:
            overall_init_idx = (ref[0] - data[0]) * 2
            idx_init = data.index(ref[0])
            data = data[idx_init:]
            type_1 = 1

        if isinstance(data_dlc[key], np.ndarray):
            datalist[type_1][key] = datalist[type_1][key][..., idx_init:]
        else:
            datalist[type_1][key] = datalist[type_1][key][idx_init:]
        if data[-1] < ref[-1]:
            idx = ref.index(data[-1])
            idx += 1
            ref = ref[:idx]
            type = 0
        elif data[-1] > ref[-1]:
            overall_final_idx = (data[-1] - ref[-1]) * 2
            idx = data.index(ref[-1])
            idx += 1
            data = data[:idx]
            type = 1
        if idx != 0:
            if isinstance(datalist[type][key], np.ndarray):
                datalist[type][key] = datalist[type][key][..., :idx]
            else:
                datalist[type][key] = datalist[type][key][:idx]

    if ref != data:
        # longest = ref if len(ref) > len(data) else data
        # smallest = data if len(ref) > len(data) else ref
        # type = 1 if len(ref) > len(data) else 0
        # for idx in longest:
        #     if idx in smallest:
        #         continue
        #     else:
        #         for key in data_dlc.keys():
        #             if isinstance(datalist[type][key], np.ndarray):
        #                 datalist[type][key] = np.delete(datalist[type][key], longest.index(idx), axis=-1)
        #             else:
        #                 datalist[type][key].pop(longest.index(idx))
        print(1)
    return datalist[0], datalist[1], overall_init_idx, overall_final_idx


def _convert_cluster_to_anato(new_cluster, data):
    anato_pos = new_cluster.process(marker_cluster_positions=data, cluster_marker_names=["M1", "M2", "M3"],
                                    save_file=False)
    return anato_pos

def _convert_cluster_to_anato_old(measurements,
                              calibration_matrix, data):

    new_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                                 measurements[4], measurements[5], calibration_matrix)

    anato_pos = new_cluster.process(marker_cluster_positions=data, cluster_marker_names=["M1", "M2", "M3"],
                                    save_file=False)
    land_dist = new_cluster.get_landmarks_distance()
    return anato_pos, land_dist


def load_all_data(participants, processed_data_path, trials=None):
    all_data = {}
    if trials is None:
        trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    for p, part in enumerate(participants):
        all_data[part] = {}
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file and "result_biomech" not in file and "processed" in file
                     and "3_crops" in file]
        trials_tmp = []
        for file in all_files:
            for trial in trials[p]:
                if trial not in file:
                    continue
                print(f"Processing participant {part}, trial : {file}")
                all_data[part][file] = load(f"{processed_data_path}/{part}/{file}")
                trials_tmp.append(trial)
        trials[p] = trials_tmp
    return all_data, trials


def dispatch_bio_results(bio_results):
    dic_result = {}
    dic_result["res_tau"] = bio_results["res_tau"]
    dic_result["jrf"] = bio_results["jrf"]
    dic_result["markers"] = bio_results["markers"]
    dic_result["state"] = bio_results["state"]
    return dic_result


def compute_error_mark(ref_mark, mark):
    # new_markers_depth_int = OfflineProcessing().butter_lowpass_filter(new_markers_depth_int, 6, 120, 4)
    err_markers = np.zeros((ref_mark.shape[1], 1))
    for i in range(ref_mark.shape[1]):
        nan_index = np.argwhere(np.isnan(ref_mark[:, i, :]))
        new_markers_depth_tmp = np.delete(mark[:, i, :], nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(ref_mark[:, i, :], nan_index, axis=1)
        nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
        new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
        err_markers[i, 0] = np.median(np.sqrt(
            np.mean(((new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000) ** 2), axis=0)))
    return list(err_markers[:, 0])

def compute_error(depth_dic, vicon_dic):

    n_markers_depth = depth_dic["markers"].shape[1]
    vicon_to_depth = depth_dic["vicon_to_depth"]
    err_markers = np.zeros((n_markers_depth, 1))
    if vicon_dic["markers"].shape[1] != depth_dic["markers"].shape[1]:
        vicon_dic["markers"] = vicon_dic["markers"][:, vicon_to_depth, :]
    # new_markers_depth_int = OfflineProcessing().butter_lowpass_filter(new_markers_depth_int, 6, 120, 4)
    for i in range(len(vicon_to_depth)):
        # ignore NaN values
        # if i not in [4, 5, 6]:
        nan_index = np.argwhere(np.isnan(vicon_dic["markers"][:, i, :]))
        new_markers_depth_tmp = np.delete(depth_dic["markers"][:, i, :], nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(vicon_dic["markers"][:, i, :], nan_index, axis=1)
        nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
        new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
        err_markers[i, 0] = np.median(np.sqrt(
            np.mean(((new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000) ** 2), axis=0)))

    err_q = []
    for i in range(depth_dic["q"].shape[0]):
        err_q.append(np.mean(np.sqrt(np.mean(((depth_dic["q"][i, :] - vicon_dic["q"][i, :]) ** 2), axis=0))))

    err_q_dot = []
    for i in range(depth_dic["q"].shape[0]):
        err_q_dot.append(np.mean(np.sqrt(np.mean(((depth_dic["q_dot"][i, :] - vicon_dic["q_dot"][i, :]) ** 2), axis=0))))

    err_q_ddot = []
    for i in range(depth_dic["q_ddot"].shape[0]):
        err_q_ddot.append(np.mean(np.sqrt(np.mean(((depth_dic["q_ddot"][i, :] - vicon_dic["q_ddot"][i, :]) ** 2), axis=0))))

    #normalize tau
    norm_tau = np.max(vicon_dic["tau"], axis=1)
    vicon_dic["tau"] = np.clip(vicon_dic["tau"] / norm_tau[:, None] * 100, 0, 100)
    norm_tau = np.max(depth_dic["tau"], axis=1)
    depth_dic["tau"] = np.clip(depth_dic["tau"] / norm_tau[:, None] * 100, 0, 100)
    err_tau = []
    for i in range(depth_dic["tau"].shape[0]):
        err_tau.append(np.mean(np.sqrt(np.mean(((depth_dic["tau"][i, :] - vicon_dic["tau"][i, :]) ** 2), axis=0))))

    # normalize muscle activation
    norm_mus_act = np.max(vicon_dic["mus_act"], axis=1)
    vicon_dic["mus_act"] = np.clip(vicon_dic["mus_act"] / norm_mus_act[:, None] * 100, 0, 100)
    norm_mus_act = np.max(depth_dic["mus_act"], axis=1)
    depth_dic["mus_act"] = np.clip(depth_dic["mus_act"] / norm_mus_act[:, None] * 100, 0, 100)
    err_mus_act = []
    for i in range(depth_dic["mus_act"].shape[0]):
        err_mus_act.append(np.mean(np.sqrt(np.mean(((depth_dic["mus_act"][i, :] - vicon_dic["mus_act"][i, :]) ** 2), axis=0))))
    err_q = [err_q[i] * 180/np.pi for i in range(len(err_q))]
    err_q_dot = [err_q_dot[i] * 180/np.pi for i in range(len(err_q_dot))]
    err_q_ddot = [err_q_ddot[i] * 180/np.pi for i in range(len(err_q_ddot))]
    return list(err_markers[:, 0]), err_q, err_q_dot, err_q_ddot, err_tau, err_mus_act

def remove_nan(data1, data2):
    mean = data1
    diff = data2
    nan_index = np.argwhere(np.isnan(mean))
    mean = np.delete(mean, nan_index, axis=0)
    diff = np.delete(diff, nan_index, axis=0)
    return mean, diff


def compute_blandt_altman(data1, data2, units="mm", title="Bland-Altman Plot", show=True, color=None, x_axis=None, markers=None, ax = None, threeshold=np.inf, no_y_label=False):
    # mean = (data1 + data2) / 2
    # diff = data1 - data2
    mean_to_plot = data1
    diff_to_plot  = data2
    mean, diff= remove_nan(data1, data2)
    # Average difference (aka the bias)
    bias = np.mean(diff)
    # Sample standard deviation
    s = np.std(diff, ddof=1)  # Use ddof=1 to get the sample standard deviation
    print(f'For the differences, μ = {bias:.4f} {units} and s = {s:.4f} {units} ')

    # Limits of agreement (LOAs)
    upper_loa = bias + 1.96 * s
    lower_loa = bias - 1.96 * s
    print(f'The limits of agreement are {upper_loa:.2f} {units} and {lower_loa:.2f} {units} ')

    # Confidence level
    C = 0.95  # 95%
    # Significance level, α
    alpha = 1 - C
    # Number of tails
    tails = 2
    # Quantile (the cumulative probability)
    q = 1 - (alpha / tails)
    # Critical z-score, calculated using the percent-point function (aka the
    # quantile function) of the normal distribution
    z_star = st.norm.ppf(q)
    print(f'95% of normally distributed data lies within {z_star}σ of the mean')
    # Limits of agreement (LOAs)
    loas = (bias - z_star * s, bias + z_star * s)

    print(f'The limits of agreement are {loas} {units} ')
    # Limits of agreement (LOAs)
    loas = st.norm.interval(C, bias, s)
    print(np.round(loas, 2))
    # Sample size
    n = data1.shape[0]
    # Degrees of freedom
    dof = n - 1
    # Standard error of the bias
    se_bias = s / np.sqrt(n)
    # Standard error of the LOAs
    se_loas = np.sqrt(3 * s ** 2 / n)

    # Confidence interval for the bias
    ci_bias = st.t.interval(C, dof, bias, se_bias)
    # Confidence interval for the lower LOA
    ci_lower_loa = st.t.interval(C, dof, loas[0], se_loas)
    # Confidence interval for the upper LOA
    ci_upper_loa = st.t.interval(C, dof, loas[1], se_loas)

    print(
        f' Lower LOA = {np.round(lower_loa, 2)}, 95% CI {np.round(ci_lower_loa, 2)}\n',
        f'Bias = {np.round(bias, 2)}, 95% CI {np.round(ci_bias, 2)}\n',
        f'Upper LOA = {np.round(upper_loa, 2)}, 95% CI {np.round(ci_upper_loa, 2)}'
    )
    if ax is None:
        plt.figure(title)
    ax = plt.axes() if ax is None else ax
    markers = markers if markers is not None else 'o'
    if color is not None:
        for i in range(len(color)):
            mean_tmp = mean_to_plot[i * len(color[i]):(i+1) * len(color[i])]
            diff_tmp = diff_to_plot[i * len(color[i]):(i+1) * len(color[i])]
            for j in range(len(mean_tmp)):
                if np.abs(diff_tmp[j]) > threeshold:
                    continue
                ax.scatter(mean_tmp[j], diff_tmp[j], c=color[i][j], s=100, alpha=0.6, marker=markers)

    # ax.scatter(mean, diff, c='k', s=20, alpha=0.6, marker='o')
    # Plot the zero line
    ax.axhline(y=0, c='k', lw=0.5)
    # Plot the bias and the limits of agreement
    ax.axhline(y=loas[1], c='grey', ls='--')
    ax.axhline(y=bias, c='grey', ls='--')
    ax.axhline(y=loas[0], c='grey', ls='--')


    # Labels
    font = 18
    ax.set_title(title, fontsize=font + 2)
    #ax.set_ylabel(f'Difference ({units} )', fontsize=font)
    if x_axis is not None:
        ax.set_xlabel(x_axis, fontsize=font)
    else:
        ax.set_xlabel(f'Mean ({units})', fontsize=font)
    ax.tick_params(axis='y', labelsize=font)
    ax.tick_params(axis='x', labelsize=font)
    if not no_y_label:
        ax.set_ylabel(f'Difference ({units})', fontsize=font)
    else:
        ax.set_ylabel("", fontsize=font)
    #ax.xticks(fontsize=font)
    #ax.yticks(fontsize=font)
    # Get axis limits
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    # Set y-axis limits
    #max_y = max(abs(bottom), abs(top))
    max_y = top
    min_y = abs(bottom)

    ax.set_ylim(-min_y, max_y)
    # Set x-axis limits
    domain = right - left
    ax.set_xlim(left, left + domain)
    # Annotations
    ax.annotate('+LOA', (right, upper_loa), (0, 7), textcoords='offset pixels', fontsize=font)
    ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -25), textcoords='offset pixels', fontsize=font)
    ax.annotate('Bias', (right, bias), (0, 7), textcoords='offset pixels', fontsize=font)
    ax.annotate(f'{bias:+4.2f}', (right, bias), (0, -25), textcoords='offset pixels', fontsize=font)
    ax.annotate('-LOA', (right, lower_loa), (0, 7), textcoords='offset pixels', fontsize=font)
    ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -25), textcoords='offset pixels', fontsize=font)

    # Confidence intervals
    ax.plot([left] * 2, list(ci_upper_loa), c='grey', ls='--', alpha=0.5)
    ax.plot([left] * 2, list(ci_bias), c='grey', ls='--', alpha=0.5)
    ax.plot([left] * 2, list(ci_lower_loa), c='grey', ls='--', alpha=0.5)
    # Confidence intervals' caps
    # x_range = [left - domain * 0.025, left + domain * 0.025]
    # ax.plot(x_range, [ci_upper_loa[1]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_upper_loa[0]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_lower_loa[1]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_lower_loa[0]] * 2, c='grey', ls='--', alpha=0.5)

    if show:
        plt.show()

    return bias, lower_loa, upper_loa


def process_cycles_offline(all_results, peaks, n_peaks=None):
    for key in ["dlc", "depth", "vicon"]:
        data_size = all_results[key]["q"].shape[1]
        dic_tmp = {}
        for key2 in all_results[key].keys():
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


def process_cycles(all_results, peaks, n_peaks=None):
    for key in all_results.keys():
        if key == "file":
            continue
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


def reorder_markers(markers, model, names):
    model_marker_names = [_convert_string(model.markerNames()[i].to_string()) for i in range(model.nbMarkers())]
    assert len(model_marker_names) == len(names)
    assert len(model_marker_names) == markers.shape[1]
    count = 0
    reordered_markers = np.zeros((markers.shape[0], len(model_marker_names), markers.shape[2]))
    final_names = []
    for i in range(len(names)):
        if names[i] == "elb":
            names[i] = "elbow"
        if _convert_string(names[i]) in model_marker_names:
            reordered_markers[:, model_marker_names.index(_convert_string(names[i])),
            :] = markers[:, count, :]
            final_names.append(model.markerNames()[i].to_string())
            count += 1
    return reordered_markers, final_names

def get_muscular_torque(x, act, model):
    """
    Get the muscular torque.
    """
    muscular_torque = np.zeros((model.nbQ(), x.shape[1]))
    states = model.stateSet()  # Get the muscle state set
    for i in range(act.shape[1]):
        for a, state in zip(act[:, i], states):
            state.setActivation(a)  # And fill it with the current value
        muscular_torque[:, i] = model.muscularJointTorque(
            states, x[: model.nbQ(), i], x[model.nbQ() : model.nbQ() * 2, i]
        ).to_array()
    return muscular_torque

def get_next_frame_from_kalman(kalman_instance=Union[None, list[Kalman]], markers_data=None, scapula_cluster=None,
                               measurement_noise_factor=100, process_noise_factor=5,
                               error_cov_post_factor=0, error_cov_pre_factor=0, rt_matrix=None,
                               n_markers=None, forward=0, idx_cluster_markers=None, params=None, in_pixel=False,  camera_converter=None,
                               convert_cluster_before_kalman=True, return_in_meter=True, fps=60):


    if in_pixel and camera_converter is None:
        raise RuntimeError("a camera converter must be provided when markers are in pixel.")
    if in_pixel and convert_cluster_before_kalman and markers_data is not None:
        markers_data = camera_converter.get_markers_pos_in_meter(markers_data[:, :, 0].T)[..., None]
    if rt_matrix is not None and markers_data is not None:
        markers_dlc_hom = np.ones((4, markers_data.shape[1], 1))
        markers_dlc_hom[:3, :, 0] = markers_data[..., 0]
        markers_data = np.dot(np.array(rt_matrix), markers_dlc_hom[:, :, 0])[:3, :, None]

    if markers_data is not None and convert_cluster_before_kalman:
        anato_from_cluster = _convert_cluster_to_anato(scapula_cluster, markers_data[:, -3:, :] * 1000) * 0.001
        markers_data = np.concatenate(
            (markers_data[:, :idx_cluster_markers + 1, :], anato_from_cluster[:3, ...], markers_data[:, idx_cluster_markers + 1:, :]),
            axis=1)
        if in_pixel:
            markers_data[:2, :, 0] = camera_converter.get_marker_pos_in_pixel(markers_data[:, :, 0].T).T
    # next_frame = markers_data
    if n_markers is None:
        if kalman_instance is not None:
            n_markers = len(kalman_instance)
        elif markers_data is not None:
            n_markers = markers_data.shape[1]
        else:
            raise ValueError("Impossible to know how many markers there are.")
    next_frame = np.zeros((3, n_markers, 1))
    kalman_instance = [None] * n_markers if kalman_instance is None else kalman_instance
    if markers_data is not None and in_pixel:
        markers_data[2, ...] *= 500
    for k in range(n_markers):
        if kalman_instance[k] is None and markers_data is not None:
            measurement_noise_factor = params[:int(markers_data.shape[1])][k]
            process_noise_factor = params[int(markers_data.shape[1]):int(markers_data.shape[1] * 2)][k]
            # error_cov_post_factor = \
            # params[int(markers_data.shape[1] * 2):int(markers_data.shape[1] * 3)][k]
            # error_cov_pre_factor = params[int(markers_data.shape[1] * 3):-1][k]
            kalman_instance[k] = Kalman(markers_data[:, k, 0], n_measures=3, n_diff=2, fps=fps,
                                        measurement_noise_factor=measurement_noise_factor,
                                        process_noise_factor=process_noise_factor,
                                        error_cov_post_factor=error_cov_post_factor,
                                        error_cov_pre_factor=error_cov_pre_factor
                                        )
            next_frame[:, k, 0] = kalman_instance[k].predict()
        elif kalman_instance[k] is not None:
            next_frame[:, k, 0] = kalman_instance[k].predict()
            if markers_data is not None:
                next_frame[:, k, 0] = kalman_instance[k].correct(markers_data[:, k, 0])
            if forward != 0:
                next_frame[:, k, 0] = kalman_instance[k].get_future_pose(dt=forward)
        else:
            raise ValueError("Unexpected error.")

    if in_pixel:
        next_frame[2, ...] /= 500
        if return_in_meter or rt_matrix is not None or not convert_cluster_before_kalman:
            next_frame = camera_converter.get_markers_pos_in_meter(next_frame[:, :, 0].T)[..., None]

    if not convert_cluster_before_kalman:
        anato_from_cluster = _convert_cluster_to_anato(scapula_cluster, next_frame[:, -3:, :] * 1000) * 0.001
        next_frame = np.concatenate(
            (next_frame[:, :idx_cluster_markers + 1, :], anato_from_cluster[:3, ...], next_frame[:, idx_cluster_markers + 1:, :]),
            axis=1)

    # if rt_matrix is not None:
    #     markers_dlc_hom = np.ones((4, next_frame.shape[1], 1))
    #     markers_dlc_hom[:3, :, 0] = next_frame[..., 0]
    #     next_frame = np.dot(np.array(rt_matrix), markers_dlc_hom[:, :, 0])[:3, :, None]
    if not return_in_meter and (rt_matrix is not None or not convert_cluster_before_kalman):
        next_frame[:2, :, 0] = camera_converter.get_marker_pos_in_pixel(next_frame[:, :, 0].T).T
    return next_frame[..., 0], kalman_instance
