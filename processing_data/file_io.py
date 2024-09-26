import os
import numpy as np
from numpy.ma.extras import notmasked_edges

from biosiglive import load, OfflineProcessing
from biosiglive.processing.msk_utils import ExternalLoads
import biorbd
from scipy.signal import find_peaks
from processing_data.data_processing_helper import (
    reorder_markers_from_names,
    get_vicon_to_depth_idx,
    check_frames,
    fill_and_interpolate,
    convert_cluster_to_anato,
    adjust_idx
)
import json

prefix = "/mnt/shared" if os.name == "posix" else "Q:/"


def load_data(data_path, part, file, filter_depth=False, markers_dic=None):
    markers_dic = {} if markers_dic is None else markers_dic
    data = load(f"{data_path}/{part}/{file}_processed_3_crops_rt.bio")
    rt = data["rt_matrix"]
    markers_depth = data["markers_depth_interpolated"]
    markers_vicon = data["truncated_markers_vicon"]
    names_from_source = [data["depth_markers_names"], data["vicon_markers_names"]]
    sensix_data = data["sensix_data_interpolated"]
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
    vicon_to_depth_idx = get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)
    emg = data["emg_proc_interpolated"]
    emg = None if not isinstance(emg, np.ndarray) else emg
    peaks, _ = find_peaks(sensix_data["crank_angle"][0, :])
    peaks = [peak for peak in peaks if sensix_data["crank_angle"][0, peak] > 6]
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
    markers_dic["depth"] = [names_from_source[0], markers_depth]
    markers_dic["vicon"] = [names_from_source[1], markers_vicon]
    markers_dic["minimal_vicon"] = [names_from_source[2], markers_minimal_vicon]
    forces = ExternalLoads()
    forces.add_external_load(
        point_of_application=[0, 0, 0],
        applied_on_body="hand_left",
        express_in_coordinate="ground",
        name="hand_pedal",
        load=np.zeros((6, 1)),
    )
    if part in ["P10", "P11", "P12", "P13", "P14"]:
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
    return markers_dic, forces, f_ext, emg, vicon_to_depth_idx, peaks, rt


def get_all_file(participants, data_dir, trial_names=None):
    all_path = []
    parts = []
    if trial_names and len(trial_names) != len(participants):
        trial_names = [trial_names for _ in participants]
    for part in participants:
        all_files = os.listdir(f"{data_dir}{os.sep}{part}")
        all_files = [f"{data_dir}{os.sep}{part}{os.sep}{file}" for file in all_files if "gear" in file and "less" not in file and "more" not in file and "result" not in file]
        final_files = all_files if not trial_names else []
        if trial_names:
            for trial in trial_names[participants.index(part)]:
                for file in all_files:
                    if trial in file:
                        final_files.append(file)
                        break
        parts.append([part for _ in final_files])
        all_path.append(final_files)
    return sum(all_path, []), sum(parts, [])


def get_dlc_data(dlc_data_path, markers_dic=None, source="dlc"):
    markers_dic = {} if markers_dic is None else markers_dic
    ordered_markers_names = ["ribs", 'ster', 'xiph', 'clavsc', 'clavac',
                              'delt', 'arml', 'epicl', 'larml', 'stylr', 'stylu', 'm1', 'm2', 'm3']
    data = load(dlc_data_path)
    reordered_markers_dlc, idx = reorder_markers_from_names(data["markers_in_meters"],
                                                              ordered_markers_names,
                                                              list(data["markers_names"][:, 0]))
    markers_dic[source] = [ordered_markers_names, reordered_markers_dlc]
    return markers_dic, data["frame_idx"]


def filter_dlc_data(markers_list, frame_idx, part, interpolation_shape, rt=None):
    data_dlc = markers_list[1]
    dlc_names = markers_list[0]
    new_markers_dlc = np.zeros((3, data_dlc.shape[1], data_dlc.shape[2] ))
    if rt is not None:
        markers_dlc_hom = np.ones((4, data_dlc.shape[1], data_dlc.shape[2]))
        markers_dlc_hom[:3, ...] = data_dlc[:3, ...]
        for k in range(new_markers_dlc.shape[2]):
            new_markers_dlc[:, :, k] = np.dot(np.array(rt), markers_dlc_hom[:, :, k])[:3, :]
    new_markers_dlc = fill_and_interpolate(data=new_markers_dlc,
                                            idx=frame_idx,
                                            shape=interpolation_shape,
                                            fill=True)
    config = "with_depth"
    measurements_dir_path = "/home/amedeoceglia/Documents/programmation/rgbd_mocap/data_collection_mesurement"
    calibration_matrix_dir = "/home/amedeoceglia/Documents/programmation/rgbd_mocap/calibration_matrix"
    measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{part}.json"))
    measurements = measurement_data[config]["measure"]
    calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[config]["calibration_matrix_name"]
    anato_from_cluster = convert_cluster_to_anato(new_markers_dlc[:, -3:, :], measurements, calibration_matrix)
    first_idx = dlc_names.index("clavac")
    new_markers_dlc = np.concatenate((new_markers_dlc[:, :first_idx + 1, :],
                                      anato_from_cluster[:3, :, :],
                                      new_markers_dlc[:, first_idx + 1:, :]), axis=1
                                     )
    new_markers_names = markers_list[0][:first_idx + 1] + ['scapaa', 'scapia', 'scapts'] + markers_list[0][first_idx + 1:]
    new_markers_dlc_filtered = np.zeros((3, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
    for i in range(3):
        new_markers_dlc_filtered[i, :8, :] = OfflineProcessing().butter_lowpass_filter(
            new_markers_dlc[i, :8, :],
            2, 120, 2)
        new_markers_dlc_filtered[i, 8:, :] = OfflineProcessing().butter_lowpass_filter(
            new_markers_dlc[i, 8:, :],
            10, 120, 2)
    markers_list[1] = new_markers_dlc_filtered
    markers_list[0] = new_markers_names
    return markers_list


def refine_markers_dlc(markers_dic, frame_idx, labeled_data_path):
    ordered_depth_markers = ['ster', 'xiph', 'clavsc', 'clavac',
                             'delt', 'arml', 'epicl', 'larml', 'stylu', 'stylr', 'm1', 'm2', 'm3']
    data = load(labeled_data_path)
    reordered_markers_depth, idx = reorder_markers_from_names(data["markers_in_meters"],
                                                              ordered_depth_markers,
                                                              list(data["markers_names"][:, 0]))
    frame_idx_dlc = frame_idx.copy()
    framer_idx_depth = data["frame_idx"].copy()
    for key in markers_dic:
        if "dlc" in key:
            data_dlc, data_labeling, idx_start, idx_end, frame_idx = check_frames(markers_dic[key][1], reordered_markers_depth,
                                                                       framer_idx_depth, frame_idx_dlc)
            markers_dic[key][1] = data_dlc
    return markers_dic, idx_start, idx_end, frame_idx


def get_data_from_sources(participant, trial_name, source_list, model_dir, model_source, filter_depth=False, live_filter=False,
                          source_to_keep=None, output_file=None):
    once_loaded = False
    dlc_frames_idx = None
    is_dlc = False
    source_to_keep = [] if source_to_keep is None else source_to_keep
    markers_dic = {}
    previous_data = None
    existing_keys = []
    forces, f_ext, emg, vicon_to_depth, peaks, rt = None, None, None, None, None, None
    root_dir = f"{prefix}{os.sep}Projet_hand_bike_markerless/RGBD/{participant}"
    directory = \
    [direc for direc in os.listdir(root_dir) if trial_name in direc and os.path.isdir(root_dir + os.sep + direc)][0]
    labeled_data_path = f"{root_dir + os.sep + directory}{os.sep}marker_pos_multi_proc_3_crops_pp.bio"

    if source_to_keep is not None and os.path.exists(output_file):
        try:
            previous_data = load(output_file)
            existing_keys = list(previous_data.keys())
        except:
            os.remove(output_file)

    for source in source_list:
        if source in source_to_keep and source in existing_keys:
            markers_dic[source] = previous_data[source]
            continue
        elif ("vicon" in source or "depth" in source) and not once_loaded:
            print(f"Processing participant {participant}, trial : {trial_name}")
            markers_dic, forces, f_ext, emg, vicon_to_depth, peaks, rt = load_data(
                prefix + "/Projet_hand_bike_markerless/process_data", participant, f"{trial_name}",
                filter_depth, markers_dic
            )
            markers_dic = {key: markers_dic[key] for key in source_list if "dlc" not in key}
            once_loaded = True
        elif "dlc" in source:
            is_dlc = True
            ratio = "0_" + source.split("_")[-1] if source.split("_")[-1] != "1" else source.split("_")[-1]
            dlc_data_path = f"{root_dir}/{directory}/marker_pos_multi_proc_3_crops_normal_500_down_b1_ribs_and_cluster_{ratio}_with_model_pp_full.bio"
            markers_dic, dlc_frames_idx = get_dlc_data(dlc_data_path, markers_dic, source)

    if os.path.isfile(labeled_data_path) and is_dlc:
        markers_dic, idx_start, idx_end, dlc_frames_idx = refine_markers_dlc(markers_dic, dlc_frames_idx, labeled_data_path)
        count = 0
        for key in markers_dic:
            if key in source_to_keep and key in existing_keys:
                continue
            if "dlc" not in key:
                markers_dic[key][1] = adjust_idx({"mark": markers_dic[key][1]}, idx_start, idx_end)[
                    "mark"]
            if "dlc" in key and live_filter:
                continue
            if "dlc" in key and not live_filter:
                markers_dic[key] = filter_dlc_data(markers_dic[key], dlc_frames_idx, participant,
                                              markers_dic["depth"][1].shape[-1], rt)
            model_path = f"{model_dir}/{participant}/model_scaled_{model_source[count]}_new_seth.bioMod"
            model_markers_names = [mark.to_string() for mark in biorbd.Model(model_path).markerNames()]
            markers_dic[key][1], _ = reorder_markers_from_names(markers_dic[key][1][:, :-3, :],
                                                                                  model_markers_names,
                                                                                  markers_dic[key][0][:-3])
            markers_dic[key][0] = model_markers_names
            count += 1
    return markers_dic, forces, f_ext, emg, vicon_to_depth, peaks, rt, dlc_frames_idx
