import numpy as np
from casadi import interp1d

from utils_old import load_data_from_dlc
from utils import convert_string
import pandas as pd
from scipy.interpolate import interp1d
from biosiglive import OfflineProcessing
from scapula_cluster.from_cluster_to_anato import ScapulaCluster
import os


def _fill_with_nan(markers, idx):
    size = idx[-1] - idx[0]
    if len(markers.shape) == 2:
        new_markers_depth = np.zeros((markers.shape[0], size))
        count = 0
        for i in range(size):
            if i + idx[0] in idx:
                new_markers_depth[:, i] = markers[:, count]
                count += 1
            else:
                new_markers_depth[:, i] = np.nan
        return new_markers_depth
    elif len(markers.shape) == 3:
        new_markers_depth = np.zeros((3, markers.shape[1], size))
        count = 0
        for i in range(size):
            if i + idx[0] in idx:
                new_markers_depth[:, :, i] = markers[:, :, count]
                count += 1
            else:
                new_markers_depth[:, :, i] = np.nan
        return new_markers_depth


def interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(3):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def get_vicon_to_depth_idx(names_depth=None, names_vicon=None):
    vicon_markers_names = [convert_string(name) for name in names_vicon]
    depth_markers_names = [convert_string(name) for name in names_depth]
    vicon_to_depth_idx = []
    for name in vicon_markers_names:
        if name in depth_markers_names:
            vicon_to_depth_idx.append(vicon_markers_names.index(name))
    return vicon_to_depth_idx


def _interpolate_2d_data(data, shape):
    x = np.linspace(0, 100, data.shape[1])
    f_mark = interp1d(x, data)
    x_new = np.linspace(0, 100, shape)
    new_data = f_mark(x_new)
    return new_data


def process_cycles(all_results, peaks, n_peaks=None, key_for_size="q_raw"):
    for key in all_results.keys():
        data_size = all_results[key][key_for_size].shape[1]
        dic_tmp = {}
        for key2 in all_results[key].keys():
            if key2 == "cycle" or key2 == "rt_matrix" or key2 == "marker_names":
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
                interp_function = _interpolate_2d_data if len(all_results[key][key2].shape) == 2 else interpolate_data
                if array_tmp is None:
                    array_tmp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], 120)
                    array_tmp = array_tmp[None, ...]
                else:
                    data_interp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], 120)
                    array_tmp = np.concatenate((array_tmp, data_interp[None, ...]), axis=0)
            dic_tmp[key2] = array_tmp
        all_results[key]["cycles"] = dic_tmp
    return all_results


def fill_and_interpolate(data, shape, idx=None, names=None, fill=True):
    data_nan = _fill_with_nan(data, idx) if fill else data
    if len(data_nan.shape) == 1:
        data_nan = data_nan.reshape((1, data_nan.shape[0]))
    names = [f"n_{i}" for i in range(data_nan.shape[-2])] if not names else names
    if len(data_nan.shape) == 2:
        data_df = pd.DataFrame(data_nan, names)
        data_filled_extr = data_df.interpolate(method='linear', axis=1)
        data_int = _interpolate_2d_data(data_filled_extr, shape)
    elif len(data_nan.shape) == 3:
        data_filled_extr = np.zeros((3, data_nan.shape[1], data_nan.shape[2]))
        for i in range(3):
            data_df = pd.DataFrame(data_nan[i, :, :], names)
            data_filled_extr[i, :, :] = data_df.interpolate(method='linear', axis=1)
        data_int = interpolate_data(data_filled_extr, shape)
    else:
        raise ValueError("Data shape not supported")
    return data_int


def get_dlc_data(data_path, model, filt, part, file, path, labeled_data_path, rt, shape, ratio, filter=True,
                 in_pixel=False):
    data_dlc, data_labelingn, dlc_names, dlc_times, idx_start, idx_end = load_data_from_dlc(labeled_data_path,
                                                                                            data_path, part, file)
    frame_idx = data_dlc["frame_idx"]
    if filter:
        if in_pixel:
            raise RuntimeError("markers in pixel asked to be filtered.")
        new_markers_dlc = np.zeros((3,
                                    data_dlc["markers_in_meters"].shape[1],
                                    data_dlc["markers_in_meters"].shape[2]
                                    ))
        markers_dlc_hom = np.ones((4, data_dlc["markers_in_meters"].shape[1], data_dlc["markers_in_meters"].shape[2]))
        markers_dlc_hom[:3, ...] = data_dlc["markers_in_meters"][:3, ...]
        for k in range(new_markers_dlc.shape[2]):
            new_markers_dlc[:, :, k] = np.dot(np.array(rt), markers_dlc_hom[:, :, k])[:3, :]
        # shape = len(frame_idx) * 2
        new_markers_dlc = fill_and_interpolate(data=new_markers_dlc,
                                               idx=frame_idx,
                                               shape=shape,
                                               fill=True)
        import json
        from utils_old import _convert_cluster_to_anato_old
        measurements_dir_path = "../data_collection_mesurement"
        calibration_matrix_dir = "../../scapula_cluster/calibration_matrix"
        measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{part}.json"
                                          ))
        measurements = measurement_data[f"with_depth"]["measure"]
        calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"][
            "calibration_matrix_name"]
        anato_from_cluster, landmarks_dist = _convert_cluster_to_anato_old(measurements,
                                                                           calibration_matrix,
                                                                           new_markers_dlc[:, -3:, :] * 1000)
        first_idx = dlc_names.index("clavac")
        new_markers_dlc = np.concatenate((new_markers_dlc[:, :first_idx + 1, :],
                                          anato_from_cluster[:3, :, :] * 0.001,
                                          new_markers_dlc[:, first_idx + 1:, :]), axis=1
                                         )
        new_markers_dlc_filtered = np.zeros((3, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
        for i in range(3):
            new_markers_dlc_filtered[i, :8, :] = OfflineProcessing().butter_lowpass_filter(
                new_markers_dlc[i, :8, :],
                2, 120, 2)
            new_markers_dlc_filtered[i, 8:, :] = OfflineProcessing().butter_lowpass_filter(
                new_markers_dlc[i, 8:, :],
                10, 120, 2)

    else:
        key = "markers_in_meters" if not in_pixel else "markers_in_pixel"
        new_markers_dlc_filtered = data_dlc[key]
    dlc_in_pixel = data_dlc["markers_in_pixel"]
    ref_in_pixel = data_labelingn["markers_in_pixel"]
    return new_markers_dlc_filtered, frame_idx, dlc_names, dlc_times, dlc_in_pixel, ref_in_pixel, idx_start, idx_end


def reorder_markers_from_names(markers_data, ordered_markers_names, markers_names):
    idx = []
    markers_names = [convert_string(name) for name in markers_names]
    for i in range(len(ordered_markers_names)):
        if markers_names[i] == "elb":
            markers_names[i] = "elbow"
        if convert_string(ordered_markers_names[i]) in markers_names:
            idx.append(markers_names.index(convert_string(ordered_markers_names[i])))
    return markers_data[:, idx], idx


def check_frames(data_dlc, data_labeling, depth_frame_idx, dlc_frame_idx):
    data = depth_frame_idx.copy()
    ref = dlc_frame_idx.copy()
    type_1 = 0
    type = 0
    idx_init = 0
    idx = 0
    overall_init_idx = None
    overall_final_idx = None
    datalist = [data_dlc, data_labeling]
    if data[0] > ref[0]:
        idx_init = ref.index(data[0])
        ref = ref[idx_init:]
        type_1 = 0
    elif data[0] < ref[0]:
        overall_init_idx = (ref[0] - data[0]) * 2
        idx_init = data.index(ref[0])
        data = data[idx_init:]
        type_1 = 1
    datalist[type_1] = datalist[type_1][..., idx_init:]
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
        datalist[type] = datalist[type][..., :idx]

    if ref != data:
        print("Warning, frames are not synchronized")
    return datalist[0], datalist[1], overall_init_idx, overall_final_idx, ref


def convert_cluster_to_anato(data, measurements=None, calibration_matrix=None, scapula_cluster=None):
    if scapula_cluster is None:
        if not measurements or not calibration_matrix:
            raise ValueError("Measurements and calibration matrix should be provided")
        scapula_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                                         measurements[4], measurements[5], calibration_matrix)

    anato_pos = scapula_cluster.process(marker_cluster_positions=data * 1000, cluster_marker_names=["M1", "M2", "M3"],
                                        save_file=False)
    return anato_pos * 0.001


def adjust_idx(data, idx_start, idx_end):
    if idx_start is None and idx_end is None:
        return data
    data_tmp = {}
    for key in data.keys():
        if "names" in key:
            data_tmp[key] = data[key]
            continue
        idx_start = 0 if idx_start is None else idx_start
        if isinstance(data[key], np.ndarray):
            idx_end = data[key].shape[-1] + 1 if idx_end is None else -idx_end
            data_tmp[key] = data[key][..., idx_start:idx_end]
        elif isinstance(data[key], list):
            data_tmp[key] = data[key][idx_start:idx_end]
        else:
            data_tmp[key] = data[key]
    return data_tmp
