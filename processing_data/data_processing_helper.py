import numpy as np
from casadi import interp1d
import pandas as pd
from scipy.interpolate import interp1d
from biosiglive import OfflineProcessing
from processing_data.scapula_cluster.from_cluster_to_anato import ScapulaCluster
import os

def convert_string(string):
    return string.lower().replace("_", "")

def process_cycles(all_results, peaks, n_peaks=None, interpolation_size=120, key_to_get_size="q"):
    for key in all_results.keys():
        if key == "file":
            continue
        data_size = all_results[key][key_to_get_size].shape[1]
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
                interp_function = _interpolate_2d_data if len(all_results[key][key2].shape) == 2 else interpolate_data
                if array_tmp is None:
                    array_tmp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], interpolation_size)
                    array_tmp = array_tmp[None, ...]
                else:
                    data_interp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], interpolation_size)
                    array_tmp = np.concatenate((array_tmp, data_interp[None, ...]), axis=0)
            dic_tmp[key2] = array_tmp
        all_results[key]["cycles"] = dic_tmp
    return all_results


def compute_error_mark(ref_mark, mark):
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


def refine_synchro(marker_full, marker_to_refine, plot_fig=True, nb_frame=200):
    error_list = []
    for i in range(nb_frame):
        marker_to_refine_tmp = marker_to_refine[:, :, :-i] if i != 0 else marker_to_refine
        marker_to_refine_tmp = interpolate_data(marker_to_refine_tmp, marker_full.shape[2])
        error_markers = compute_error_mark(
            marker_full[:, ...], marker_to_refine_tmp[:, ...])
        error_tmp = np.abs(np.mean(error_markers))
        error_list.append(error_tmp)
    idx = error_list.index(min(error_list))
    marker_to_refine_tmp = marker_to_refine[:, :, :-idx] if idx != 0 else marker_to_refine[:, :, :]
    marker_to_refine_tmp = interpolate_data(marker_to_refine_tmp, marker_full.shape[2])
    if plot_fig:
        import matplotlib.pyplot as plt
        plt.figure("refine synchro")
        for i in range(marker_to_refine_tmp.shape[1]):
            plt.subplot(4, 4, i + 1)
            for j in range(0, 3):
                plt.plot(marker_to_refine_tmp[j, i, :], "b")
                plt.plot(marker_full[j, i, :], 'r')
    print("idx to refine synchro : ", idx, "error", min(error_list))
    return marker_to_refine_tmp, idx


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
        if measurements is None or calibration_matrix is None:
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
