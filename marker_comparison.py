import os
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
try:
    from msk_utils import process_all_frames
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



def compute_error(markers_depth, markers_vicon, vicon_to_depth_idx, state_depth, state_vicon):
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
        # if i not in [3, 4, 5, 8]:
        err_q.append(np.mean(np.sqrt(np.mean(((state_depth[i, :] - state_vicon[i, :]) ** 2), axis=0))))
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
    for i in range(3):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def refine_synchro(depth_markers, vicon_markers, vicon_to_depth_idx):
    error = np.inf
    depth_markers_tmp = np.zeros((3, depth_markers.shape[1], depth_markers.shape[2]))
    vicon_markers_tmp = np.zeros((3, vicon_markers.shape[1], vicon_markers.shape[2]))
    idx = 0
    for i in range(30):
        vicon_markers_tmp = vicon_markers[:, :, :-i] if i != 0 else vicon_markers
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

def load_data(part, file, end_idx=None):
    data = load(f"{processed_data_path}/{part}/{file}")
    markers_depth = data["markers_depth_interpolated"][..., :end_idx] if end_idx else data["markers_depth_interpolated"]
    markers_vicon = data["truncated_markers_vicon"][..., :end_idx] if end_idx else data["truncated_markers_vicon"]
    sensix_data = data["sensix_data_interpolated"][:, :end_idx] if end_idx else data["sensix_data_interpolated"]
    depth_markers_names = data["depth_markers_names"]
    vicon_markers_names = data["vicon_markers_names"]
    vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)
    markers_depth, markers_vicon, idx = refine_synchro(markers_depth, markers_vicon, vicon_to_depth_idx)
    emg = data["emg_proc_interpolated"][:, :end_idx - idx] if end_idx else data["emg_proc_interpolated"]

    markers_minimal_vicon = markers_vicon[:, vicon_to_depth_idx, :]
    markers_depth_filtered = np.zeros((3, markers_depth.shape[1], markers_depth.shape[2]))
    for i in range(3):
        markers_depth_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(markers_depth[i, :, :],
                                                                                    4, 120, 4)
    markers_from_source = [markers_depth, markers_vicon, markers_minimal_vicon]
    forces = ExternalLoads()
    forces.add_external_load(
        point_of_application=[0, 0, 0],
        applied_on_body="radius_left_pro_sup_left",
        express_in_coordinate="ground",
        name="hand_pedal",
        load=np.zeros((6, 1)),
    )
    f_ext = np.array([sensix_data["RMY"],
                      -sensix_data["RMX"],
                      sensix_data["RMZ"],
                      sensix_data["RFY"],
                      -sensix_data["RFX"],
                      sensix_data["RFZ"]])
    return markers_from_source, forces, f_ext, emg

def main(model_dir, participants, processed_data_path):
    model_paths = [model_dir + "wu_bras_gauche_depth.bioMod", model_dir + "wu_bras_gauche_vicon.bioMod",
                   model_dir + "wu_bras_gauche_depth.bioMod"]
    source = ["depth", "vicon", "minimal_vicon"]
    for part in participants:
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file]
        for file in all_files:
            markers_from_source, forces, f_ext, emg = load_data(part, file, end_idx=150)
            if len(model_paths) != len(markers_from_source):
                raise ValueError("The number of models and the number of sources are different")
            for s in range(len(markers_from_source)):
                msk_function = MskFunctions(model=model_paths[s], data_buffer_size=6, system_rate=120)
                result_biomech = process_all_frames(markers_from_source[s][:, :-3, :], msk_function,
                                                    forces, (1, 1), emg,
                f_ext, save_data=False, data_path=f"{processed_data_path}/{part}/result_biomech_{file}_{source[s]}.bio")


if __name__ == '__main__':
    model_dir = "models/"
    participants = ["P9"]#, "P10", "P11", "P12", "P13", "P14"]  # ,"P9", "P10",
    processed_data_path = "Q:\Projet_hand_bike_markerless/process_data"
    main(model_dir, participants, processed_data_path)