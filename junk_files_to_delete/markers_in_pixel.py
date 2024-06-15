from pathlib import Path
import os

import matplotlib.pyplot as plt

from biosiglive import load
from utils import load_data, _get_vicon_to_depth_idx, _convert_string
from utils import *


def compute_error(data, ref):
    shape_idx = 1 if data.shape[0] == 2 else 0
    n_data = data.shape[shape_idx]
    err = np.zeros((n_data))
    for i in range(n_data):
        # remove nan values
        if len(data.shape) == 3:
            nan_index = np.argwhere(np.isnan(ref[:, i, :]))
            data_tmp = np.delete(data[:, i, :], nan_index, axis=1)
            ref_tmp = np.delete(ref[:, i, :], nan_index, axis=1)
            err[i] = np.mean(np.sqrt(np.median(((data_tmp - ref_tmp) ** 2), axis=0)))
        else:
            nan_index = np.argwhere(np.isnan(ref[i, :]))
            data_tmp = np.delete(data[i, :], nan_index, axis=0)
            ref_tmp = np.delete(ref[i, :], nan_index, axis=0)
            err[i] = np.mean(np.sqrt(np.median(((data_tmp - ref_tmp) ** 2), axis=0)))
    return err


def compute_std(data, ref):
    shape_idx = 1 if data.shape[0] == 2 else 0
    n_data = data.shape[shape_idx]
    err = np.zeros((n_data))
    for i in range(n_data):
        # remove nan values
        if len(data.shape) == 3:
            nan_index = np.argwhere(np.isnan(ref[:, i, :]))
            data_tmp = np.delete(data[:, i, :], nan_index, axis=1)
            ref_tmp = np.delete(ref[:, i, :], nan_index, axis=1)
            err[i] = np.mean(np.std(data_tmp - ref_tmp, axis=1))
        else:
            nan_index = np.argwhere(np.isnan(data[i, :]))
            data_tmp = np.delete(data[i, :], nan_index, axis=0)
            ref_tmp = np.delete(ref[i, :], nan_index, axis=0)
            err[i] = np.mean(np.std(data_tmp - ref_tmp, axis=0))
    return err


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_end_frame(part, file):
    end_frame = None
    if part == "P12" and "gear_10" in file:
        end_frame = 11870
    elif part == "P12" and "gear_15" in file:
        end_frame = 10220
    elif part == "P12" and "gear_20" in file:
        end_frame = 10130
    elif part == "P11" and "gear_20" in file:
        end_frame = 8970
    return end_frame


if __name__ == '__main__':
    # participants = ["P9", "P10",  "P11", "P12", "P13", "P14", "P15", "P16"]
    participants = [f"P{i}" for i in range(9, 17)]
    participants.pop(participants.index("P12"))
    trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    # trials[-1] = ["gear_10"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))

    # plt.show()
    # all_data, trials = load_results(participants,
    #                                 "Q://Projet_hand_bike_markerless/process_data",
    #                                 file_name="normal_alone", recompute_cycles=False)

    keys = ["markers_in_pixel"]  # , "q_dot", "q_ddot", "tau", "mus_act", "mus_force"]
    factors = [1, 180 / np.pi]  # , 180 / np.pi, 180 / np.pi, 1, 100, 1]
    units = ["°", "°/s", "°/s²", "N.m", "%", "N"]
    source = ["minimal_vicon", "minimal_vicon", "depth"]
    to_compare_source = ["depth", "dlc", "dlc"]
    # plot the colors
    # colors = ["b", "orange", "g"]
    # keys = ["q"]
    all_rmse = []
    all_std = []
    outliers = [None] * 3
    for k, key in enumerate(keys):
        all_colors = []
        shape_idx = 1 if key == "markers" else 0
        n_key = 13
        means_file = np.ndarray((len(participants) * n_key,))
        diffs_file = np.ndarray((len(participants) * n_key,))
        rmse = np.ndarray((len(participants) * n_key,))
        std = np.ndarray((len(participants) * n_key,))
        for p, part in enumerate(participants):
            means = np.ndarray((n_key, len(trials[p])))
            diffs = np.ndarray((n_key, len(trials[p])))
            all_colors.append([colors[p]] * n_key)
            rmse_file = np.ndarray((n_key, len(trials[p])))
            std_file = np.ndarray((n_key, len(trials[p])))
            all_files = os.listdir(f"Q://Projet_hand_bike_markerless/RGBD/{part}")
            # all_files = [file for file in all_files if "gear" in file and "result_biomech" not in file and "3_crops" in file]
            all_files = [file for file in all_files if "gear" in file and "less" not in file and "more" not in file]
            for f, file in enumerate(all_files):
                path = f"Q://Projet_hand_bike_markerless/RGBD{os.sep}{part}{os.sep}{file}"
                labeled_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_pp.bio"
                dlc_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_normal_alone_pp.bio"
                data_dlc, data_labeling = load_data_from_dlc(labeled_data_path, dlc_data_path, part, file)
                if data_dlc["markers_in_pixel"].shape[2] < data_labeling["markers_in_pixel"].shape[2]:
                    data_labeling["markers_in_pixel"] = data_labeling["markers_in_pixel"][:, :, :data_dlc["markers_in_pixel"].shape[2]]
                elif data_dlc["markers_in_pixel"].shape[2] > data_labeling["markers_in_pixel"].shape[2]:
                    data_dlc["markers_in_pixel"] = data_dlc["markers_in_pixel"][:, :, :data_labeling["markers_in_pixel"].shape[2]]
                to_compare = data_dlc["markers_in_pixel"][:2, :, :]
                ref_data = data_labeling["markers_in_pixel"][:2, :, :]
                rmse_file[:, f] = compute_error(to_compare * factors[k],
                                                   ref_data * factors[k])
                std_file[:, f] = compute_std(to_compare * factors[k],
                                                ref_data * factors[k])


            rmse[ n_key * p: n_key * (p + 1)] = np.mean(rmse_file[ :, :], axis=1)
            std[ n_key * p: n_key * (p + 1)] = np.mean(std_file[:, :], axis=1)
        all_rmse.append(rmse.mean().round(2))
        all_std.append(std.mean().round(2))

    print(r"""
    \begin{table}[h]
    \caption{Root Mean Square Deviation (RMSD), along with Bland-Altman limit of agreement (LOA) and Bland-Altman Bias,
     of the biomechanical outcomes using both Vicon-based methods, with redundancy and minimal, as reference standards.}
    \centering
    \begin{tabular}{l|l|cc|cc|c}
    \hline
         & & \multicolumn{2}{c|}{RMSD $\pm$ SD} & \multicolumn{2}{c|}{LOA} & Bias \\
         &  &  & & Low & High &  \\
         \hline
         """
          "\multirow{3}*{Markers (mm)} "
          "& depth vs vicon &" + f" {all_rmse}& {all_std}" + r"\\" + "\n"
                                                                     
          # "\multirow{3}*{Joint velocity (\degree~/s)} "
          # "& RGBD vs redundant &" + f" {all_rmse[1][0]}& {all_std[1][0]} & {all_loa[1][0][0]}& {all_loa[1][0][1]}& {all_bias[1][0]}" + r"\\" + "\n" 
          # "& minimal vs redundant &" + f" {all_rmse[1][1]}& {all_std[1][1]}  & {all_loa[1][1][0]}& {all_loa[1][1][1]}& {all_bias[1][1]}" + r"\\" + "\n" 
          # "& RGBD vs minimal &" + f" {all_rmse[1][2]}& {all_std[1][2]}  & {all_loa[1][2][0]}& {all_loa[1][2][1]}& {all_bias[1][2]}"+ r"\\" + "\n" +  r" \hdashline" + "\n"

          #   "\multirow{3}*{Joint torques (N.m)} "
          # "& RGBD vs redundant &" + f" {all_rmse[3][0]}& {all_std[3][0]} & {all_loa[3][0][0]}& {all_loa[3][0][1]}& {all_bias[3][0]}" + r"\\" + "\n" 
          # "& minimal vs redundant &" + f" {all_rmse[3][1]}& {all_std[3][1]}  & {all_loa[3][1][0]}& {all_loa[3][1][1]}& {all_bias[3][1]}" + r"\\" + "\n" 
          # "& RGBD vs minimal &" + f" {all_rmse[3][2]}& {all_std[3][2]}  & {all_loa[3][2][0]}& {all_loa[3][2][1]}& {all_bias[3][2]}"+ r"\\" + "\n" +  r" \hdashline" + "\n"
          #   
          # "\multirow{3}*{Muscle forces (N)} "
          # "& RGBD vs redundant &" + f" {all_rmse[5][0]}& {all_std[5][0]} & {all_loa[5][0][0]}& {all_loa[5][0][1]}& {all_bias[5][0]}" + r"\\" + "\n" 
          # "& minimal vs redundant &" + f" {all_rmse[5][1]}& {all_std[5][1]}  & {all_loa[5][1][0]}& {all_loa[5][1][1]}& {all_bias[5][1]}" + r"\\" + "\n" 
          # "& RGBD vs minimal &" + f" {all_rmse[5][2]}& {all_std[5][2]}  & {all_loa[5][2][0]}& {all_loa[5][2][1]}& {all_bias[5][2]}"+ r"\\" + "\n" + r"\hline" + "\n"                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  r"""                                                                                                      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \end{tabular}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \label{tab:errors}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \end{table}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         """)
    plt.show()
