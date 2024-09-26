from pathlib import Path
import os

import matplotlib.pyplot as plt

from biosiglive import load
from utils_old import load_data, _get_vicon_to_depth_idx, _convert_string
from utils_old import *


def compute_error(data, ref):
    shape_idx = 1 if data.shape[0] == 3 else 0
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
    shape_idx = 1 if data.shape[0] == 3 else 0
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
    #participants.pop(participants.index("P12"))
    #trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    #trials[-1] = ["gear_10"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))

    plt.figure("colors")
    for i in range(len(participants)):
        plt.scatter(i, i, color=colors[i], s=200, alpha=0.5)
    plt.legend(participants)
    # plt.show()
    all_data, trials = load_results(participants,
                            "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            file_name="normal_times_three_filtered_all_dofs", recompute_cycles=False)

    keys = ["tracked_markers", "q_raw", "q_dot"]  # "q_ddot", "tau", "mus_act", "mus_force"]
    factors = [1000, 180 / np.pi, 180 / np.pi]  #, 180 / np.pi, 1, 100, 1]
    units = ["°", "°/s", "°/s²", "N.m", "%", "N"]
    source = ["vicon", "vicon", "depth"]
    to_compare_source = ["depth", "dlc", "dlc"]
    # plot the colors
    n_comparison = len(to_compare_source)
    #colors = ["b", "orange", "g"]
    # keys = ["q"]
    all_rmse = []
    all_std = []
    all_bias = [None] * len(keys)
    all_loa = [None] * len(keys)
    outliers = [None] * 3
    for k, key in enumerate(keys):
        all_bias[k] = []
        all_loa[k] = []
        all_colors = []
        shape_idx = 1 if "markers" in key else 0
        n_key = all_data[participants[0]][list(all_data[participants[0]].keys())[0]]["depth"][key].shape[shape_idx]
        means_file = np.ndarray((n_comparison, len(participants) * n_key))
        diffs_file = np.ndarray((n_comparison, len(participants) * n_key))
        rmse = np.ndarray((n_comparison, len(participants) * n_key))
        std = np.ndarray((n_comparison, len(participants) * n_key))
        for p, part in enumerate(all_data.keys()):
            means = np.ndarray((n_comparison, n_key, len(trials[p])))
            diffs = np.ndarray((n_comparison, n_key, len(trials[p])))
            all_colors.append([colors[p]] * n_key)
            rmse_file = np.ndarray((n_comparison, n_key, len(trials[p])))
            std_file = np.ndarray((n_comparison, n_key, len(trials[p])))
            for f, file in enumerate(all_data[part].keys()):
                for j in range(n_comparison):
                    end_frame = get_end_frame(part, file)
                    source_tmp = "minimal_vicon" if "markers" in key and "vicon" in source[j] else source[j]
                    if key == "tracked_markers" and "dlc" in to_compare_source[j]:
                        markers_names = all_data[part][file][to_compare_source[j]]["marker_names"]
                        to_compare = all_data[part][file][to_compare_source[j]][key][...,
                                     :end_frame] if end_frame is not None else \
                        all_data[part][file][to_compare_source[j]][key]
                        idx_scap_ia = markers_names.index("SCAP_IA")
                        idx_scap_ts = markers_names.index("SCAP_TS")
                        # exchange _IA and _TS
                        to_compare[:, idx_scap_ia, :], to_compare[:, idx_scap_ts, :] = to_compare[:, idx_scap_ts, :], to_compare[
                            :, idx_scap_ia, :]
                        idx_to_remove = markers_names.index("ribs")
                        to_compare = np.delete(to_compare, idx_to_remove, axis=1)
                    else:
                        to_compare = all_data[part][file][to_compare_source[j]][key][..., :end_frame] if end_frame is not None else all_data[part][file][to_compare_source[j]][key]
                    ref_data = all_data[part][file][source_tmp][key][..., :end_frame] if end_frame is not None else \
                        all_data[part][file][source_tmp][key]
                    rmse_file[j, :, f] = compute_error(to_compare * factors[k],
                                                       ref_data * factors[k])
                    std_file[j, :, f] = compute_std(to_compare * factors[k],
                                                    ref_data * factors[k])
                    sum_minimal = (to_compare + ref_data) / 2
                    dif_minimal = to_compare - ref_data
                    nan_idx = np.argwhere(np.isnan(sum_minimal))
                    print(part, file, nan_idx.shape)
                    nan_idx_bis = np.argwhere(np.isnan(dif_minimal))
                    axis = 2 if "markers" in key else 1
                    nan_idx = np.unique(np.concatenate((nan_idx, nan_idx_bis), axis=1))
                    sum_minimal = np.delete(sum_minimal, nan_idx, axis=axis)
                    dif_minimal = np.delete(dif_minimal, nan_idx, axis=axis)
                    # if key == "q":
                    #     for m in range(dif_minimal.shape[0]):
                    #         dif_minimal_tmp = dif_minimal[m, :] * factors[k]
                    #         dif_minimal_tmp_clipped = dif_minimal_tmp[np.abs(dif_minimal_tmp) < 40]
                    #         sum_minimal_tmp = sum_minimal[m, :] * factors[k]
                    #         sum_minimal_tmp_clipped = sum_minimal_tmp[np.abs(dif_minimal_tmp) < 40]
                    #         print(part, file, sum_minimal_tmp_clipped.shape, dif_minimal_tmp_clipped.shape)
                    #         means[j, m, f] = np.mean(sum_minimal_tmp_clipped)
                    #         diffs[j, m, f] = np.mean(dif_minimal_tmp_clipped)
                    # else:
                    if "markers" in key:
                        sum_minimal = np.mean(sum_minimal, axis=0)
                        dif_minimal = np.mean(dif_minimal, axis=0)
                    means[j, :, f] = np.mean(sum_minimal, axis=1) * factors[k]
                    diffs[j, :, f] = np.mean(dif_minimal, axis=1) * factors[k]

            for j in range(n_comparison):
                means_file[j, n_key * p: n_key * (p+1)] = np.mean(means[j, :, :], axis=1)
                diffs_file[j, n_key * p: n_key * (p+1)] = np.mean(diffs[j, :, :], axis=1)
                rmse[j, n_key * p: n_key * (p + 1)] = np.mean(rmse_file[j, :, :], axis=1)
                std[j, n_key * p: n_key * (p + 1)] = np.mean(std_file[j, :, :], axis=1)
        all_rmse.append(rmse.mean(axis=1).round(2))
        all_std.append(std.mean(axis=1).round(2))
        bias, lower_loa, upper_loa = compute_blandt_altman(means_file[0, :], diffs_file[0, :],
                              units=units[k], title="Bland-Altman Plot for " + key + " depth vs vicon",
                              show=False, color=all_colors)
        all_bias[k].append(np.round(bias, 2))
        all_loa[k].append([np.round(lower_loa, 2), np.round(upper_loa, 2)])
        bias, lower_loa, upper_loa = compute_blandt_altman(means_file[1, :], diffs_file[1, :], units=units[k],
                              title="Bland-Altman Plot for " + key + " dlc vs vicon",
                              show=False, color=all_colors)

        all_bias[k].append(np.round(bias, 2))
        all_loa[k].append([np.round(lower_loa, 2), np.round(upper_loa, 2)])
        bias, lower_loa, upper_loa = compute_blandt_altman(means_file[2, :], diffs_file[2, :], units=units[k],
                              title="Bland-Altman Plot for " + key + " dlc vs depth",
                              show=False, color=all_colors)

        all_bias[k].append(np.round(bias, 2))
        all_loa[k].append([np.round(lower_loa, 2), np.round(upper_loa, 2)])

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
          "& depth vs vicon &" + f" {all_rmse[0][0]}& {all_std[0][0]} & {all_loa[0][0][0]}& {all_loa[0][0][1]}& {all_bias[0][0]}" + r"\\" + "\n" 
          "& dlc vs vicon &" + f" {all_rmse[0][1]}& {all_std[0][1]}  & {all_loa[0][1][0]}& {all_loa[0][1][1]}& {all_bias[0][1]}" + r"\\" + "\n" 
          "& dlc vs depth &" + f" {all_rmse[0][2]}& {all_std[0][2]}  & {all_loa[0][2][0]}& {all_loa[0][2][1]}& {all_bias[0][2]}"+ r"\\" + "\n" +  r" \hdashline" + "\n"
          "\multirow{3}*{Joint angles (\degree~)} "
          "& depth vs vicon &" + f" {all_rmse[1][0]}& {all_std[1][0]} & {all_loa[1][0][0]}& {all_loa[1][0][1]}& {all_bias[1][0]}" + r"\\" + "\n" 
          "& dlc vs vicon &" + f" {all_rmse[1][1]}& {all_std[1][1]}  & {all_loa[1][1][0]}& {all_loa[1][1][1]}& {all_bias[1][1]}" + r"\\" + "\n" 
          "& dlc vs depth &" + f" {all_rmse[1][2]}& {all_std[1][2]}  & {all_loa[1][2][0]}& {all_loa[1][2][1]}& {all_bias[1][2]}"+ r"\\" + "\n" +  r" \hdashline" + "\n"                                                                                                                                                       
          "\multirow{3}*{Joint velocity (\degree~/s)} "
          "& RGBD vs redundant &" + f" {all_rmse[2][0]}& {all_std[2][0]} & {all_loa[2][0][0]}& {all_loa[2][0][1]}& {all_bias[2][0]}" + r"\\" + "\n" 
          "& minimal vs redundant &" + f" {all_rmse[2][1]}& {all_std[2][1]}  & {all_loa[2][1][0]}& {all_loa[2][1][1]}& {all_bias[2][1]}" + r"\\" + "\n" 
          "& RGBD vs minimal &" + f" {all_rmse[2][2]}& {all_std[2][2]}  & {all_loa[2][2][0]}& {all_loa[2][2][1]}& {all_bias[2][2]}"+ r"\\" + "\n" +  r" \hdashline" + "\n"
                                                                                                                                                                      
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
