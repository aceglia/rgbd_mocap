from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from biosiglive import load, save
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
    #participants.pop(participants.index("P11"))
    #trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    #trials[-1] = ["gear_10"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))

    plt.figure("colors")
    for i in range(len(participants)):
        plt.scatter(i, i, color=colors[i], s=200, alpha=0.5)
    plt.legend(participants)
    # plt.show()
    reload_data = False
    if reload_data:
        all_data, trials = load_results(participants,
                                "/mnt/shared/Projet_hand_bike_markerless/process_data",
                                file_name="normal_500_down_b1_no_root.bio", recompute_cycles=False,
                                        to_exclude=["live_filt"],
                                        #  trials_to_exclude=[["P13", "gear_5"],
                                        # #                    ["P13", "gear_15"],
                                        #                     ["P12", "gear_15"],
                                        #                     ["P11", "gear_20"]]
                                        )
        save(all_data, "_all_data_tmp.bio", safe=False)
    else:
        all_data = load("_all_data_tmp.bio")

    all_errors_std = []
    for part in all_data.keys():
        error_tmp = []
        std_tmp = []
        for key in all_data[part].keys():
            data_tmp = all_data[part][key]
            data_tmp["dlc_in_pixel"] = data_tmp["dlc_in_pixel"][:, 1:, :]
            data_tmp["dlc_in_pixel"][:, 8, :], data_tmp["dlc_in_pixel"][:, 9, :] = data_tmp["dlc_in_pixel"][:, 9, :], \
            data_tmp["dlc_in_pixel"][:, 8, :]
            fix_len = min(data_tmp["ref_in_pixel"].shape[-1], data_tmp["dlc_in_pixel"].shape[-1])
            error_tmp.append(compute_error(data_tmp["ref_in_pixel"][:2, :, :fix_len], data_tmp["dlc_in_pixel"][:2, :, :fix_len]))
            std_tmp.append(compute_std(data_tmp["ref_in_pixel"][:2, :, :fix_len], data_tmp["dlc_in_pixel"][:2, :, :fix_len]))
        all_errors_std.append([np.round(np.mean(error_tmp), 2), np.round(np.mean(std_tmp), 2)])
    mean_errors = np.mean(np.array(all_errors_std)[:, 0])
    variance = np.abs(np.repeat(mean_errors, 8) - np.array(all_errors_std)[:, 0])
    print(variance)
    var = np.var(np.array(all_errors_std)[:, 0])
    print(mean_errors, var)
    for er in all_errors_std:
        print("err", er[0], "std", er[1])
print(
    r"""
    Participant & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\
    Gender & M & M & M & F & F & M & F & M \\ """ + "\n"
    "\par{Mean (pixel)}&" + f"{all_errors_std[0][0]:0,.2f} & {all_errors_std[1][0]:0,.2f} "
                            f" & {all_errors_std[2][0]:0,.2f}  & {all_errors_std[3][0]:0,.2f}  "
                            f"& {all_errors_std[4][0]:0,.2f}  & {all_errors_std[5][0]:0,.2f}  &"
                            f" {all_errors_std[6][0]:0,.2f}  & {all_errors_std[7][0]:0,.2f} " + r"\\" + "\n"
    "\par{STD (pixel)}&"  + f"{all_errors_std[0][1]:0,.2f} & {all_errors_std[1][1]:0,.2f} "
                            f" & {all_errors_std[2][1]:0,.2f}  & {all_errors_std[3][1]:0,.2f}  "
                            f"& {all_errors_std[4][1]:0,.2f}  & {all_errors_std[5][1]:0,.2f}  &"
                            f" {all_errors_std[6][1]:0,.2f}  & {all_errors_std[7][1]:0,.2f} " + r"\\"
r"""
\begin{table}[ht]
\caption{Root-mean-square error (RMSE) and standard deviation (SD) of the error in pixel between annotated image and detection for each model.}
\centering
\begin{tabular}{lccc}
\hline
Participant & Gender & Deviation (pixel)\\
\hline """ + "\n"
f"Participant 1 & M &  {variance[0]: .2f} " + r"\\" + "\n"
"\hdashline"+ "\n"
f"Participant 2 & M &  {variance[1]: .2f} " + r"\\" + "\n"
"\hdashline"+ "\n"
f"Participant 3 & M &  {variance[2]: .2f} " + r"\\" + "\n"
"\hdashline"+ "\n"
f"Participant 4 & F &  {variance[3]: .2f} " + r"\\" + "\n"
"\hdashline"+ "\n"
f"Participant 5 & F &  {variance[4]: .2f} " + r"\\" + "\n"
"\hdashline"+ "\n"
f"Participant 6 & M &  {variance[5]: .2f} " + r"\\" + "\n"
"\hdashline"+ "\n"
f"Participant 7 & F &  {variance[6]: .2f} " + r"\\" + "\n"
"\hdashline"+ "\n"
f"Participant 8 & M &  {variance[7]: .2f} " + r"\\" + "\n"
"\hdashline\n"
r"RMSE for all model & & \textbf{ " + f"{np.round(np.mean(np.array(all_errors_std)[:, 0]), 2): .2f}" + r"}\\" + "\n"
"\hline"
"""
\end{tabular}
\label{tab:model_variability}
\end{table}""")