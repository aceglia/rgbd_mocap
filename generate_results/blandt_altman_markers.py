from pathlib import Path
import os

import numpy as np

from biosiglive import load
from utils_old import load_data, _get_vicon_to_depth_idx, _convert_string
from utils_old import *
from RMSE import compute_error, compute_std

if __name__ == '__main__':
    participants = [ "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    #trials = [["gear_5", "gear_10"]] * len(participants)
    all_data, trials = load_all_data(participants,
                              "Q:\Projet_hand_bike_markerless\process_data",
                            )
    key = ["markers"]
    n_key = all_data[participants[0]][list(all_data[participants[0]].keys())[0]]["markers_depth"].shape[1]
    means_file = np.ndarray((len(participants) * n_key))
    diffs_file = np.ndarray((len(participants) * n_key))
    all_colors = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
    for p, part in enumerate(all_data.keys()):
        means = None
        diffs = None
        all_colors.append([colors[p]] * n_key)
        for f, file in enumerate(all_data[part].keys()):
            markers_depth, markers_vicon, vicon_to_depth = load_in_markers_ref(all_data[part][file])
            sum_minimal = (markers_depth[2, :, :] + markers_vicon[2, vicon_to_depth, :]) / 2
            dif_minimal = markers_depth - markers_vicon[:, vicon_to_depth, :]
            nan_idx = np.argwhere(np.isnan(sum_minimal))
            sum_minimal = np.delete(sum_minimal, nan_idx, axis=1)
            dif_minimal = np.delete(dif_minimal, nan_idx, axis=2)
            # outliers_idx = np.argwhere(np.abs(dif_minimal) > 0.015)
            # sum_minimal = np.delete(sum_minimal, outliers_idx, axis=1)
            # dif_minimal = np.delete(dif_minimal, outliers_idx, axis=2)
            if np.max(np.abs(np.mean(dif_minimal, axis=0).mean(axis=1))) > 0.025:
                print("outliers", part, file)
                continue
            if means is None:
                means = np.mean(sum_minimal, axis=1) * 1000
                means = means[:, np.newaxis]
            else:
                means = np.append(means, (np.mean(sum_minimal, axis=1) * 1000)[:, np.newaxis], axis=1)
            if diffs is None:
                diffs = np.mean(dif_minimal, axis=0).mean(axis=1) * 1000
                diffs = diffs[:, np.newaxis]
            else:
                diffs = np.append(diffs, (np.mean(dif_minimal, axis=0).mean(axis=1) * 1000)[:, np.newaxis], axis=1)
        means_file[n_key * p: n_key * (p + 1)] = np.mean(means, axis=1)
        diffs_file[n_key * p: n_key * (p + 1)] = np.mean(diffs, axis=1)

    compute_blandt_altman(means_file, diffs_file, units="mm", title="Bland-Altman Plot for markers positions", color=all_colors, show=True, x_axis="Mean on z axis (mm)")

