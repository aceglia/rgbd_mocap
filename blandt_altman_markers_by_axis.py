from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from biosiglive import load
from utils import load_data, _get_vicon_to_depth_idx, _convert_string
from utils import *

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    #trials = [["gear_5", "gear_10"]] * len(participants)
    all_data, trials = load_all_data(participants,
                              "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            )

    scatter_markers = ["o", "s", "v"]
    title = ["a) Image plane", "b) Depth value", "c) 3D space"]
    all_means_file = []
    all_diffs_file = []
    all_colors = []
    fig = plt.figure("bland_alt mark")
    ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0))
    ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 1), sharey=ax1)
    ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), rowspan=1, colspan=2)
    axes = [ax1, ax2, ax3]
    for j in range(0, 3):
        key = ["markers"]
        n_key = all_data[participants[0]][list(all_data[participants[0]].keys())[0]]["markers_depth"].shape[1]
        means_file = np.ndarray((len(participants) * n_key))
        diffs_file = np.ndarray((len(participants) * n_key))
        all_colors = []
        colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
        for p, part in enumerate(all_data.keys()):
            # means = np.ndarray((n_key, len(trials[p])))
            # diffs = np.ndarray((n_key, len(trials[p])))
            means = None
            diffs = None
            all_colors.append([colors[p]] * n_key)
            for f, file in enumerate(all_data[part].keys()):
                markers_depth, markers_vicon, vicon_to_depth = load_in_markers_ref(all_data[part][file])
                sum_minimal = (markers_depth[2, :, :] + markers_vicon[2, vicon_to_depth, :]) / 2
                if j == 1:
                    dif_minimal = markers_depth[j, ...] - markers_vicon[j, vicon_to_depth, :]
                    nan_idx = np.argwhere(np.isnan(sum_minimal))
                    non_visible_idx = None
                    sum_minimal = np.delete(sum_minimal, nan_idx, axis=1)
                    dif_minimal = np.delete(dif_minimal, nan_idx, axis=1)
                    if np.abs(np.mean(dif_minimal, axis=1)).max() > 0.05:
                        print(part, file)
                        continue
                    means = (np.mean(sum_minimal, axis=1) * 1000)[:, None] if means is None else np.append(means,
                                                                                                           (np.mean(
                                                                                                               sum_minimal,
                                                                                                               axis=1) * 1000)[
                                                                                                           :, None],
                                                                                                           axis=1)
                    diffs = (np.mean(dif_minimal, axis=1) * 1000)[:, None] if diffs is None else np.append(diffs,
                                                                                                           (np.mean(
                                                                                                               dif_minimal,
                                                                                                               axis=1) * 1000)[
                                                                                                           :, None],
                                                                                                           axis=1)
                else:

                    if j == 2:
                        dif_minimal = markers_depth - markers_vicon[:, vicon_to_depth, :]
                    else:
                        dif_minimal = markers_depth[:2, :, :] - markers_vicon[:2, vicon_to_depth, :]
                    try:
                        if np.abs(np.mean(dif_minimal, axis=2).mean(axis=0)).max() > 0.05:
                            print(part, file)
                    except:
                        pass
                    nan_idx = np.argwhere(np.isnan(sum_minimal))
                    non_visible_idx = None
                    sum_minimal = np.delete(sum_minimal, nan_idx, axis=1)
                    nan_idx = np.argwhere(np.isnan(dif_minimal))
                    dif_minimal = np.delete(dif_minimal, nan_idx, axis=2)
                    means = (np.mean(sum_minimal, axis=1) * 1000)[:, None] if means is None else np.append(means, (np.mean(sum_minimal, axis=1) * 1000)[:, None], axis=1)
                    diffs = (np.mean(dif_minimal, axis=0).mean(axis=1) * 1000)[:, None] if diffs is None else np.append(diffs, (np.mean(dif_minimal, axis=0).mean(axis=1) * 1000)[:, None], axis=1)

            means_file[n_key * p: n_key * (p + 1)] = np.mean(means, axis=1)
            diffs_file[n_key * p: n_key * (p + 1)] = np.mean(diffs, axis=1)
        all_means_file.append(means_file)
        all_diffs_file.append(diffs_file)
        bias, lower_loa, upper_loa = compute_blandt_altman(means_file, diffs_file, units="mm", title=title[j],
                              color=all_colors, show=False, x_axis="Mean on z axis (mm)", ax=axes[j])

    plt.show()
