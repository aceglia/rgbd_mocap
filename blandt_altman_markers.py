from pathlib import Path
import os
from biosiglive import load
from utils import load_data, _get_vicon_to_depth_idx, _convert_string
from utils import *

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P13", "P14", "P15", "P16"]
    #trials = [["gear_5", "gear_10"]] * len(participants)
    all_data, trials = load_all_data(participants,
                              "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            )
    key = ["markers"]
    n_key = all_data[participants[0]][list(all_data[participants[0]].keys())[0]]["markers_depth"].shape[1]
    means_file = np.ndarray((len(participants) * n_key))
    diffs_file = np.ndarray((len(participants) * n_key))
    all_colors = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
    for p, part in enumerate(all_data.keys()):
        means = np.ndarray((n_key, len(trials[p])))
        diffs = np.ndarray((n_key, len(trials[p])))
        all_colors.append([colors[p]] * n_key)
        for f, file in enumerate(all_data[part].keys()):
            markers_depth, markers_vicon, vicon_to_depth = load_in_markers_ref(all_data[part][file])
            sum_minimal = (markers_depth[2, :, :] + markers_vicon[2, vicon_to_depth, :]) / 2
            dif_minimal = markers_depth - markers_vicon[:, vicon_to_depth, :]
            nan_idx = np.argwhere(np.isnan(sum_minimal))
            sum_minimal = np.delete(sum_minimal, nan_idx, axis=1)
            dif_minimal = np.delete(dif_minimal, nan_idx, axis=2)
            means[:, f] = np.mean(sum_minimal, axis=1) * 1000
            diffs[:, f] = np.mean(dif_minimal, axis=0).mean(axis=1) * 1000
        means_file[n_key * p: n_key * (p + 1)] = np.mean(means, axis=1)
        diffs_file[n_key * p: n_key * (p + 1)] = np.mean(diffs, axis=1)

    compute_blandt_altman(means_file, diffs_file, units="mm", title="Bland-Altman Plot for markers positions", color=all_colors, show=True, x_axis="Mean on z axis (mm)")

