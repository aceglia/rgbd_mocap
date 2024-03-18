from pathlib import Path
import os
from biosiglive import load
from utils import load_data, _get_vicon_to_depth_idx, _convert_string
from utils import *


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P13", "P14", "P15", "P16"]
    #trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    #trials[-1] = ["gear_10"]

    all_data, trials = load_results(participants,
                            "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            )

    keys = ["markers", "q", "q_dot", "q_ddot", "tau", "mus_act", "mus_force"]
    factors = [1000, 180 / np.pi, 180 / np.pi, 180 / np.pi, 1, 100, 1]
    units = ["mm", "°", "°/s", "°/s²", "N.m", "%", "N"]
    source = ["vicon", "minimal_vicon", "minimal_vicon"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
    #colors = ["b", "orange", "g"]
    # keys = ["q"]

    for k, key in enumerate(keys):
        all_colors = []
        shape_idx = 1 if key == "markers" else 0
        n_key = all_data[participants[0]][list(all_data[participants[0]].keys())[0]]["depth"][key].shape[shape_idx]
        means_file = np.ndarray((2, len(participants) * n_key))
        diffs_file = np.ndarray((2, len(participants) * n_key))
        for p, part in enumerate(all_data.keys()):
            means = np.ndarray((2, n_key, len(trials[p])))
            diffs = np.ndarray((2, n_key, len(trials[p])))
            all_colors.append([colors[p]] * n_key)

            for f, file in enumerate(all_data[part].keys()):
                for j in range(2):
                    idx = j + 1 if key == "markers" else j
                    sum_minimal = (all_data[part][file]["depth"][key] + all_data[part][file][source[idx]][
                        key]) / 2
                    dif_minimal = all_data[part][file]["depth"][key] - all_data[part][file][source[idx]][
                        key]
                    if key == "markers":
                        nan_idx = np.argwhere(np.isnan(sum_minimal))
                        sum_minimal = np.delete(sum_minimal, nan_idx, axis=2)
                        dif_minimal = np.delete(dif_minimal, nan_idx, axis=2)
                        means[j, :, f] = np.mean(sum_minimal, axis=0).mean(axis=1) * 1000
                        diffs[j, :, f] = np.mean(dif_minimal, axis=0).mean(axis=1) * 1000
                    else:
                        if key == "q":
                            outliers = np.argwhere(np.abs(dif_minimal) * factors[k] > 30)
                            dif_minimal = np.delete(dif_minimal, outliers, axis=1)
                            sum_minimal = np.delete(sum_minimal, outliers, axis=1)
                        means[j, :, f] = np.mean(sum_minimal, axis=1) * factors[k]
                        diffs[j, :, f] = np.mean(dif_minimal, axis=1) * factors[k]
                        if key == "q":
                            if np.max(diffs[j, :, f]) > 15:
                                print("diffs for participant", part, " for trial ", file, " : ", diffs[j, :, f])
            for j in range(2):
                means_file[j, n_key * p: n_key * (p+1)] = np.mean(means[j, :, :], axis=1)
                diffs_file[j, n_key * p: n_key * (p+1)] = np.mean(diffs[j, :, :], axis=1)

        compute_blandt_altman(means_file[0, :], diffs_file[0, :],
                              units=units[k], title="Bland-Altman Plot for " + key + " with redundancy",
                              show=False, color=all_colors)
        compute_blandt_altman(means_file[1, :], diffs_file[1, :], units=units[k],
                              title="Bland-Altman Plot for " + key + " without redundancy",
                              show=False, color=all_colors)
    plt.show()
