import numpy as np
import matplotlib.pyplot as plt
from utils_old import load_results


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


if __name__ == "__main__":
    participants = ["P9"]  # , "P10", "P11", "P13", "P14", "P15", "P16"]
    # trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    # trials[-1] = ["gear_10"]
    all_data, trials = load_results(
        participants, "Q:/Projet_hand_bike_markerless/process_data", file_name="normal_alone", recompute_cycles=False
    )
    keys = ["markers", "q_raw"]  # , "q_dot", "q_ddot", "tau", "mus_act", "mus_force"]
    factors = [1000, 180 / np.pi, 180 / np.pi, 180 / np.pi, 1, 100, 1]
    units = ["mm", "°", "°/s", "°/s²", "N.m", "%"]
    source = ["depth", "minimal_vicon", "minimal_vicon"]

    for k, key in enumerate(keys):
        all_colors = []
        shape_idx = 1 if key == "markers" else 0
        n_key = all_data[participants[0]][list(all_data[participants[0]].keys())[0]]["depth"][key].shape[shape_idx]
        rmse = np.ndarray((2, len(participants) * n_key))
        std = np.ndarray((2, len(participants) * n_key))
        for p, part in enumerate(all_data.keys()):
            rmse_file = np.ndarray((2, n_key, len(trials[p])))
            std_file = np.ndarray((2, n_key, len(trials[p])))
            for f, file in enumerate(all_data[part].keys()):
                for j in range(2):
                    idx = j + 1 if key == "markers" else j
                    rmse_file[j, :, f] = compute_error(
                        all_data[part][file]["depth"][key] * factors[k],
                        all_data[part][file][source[idx]][key] * factors[k],
                    )
                    std_file[j, :, f] = compute_std(
                        all_data[part][file]["depth"][key] * factors[k],
                        all_data[part][file][source[idx]][key] * factors[k],
                    )
            for j in range(2):
                rmse[j, n_key * p : n_key * (p + 1)] = np.mean(rmse_file[j, :, :], axis=1)
                std[j, n_key * p : n_key * (p + 1)] = np.mean(std_file[j, :, :], axis=1)

        print(
            "RMSE for",
            key,
            ":",
            "vicon:",
            np.mean(rmse[0, :]),
            "+/-",
            np.mean(std[0, :]),
            "minimal_vicon:",
            np.mean(rmse[1, :]),
            "+/-",
            np.mean(std[1, :]),
        )
