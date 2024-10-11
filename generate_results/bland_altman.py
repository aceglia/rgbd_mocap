from biosiglive import load, save
from processing_data.file_io import get_all_file
import numpy as np
import matplotlib.pyplot as plt
import os

prefix = "/mnt/shared"


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


def _compute_part_ba_():
    pass


def remove_outliers(data, m=3, plot=False):
    final_data = []
    for i in range(data.shape[0]):
        upper_quartile = np.percentile(data[i], 75)
        lower_quartile = np.percentile(data[i], 25)
        IQR = (upper_quartile - lower_quartile) * 1.5
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        result = data[i][np.where((data[i] >= quartileSet[0]) & (data[i] <= quartileSet[1]))]
        final_data.append(result.tolist())
        # final_data.append(data[i][np.abs(data[i]) < np.abs(np.median(data[i])) + 3 * np.std(data[i], ddof=1)].tolist())
        # if plot:
        # plt.figure()
        # plt.hist(data[i].flatten(), bins=100)
        # #plt.vlines(quartileSet, 0, 100, color='r', linestyle='--')
        # plt.vlines(np.mean(data[i]) + 3 * np.std(data[i], ddof=1), 0, 100, color='r', linestyle='--')
        # plt.vlines(np.mean(data[i]) - 3 * np.std(data[i], ddof=1), 0, 100, color='r', linestyle='--')

        # plt.show()
    return final_data


def _compute_bland_altman(participants, all_data, files, key, factor, unit, source, to_compare_source):
    comparison_tab = ["RGBD vs redundant", "minimal vs redundant", "RGBD vs minimal"]
    for j in range(len(to_compare_source)):
        all_diff = []
        all_mean = []
        all_rmse = []
        all_std = []
        file_counter = 0
        for part, data in zip(participants, all_data):
            if part == "P126":
                continue
            end_frame = get_end_frame(part, files[file_counter])
            file_counter += 1
            to_compare = (
                data[to_compare_source[j]][key][..., :end_frame]
                if end_frame is not None
                else data[to_compare_source[j]][key]
            )
            ref_data = data[source[j]][key][..., :end_frame] if end_frame is not None else data[source[j]][key]
            dif_tmp = remove_outliers(ref_data - to_compare, m=6)
            all_rmse.append(np.mean([np.sqrt(np.mean(np.array(d) ** 2)) for d in dif_tmp]))
            all_std.append(np.mean([np.std(np.array(d), ddof=1) for d in dif_tmp]))
            all_diff.append(sum(dif_tmp, []))
        dif = sum(all_diff, [])
        mean = sum(all_mean, [])
        diff = np.array(dif)
        mean = np.array(mean)
        rmse = np.mean(all_rmse) * factor
        std = np.mean(all_std) * factor

        # compute Bland-Altman
        bias = np.mean(diff) * factor
        low_limit = (np.mean(diff) - 1.96 * np.std(diff)) * factor
        high_limit = (np.mean(diff) + 1.96 * np.std(diff)) * factor
        print(
            f"& {comparison_tab[j]} &"
            + f" {rmse:.2f}& {std:.2f} & {low_limit:.2f}& {high_limit:.2f}& {bias:.2f}"
            + r"\\"
        )


def _save_tmp_file(participants, all_files, name_to_save="tmp.bio"):
    dict_data = {}
    all_data_list = []
    for part, file in zip(participants, all_files):
        data = load(file)
        all_data_list.append(data)
    dict_data["data"] = all_data_list
    dict_data["participants"] = participants
    save(dict_data, name_to_save, safe=False)


def load_all_data(participants, all_files, name_to_load="tmp.bio", reload=False):
    if os.path.exists(name_to_load) and not reload:
        print(f"Loading pre-processed data on path {name_to_load}")
        dict_data = load(name_to_load)
        all_data_list = dict_data["data"]
        participants = dict_data["participants"]
        return all_data_list, participants
    else:
        _save_tmp_file(participants, all_files, name_to_save=name_to_load)
        return load_all_data(participants, all_files, name_to_load=name_to_load)


if __name__ == "__main__":
    # Load the data
    participants = [f"P{i}" for i in range(9, 17)]
    processed_data_path = prefix + "/Projet_hand_bike_markerless/process_data"
    file_name = "kalman_proc.bio"
    all_files, mapped_part = get_all_file(participants, processed_data_path, to_include=[file_name])
    all_data, participants = load_all_data(
        mapped_part, all_files, name_to_load=f"_{file_name[:-4]}_tmp.bio", reload=False
    )
    keys = ["q", "tau", "mus_force"]
    factors = [180 / np.pi, 1, 1]
    units = ["°", "N.m", "N"]
    source = ["vicon", "vicon", "minimal_vicon"]
    to_compare_source = ["depth", "minimal_vicon", "depth"]
    key_for_tab = ["Joint angles (\degree)", "Joint torques (N.m)", "Muscle force (N)"]
    print(
        r"""
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
    )

    for k, key in enumerate(keys):
        print(f"\multirow{3}*{key_for_tab[k]} ")
        _compute_bland_altman(participants, all_data, all_files, key, factors[k], units[k], source, to_compare_source)
        print(r" \hdashline")
