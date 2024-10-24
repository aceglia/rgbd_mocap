from biosiglive import load, save
from processing_data.file_io import get_all_file
import numpy as np
import scipy.stats as st
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


def remove_outliers(data, data_ref, m=3, plot=False):
    final_data = []
    final_ref = []
    for i in range(data.shape[0]):
        # upper_quartile = np.percentile(data[i], 75)
        # lower_quartile = np.percentile(data[i], 25)
        # IQR = (upper_quartile - lower_quartile) * 1.5
        # quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        # result = data[i][np.where((data[i] >= quartileSet[0]) & (data[i] <= quartileSet[1]))]
        # ref = data_ref[i][np.where((data[i] >= quartileSet[0]) & (data[i] <= quartileSet[1]))]
        # final_data.append(result.tolist())
        # final_ref.append(ref.tolist())
        final_data.append(data[i][np.abs(data[i]) < np.abs(np.median(data[i])) + 3 * np.std(data[i], ddof=1)].tolist())
        # if plot:
        # plt.figure()
        # plt.hist(data[i].flatten(), bins=100)
        # #plt.vlines(quartileSet, 0, 100, color='r', linestyle='--')
        # plt.vlines(np.mean(data[i]) + 3 * np.std(data[i], ddof=1), 0, 100, color='r', linestyle='--')
        # plt.vlines(np.mean(data[i]) - 3 * np.std(data[i], ddof=1), 0, 100, color='r', linestyle='--')

        # plt.show()
    return final_data, final_ref


def get_confidence_interval(bias, s, n, ci_level=0.95):
    # Quantile (the cumulative probability)
    q = 1 - ((1 - ci_level) / 2)
    # Critical z-score, calculated using the percent-point function (aka the
    # quantile function) of the normal distribution
    z_star = st.norm.ppf(q)
    # print(f"95% of normally distributed data lies within {z_star}σ of the mean")
    # Limits of agreement (LOAs)
    loas = st.norm.interval(ci_level, bias, s)
    # print(np.round(loas, 2))
    # Degrees of freedom
    dof = n - 1
    # Standard error of the bias
    se_bias = s / np.sqrt(n)
    se_loas = np.sqrt(3 * s**2 / n)
    ci_bias = st.t.interval(ci_level, dof, bias, se_bias)
    ci_lower_loa = st.t.interval(ci_level, dof, loas[0], se_loas)
    ci_upper_loa = st.t.interval(ci_level, dof, loas[1], se_loas)
    return ci_bias, ci_lower_loa, ci_upper_loa, loas


def _compute_bland_altman(participants, all_data, files, key, factor, unit, source, to_compare_source, plot=False):
    comparison_tab = ["RGBD vs redundant", "minimal vs redundant", "RGBD vs minimal"]
    if key == "q":
        plot = True

    for j in range(len(to_compare_source)):
        if plot:
            plt.figure(f"{key}_{comparison_tab[j]}")
        all_diff = []
        all_mean = []
        all_rmse = []
        all_std = []
        file_counter = 0
        colors = plt.cm.get_cmap("tab20", len(participants))
        p_prev = participants[0]
        count_part = 0
        for part, data in zip(participants, all_data):
            if part == "P10":
                continue
            end_frame = get_end_frame(part, files[file_counter])
            file_counter += 1
            to_compare = (
                data[to_compare_source[j]][key][..., :end_frame]
                if end_frame is not None
                else data[to_compare_source[j]][key]
            )
            ref_data = data[source[j]][key][..., :end_frame] if end_frame is not None else data[source[j]][key]
            # to_compare, ref_data = remove_outliers(to_compare, ref_data, m=6)
            # dif_tmp = [np.mean((np.array(ref) - np.array(comp))) for ref, comp in zip(ref_data, to_compare)]
            mean_tmp = [(np.array(ref) + np.array(comp) / 2).tolist() for ref, comp in zip(ref_data, to_compare)]
            dif_tmp = [(np.array(ref) - np.array(comp)).tolist() for ref, comp in zip(ref_data, to_compare)]
            all_rmse.append(np.mean([np.sqrt(np.mean(np.array(d) ** 2)) for d in dif_tmp]))
            all_std.append(np.mean([np.std(np.array(d), ddof=1) for d in dif_tmp]))
            all_diff.append(sum(dif_tmp, []))
            all_mean.append(sum(mean_tmp, []))
            # all_diff.append(dif_tmp)
            # all_mean.append(mean_tmp)
            count_part = count_part + 1 if p_prev != part else count_part
            p_prev = part
            if plot:
                plt.scatter(sum(mean_tmp, []), sum(dif_tmp, []), color=colors(count_part), label=part)
        dif = sum(all_diff, [])
        mean = sum(all_mean, [])
        diff = np.array(dif)
        mean = np.array(mean)
        rmse = np.mean(all_rmse) * factor
        std = np.mean(all_std) * factor

        # compute Bland-Altman
        bias = np.mean(diff) * factor
        # low_limit = (np.mean(diff) - 1.96 * np.std(diff)) * factor
        # high_limit = (np.mean(diff) + 1.96 * np.std(diff)) * factor
        # compute confidence interval
        ci_bias, ci_lower_loa, ci_upper_loa, loas = get_confidence_interval(
            bias, np.std(diff * factor, ddof=1), len(dif), ci_level=0.95
        )
        low_limit = loas[0]
        high_limit = loas[1]
        # print(f"Confidence interval for the bias: {ci_bias}"
        #      f"\nConfidence interval for the lower LOA: {ci_lower_loa}"
        #      f"\nConfidence interval for the upper LOA: {ci_upper_loa}"
        #      )
        print(
            f"& {comparison_tab[j]} &"
            + f" {rmse:.2f}& {std:.2f} & {low_limit:.2f}& {high_limit:.2f}& {bias:.2f}"
            + r"\\"
        )


def _save_tmp_file(participants, all_files, name_to_save="tmp.bio"):
    dict_data = {}
    all_data_list = []
    for part, file in zip(participants, all_files):
        print(part, file)
        if part == "P15" and "gear_5" in file:
            continue
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
    file_name = "kalman_proc_new.bio"
    all_files, mapped_part = get_all_file(participants, processed_data_path, to_include=[file_name], is_dir=False)
    all_data, participants = load_all_data(
        mapped_part, all_files, name_to_load=f"_{file_name[:-4]}_tmp.bio", reload=False
    )
    keys = ["q", "tau", "mus_force"]
    factors = [180 / np.pi, 1, 1]
    units = ["°", "N.m", "N"]
    source = ["vicon", "vicon", "minimal_vicon"]
    to_compare_source = ["depth", "minimal_vicon", "depth"]
    key_for_tab = ["Joint angles (\degree)", "Joint torques (N.m)", "Muscle force (N)"]
    colors = plt.cm.get_cmap("tab20", len(participants))
    plt.figure("legend")
    plotted_parts = []
    for i, part in enumerate(participants):
        if part in plotted_parts:
            continue
        plt.scatter([], [], color=colors(i), label=part)
        plotted_parts.append(part)
    plt.legend()
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
    plt.show()
