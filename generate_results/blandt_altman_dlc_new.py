import scipy
import numpy as np

import matplotlib.pyplot as plt

from biosiglive import load, save
from utils_old import load_results #, compute_blandt_altman
from processing_data.data_processing_helper import compute_blandt_altman, fill_and_interpolate



def compute_error(data, ref):
    shape_idx = 1 if data.shape[0] == 3 else 0
    n_data = data.shape[shape_idx]
    err = np.zeros((n_data))
    for i in range(n_data):
        if len(data.shape) == 3:
            err[i] = np.nanmean(np.sqrt(np.nanmedian(((data[:, i, :] - ref[:, i, :]) ** 2), axis=0)))
        else:
            err[i] = np.nanmean(np.sqrt(np.nanmedian(((data[i, :] - ref[i, :]) ** 2), axis=0)))
        # remove nan values
        #if len(data.shape) == 3:
        #    #nan_index = np.argwhere(np.isnan(ref[:, i, :]))
        #    #data_tmp = np.delete(data[:, i, :], nan_index, axis=1)
        #    #ref_tmp = np.delete(ref[:, i, :], nan_index, axis=1)
        #    err[i] = np.mean(np.sqrt(np.nanmedian(((data - ref) ** 2), axis=0)))
        #else:
        #    #nan_index = np.argwhere(np.isnan(ref[i, :]))
        #    #data_tmp = np.delete(data[i, :], nan_index, axis=0)
        #    #ref_tmp = np.delete(ref[i, :], nan_index, axis=0)
        #    err[i] = np.mean(np.sqrt(np.median(((data - ref) ** 2), axis=0)))
    return err


def compute_std(data, ref):
    shape_idx = 1 if data.shape[0] == 3 else 0
    n_data = data.shape[shape_idx]
    err = np.zeros((n_data))
    for i in range(n_data):
        if len(data.shape) == 3:
            err[i] = np.nanmean(np.nanstd(data[:, i, :] - ref[:, i, :], axis=1))
        else:
            err[i] = np.nanmean(np.nanstd(data[i, :] - ref[i, :], axis=0))

    #    # remove nan values
    #    if len(data.shape) == 3:
    #        nan_index = np.argwhere(np.isnan(ref[:, i, :]))
    #        data_tmp = np.delete(data[:, i, :], nan_index, axis=1)
    #        ref_tmp = np.delete(ref[:, i, :], nan_index, axis=1)
    #        err[i] = np.mean(np.std(data_tmp - ref_tmp, axis=1))
    #    else:
    #        nan_index = np.argwhere(np.isnan(data[i, :]))
    #        data_tmp = np.delete(data[i, :], nan_index, axis=0)
    #        ref_tmp = np.delete(ref[i, :], nan_index, axis=0)
    #        err[i] = np.mean(np.std(data_tmp - ref_tmp, axis=0))
    return err


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_by_cycles(data, cycles_idx):
    subarrays = np.split(data, cycles_idx, axis=-1)[1:-1]
    subarrays = np.array([fill_and_interpolate(subarray, 100, fill=False) for subarray in subarrays])
    subarrays = np.mean(subarrays, axis=0)
    return subarrays


def get_end_frame(part, file):
    end_frame = None
    if part == "P12" and "gear_10" in file:
        end_frame = 11870
    elif part == "P12" and "gear_15" in file:
        end_frame = 10220
    elif part == "P12" and "gear_20" in file:
        end_frame = 10130
    elif part == "P11" and "gear_20" in file:
        end_frame = 10200  # 8970
    return end_frame


if __name__ == "__main__":
    participants = [f"P{i}" for i in range(9, 17)]
    reload_data = False
    if reload_data:
        all_data, trials = load_results(
            participants,
            "/mnt/shared/Projet_hand_bike_markerless/process_data",
            file_name="normal_500_down_b1_no_root.bio",
        )
        save(all_data, "_all_data_tmp.bio", safe=False)
    else:
        all_data = load("_all_data_tmp.bio")

    all_data_tmp = all_data.copy()
    participants = [f"P{i}" for i in range(9, 17)]
    keys = ["markers", "q", "q_dot"]  # "q_ddot", "tau", "mus_act", "mus_force"]
    factors = [1000, 180 / np.pi, 180 / np.pi]  # , 180 / np.pi, 1, 100, 1]
    units = ["mm", "°", "°/s"]
    source = ["vicon"]
    to_compare_source = ["dlc_1"]
    computation_method = "mean"  # "by_frame", "mean"
    all_diff = None
    all_mean = None
    n_comparison = len(to_compare_source)
    for k, key in enumerate(keys):
        all_colors = []
        shape_idx = 1 if ("markers" in key or "center" in key) else 0
        n_key = all_data[participants[0]][list(all_data[participants[0]].keys())[0]]["dlc_1"][key].shape[shape_idx]
        if key == "markers":
            n_key -= 1
        # if key == "q" or key =="q_dot":
        #     n_key -= 2
        n_frame = 100
        init_frame = 20
        for p, part in enumerate(all_data.keys()):
            for f, file in enumerate(all_data[part].keys()):
                end_frame = get_end_frame(part, file)
                source_tmp = "vicon" if "markers" in key and "vicon" in source[0] else source[0]
                end_frame = end_frame if end_frame is not None else all_data[part][file][to_compare_source[0]]["q"].shape[-1]
                if computation_method == "by_cycle":
                    find_peak = scipy.signal.find_peaks(all_data[part][file][source[0]]["q"][-2, init_frame:end_frame], height=0.01,
                                                        distance=100)
                    cycles_idx = find_peak[0]
                    find_peak = scipy.signal.find_peaks(all_data[part][file][to_compare_source[0]]["q"][-2, init_frame:end_frame], height=0.01,
                                                        distance=100)
                    cycles_idx_dlc = find_peak[0]
                if key == "markers" and "dlc" in to_compare_source[0]:
                    dlc_mark_tmp = all_data[part][file][to_compare_source[0]]["markers"][:, :, :].copy()
                    dlc_mark_tmp = np.delete(dlc_mark_tmp, all_data[part][file][to_compare_source[0]]["marker_names"].index("technical_marker"),
                                             axis=1)
                    idx = [0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16]
                    to_compare = dlc_mark_tmp[..., init_frame:end_frame]
                    ref_data = all_data[part][file][source[0]]["markers"][:, idx, :]
                    ref_data = ref_data[..., init_frame:end_frame]
                else:
                    to_compare = all_data[part][file][to_compare_source[0]][key][..., init_frame:end_frame]
                    ref_data = all_data[part][file][source_tmp][key][..., init_frame:end_frame]
                if key == "q" or key == "q_dot":
                    to_compare = to_compare[3:16, ...]
                    ref_data = ref_data[3:16, ...]
                if computation_method == "by_cycle":
                    to_compare = get_by_cycles(to_compare, cycles_idx_dlc)
                    ref_data = get_by_cycles(ref_data, cycles_idx)
                to_compare = to_compare * factors[k]
                ref_data = ref_data * factors[k]
                sum_minimal = (to_compare + ref_data) / 2
                dif_minimal = to_compare - ref_data
                if "markers" in key:
                    sum_minimal = np.nanmean(sum_minimal, axis=0)
                    dif_minimal = np.nanmean(dif_minimal, axis=0)
                if computation_method == "mean":
                    sum_minimal = np.mean(sum_minimal, axis=-1)
                    dif_minimal = np.mean(dif_minimal, axis=-1)
                if all_diff is None:
                    all_diff = dif_minimal.flatten()
                else:
                    all_diff = np.concatenate((all_diff, dif_minimal.flatten()))

                if all_mean is None:
                    all_mean = sum_minimal.flatten()
                else:
                    all_mean = np.concatenate((all_mean, sum_minimal.flatten()), axis=0)
        bias, lower_loa, upper_loa, (ci_0, ci_1) = compute_blandt_altman(
            all_mean,
            all_diff,
            units=units[k],
            title="Bland-Altman Plot for " + key + "1.0",
            show=False,
            plot=computation_method == "mean",
        )

        all_bias = np.round(bias, 2)
        all_loa = [np.round(lower_loa, 2), np.round(upper_loa, 2)]
        print(f"{key}: Bias = {all_bias}, LOA = {all_loa}", "CI", ci_1[1]-ci_1[0])
    if computation_method == "mean":
        plt.show()

#     print(
#         r"""
#     \begin{table*}[h]
#     \caption{Root Mean Square Deviation (RMSD), along with Bland-Altman limit of agreement (LOA) and Bland-Altman Bias,
#      of the biomechanical outcomes using both Vicon-based methods, with redundancy and minimal, as reference standards.}
#     \centering
#     \begin{tabular}{l|l|cc|cc|c}
#     \hline
#          & Ratio & \multicolumn{2}{c|}{RMSE/D SD} & \multicolumn{2}{c|}{LOA} & Bias \\
#          &  &  & & Low & High &  \\
#          \hline
#          """
#         "\multirow{3}*{Markers (mm)} "
#         "&1.0   &"
#         + f"  & {all_loa[0][0]: .2f}& {all_loa[0][1]: .2f}& {all_bias[0]: .2f}"
#         + r"\\"
#         + "\n"
#         + r" \hdashline"
#         + "\n"
#         "\multirow{3}*{Joint angles (\degree)} "
#         "& 1.0    &"
#         + f" {all_rmse[1][0]: .2f}& {all_std[1][0]: .2f}  & {all_loa[1][0][0]: .2f}& {all_loa[1][0][1]: .2f}& {all_bias[1][0]: .2f}"
#         + r"\\"
#         + "\n"
#         + r" \hdashline"
#         + "\n"
#         "\multirow{3}*{Joint velocity (\degree/s)} "
#         "& 1.0    &"
#         + f" {all_rmse[2][0]: .2f}& {all_std[2][0]: .2f}  & {all_loa[2][0][0]: .2f}& {all_loa[2][0][1]: .2f}& {all_bias[2][0]: .2f}"
#         + r"\\"
#         + "\n"
#         + r" \hline"
#         + "\n"
#         r"""\end{tabular}
# \label{tab:errors}
# \end{table*}
# """
#     )
#     plt.show()
