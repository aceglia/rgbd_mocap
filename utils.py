from biosiglive import load
import os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from biosiglive.processing.msk_utils import ExternalLoads
from biosiglive import OfflineProcessing
from scipy.signal import find_peaks


def _load_data(data_path, part, file, end_idx=None):
    data = load(f"{data_path}/{part}/{file}")
    names_from_source = [data["depth_markers_names"], data["vicon_markers_names"]]
    depth_markers_names = data["depth_markers_names"]
    idx_ts = depth_markers_names.index("scapts")
    idx_ai = depth_markers_names.index("scapia")
    depth_markers_names[idx_ts] = "scapia"
    depth_markers_names[idx_ai] = "scapts"
    vicon_markers_names = data["vicon_markers_names"]
    idx_ts = vicon_markers_names.index("scapts")
    idx_ai = vicon_markers_names.index("scapia")
    vicon_markers_names[idx_ts] = "scapia"
    vicon_markers_names[idx_ai] = "scapts"
    vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)
    return names_from_source, vicon_to_depth_idx


def _convert_string(string):
    return string.lower().replace("_", "")


def _get_vicon_to_depth_idx(names_depth=None, names_vicon=None):
    vicon_markers_names = [_convert_string(name) for name in names_vicon]
    depth_markers_names = [_convert_string(name) for name in names_depth]
    vicon_to_depth_idx = []
    for name in vicon_markers_names:
        if name in depth_markers_names:
            vicon_to_depth_idx.append(vicon_markers_names.index(name))
    return vicon_to_depth_idx


def load_results(participants, processed_data_path, trials=None, file_name="", to_exclude=None):
    if trials is None:
        trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    all_data = {}
    for p, part in enumerate(participants):
        all_data[part] = {}
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file and "result_biomech" in file and file_name in file
                     and "3_crops" in file and "3_crops_3_crops" not in file]
        trials_tmp = []
        for file in all_files:
            for trial in trials[p]:
                if trial not in file:
                    continue
                print(f"Processing participant {part}, trial : {file}")
                all_data[part][file] = load(f"{processed_data_path}/{part}/{file}")
                trials_tmp.append(trial)
        trials[p] = trials_tmp
    return all_data, trials


def load_in_markers_ref(data):
    markers_depth = data["markers_depth_initial"]
    markers_vicon = data["markers_vicon_rotated"]
    depth_markers_names = data["depth_markers_names"]
    idx_ts = depth_markers_names.index("scapts")
    idx_ai = depth_markers_names.index("scapia")
    depth_markers_names[idx_ts] = "scapia"
    depth_markers_names[idx_ai] = "scapts"
    vicon_markers_names = data["vicon_markers_names"]
    idx_ts = vicon_markers_names.index("scapts")
    idx_ai = vicon_markers_names.index("scapia")
    vicon_markers_names[idx_ts] = "scapia"
    vicon_markers_names[idx_ai] = "scapts"
    vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)
    return markers_depth, markers_vicon, vicon_to_depth_idx


def load_data(data_path, part, file, filter_depth, end_idx=None, ):
    data = load(f"{data_path}/{part}/{file}")

    markers_depth = data["markers_depth_interpolated"]
    is_visible = np.repeat(data["is_visible"][np.newaxis, :, :], 3, axis=0)
    markers_vicon = data["truncated_markers_vicon"]
    names_from_source = [data["depth_markers_names"], data["vicon_markers_names"]]
    sensix_data = data["sensix_data_interpolated"]
    # put nan at idx where the marker is not visible
    depth_markers_names = data["depth_markers_names"]
    idx_ts = depth_markers_names.index("scapts")
    idx_ai = depth_markers_names.index("scapia")
    depth_markers_names[idx_ts] = "scapia"
    depth_markers_names[idx_ai] = "scapts"
    vicon_markers_names = data["vicon_markers_names"]
    idx_ts = vicon_markers_names.index("scapts")
    idx_ai = vicon_markers_names.index("scapia")
    vicon_markers_names[idx_ts] = "scapia"
    vicon_markers_names[idx_ai] = "scapts"
    vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)

    emg = data["emg_proc_interpolated"]
    if not isinstance(emg, np.ndarray):
        emg = None
    # find peak in crank angle
    peaks, _ = find_peaks(sensix_data["crank_angle"][0, :])
    peaks = [peak for peak in peaks if sensix_data["crank_angle"][0, peak] > 6]
    # plt.plot(sensix_data["crank_angle"][0, :])
    # plt.plot(peaks, sensix_data["crank_angle"][0, peaks], "x")
    # plt.show()

    markers_minimal_vicon = markers_vicon[:, vicon_to_depth_idx, :]
    names_from_source.append(np.array(vicon_markers_names)[vicon_to_depth_idx])
    markers_depth_filtered = np.zeros((3, markers_depth.shape[1], markers_depth.shape[2]))
    for i in range(3):
        markers_depth_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(markers_depth[i, :, :],
                                                                                    4, 120, 4)
    depth = markers_depth_filtered if filter_depth else markers_depth
    markers_from_source = [depth, markers_vicon, markers_minimal_vicon]
    # plt.figure("markers")
    # for i in range(markers_depth_filtered.shape[1]):
    #     plt.subplot(4, ceil(markers_depth_filtered.shape[1] / 4), i + 1)
    #     for j in range(3):
    #         plt.plot(markers_depth_filtered[j, i, :])
    #         plt.plot(markers_vicon[j, vicon_to_depth_idx[i], :])
    #         plt.plot(markers_minimal_vicon[j, i, :])
    #         plt.plot(peaks, markers_minimal_vicon[j, i, peaks], "x")
    #
    # plt.show()
    forces = ExternalLoads()
    forces.add_external_load(
        point_of_application=[0, 0, 0],
        applied_on_body="radius_left_pro_sup_left",
        express_in_coordinate="ground",
        name="hand_pedal",
        load=np.zeros((6, 1)),
    )
    f_ext = np.array([sensix_data["RMY"],
                      sensix_data["RMX"],
                      sensix_data["RMZ"],
                      sensix_data["RFY"],
                      sensix_data["RFX"],
                      sensix_data["RFZ"]])
    if part not in ["P9", "P10", "P11", "P13"]:
        f_ext = -f_ext
    f_ext = f_ext[:, 0, :]
    return markers_from_source, names_from_source, forces, f_ext, emg, vicon_to_depth_idx, peaks


def load_all_data(participants, processed_data_path, trials=None):
    all_data = {}
    if trials is None:
        trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    for p, part in enumerate(participants):
        all_data[part] = {}
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file and "result_biomech" not in file and "processed" in file
                     and "3_crops" in file]
        trials_tmp = []
        for file in all_files:
            for trial in trials[p]:
                if trial not in file:
                    continue
                print(f"Processing participant {part}, trial : {file}")
                all_data[part][file] = load(f"{processed_data_path}/{part}/{file}")
                trials_tmp.append(trial)
        trials[p] = trials_tmp
    return all_data, trials


def dispatch_bio_results(bio_results):
    dic_result = {}
    dic_result["res_tau"] = bio_results["res_tau"]
    dic_result["jrf"] = bio_results["jrf"]
    dic_result["markers"] = bio_results["markers"]
    dic_result["state"] = bio_results["state"]
    return dic_result


def compute_error(depth_dic, vicon_dic):

    n_markers_depth = depth_dic["markers"].shape[1]
    vicon_to_depth = depth_dic["vicon_to_depth"]
    err_markers = np.zeros((n_markers_depth, 1))
    if vicon_dic["markers"].shape[1] != depth_dic["markers"].shape[1]:
        vicon_dic["markers"] = vicon_dic["markers"][:, vicon_to_depth, :]
    # new_markers_depth_int = OfflineProcessing().butter_lowpass_filter(new_markers_depth_int, 6, 120, 4)
    for i in range(len(vicon_to_depth)):
        # ignore NaN values
        # if i not in [4, 5, 6]:
        nan_index = np.argwhere(np.isnan(vicon_dic["markers"][:, i, :]))
        new_markers_depth_tmp = np.delete(depth_dic["markers"][:, i, :], nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(vicon_dic["markers"][:, i, :], nan_index, axis=1)
        nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
        new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
        err_markers[i, 0] = np.median(np.sqrt(
            np.mean(((new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000) ** 2), axis=0)))

    err_q = []
    for i in range(depth_dic["q"].shape[0]):
        err_q.append(np.mean(np.sqrt(np.mean(((depth_dic["q"][i, :] - vicon_dic["q"][i, :]) ** 2), axis=0))))

    err_q_dot = []
    for i in range(depth_dic["q"].shape[0]):
        err_q_dot.append(np.mean(np.sqrt(np.mean(((depth_dic["q_dot"][i, :] - vicon_dic["q_dot"][i, :]) ** 2), axis=0))))

    err_q_ddot = []
    for i in range(depth_dic["q_ddot"].shape[0]):
        err_q_ddot.append(np.mean(np.sqrt(np.mean(((depth_dic["q_ddot"][i, :] - vicon_dic["q_ddot"][i, :]) ** 2), axis=0))))

    #normalize tau
    norm_tau = np.max(vicon_dic["tau"], axis=1)
    vicon_dic["tau"] = np.clip(vicon_dic["tau"] / norm_tau[:, None] * 100, 0, 100)
    norm_tau = np.max(depth_dic["tau"], axis=1)
    depth_dic["tau"] = np.clip(depth_dic["tau"] / norm_tau[:, None] * 100, 0, 100)
    err_tau = []
    for i in range(depth_dic["tau"].shape[0]):
        err_tau.append(np.mean(np.sqrt(np.mean(((depth_dic["tau"][i, :] - vicon_dic["tau"][i, :]) ** 2), axis=0))))

    # normalize muscle activation
    norm_mus_act = np.max(vicon_dic["mus_act"], axis=1)
    vicon_dic["mus_act"] = np.clip(vicon_dic["mus_act"] / norm_mus_act[:, None] * 100, 0, 100)
    norm_mus_act = np.max(depth_dic["mus_act"], axis=1)
    depth_dic["mus_act"] = np.clip(depth_dic["mus_act"] / norm_mus_act[:, None] * 100, 0, 100)
    err_mus_act = []
    for i in range(depth_dic["mus_act"].shape[0]):
        err_mus_act.append(np.mean(np.sqrt(np.mean(((depth_dic["mus_act"][i, :] - vicon_dic["mus_act"][i, :]) ** 2), axis=0))))
    err_q = [err_q[i] * 180/np.pi for i in range(len(err_q))]
    err_q_dot = [err_q_dot[i] * 180/np.pi for i in range(len(err_q_dot))]
    err_q_ddot = [err_q_ddot[i] * 180/np.pi for i in range(len(err_q_ddot))]
    return list(err_markers[:, 0]), err_q, err_q_dot, err_q_ddot, err_tau, err_mus_act

def remove_nan(data1, data2):
    mean = data1
    diff = data2
    nan_index = np.argwhere(np.isnan(mean))
    mean = np.delete(mean, nan_index, axis=0)
    diff = np.delete(diff, nan_index, axis=0)
    return mean, diff


def compute_blandt_altman(data1, data2, units="mm", title="Bland-Altman Plot", show=True, color=None, x_axis=None, markers=None, ax = None):
    # mean = (data1 + data2) / 2
    # diff = data1 - data2
    mean_to_plot = data1
    diff_to_plot  = data2
    mean, diff= remove_nan(data1, data2)
    # Average difference (aka the bias)
    bias = np.mean(diff)
    # Sample standard deviation
    s = np.std(diff, ddof=1)  # Use ddof=1 to get the sample standard deviation
    print(f'For the differences, μ = {bias:.4f} {units} and s = {s:.4f} {units} ')

    # Limits of agreement (LOAs)
    upper_loa = bias + 1.96 * s
    lower_loa = bias - 1.96 * s
    print(f'The limits of agreement are {upper_loa:.2f} {units} and {lower_loa:.2f} {units} ')

    # Confidence level
    C = 0.95  # 95%
    # Significance level, α
    alpha = 1 - C
    # Number of tails
    tails = 2
    # Quantile (the cumulative probability)
    q = 1 - (alpha / tails)
    # Critical z-score, calculated using the percent-point function (aka the
    # quantile function) of the normal distribution
    z_star = st.norm.ppf(q)
    print(f'95% of normally distributed data lies within {z_star}σ of the mean')
    # Limits of agreement (LOAs)
    loas = (bias - z_star * s, bias + z_star * s)

    print(f'The limits of agreement are {loas} {units} ')
    # Limits of agreement (LOAs)
    loas = st.norm.interval(C, bias, s)
    print(np.round(loas, 2))
    # Sample size
    n = data1.shape[0]
    # Degrees of freedom
    dof = n - 1
    # Standard error of the bias
    se_bias = s / np.sqrt(n)
    # Standard error of the LOAs
    se_loas = np.sqrt(3 * s ** 2 / n)

    # Confidence interval for the bias
    ci_bias = st.t.interval(C, dof, bias, se_bias)
    # Confidence interval for the lower LOA
    ci_lower_loa = st.t.interval(C, dof, loas[0], se_loas)
    # Confidence interval for the upper LOA
    ci_upper_loa = st.t.interval(C, dof, loas[1], se_loas)

    print(
        f' Lower LOA = {np.round(lower_loa, 2)}, 95% CI {np.round(ci_lower_loa, 2)}\n',
        f'Bias = {np.round(bias, 2)}, 95% CI {np.round(ci_bias, 2)}\n',
        f'Upper LOA = {np.round(upper_loa, 2)}, 95% CI {np.round(ci_upper_loa, 2)}'
    )
    if ax is None:
        plt.figure(title)
    ax = plt.axes() if ax is None else ax
    markers = markers if markers is not None else 'o'
    if color is not None:
        for i in range(len(color)):
            ax.scatter(mean_to_plot[i * len(color[i]):(i+1) * len(color[i])],
                       diff_to_plot[i * len(color[i]):(i+1) * len(color[i])], c=color[i], s=100, alpha=0.6, marker=markers)
    # ax.scatter(mean, diff, c='k', s=20, alpha=0.6, marker='o')
    # Plot the zero line
    ax.axhline(y=0, c='k', lw=0.5)
    # Plot the bias and the limits of agreement
    ax.axhline(y=loas[1], c='grey', ls='--')
    ax.axhline(y=bias, c='grey', ls='--')
    ax.axhline(y=loas[0], c='grey', ls='--')
    # Labels
    font = 18
    ax.set_title(title, fontsize=font + 2)
    if x_axis is not None:
        ax.set_xlabel(x_axis, fontsize=font)
    else:
        ax.set_xlabel(f'Mean ({units} )', fontsize=font)
    ax.set_ylabel(f'Difference ({units} )', fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    # Get axis limits
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    # Set y-axis limits
    #max_y = max(abs(bottom), abs(top))
    max_y = top
    min_y = abs(bottom)

    ax.set_ylim(-min_y, max_y)
    # Set x-axis limits
    domain = right - left
    ax.set_xlim(left, left + domain)
    # Annotations
    ax.annotate('+LOA', (right, upper_loa), (0, 7), textcoords='offset pixels', fontsize=font)
    ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -25), textcoords='offset pixels', fontsize=font)
    ax.annotate('Bias', (right, bias), (0, 7), textcoords='offset pixels', fontsize=font)
    ax.annotate(f'{bias:+4.2f}', (right, bias), (0, -25), textcoords='offset pixels', fontsize=font)
    ax.annotate('-LOA', (right, lower_loa), (0, 7), textcoords='offset pixels', fontsize=font)
    ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -25), textcoords='offset pixels', fontsize=font)

    # Confidence intervals
    ax.plot([left] * 2, list(ci_upper_loa), c='grey', ls='--', alpha=0.5)
    ax.plot([left] * 2, list(ci_bias), c='grey', ls='--', alpha=0.5)
    ax.plot([left] * 2, list(ci_lower_loa), c='grey', ls='--', alpha=0.5)
    # Confidence intervals' caps
    # x_range = [left - domain * 0.025, left + domain * 0.025]
    # ax.plot(x_range, [ci_upper_loa[1]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_upper_loa[0]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_lower_loa[1]] * 2, c='grey', ls='--', alpha=0.5)
    # ax.plot(x_range, [ci_lower_loa[0]] * 2, c='grey', ls='--', alpha=0.5)

    if show:
        plt.show()

    return bias, lower_loa, upper_loa
