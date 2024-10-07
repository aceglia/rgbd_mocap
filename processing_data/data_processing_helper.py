import numpy as np
from casadi import interp1d
import pandas as pd
from scipy.interpolate import interp1d
from biosiglive import OfflineProcessing
from processing_data.scapula_cluster.from_cluster_to_anato import ScapulaCluster
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import os

def convert_string(string):
    return string.lower().replace("_", "")


def compute_blandt_altman(data1, data2, units="mm", title="Bland-Altman Plot", show=True, color=None, x_axis=None,
                          markers=None, ax=None, threeshold=np.inf, no_y_label=False):
    def _remove_nan(data1, data2):
        mean = data1
        diff = data2
        nan_index = np.argwhere(np.isnan(mean))
        mean = np.delete(mean, nan_index, axis=0)
        diff = np.delete(diff, nan_index, axis=0)
        return mean, diff
    # mean = (data1 + data2) / 2
    # diff = data1 - data2
    mean_to_plot = data1
    diff_to_plot = data2
    mean, diff = _remove_nan(data1, data2)
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
            mean_tmp = mean_to_plot[i * len(color[i]):(i + 1) * len(color[i])]
            diff_tmp = diff_to_plot[i * len(color[i]):(i + 1) * len(color[i])]
            for j in range(len(mean_tmp)):
                if np.abs(diff_tmp[j]) > threeshold:
                    continue
                ax.scatter(mean_tmp[j], diff_tmp[j], c=color[i][j], s=100, alpha=0.6, marker=markers)

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
    # ax.set_ylabel(f'Difference ({units} )', fontsize=font)
    if x_axis is not None:
        ax.set_xlabel(x_axis, fontsize=font)
    else:
        ax.set_xlabel(f'Mean ({units})', fontsize=font)
    ax.tick_params(axis='y', labelsize=font)
    ax.tick_params(axis='x', labelsize=font)
    if not no_y_label:
        ax.set_ylabel(f'Difference ({units})', fontsize=font)
    else:
        ax.set_ylabel("", fontsize=font)
    # ax.xticks(fontsize=font)
    # ax.yticks(fontsize=font)
    # Get axis limits
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    # Set y-axis limits
    # max_y = max(abs(bottom), abs(top))
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
    x_range = [left - domain * 0.025, left + domain * 0.025]
    ax.plot(x_range, [ci_upper_loa[1]] * 2, c='grey', ls='--', alpha=0.5)
    ax.plot(x_range, [ci_upper_loa[0]] * 2, c='grey', ls='--', alpha=0.5)
    ax.plot(x_range, [ci_bias[1]] * 2, c='grey', ls='--', alpha=0.5)
    ax.plot(x_range, [ci_bias[0]] * 2, c='grey', ls='--', alpha=0.5)
    ax.plot(x_range, [ci_lower_loa[1]] * 2, c='grey', ls='--', alpha=0.5)
    ax.plot(x_range, [ci_lower_loa[0]] * 2, c='grey', ls='--', alpha=0.5)

    if show:
        plt.show()

    return bias, lower_loa, upper_loa

def process_cycles(all_results, peaks, n_peaks=None, interpolation_size=120, key_to_get_size="q"):
    for key in all_results.keys():
        if key == "file":
            continue
        data_size = all_results[key][key_to_get_size].shape[1]
        dic_tmp = {}
        for key2 in all_results[key].keys():
            if key2 == "cycle" or key2 == "rt_matrix":
                continue
            array_tmp = None
            if not isinstance(all_results[key][key2], np.ndarray):
                dic_tmp[key2] = []
                continue
            if n_peaks and n_peaks > len(peaks) - 1:
                raise ValueError("n_peaks should be less than the number of peaks")
            for k in range(len(peaks) - 1):
                if peaks[k + 1] > data_size:
                    break
                interp_function = _interpolate_2d_data if len(all_results[key][key2].shape) == 2 else interpolate_data
                if array_tmp is None:
                    array_tmp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], interpolation_size)
                    array_tmp = array_tmp[None, ...]
                else:
                    data_interp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], interpolation_size)
                    array_tmp = np.concatenate((array_tmp, data_interp[None, ...]), axis=0)
            dic_tmp[key2] = array_tmp
        all_results[key]["cycles"] = dic_tmp
    return all_results


def compute_error_mark(ref_mark, mark):
    err_markers = np.zeros((ref_mark.shape[1], 1))
    for i in range(ref_mark.shape[1]):
        nan_index = np.argwhere(np.isnan(ref_mark[:, i, :]))
        new_markers_depth_tmp = np.delete(mark[:, i, :], nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(ref_mark[:, i, :], nan_index, axis=1)
        nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
        new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
        err_markers[i, 0] = np.median(np.sqrt(
            np.mean(((new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000) ** 2), axis=0)))
    return list(err_markers[:, 0])


def refine_synchro(marker_full, marker_to_refine, plot_fig=True, nb_frame=200):
    error_list = []
    for i in range(nb_frame):
        marker_to_refine_tmp = marker_to_refine[:, :, :-i] if i != 0 else marker_to_refine
        marker_to_refine_tmp = interpolate_data(marker_to_refine_tmp, marker_full.shape[2])
        error_markers = compute_error_mark(
            marker_full[:, ...], marker_to_refine_tmp[:, ...])
        error_tmp = np.abs(np.mean(error_markers))
        error_list.append(error_tmp)
    idx = error_list.index(min(error_list))
    marker_to_refine_tmp = marker_to_refine[:, :, :-idx] if idx != 0 else marker_to_refine[:, :, :]
    marker_to_refine_tmp = interpolate_data(marker_to_refine_tmp, marker_full.shape[2])
    if plot_fig:
        import matplotlib.pyplot as plt
        plt.figure("refine synchro")
        for i in range(marker_to_refine_tmp.shape[1]):
            plt.subplot(4, 4, i + 1)
            for j in range(0, 3):
                plt.plot(marker_to_refine_tmp[j, i, :], "b")
                plt.plot(marker_full[j, i, :], 'r')
    print("idx to refine synchro : ", idx, "error", min(error_list))
    return marker_to_refine_tmp, idx


def _fill_with_nan(markers, idx):
    size = idx[-1] - idx[0]
    if len(markers.shape) == 2:
        new_markers_depth = np.zeros((markers.shape[0], size))
        count = 0
        for i in range(size):
            if i + idx[0] in idx:
                new_markers_depth[:, i] = markers[:, count]
                count += 1
            else:
                new_markers_depth[:, i] = np.nan
        return new_markers_depth
    elif len(markers.shape) == 3:
        new_markers_depth = np.zeros((3, markers.shape[1], size))
        count = 0
        for i in range(size):
            if i + idx[0] in idx:
                new_markers_depth[:, :, i] = markers[:, :, count]
                count += 1
            else:
                new_markers_depth[:, :, i] = np.nan
        return new_markers_depth


def interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(3):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def get_vicon_to_depth_idx(names_depth=None, names_vicon=None):
    vicon_markers_names = [convert_string(name) for name in names_vicon]
    depth_markers_names = [convert_string(name) for name in names_depth]
    vicon_to_depth_idx = []
    for name in vicon_markers_names:
        if name in depth_markers_names:
            vicon_to_depth_idx.append(vicon_markers_names.index(name))
    return vicon_to_depth_idx


def _interpolate_2d_data(data, shape):
    x = np.linspace(0, 100, data.shape[1])
    f_mark = interp1d(x, data)
    x_new = np.linspace(0, 100, shape)
    new_data = f_mark(x_new)
    return new_data


def fill_and_interpolate(data, shape, idx=None, names=None, fill=True):
    data_nan = _fill_with_nan(data, idx) if fill else data
    if len(data_nan.shape) == 1:
        data_nan = data_nan.reshape((1, data_nan.shape[0]))
    names = [f"n_{i}" for i in range(data_nan.shape[-2])] if not names else names
    if len(data_nan.shape) == 2:
        data_df = pd.DataFrame(data_nan, names)
        data_filled_extr = data_df.interpolate(method='linear', axis=1)
        data_int = _interpolate_2d_data(data_filled_extr, shape)
    elif len(data_nan.shape) == 3:
        data_filled_extr = np.zeros((3, data_nan.shape[1], data_nan.shape[2]))
        for i in range(3):
            data_df = pd.DataFrame(data_nan[i, :, :], names)
            data_filled_extr[i, :, :] = data_df.interpolate(method='linear', axis=1)
        data_int = interpolate_data(data_filled_extr, shape)
    else:
        raise ValueError("Data shape not supported")
    return data_int


def reorder_markers_from_names(markers_data, ordered_markers_names, markers_names):
    idx = []
    markers_names = [convert_string(name) for name in markers_names]
    for i in range(len(ordered_markers_names)):
        if markers_names[i] == "elb":
            markers_names[i] = "elbow"
        if convert_string(ordered_markers_names[i]) in markers_names:
            idx.append(markers_names.index(convert_string(ordered_markers_names[i])))
    return markers_data[:, idx], idx


def check_frames(data_dlc, data_labeling, depth_frame_idx, dlc_frame_idx):
    data = depth_frame_idx.copy()
    ref = dlc_frame_idx.copy()
    type_1 = 0
    type = 0
    idx_init = 0
    idx = 0
    overall_init_idx = None
    overall_final_idx = None
    datalist = [data_dlc, data_labeling]
    if data[0] > ref[0]:
        idx_init = ref.index(data[0])
        ref = ref[idx_init:]
        type_1 = 0
    elif data[0] < ref[0]:
        overall_init_idx = (ref[0] - data[0]) * 2
        idx_init = data.index(ref[0])
        data = data[idx_init:]
        type_1 = 1
    datalist[type_1] = datalist[type_1][..., idx_init:]
    if data[-1] < ref[-1]:
        idx = ref.index(data[-1])
        idx += 1
        ref = ref[:idx]
        type = 0
    elif data[-1] > ref[-1]:
        overall_final_idx = (data[-1] - ref[-1]) * 2
        idx = data.index(ref[-1])
        idx += 1
        data = data[:idx]
        type = 1
    if idx != 0:
        datalist[type] = datalist[type][..., :idx]

    if ref != data:
        print("Warning, frames are not synchronized")
    return datalist[0], datalist[1], overall_init_idx, overall_final_idx, ref


def convert_cluster_to_anato(data, measurements=None, calibration_matrix=None, scapula_cluster=None):
    if scapula_cluster is None:
        if measurements is None or calibration_matrix is None:
            raise ValueError("Measurements and calibration matrix should be provided")
        scapula_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                                         measurements[4], measurements[5], calibration_matrix)

    anato_pos = scapula_cluster.process(marker_cluster_positions=data * 1000, cluster_marker_names=["M1", "M2", "M3"],
                                        save_file=False)
    return anato_pos * 0.001


def adjust_idx(data, idx_start, idx_end):
    if idx_start is None and idx_end is None:
        return data
    data_tmp = {}
    for key in data.keys():
        if "names" in key:
            data_tmp[key] = data[key]
            continue
        idx_start = 0 if idx_start is None else idx_start
        if isinstance(data[key], np.ndarray):
            idx_end = data[key].shape[-1] + 1 if idx_end is None else -idx_end
            data_tmp[key] = data[key][..., idx_start:idx_end]
        elif isinstance(data[key], list):
            data_tmp[key] = data[key][idx_start:idx_end]
        else:
            data_tmp[key] = data[key]
    return data_tmp
