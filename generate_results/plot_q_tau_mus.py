import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from utils_old import *
import seaborn as sns


def plot_results(
    all_results,
    track_idx,
    vicon_to_depth,
    sources=("depth", "vicon", "minimal_vicon"),
    stop_frame=None,
    cycle=False,
    init_subplots=None,
):
    joints_names = [
        "Pro/retraction",
        "Depression/Elevation",
        "Pro/retraction",
        "Lateral/medial rotation",
        "Tilt",
        "Plane of elevation",
        "Elevation",
        "Axial rotation",
        "Flexion/extension",
        "Pronation/supination",
    ]

    sources = []
    results_from_sources = []
    if init_subplots is None:
        final_subplots = []
    else:
        final_subplots = init_subplots
    for key in all_results.keys():
        sources.append(key)
        (
            results_from_sources.append(all_results[key])
            if not cycle
            else results_from_sources.append(all_results[key]["cycles"])
        )
        print(f"mean time for source: {key} ", np.mean(all_results[key]["time"]["tot"]))

    if cycle:
        results_from_sources_tmp = []
        for result in results_from_sources:
            dic_tmp = {}
            for key in result.keys():
                if isinstance(result[key], np.ndarray):
                    # dic_tmp[key] = np.median(result[key][:1, ...], axis=0)
                    dic_tmp[key] = result[key][0, ...]

                else:
                    dic_tmp[key] = result[key]
            results_from_sources_tmp.append(dic_tmp)
        results_from_sources = results_from_sources_tmp

    font_size = 18
    factors = [180 / np.pi, 180 / np.pi, 180 / np.pi, 1]
    segments = [
        "Clavicle",
        "Clavicle",
        "Scapula",
        "Scapula",
        "Scapula",
        "Humerus",
        "Humerus",
        "Humerus",
        "Forearm",
        "Forearm",
    ]

    metrics = ["Joint angle (°)", "Joint angular velocity (°/s)", "Joint angular acceleration (°/s²)", "Torque (N.m)"]
    for p, plt_name in enumerate(["q", "q_dot", "q_ddot", "tau"]):
        factor = factors[p]
        t = np.linspace(0, 100, results_from_sources[0][plt_name].shape[1])
        color = ["b", "r", "r"]
        line = ["-", "-", "--"]
        fig = plt.figure(num=plt_name, constrained_layout=False)
        if init_subplots is not None:
            subplots = init_subplots[p]
        else:
            subplots = fig.subplots(4, 3, sharex=False, sharey=False)
            final_subplots.append(subplots)
        count = 0
        for i in range(results_from_sources[0][plt_name].shape[0] + 2):
            if i in [2, 11]:
                subplots.flat[i].remove()
                continue
            ax = subplots.flat[i]
            for k in range(len(results_from_sources)):
                ax.plot(t, results_from_sources[k][plt_name][count, :] * factor, line[k], color=color[k], alpha=0.7)
            ax.set_title(joints_names[count], fontsize=font_size)
            ax.tick_params(axis="y", labelsize=font_size - 2)
            if i not in [8, 9, 10]:
                ax.set_xticks([])
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Mean cycle (%)", fontsize=font_size)
                ax.tick_params(axis="x", labelsize=font_size - 2)
            if i in [0, 3, 6, 9]:
                ax.set_ylabel(segments[count] + "\n\n" + metrics[p], fontsize=font_size, rotation=90)
                ax.tick_params(axis="y", labelsize=font_size - 2)
            count += 1
        fig.legend(
            ["RGBD-based", "redundant-Vicon-based", "minimal-Vicon-based"],
            loc="upper right",
            bbox_to_anchor=(0.98, 0.95),
            fontsize=font_size,
            frameon=False,
        )
        fig.align_ylabels(subplots)
        # fig.tight_layout()
    # subplots = None
    ax = None
    fig = None
    muscle_names = [
        "Trapezius (superior)",
        "Deltoid (anterior)",
        "Deltoid (medial)",
        "Deltoid (posterior)",
        "Biceps brachii",
        "Triceps brachii",
        "Supraspinatus",
        "Infraspinatus",
        "Subscapularis",
    ]
    muscle_model_format_names = [
        "TRP2_left",
        "DELT1_left",
        "DELT2_left",
        "DELT3_left",
        "bic_l_left",
        "tric_long_left",
        "SUPSP_left",
        "INFSP_left",
        "SUBSC_left",
    ]
    import biorbd

    model = biorbd.Model("/mnt/shared/Projet_hand_bike_markerless/RGBD/P9/model_scaled_depth.bioMod")
    muscle_model_names = [model.muscleNames()[i].to_string() for i in range(model.nbMuscles())]
    t = np.linspace(0, 100, results_from_sources[0]["mus_force"].shape[1])
    color = ["b", "r", "r"]
    line = ["-", "-", "--"]
    # plot joints
    fig = plt.figure(num="Muscle forces", constrained_layout=False)
    if init_subplots is not None:
        subplots = init_subplots[-1]
    else:
        subplots = fig.subplots(3, ceil(len(muscle_names) / 3), sharex=False, sharey=False)
        final_subplots.append(subplots)
    for i in range(len(muscle_names)):
        ax = subplots.flat[i]
        for k in range(len(results_from_sources)):
            idx = muscle_model_names.index(muscle_model_format_names[i])
            ax.plot(t, results_from_sources[k]["mus_force"][idx, :], line[k], color=color[k], alpha=0.7)
        ax.set_title(muscle_names[i], fontsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size - 2)
        if i not in [6, 7, 8]:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Mean cycle (%)", fontsize=font_size)
            ax.tick_params(axis="x", labelsize=font_size - 2)
        if i in [0, 3, 6]:
            ax.set_ylabel("Muscle force (N)", fontsize=font_size, rotation=90)
            ax.tick_params(axis="y", labelsize=font_size - 2)
    fig.legend(
        ["RGBD-based", "redundant-Vicon-based", "minimal-Vicon-based"],
        loc="upper center",
        fontsize=font_size,
        frameon=False,
        ncol=3,
    )
    fig.align_ylabels(subplots)
    return final_subplots


if __name__ == "__main__":
    participants = ["P13", "P14"]
    trials = [["gear_20"]] * len(participants)
    all_data, _ = load_results(participants, "/mnt/shared/Projet_hand_bike_markerless/process_data", trials)
    all_errors_minimal = []
    all_errors_vicon = []
    count = 0
    ax = None
    subplots = None
    for part in all_data.keys():
        for file in all_data[part].keys():
            subplots = plot_results(
                all_data[part][file],
                all_data[part][file]["depth"]["track_idx"],
                all_data[part][file]["depth"]["vicon_to_depth"],
                sources=("depth", "vicon", "minimal_vicon"),
                stop_frame=None,
                cycle=True,
                init_subplots=subplots,
            )
    plt.show()
