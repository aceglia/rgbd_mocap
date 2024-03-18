import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from utils import *
import seaborn as sns

def plot_results(all_results, track_idx, vicon_to_depth, sources=("depth", "vicon", "minimal_vicon"),
                 stop_frame=None, cycle=False, figs=None):
    joints_names = ["Pro/retraction", "Depression/Elevation",
                    "Pro/retraction", "Lateral/medial rotation", "Tilt",
                    "Plane of elevation", "Elevation", "Axial rotation",
                    "Flexion/extension", "Pronation/supination"]

    sources = []
    results_from_sources = []
    for key in all_results.keys():
        sources.append(key)
        results_from_sources.append(all_results[key]) if not cycle else results_from_sources.append(all_results[key]["cycles"])
        print(f"mean time for source: {key} ", np.mean(all_results[key]["time"]["tot"]))

    # plot markers
    # check if figure exists

    stop_frame = results_from_sources[0]["markers"].shape[2] if stop_frame is None else stop_frame
    color = ["b", "orange", "g"]
    line = ["-", "-", "-"]

    plt.figure("markers")
    for i in range(results_from_sources[0]["markers"].shape[1]):
        plt.subplot(4, ceil(results_from_sources[0]["markers"].shape[1] / 4), i+1)
        for j in range(3):
            for k in range(len(results_from_sources)):
                idx = vicon_to_depth[i] if sources[k] == "vicon" else i
                plt.plot(results_from_sources[k]["markers"][j, idx, :stop_frame], line[k], color=color[k])

        plt.legend(sources)
    font_size = 18
    factors = [180 / np.pi, 180 / np.pi, 180 / np.pi, 1]
    segments = ["Clavicle", "Clavicle",
                "Scapula", "Scapula", "Scapula",
                "Humerus", "Humerus", "Humerus",
                "Forearm", "Forearm"]
    metrics = ["Joint angle (°)", "Joint angular velocity (°/s)", "Joint angular acceleration (°/s²)", "Torque (N.m)"]
    for p, plt_name in enumerate(["q", "q_dot", "q_ddot", "tau"]):
        factor = factors[p]
        t = np.linspace(0, 100, results_from_sources[0][plt_name].shape[1])
        color = sns.color_palette("tab10", len(sources))
        color = ["b", "r", "r"]
        line = ["-", "-", "--"]
        # plot joints
        fig = plt.figure(num=plt_name, constrained_layout=False)
        # subfigs = fig.subfigures(4, 3, width_ratios=[1, 1, 1], wspace=.02)
        sublots = fig.subplots(4, 3, sharex=False, sharey=False)
        count = 0
        for i in range(results_from_sources[0][plt_name].shape[0] + 2):
            if i in [2, 11]:
                # if i == 11:
                #     ax = subfigs.flat[i].subplots(1, 1)
                #     ax.legend(["RGBD-based", "Vicon-based with redundancy", "Vicon-based without redundancy"],
                #               loc='center right', bbox_to_anchor=(1, 0.5), )
                sublots.flat[i].remove()

                continue
            # if i == 10:
            #     ax = subfigs.flat[i].subplots(1, 2)
            # else:

            ax = sublots.flat[i]
            for k in range(len(results_from_sources)):
                ax.plot(t, results_from_sources[k][plt_name][count, :] * factor, line[k], color=color[k], alpha=0.7)
            ax.set_title(joints_names[count], fontsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size - 2)
            if i not in [8, 9, 10]:
                ax.set_xticks([])
                ax.set_xticklabels([])
                # ax.set_xlabel(" ", fontsize=font_size)
            else:
                ax.set_xlabel("Mean cycle (%)", fontsize=font_size)
                ax.tick_params(axis='x', labelsize=font_size - 2)
            if i in [0, 3, 6, 9]:
                ax.set_ylabel(segments[count] + "\n\n" + metrics[p], fontsize=font_size, rotation=90)
                # sec_axis = ax.secondary_yaxis(0.5)
                # sec_axis.set_ylabel(segments[count], fontsize=font_size - 2, rotation=90)
                # sec_axis.set_yticks([])
                # sec_axis.set_yticklabels([])
                ax.tick_params(axis='y', labelsize=font_size - 2)
            # if i == 10:
            #     ax.legend(sources, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size - 4)
            # plt.grid(True)
            count += 1
        fig.legend(["RGBD-based", "Vicon-based with redundancy", "Vicon-based without redundancy"],
                   loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=font_size, frameon=False)
        fig.align_ylabels(sublots)
        #fig.tight_layout()

    # plot muscle activations
    if not isinstance(results_from_sources[0]["mus_act"], list):
        plt.figure("muscle activations")
        for i in range(results_from_sources[0]["mus_act"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["mus_act"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["mus_act"][i, :], line[k], color=color[k])
            if not isinstance(results_from_sources[0]["emg_proc"], list):
                if i in track_idx:
                    plt.plot(results_from_sources[0]["emg_proc"][track_idx.index(i), :stop_frame])
            plt.legend(sources)

    if not isinstance(results_from_sources[0]["mus_force"], list):
        plt.figure("muscle forces")
        for i in range(results_from_sources[0]["mus_force"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["mus_force"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["mus_force"][i, :], line[k], color=color[k])
            plt.legend(sources)

    if not isinstance(results_from_sources[0]["res_tau"], list):
        # plot residual tau
        plt.figure("residual tau")
        for i in range(results_from_sources[0]["res_tau"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["res_tau"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["res_tau"][i, :], line[k], color=color[k])
            plt.legend(sources)


if __name__ == '__main__':

    participants = ["P14", "P15", "P16"]
    trials = [["gear_15"]] * len(participants)
    all_data, _ = load_results(participants,
                            "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            trials)
    all_errors_minimal = []
    all_errors_vicon = []
    count = 0
    ax = None
    for part in all_data.keys():
        for file in all_data[part].keys():
            plot_results(all_data[part][file],
                         all_data[part][file]["depth"]["track_idx"],
                         all_data[part][file]["depth"]["vicon_to_depth"], sources=("depth", "vicon", "minimal_vicon"),
                 stop_frame=None, cycle=True)
    plt.show()