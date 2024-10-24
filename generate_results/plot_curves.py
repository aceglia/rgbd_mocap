from math import ceil

import numpy as np

from utils_old import *


def plot_results(
    all_results,
    track_idx=None,
    to_plot=None,
    vicon_to_depth=None,
    sources=("depth", "vicon", "minimal_vicon"),
    stop_frame=None,
    cycle=False,
    init_subplots=None,
    fig_suffix="",
    trial_name="",
    count=None,
    n_cycle=None,
):
    init_joints_names = [
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

    # sources = []
    results_from_sources = []
    for key in sources:
        # sources.append(key)
        (
            results_from_sources.append(all_results[key])
            if not cycle
            else results_from_sources.append(all_results[key]["cycles"])
        )
        # print(f"mean time for source: {key} ", np.mean(all_results[key]["time"]["tot"][1:]))

    # if cycle:
    def map_activation(emg_proc, map_idx):
        act = np.zeros((len(map_idx), int(emg_proc.shape[1])))
        for i in range(len(map_idx)):
            act[i, :] = emg_proc[map_idx[i], :]
        return act
    results_from_sources_tmp = []
    for result in results_from_sources:
        dic_tmp = {}
        for key in result.keys():
            dic_tmp[key] = {}
            if isinstance(result[key], np.ndarray):
                if cycle:
                    if n_cycle:
                        dic_tmp[key]["mean"] = np.mean(result[key][:n_cycle, ...], axis=0)
                        dic_tmp[key]["std"] = np.std(result[key][:n_cycle, ...], axis=0)
                    else:
                        dic_tmp[key]["mean"] = np.mean(result[key][:, ...], axis=0)
                        dic_tmp[key]["std"] = np.std(result[key][:, ...], axis=0)
                else:
                    dic_tmp[key]["mean"] = result[key]
                    dic_tmp[key]["std"] = result[key]
                # dic_tmp[key] = result[key][0, ...]
            else:
                dic_tmp[key] = result[key]
        results_from_sources_tmp.append(dic_tmp)
    results_from_sources = results_from_sources_tmp

    models = [
        f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_depth_new_seth_old.bioMod",
        f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_vicon_new_seth_old.bioMod",
        f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_minimal_vicon_new_seth_old.bioMod",
    ]

    # import bioviz
    # b = bioviz.Viz(f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/models/{trial_name}_model_scaled_dlc_ribs_new_seth.bioMod")
    # b.load_movement(results_from_sources[0]["q_raw"]["mean"])
    # b.exec()

    # stop_frame = results_from_sources[0]["markers"].shape[2] if stop_frame is None else stop_frame
    # color = ["b", "orange", "g"]
    # line = ["-", "-", "-"]
    # idx = [idx_tmp for idx_tmp in range(len(sources)) if "vicon" in sources[idx_tmp]][0]
    # idx_dlc = [idx_tmp for idx_tmp in range(len(sources)) if "dlc" in sources[idx_tmp]][0]
    # plt.figure("markers" + fig_suffix)
    # from utils_old import _reorder_markers_from_names
    # idx_scap_ia = all_results[sources[idx_dlc]]["marker_names"].index("SCAP_IA")
    # idx_scap_ts = all_results[sources[idx_dlc]]["marker_names"].index("SCAP_TS")
    # all_results[sources[idx_dlc]]["marker_names"][idx_scap_ia] = "SCAP_TS"
    # all_results[sources[idx_dlc]]["marker_names"][idx_scap_ts] = "SCAP_IA"
    # if sources[idx] != "vicon":
    #     reordered_dlc, _ = _reorder_markers_from_names(results_from_sources[idx_dlc]["tracked_markers"]["mean"],
    #                                                 ordered_markers_names=all_results[sources[idx]]["marker_names"],
    #                                                 markers_names=all_results[sources[idx_dlc]]["marker_names"])
    #     results_from_sources[idx_dlc]["tracked_markers"]["mean"] = reordered_dlc
    #     count = 0
    #     for i in range(results_from_sources[idx]["tracked_markers"]["mean"].shape[1]):
    #         plt.subplot(4, ceil(results_from_sources[idx]["markers"]["mean"].shape[1] / 4), i+1)
    #         for k in range(len(results_from_sources)):
    #             for j in range(3):
    #                 if k ==0:
    #                     plt.plot(results_from_sources[k]["tracked_markers"]["mean"][j, i, :] * 1000, line[k], color=color[k])
    #                 else:
    #                     plt.plot(results_from_sources[k]["tracked_markers"]["mean"][j, i, :] * 1000, line[k], color=color[k])
    #         # plt.title(all_results[sources[0]]["marker_names"][count] + all_results[sources[1]]["marker_names"][i])
    #         plt.legend(sources)
    # plt.figure("COR")
    # if cycle:
    #     t = np.linspace(0, 100, results_from_sources[0]["center_of_rot"]["mean"].shape[-1])
    #     final_idx = None
    # else:
    #     final_idx = n_cycle * 120 if n_cycle else results_from_sources[0]["center_of_rot"]["mean"].shape[-1]
    #     if final_idx > results_from_sources[0]["center_of_rot"]["mean"].shape[-1]:
    #         final_idx = results_from_sources[0]["center_of_rot"]["mean"].shape[-1]
    #     t = np.linspace(0, final_idx, final_idx)
    # for i in range(results_from_sources[idx]["center_of_rot"]["mean"].shape[1]):
    #     plt.subplot(4, ceil(results_from_sources[idx]["center_of_rot"]["mean"].shape[1] / 4), i+1)
    #     for k in range(len(results_from_sources)):
    #         for j in range(3):
    #             #if k ==0:
    #             # plt.fill_between(t, (
    #             #         results_from_sources[k]["center_of_rot" ]["mean"][j, i, :]*1000 - results_from_sources[k]["center_of_rot"][
    #             #                                                                   "std"][j, i, :]) * 1000,
    #             #                 (results_from_sources[k]["center_of_rot"]["mean"][j, i, :]*1000 +
    #             #                  results_from_sources[k]["center_of_rot"]["std"][j, i, :]) * 1000,
    #             #                 color=color[k], alpha=0.3)
    #             plt.plot(t, results_from_sources[k]["center_of_rot"]["mean"][j, i, :] * 1000, line[k], color=color[k])
    #             #else:
    #             #    plt.plot(results_from_sources[k]["center_of_rot"]["mean"][j, i, :] * 1000, line[k], color=color[k])
    #     # plt.title(all_results[sources[0]]["marker_names"][count] + all_results[sources[1]]["marker_names"][i])
    #     plt.legend(sources)
    font_size = 18
    factors = [180 / np.pi, 180 / np.pi, 1]
    init_segments = [
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

    metrics = ["Joint angle (°)", "Joint angular velocity (°/s)", "Torque (N.m)"]
    # plot_names = ["q_raw"]# , "q_dot", "q_ddot", "tau"]
    plot_names = to_plot
    for p, plt_name in enumerate(plot_names):
        factor = factors[p]
        if cycle:
            t = np.linspace(0, 100, results_from_sources[0][plt_name]["mean"].shape[1])
            final_idx = None
        else:
            final_idx = n_cycle * 120 if n_cycle else results_from_sources[0][plt_name]["mean"].shape[1]
            if final_idx > results_from_sources[0][plt_name]["mean"].shape[1]:
                final_idx = results_from_sources[0][plt_name]["mean"].shape[1]
            t = np.linspace(0, final_idx, final_idx)
        color = ["b", "r", "g"]
        line = ["-", "-", "-"]
        fig = plt.figure(num=plt_name + fig_suffix, constrained_layout=False)
        subplots = fig.subplots(
            (results_from_sources[0][plt_name]["mean"].shape[0] + 2) // 3, 3, sharex=False, sharey=False
        )
        count = 0
        for i in range(results_from_sources[0][plt_name]["mean"].shape[0] + 2):
            to_add = results_from_sources[0][plt_name]["mean"].shape[0] - 10
            if results_from_sources[0][plt_name]["mean"].shape[0] != len(init_joints_names):
                joints_names = ["TX", "Ty", "Tz", "list", "tils", "rot"] + init_joints_names
                segments = ["Thorax"] * 6 + init_segments
            else:
                joints_names = init_joints_names
                segments = init_segments
            if i in [2 + to_add, 11 + to_add]:
                subplots.flat[i].remove()
                continue
            ax = subplots.flat[i]
            for k in range(len(results_from_sources)):
                if cycle:
                    ax.fill_between(
                        t,
                        (
                            results_from_sources[k][plt_name]["mean"][count, :]
                            - results_from_sources[k][plt_name]["std"][count, :]
                        )
                        * factor,
                        (
                            results_from_sources[k][plt_name]["mean"][count, :]
                            + results_from_sources[k][plt_name]["std"][count, :]
                        )
                        * factor,
                        color=color[k],
                        alpha=0.3,
                    )
                    ax.plot(
                        t,
                        results_from_sources[k][plt_name]["mean"][count, :] * factor,
                        line[k],
                        color=color[k],
                        alpha=0.7,
                    )
                else:
                    ax.plot(
                        t,
                        results_from_sources[k][plt_name]["mean"][count, :final_idx] * factor,
                        line[k],
                        color=color[k],
                        alpha=0.7,
                    )
            ax.set_title(joints_names[count], fontsize=font_size)
            ax.tick_params(axis="y", labelsize=font_size - 2)
            if i not in [8 + to_add, 9 + to_add, 10 + to_add]:
                ax.set_xticks([])
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Mean cycle (%)", fontsize=font_size)
                ax.tick_params(axis="x", labelsize=font_size - 2)
            if i in [0 + to_add, 3 + to_add, 6 + to_add, 9 + to_add]:
                ax.set_ylabel(segments[count] + "\n\n" + metrics[p], fontsize=font_size, rotation=90)
                ax.tick_params(axis="y", labelsize=font_size - 2)
            if cycle:
                ax.set_xlim(0, 100)
            count += 1
        fig.legend(sources, loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=font_size, frameon=False)
        # fig.align_ylabels(subplots)
        # fig.tight_layout()
    subplots = None
    ax = None
    fig = None
    muscle_names = [
        "Deltoid (anterior)",
        "Deltoid (medial)",
        "Deltoid (posterior)",
        "Trapezius (superior)",
        "Biceps brachii",
        "Triceps brachii",
        "Supraspinatus",
        "Infraspinatus",
        "Subscapularis",
    ]
    # muscle_model_format_names = [
    #     "DELT1_left",
    #     "DELT2_left",
    #     "DELT3_left",
    #     "TRP2_left",
    #     "bic_l_left",
    #     "tric_long_left",
    #     "SUPSP_left",
    #     "INFSP_left",
    #     "SUBSC_left",
    #     ]
    muscle_model_format_names = ["DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                 'DeltoideusScapula_P',
                 'TrapeziusClavicle_S',
                 "BIC_long",
                 "TRI_lat",
                 "Infraspinatus_S",
                 "Subscapularis_M",
                 "Supraspinatus_A",
                 # "PectoralisMajor",
                 # "LatissimusDorsi",

                 ]
    import biorbd
    model = biorbd.Model(models[0])
    muscle_model_names = [model.muscleNames()[i].to_string() for i in range(model.nbMuscles())]
    t = np.linspace(0, 100, results_from_sources[0]["mus_force"]["mean"].shape[1])
    # plot joints
    fig = plt.figure(num="Muscle forces" + fig_suffix, constrained_layout=False)
    subplots = fig.subplots(3, ceil(len(muscle_names) / 3), sharex=False, sharey=False)
    for i in range(len(muscle_names)):
        ax = subplots.flat[i]
        for k in range(len(results_from_sources)):
            idx = muscle_model_names.index(muscle_model_format_names[i])
            ax.plot(t, results_from_sources[k]["mus_force"]["mean"][idx, :], line[k], color=color[k], alpha=0.7)
            ax.fill_between(t, (results_from_sources[k]["mus_force"]["mean"][idx, :] - results_from_sources[k]["mus_force"]["std"][idx, :]),
                            (results_from_sources[k]["mus_force"]["mean"][idx, :] + results_from_sources[k]["mus_force"]["std"][idx, :]),
                            color=color[k], alpha=0.3)
        ax.set_title(muscle_names[i], fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size - 2)
        if i not in [6, 7, 8]:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Mean cycle (%)", fontsize=font_size)
            ax.tick_params(axis='x', labelsize=font_size - 2)
        if i in [0, 3, 6]:
            ax.set_ylabel("Muscle force (N)", fontsize=font_size, rotation=90)
            ax.tick_params(axis='y', labelsize=font_size - 2)
    fig.legend(["RGBD-based", "redundant-Vicon-based", "minimal-Vicon-based"],
               loc='upper center', fontsize=font_size, frameon=False, ncol=3) #bbox_to_anchor=(0.98, 0.95)
    # fig.align_ylabels(subplots)
    all_names = [name.to_string() for name in model.muscleNames()]

    # plot muscle activations
    emg = all_results["shared"]["emg"]
    track_idx = [29, 33, 34, 31, 25, 1, 23, 13, 12]
    map_idx = [0, 1, 1, 2, 3, 4, 5, 6, 7]
    def map_activation(emg_proc, map_idx):
        act = np.zeros((len(map_idx), int(emg_proc.shape[1])))
        for i in range(len(map_idx)):
            act[i, :] = emg_proc[map_idx[i], :]
        return act
    emg = map_activation(emg, map_idx)
    if not isinstance(results_from_sources[0]["mus_act"]["mean"], list):
        plt.figure("muscle act")
        for i in range(results_from_sources[0]["mus_act"]["mean"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["mus_act"]["mean"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["mus_act"]["mean"][i, :])
            if not isinstance(emg, list):
                if i in track_idx:
                    plt.plot(emg[track_idx.index(i), :])
            plt.legend()
    #
    # if not isinstance(results_from_sources[0]["mus_force"], list):
    #     plt.figure("muscle forces")
    #     for i in range(results_from_sources[0]["mus_force"].shape[0]):
    #         plt.subplot(4, ceil(results_from_sources[0]["mus_force"].shape[0] / 4), i+1)
    #         for k in range(len(results_from_sources)):
    #             plt.plot(results_from_sources[k]["mus_force"][i, :], line[k], color=color[k])
    #         plt.legend(sources)

    # if not isinstance(results_from_sources[0]["res_tau"]["mean"], list):
    #     # plot residual tau
    #     fig  = plt.figure("residual tau")
    #     for i in range(results_from_sources[0]["res_tau"]["mean"].shape[0]):
    #         plt.subplot(4, ceil(results_from_sources[0]["res_tau"]["mean"].shape[0] / 4), i+1)
    #         for k in range(len(results_from_sources)):
    #             plt.plot(results_from_sources[k]["mus_force"][i, :], line[k], color=color[k])
    #         plt.legend(sources)

    # if not isinstance(results_from_sources[0]["res_tau"]["mean"], list):
    #     # plot residual tau
    #     fig  = plt.figure("residual tau")
    #     for i in range(results_from_sources[0]["res_tau"]["mean"].shape[0]):
    #         plt.subplot(4, ceil(results_from_sources[0]["res_tau"]["mean"].shape[0] / 4), i+1)
    #         for k in range(len(results_from_sources)):
    #             plt.plot(results_from_sources[k]["res_tau"]["mean"][i, :], line[k], color=color[k])
    #         fig.legend(["RGBD-based", "redundant-Vicon-based", "minimal-Vicon-based"],
    #                    loc='upper center', fontsize=font_size, frameon=False, ncol=3)  # bbox_to_anchor=(0.98, 0.95)


if __name__ == "__main__":
    participants = ["P14"]  # , "P11", "P12", "P13", "P14", "P15", "P16"]
    trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(
        participants
    )  # , "gear_15", "gear_20"]] * len(participants)
    trials = [["gear_10"]] * len(participants)
    all_data, _ = load_results(
        participants,
        # "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files/process_data",
        "/mnt/shared/Projet_hand_bike_markerless/process_data",
        file_name="kalman_proc_new.bio",
        trials=trials,
        # file_name="seth_new_model.bio", trials=trials,
        recompute_cycles=False,
    )
    # load_results(participants,
    #                     "/mnt/shared/Projet_hand_bike_markerless/process_data",
    #                     trials, file_name="_seth")
    # load_results(participants,
    #                     "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files/process_data",
    #                     file_name="3_crops_seth_full", trials=trials)

    count = 0
    all_errors_minimal = []
    all_errors_vicon = []
    for part in all_data.keys():
        for f, file in enumerate(all_data[part].keys()):
            print(file)
            plot_results(
                all_data[part][file],
                # all_data[part][file]["depth"]["track_idx"],
                # all_data[part][file]["depth"]["vicon_to_depth"],
                to_plot=["q", "q_dot", "tau"],
                # sources=("depth", "minimal_vicon", "vicon"),
                stop_frame=None,
                cycle=True,
                trial_name=trials[0][f],
                fig_suffix="_" + str(count),
                n_cycle=None,
            )
            count += 1
            plt.show()

    # all_data, _ = load_results(participants,
    #                         "/mnt/shared/Projet_hand_bike_markerless/process_data",
    #                         trials, "wt_kalman")
    # for part in all_data.keys():
    #     for file in all_data[part].keys():
    #         plot_results(all_data[part][file],
    #                      all_data[part][file]["depth"]["track_idx"],
    #                      all_data[part][file]["depth"]["vicon_to_depth"], sources=("depth", "vicon", "minimal_vicon"),
    #              stop_frame=None, cycle=False, fig_suffix="_wt_kalman")
    plt.show()
