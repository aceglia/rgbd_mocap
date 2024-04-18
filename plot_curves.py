from math import ceil
from utils import *


def plot_results(all_results, track_idx, vicon_to_depth, sources=("depth", "vicon", "minimal_vicon"),
                 stop_frame=None, cycle=False, init_subplots=None, fig_suffix="", trial_name="", count=None,
                 n_cycle=None):
    joints_names = ["Pro/retraction", "Depression/Elevation",
                    "Pro/retraction", "Lateral/medial rotation", "Tilt",
                    "Plane of elevation", "Elevation", "Axial rotation",
                    "Flexion/extension", "Pronation/supination"]

    sources = []
    results_from_sources = []
    for key in all_results.keys():
        sources.append(key)
        results_from_sources.append(all_results[key]) if not cycle else results_from_sources.append(all_results[key]["cycles"])
        print(f"mean time for source: {key} ", np.mean(all_results[key]["time"]["tot"][1:]))

    #if cycle:
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

    models = [f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_depth_seth.bioMod",
              f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_vicon_seth.bioMod",
              f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_minimal_vicon_seth.bioMod"]

    # import bioviz
    # b = bioviz.Viz(f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/gear_10_processed_3_model_scaled_vicon.bioMod")
    # b.load_movement(results_from_sources[1]["q"])
    # b.exec()

    # stop_frame = results_from_sources[0]["markers"].shape[2] if stop_frame is None else stop_frame
    color = ["b", "orange", "g"]
    line = ["-", "-", "-"]

    plt.figure("markers" + fig_suffix)
    for i in range(results_from_sources[0]["markers"]["mean"].shape[1]):
        plt.subplot(4, ceil(results_from_sources[0]["markers"]["mean"].shape[1] / 4), i+1)
        for j in range(3):
            for k in range(len(results_from_sources)):
                idx = vicon_to_depth[i] if sources[k] == "vicon" else i
                plt.plot(results_from_sources[k]["markers"]["mean"][j, idx, :], line[k], color=color[k])

        plt.legend(sources)
    font_size = 18
    factors = [180 / np.pi, 180 / np.pi, 180 / np.pi, 1]
    segments = ["Clavicle", "Clavicle",
                "Scapula", "Scapula", "Scapula",
                "Humerus", "Humerus", "Humerus",
                "Forearm", "Forearm"]

    metrics = ["Joint angle (°)", "Joint angular velocity (°/s)", "Joint angular acceleration (°/s²)", "Torque (N.m)"]
    plot_names = ["q"]#, "q_dot", "q_ddot", "tau"]
    for p, plt_name in enumerate(plot_names):
        factor = factors[p]
        if cycle:
            t = np.linspace(0, 100, results_from_sources[0][plt_name]["mean"].shape[1])
        else:
            t = np.linspace(0, results_from_sources[0][plt_name]["mean"].shape[1], results_from_sources[0][plt_name]["mean"].shape[1])
        color = ["b", "r", "g"]
        line = ["-", "-", "-"]
        fig = plt.figure(num=plt_name + fig_suffix, constrained_layout=False)
        subplots = fig.subplots(4, 3, sharex=False, sharey=False)
        count = 0
        for i in range(results_from_sources[0][plt_name]["mean"].shape[0] + 2):
            if i in [2, 11]:
                subplots.flat[i].remove()
                continue
            ax = subplots.flat[i]
            for k in range(len(results_from_sources)):
                if cycle:
                    ax.fill_between(t, (
                                results_from_sources[k][plt_name]["mean"][count, :] - results_from_sources[k][plt_name][
                                                                                          "std"][count, :]) * factor,
                                                            (results_from_sources[k][plt_name]["mean"][count, :] + results_from_sources[k][plt_name]["std"][count, :]) * factor,
                                                            color=color[k], alpha=0.3)
                ax.plot(t, results_from_sources[k][plt_name]["mean"][count, :] * factor, line[k], color=color[k], alpha=0.7)
            ax.set_title(joints_names[count], fontsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size - 2)
            if i not in [8, 9, 10]:
                ax.set_xticks([])
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Mean cycle (%)", fontsize=font_size)
                ax.tick_params(axis='x', labelsize=font_size - 2)
            if i in [0, 3, 6, 9]:
                ax.set_ylabel(segments[count] + "\n\n" + metrics[p], fontsize=font_size, rotation=90)
                ax.tick_params(axis='y', labelsize=font_size - 2)
            count += 1
        fig.legend(["RGBD-based", "redundant-Vicon-based", "minimal-Vicon-based"],
                   loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=font_size, frameon=False)
        # fig.align_ylabels(subplots)
        #fig.tight_layout()
    # subplots = None
    # ax = None
    # fig = None
    # muscle_names = [
    #     "Deltoid (anterior)",
    #     "Deltoid (medial)",
    #     "Deltoid (posterior)",
    #     "Trapezius (superior)",
    #     "Biceps brachii",
    #     "Triceps brachii",
    #     "Supraspinatus",
    #     "Infraspinatus",
    #     "Subscapularis",
    # ]
    # # muscle_model_format_names = [
    # #     "DELT1_left",
    # #     "DELT2_left",
    # #     "DELT3_left",
    # #     "TRP2_left",
    # #     "bic_l_left",
    # #     "tric_long_left",
    # #     "SUPSP_left",
    # #     "INFSP_left",
    # #     "SUBSC_left",
    # #     ]
    # muscle_model_format_names = ["DeltoideusClavicle_A",
    #              'DeltoideusScapula_M',
    #              'DeltoideusScapula_P',
    #              'TrapeziusClavicle_S',
    #              "BIC_long",
    #              "TRI_lat",
    #              "Infraspinatus_S",
    #              "Subscapularis_M",
    #              "Supraspinatus_A",
    #              # "PectoralisMajor",
    #              # "LatissimusDorsi",
    #
    #              ]
    # import biorbd
    # model = biorbd.Model(models[0])
    # muscle_model_names = [model.muscleNames()[i].to_string() for i in range(model.nbMuscles())]
    # t = np.linspace(0, 100, results_from_sources[0]["mus_force"]["mean"].shape[1])
    # # plot joints
    # fig = plt.figure(num="Muscle forces" + fig_suffix, constrained_layout=False)
    # subplots = fig.subplots(3, ceil(len(muscle_names) / 3), sharex=False, sharey=False)
    # for i in range(len(muscle_names)):
    #     ax = subplots.flat[i]
    #     for k in range(len(results_from_sources)):
    #         idx = muscle_model_names.index(muscle_model_format_names[i])
    #         ax.plot(t, results_from_sources[k]["mus_force"]["mean"][idx, :], line[k], color=color[k], alpha=0.7)
    #         ax.fill_between(t, (results_from_sources[k]["mus_force"]["mean"][idx, :] - results_from_sources[k]["mus_force"]["std"][idx, :]),
    #                         (results_from_sources[k]["mus_force"]["mean"][idx, :] + results_from_sources[k]["mus_force"]["std"][idx, :]),
    #                         color=color[k], alpha=0.3)
    #     ax.set_title(muscle_names[i], fontsize=font_size)
    #     ax.tick_params(axis='y', labelsize=font_size - 2)
    #     if i not in [6, 7, 8]:
    #         ax.set_xticks([])
    #         ax.set_xticklabels([])
    #     else:
    #         ax.set_xlabel("Mean cycle (%)", fontsize=font_size)
    #         ax.tick_params(axis='x', labelsize=font_size - 2)
    #     if i in [0, 3, 6]:
    #         ax.set_ylabel("Muscle force (N)", fontsize=font_size, rotation=90)
    #         ax.tick_params(axis='y', labelsize=font_size - 2)
    # fig.legend(["RGBD-based", "redundant-Vicon-based", "minimal-Vicon-based"],
    #            loc='upper center', fontsize=font_size, frameon=False, ncol=3) #bbox_to_anchor=(0.98, 0.95)
    # fig.align_ylabels(subplots)
    # all_names = [name.to_string() for name in model.muscleNames()]

    # plot muscle activations
    # if not isinstance(results_from_sources[0]["mus_act"]["mean"], list):
    #     plt.figure("muscle torque")
    #     for i in range(results_from_sources[0]["mus_act"]["mean"].shape[0]):
    #         plt.subplot(4, ceil(results_from_sources[0]["mus_act"]["mean"].shape[0] / 4), i+1)
    #         for k in range(len(results_from_sources)):
    #             plt.plot(results_from_sources[k]["mus_act"]["mean"][i, :], line[k], color=color[k], label=all_names[i])
    #             # plt.suptitle(all_names[i], fontsize=font_size)
    #
    #         if not isinstance(results_from_sources[0]["emg_proc"]["mean"], list):
    #             if i in track_idx:
    #                 plt.plot(results_from_sources[0]["emg_proc"]["mean"][track_idx.index(i), :stop_frame])
    #         plt.legend()
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

    if not isinstance(results_from_sources[0]["res_tau"]["mean"], list):
        # plot residual tau
        fig  = plt.figure("residual tau")
        for i in range(results_from_sources[0]["res_tau"]["mean"].shape[0]):
            plt.subplot(4, ceil(results_from_sources[0]["res_tau"]["mean"].shape[0] / 4), i+1)
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["res_tau"]["mean"][i, :], line[k], color=color[k])
            fig.legend(["RGBD-based", "redundant-Vicon-based", "minimal-Vicon-based"],
                       loc='upper center', fontsize=font_size, frameon=False, ncol=3)  # bbox_to_anchor=(0.98, 0.95)


    participants = ["P11"]#, "P11", "P12", "P13", "P14", "P15", "P16"]
    trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)#, "gear_15", "gear_20"]] * len(participants)
    trials = [["gear_5"]] * len(participants)
    all_data, _ = load_results(participants,
                            # "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files/process_data",
                               "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            file_name="3_crops_seth_full", trials=trials,
                               recompute_cycles=True)
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
            plot_results(all_data[part][file],
                         all_data[part][file]["depth"]["track_idx"],
                         all_data[part][file]["depth"]["vicon_to_depth"], sources=("depth", "vicon", "minimal_vicon"),
                 stop_frame=None, cycle=False, trial_name=trials[0][f], fig_suffix="_" + str(count),
                         n_cycle=88)
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