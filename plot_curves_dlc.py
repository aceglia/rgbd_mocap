from math import ceil
from utils import *


def plot_results(all_results, cycle=False, fig_suffix="", trial_name="", n_cycle=None):
    joints_names = ["Pro/retraction", "Depression/Elevation",
                    "Pro/retraction", "Lateral/medial rotation", "Tilt",
                    "Plane of elevation", "Elevation", "Axial rotation",
                    "Flexion/extension", "Pronation/supination"]

    sources = []
    results_from_sources = []
    for key in ["dlc", "depth", "vicon"]:
        sources.append(key)
        results_from_sources.append(all_results[key]) if not cycle else results_from_sources.append(all_results[key]["cycles"])
    #if cycle:
    results_from_sources_tmp = []
    for result in results_from_sources:
        dic_tmp = {}
        for key in result.keys():
            dic_tmp[key] = {}
            if isinstance(result[key], np.ndarray):
                if cycle:
                    if key == "q":
                        result[key] = result[key][:, 6:, :]
                    if n_cycle:
                        dic_tmp[key]["mean"] = np.median(result[key][:n_cycle, ...], axis=0)
                        dic_tmp[key]["std"] = np.std(result[key][:n_cycle, ...], axis=0)
                    else:
                        dic_tmp[key]["mean"] = np.median(result[key][:, ...], axis=0)
                        dic_tmp[key]["std"] = np.std(result[key][:, ...], axis=0)
                else:

                    if key == "q":
                        result[key] = result[key][6:, :]
                    dic_tmp[key]["mean"] = result[key]
                    dic_tmp[key]["std"] = result[key]
                # dic_tmp[key] = result[key][0, ...]
            else:
                dic_tmp[key] = result[key]
        results_from_sources_tmp.append(dic_tmp)
    results_from_sources = results_from_sources_tmp

    color = ["b", "orange", "g"]
    line = ["-", "-", "-"]

    plt.figure("markers" + fig_suffix)
    for i in range(results_from_sources[0]["markers"]["mean"].shape[1]):
        plt.subplot(4, ceil(results_from_sources[0]["markers"]["mean"].shape[1] / 4), i+1)
        for j in range(3):
            for k in range(len(results_from_sources)):
                plt.plot(results_from_sources[k]["markers"]["mean"][j, i, :], line[k], color=color[k])

        plt.legend(sources)
    font_size = 18
    factors = [1, 180 / np.pi, 180 / np.pi, 180 / np.pi, 1]
    segments = ["Clavicle", "Clavicle",
                "Scapula", "Scapula", "Scapula",
                "Humerus", "Humerus", "Humerus",
                "Forearm", "Forearm"]

    metrics = ["Joint angle (°)"]
    plot_names = ["q"]
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
                # if k == 1:
                #     continue
                if cycle:
                    ax.fill_between(t, (
                                results_from_sources[k][plt_name]["mean"][count, :] - results_from_sources[k][plt_name][
                                                                                          "std"][count, :]) * factor,
                                                            (results_from_sources[k][plt_name]["mean"][count, :] + results_from_sources[k][plt_name]["std"][count, :]) * factor,
                                                            color=color[k], alpha=0.3)
                ax.plot(t, results_from_sources[k][plt_name]["mean"][count, :] * factor, line[k], color=color[k], alpha=0.7)
            ax.set_title(joints_names[count], fontsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size - 2)
            ax.set_xlim(0, 100)
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
        fig.legend(["DLC", "Labbelled", "Vicon-based"],
                   loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=font_size, frameon=False)
        plt.xlim(0, 100)

        # fig.align_ylabels(subplots)
        #fig.tight_layout()
    subplots = None
    ax = None
    fig = None


if __name__ == '__main__':
    participants = ["P15"]#, "P11", "P12", "P13", "P14", "P15", "P16"]
    # trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)#, "gear_15", "gear_20"]] * len(participants)
    trials = [["gear_10"]] * len(participants)
    all_data, _ = load_results_offline(participants,
                                    "Q://Projet_hand_bike_markerless/RGBD",
                                    file_name="normal_alone", recompute_cycles=True, trials=trials)
    count = 0
    all_errors_minimal = []
    all_errors_vicon = []
    for part in all_data.keys():
        for f, file in enumerate(all_data[part].keys()):
            print(file)
            plot_results(all_data[part][file], cycle=True, trial_name=trials[0][f], fig_suffix="_" + str(count),
                         n_cycle=80)
            count += 1
            plt.show()
