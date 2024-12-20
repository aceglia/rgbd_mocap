import numpy as np
from utils_old import load_results
import matplotlib.pyplot as plt


def plot_cycle(data_dic, participant, trial):
    if data_dic[participant] == {}:
        print(f"Participant {participant} has no data")
        return
    fig = plt.figure("Markers 3D")
    ax = fig.add_subplot(111, projection="3d")
    # file_name = f"result_biomech_{trial}_processed_3_crops_seth_full.bio"
    file_name = f"result_biomech_{trial}_for_params_new_mvc.bio"
    dlc_markers = data_dic[participant][file_name]["dlc_1"]["cycles"]["markers"] * 1000
    # depth_markers = np.mean(data_dic[participant][file_name]["depth"]["cycles"]["markers"] * 1000, axis=0)
    vicon_markers = data_dic[participant][file_name]["minimal_vicon"]["cycles"]["markers"] * 1000
    dlc_markers = np.concatenate((dlc_markers[:, :, :2], dlc_markers[:, :, 3:]), axis=2)
    # vicon_to_depth = data_dic[participant][file_name]["vicon"]["vicon_to_depth"]
    n_cycles = 50
    ax.set_box_aspect([1, 1, 1])
    for k in range(0, n_cycles):
        for i in range(0, vicon_markers.shape[2]):
            # for j in range(0, len(vicon_to_depth)):
            #     ax.scatter(vicon_markers[0, vicon_to_depth[j], :],
            #             vicon_markers[1, vicon_to_depth[j], :],
            #             vicon_markers[2, vicon_to_depth[j], :], c='r')
            # ax.scatter(depth_markers[0, i, :], depth_markers[1, i, :], depth_markers[2, i, :], c="b")
            ax.scatter(vicon_markers[k, 0, i, :], vicon_markers[k, 1, i, :], vicon_markers[k, 2, i, :], c="r")
            ax.scatter(dlc_markers[k, 0, i, :], dlc_markers[k, 1, i, :], dlc_markers[k, 2, i, :], c="g")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    plt.legend(["Vicon markers", "DLC markers"])
    plt.show()


if __name__ == "__main__":
    participants = ["P10"]
    trials = [["gear_10"]]
    all_data, _ = load_results(
        participants,
        "/mnt/shared/Projet_hand_bike_markerless/process_data",
        trials,
        file_name="result_biomech_gear_10_for_params_new_mvc.bio",
        recompute_cycles=False,
    )
    plot_cycle(all_data, "P10", trials[0][0])
