from utils_old import *
import matplotlib.pyplot as plt


def plot_cycle(data_dic, participant, trial):
    if data_dic[participant] == {}:
        print(f"Participant {participant} has no data")
        return
    fig = plt.figure("Markers 3D")
    ax = fig.add_subplot(111, projection="3d")
    # file_name = f"result_biomech_{trial}_processed_3_crops_seth_full.bio"
    file_name = f"result_biomech_{trial}_normal_alone.bio"
    dlc_markers = np.mean(data_dic[participant][file_name]["dlc"]["cycles"]["markers"] * 1000, axis=0)
    depth_markers = np.mean(data_dic[participant][file_name]["depth"]["cycles"]["markers"] * 1000, axis=0)
    vicon_markers = np.mean(data_dic[participant][file_name]["minimal_vicon"]["cycles"]["markers"] * 1000, axis=0)

    # vicon_to_depth = data_dic[participant][file_name]["vicon"]["vicon_to_depth"]
    ax.set_box_aspect([1, 1, 1])
    for i in range(0, vicon_markers.shape[1]):
        # for j in range(0, len(vicon_to_depth)):
        #     ax.scatter(vicon_markers[0, vicon_to_depth[j], :],
        #             vicon_markers[1, vicon_to_depth[j], :],
        #             vicon_markers[2, vicon_to_depth[j], :], c='r')
        ax.scatter(depth_markers[0, i, :], depth_markers[1, i, :], depth_markers[2, i, :], c="b")
        ax.scatter(vicon_markers[0, i, :], vicon_markers[1, i, :], vicon_markers[2, i, :], c="r")
        ax.scatter(dlc_markers[0, i, :], dlc_markers[1, i, :], dlc_markers[2, i, :], c="g")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    plt.legend(["Labelled markers", "Vicon markers", "DLC markers"])
    plt.show()


if __name__ == "__main__":
    participants = ["P9"]
    trials = [["gear_10"]]
    all_data, _ = load_results(
        participants,
        "Q://Projet_hand_bike_markerless/process_data",
        trials,
        file_name="_normal_alone",
        recompute_cycles=False,
    )
    plot_cycle(all_data, "P9", trials[0][0])
