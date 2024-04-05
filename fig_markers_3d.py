from utils import *
import matplotlib.pyplot as plt


def plot_cycle(data_dic, participant, trial):
    if data_dic[participant] == {}:
        print(f"Participant {participant} has no data")
        return
    fig = plt.figure("Markers 3D")
    ax = fig.add_subplot(111, projection='3d')
    vicon_markers = np.mean(data_dic[participant][f"result_biomech_{trial}_processed_3_crops_wt_filter.bio"]["vicon"]["cycles"]["markers"] * 1000, axis=0)
    depth_markers = np.mean(data_dic[participant][f"result_biomech_{trial}_processed_3_crops_wt_filter.bio"]["depth"]["cycles"]["markers"] * 1000, axis=0)
    vicon_to_depth = data_dic[participant][f"result_biomech_{trial}_processed_3_crops_wt_filter.bio"]["vicon"]["vicon_to_depth"]
    ax.set_box_aspect([1, 1, 1])
    for i in range(0, len(vicon_markers)):
        for j in range(0, len(vicon_to_depth)):
            ax.scatter(vicon_markers[0, vicon_to_depth[j], :],
                    vicon_markers[1, vicon_to_depth[j], :],
                    vicon_markers[2, vicon_to_depth[j], :], c='r')
            ax.scatter(depth_markers[0, j, :], depth_markers[1, j, :], depth_markers[2, j, :], c='b')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.legend(["Minimal vicon-based", "RGBD-based"])
    plt.show()



if __name__ == '__main__':
    participants = ["P9"]
    trials = [["gear_10"]]
    all_data, _ = load_results(participants,
                            "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            trials)
    plot_cycle(all_data, "P9", trials[0][0])
