import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d
from biosiglive.processing.data_processing import OfflineProcessing, RealTimeProcessing
from biosiglive import MskFunctions, InverseKinematicsMethods, load
from utils import load_all_data
from rgbd_mocap.tracking.kalman import Kalman


def _convert_cluster_to_anato(new_cluster, data):
    anato_pos = new_cluster.process(marker_cluster_positions=data, cluster_marker_names=["M1", "M2", "M3"],
                                    save_file=False)
    anato_pos_ordered = np.zeros_like(anato_pos)
    anato_pos_ordered[:, 0, :] = anato_pos[:, 0, :]
    anato_pos_ordered[:, 1, :] = anato_pos[:, 2, :]
    anato_pos_ordered[:, 2, :] = anato_pos[:, 1, :]
    return anato_pos

def _interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(3):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int



if __name__ == '__main__':
    participants = ["P9"]
    trials = [["gear_10"]]
    # all_data, trials = load_all_data(participants,
    #                 "/mnt/shared/Projet_hand_bike_markerless/process_data",
    #                                  trials
    #                                 )
    import json, os
    from scapula_cluster.from_cluster_to_anato import ScapulaCluster
    measurements_dir_path = "data_collection_mesurement"
    calibration_matrix_dir = "../scapula_cluster/calibration_matrix"
    measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_P9.json"))
    measurements = measurement_data[f"with_depth"]["measure"]
    calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"][
        "calibration_matrix_name"]
    new_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                                 measurements[4], measurements[5], calibration_matrix)


    n_window = 7
    for participant in participants:
        for trial in trials:
            # markers = all_data[participant][trial]["markers_depth_interpolated"]
            data = load(f"D:\Documents\marker_pos_multi_proc_3_crops_normal_alone_pp.bio")
            markers = data["markers_in_meters"]
            marker_to_process = np.zeros((3, markers.shape[1], markers.shape[2]))
            # markers_depth_filtered = np.zeros((3, markers.shape[1], markers.shape[2]))
            # for i in range(3):
            #     markers_depth_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(markers[i, :, :],
            #                                                                                 4, 120, 4)
            # markers = markers_depth_filtered

            # marker_to_process_ma = np.zeros((3, markers.shape[1], markers.shape[2]))
            # marker_to_process_ma_rt = np.zeros((3, markers.shape[1], markers.shape[2]))
            marker_to_process_kalman = np.zeros((3, markers.shape[1], markers.shape[2]))

            markers_process = [RealTimeProcessing(60, n_window), RealTimeProcessing(60, n_window), RealTimeProcessing(60, n_window)]
            #
            # for i in range(markers.shape[2]):
            #     for j in range(3):
            #         marker_to_process_ma_rt[j, :, i] = markers_process[j].process_emg(
            #                         markers[j, :, i:i+1],
            #                         moving_average=True,
            #                         band_pass_filter=False,
            #                         centering=False,
            #                         absolute_value=False,
            #                         moving_average_window=n_window,
            #             # window_weights=[1,1,1,1,1,1, 1,5,5,5,5,5,5,5]
            #                     )[:, -1]
            # markers = marker_to_process_ma_rt[:, :, 14:]
            #
            # msk_func = MskFunctions("/mnt/shared/Projet_hand_bike_markerless/process_data/P9/models/gear_20_processed_3_model_scaled_depth.bioMod",
            #                         markers.shape[2], 120)
            # q, _, _ = msk_func.compute_inverse_kinematics(markers[:, :-3, :],
            #                                               InverseKinematicsMethods.BiorbdLeastSquare,
            #                                               kalman_freq=120
            #                                               )
            # msk_func = MskFunctions("/mnt/shared/Projet_hand_bike_markerless/process_data/P9/models/gear_20_processed_3_model_scaled_depth.bioMod",
            #                         markers.shape[2], 120)
            # initial_guess = [q[:, 0], np.zeros_like(q)[:, 0], np.zeros_like(q)[:, 0]]
            # q, q_dot, qddot = msk_func.compute_inverse_kinematics(markers[:, :-3, :],
            #                                                       InverseKinematicsMethods.BiorbdKalman,
            #                                                       kalman_freq=120,
            #                                                       initial_guess=initial_guess)
            #
            # # qdot_df = np.diff(q, axis=1)
            # # qddot_df = np.diff(qdot_df, axis=1)
            # q_dot_man = np.zeros((q_dot.shape[0], q_dot.shape[1]))
            # q_ddot_man = np.zeros((qddot.shape[0], qddot.shape[1]))
            # for i in range(1, q_dot.shape[1] - 1):
            #     q_dot_man[:, i] = (q[:, i+1] - q[:, i-1]) / (2/120)
            # for i in range(1, qddot.shape[1]):
            #     q_ddot_man[:, i] = (q_dot[:, i] - q_dot[:, i-1])/(1/120)
            #
            # plt.figure("q")
            # for i in range(q.shape[0]):
            #     plt.subplot(4, q.shape[0]//4 + 1, i+1)
            #     plt.plot(q[i, :] * 180 / np.pi, label=f"q{i}")
            #
            # plt.figure("q_dot")
            # for i in range(q_dot.shape[0]):
            #     plt.subplot(4, q_dot.shape[0]//4 + 1, i+1)
            #     plt.plot(q_dot[i, :] * 180 / np.pi, "r")
            #     plt.plot(q_dot_man[i, :]* 180 / np.pi,"b")
            # plt.figure("q_ddot")
            # for i in range(qddot.shape[0]):
            #     plt.subplot(4, qddot.shape[0]//4 + 1, i+1)
            #     plt.plot(qddot[i, 1:]* 180 / np.pi, "r")
            #     plt.plot(q_ddot_man[i, :]* 180 / np.pi,"b")
            # plt.show()

            # interp_markers = _interpolate_data(markers, int(markers.shape[2] * 2))

            marker_to_process_ma = np.zeros((3, markers.shape[1], markers.shape[2]))
            markers_cluster_rt = np.zeros((3, markers.shape[1] + 3, markers.shape[2]))
            marker_to_process_ma_rt = np.zeros((3, markers.shape[1], markers.shape[2]))
            all_kalman = []
            t_kalman = []
            t_rt_process = []
            t_markers = []
            markers_cluster = np.zeros((3, markers.shape[1] + 3, markers.shape[2]))

            anato_from_cluster = _convert_cluster_to_anato(new_cluster, markers[:, -3:, :] * 1000)
            anato_tmp = anato_from_cluster.copy()
            anato_from_cluster[:, 0, :] = anato_tmp[:, 0, :]
            anato_from_cluster[:, 1, :] = anato_tmp[:, 2, :]
            anato_from_cluster[:, 2, :] = anato_tmp[:, 1, :]

            first_idx = 3
            markers_cluster[:, :-3, :] = markers.copy()
            markers_cluster[:, -3:, :] = anato_from_cluster[:3, :, :] * 0.001

            for k in range(markers.shape[1]):
                all_kalman.append(Kalman(markers[:, k, 0], n_measures=3, n_states=6))
            for i in range(markers.shape[2]):

                # if i < markers.shape[2] - 1:
                for k in range(markers.shape[1]):
                    if i < markers.shape[2] - 1:
                        all_kalman[k].correct(markers[:, k, i])
                        marker_to_process_kalman[:, k, i + 1] = all_kalman[k].predict()

                    elif i == 0:
                        marker_to_process_kalman[:, k, i] = markers[:, k, i]
                        all_kalman[k].init_kalman(markers[i])
                        marker_to_process_kalman[:, k, i+1] = all_kalman[k].last_predicted_pos
                        all_kalman[k].correct(markers[:, k, i])
                t_kalman.append(i+1)
                anato_from_cluster = _convert_cluster_to_anato(new_cluster, marker_to_process_kalman[:, -3:, i:i+1] * 1000)
                anato_tmp = anato_from_cluster.copy()
                anato_from_cluster[:, 0, :] = anato_tmp[:, 0, :]
                anato_from_cluster[:, 1, :] = anato_tmp[:, 2, :]
                anato_from_cluster[:, 2, :] = anato_tmp[:, 1, :]

                first_idx = 3
                markers_cluster_rt[:, :-3, :] = marker_to_process_kalman.copy()
                markers_cluster_rt[:, -3:, i] = anato_from_cluster[:3, :, 0] * 0.001
                for j in range(3):
                    marker_to_process_ma_rt[j, :, i] = markers_process[j].process_emg(
                                    markers[j, :, i:i+1],
                                    moving_average=True,
                                    band_pass_filter=False,
                                    centering=False,
                                    absolute_value=False,
                                    moving_average_window=n_window,
                        # window_weights=[1,1,1,1,1,1, 1,5,5,5,5,5,5,5]
                                )[:, -1]
                t_rt_process.append(i)
                t_markers.append(i)
                # if i < n_window:
                #     marker_to_process[:, :, i] = 0
                # else:
                #     # for j in range(3):
                #     #     marker_to_process[j, :, i] = OfflineProcessing().butter_lowpass_filter(
                #     #         markers[j, :, i-n_window:i+1], 4, 120)[:, -1]
                #     for j in range(3):
                #         marker_to_process_ma[j, :, i] = np.mean(markers[j, :, i-n_window:i+1], axis=1)
            # markers_cluster_rt = markers_cluster_rt[:, :-3, :]
            for i in range(marker_to_process.shape[1]):
                plt.subplot(4, markers_cluster_rt.shape[1]//4 + 1, i+1)
                for j in range(3):
                    # plt.plot(marker_to_process[j, i, :], "r", alpha=0.5)
                    plt.plot(t_markers, markers[j, i, :], "g", alpha=0.5)
                    # plt.plot(markers_depth_filtered[j, i, :], "b", alpha=0.5)
                    # plt.plot(marker_to_process_ma[j, i, :], "y", alpha=0.5)
                    plt.plot(t_rt_process, marker_to_process_ma_rt[j, i, :], "c", alpha=0.5)
                    plt.plot(t_kalman[:], marker_to_process_kalman[j, i, :], "r", alpha=0.5)
            for i in range(markers_cluster_rt.shape[1]):
                plt.subplot(4, markers_cluster_rt.shape[1]//4 + 1, i+1)
                for j in range(3):
                    plt.plot(t_rt_process, markers_cluster_rt[j, i, :], "b", alpha=0.5)
                    plt.plot(t_rt_process, markers_cluster[j, i, :], "y", alpha=0.5)

            plt.show()


