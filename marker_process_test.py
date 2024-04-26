import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d
from biosiglive.processing.data_processing import OfflineProcessing, RealTimeProcessing
from biosiglive import MskFunctions, InverseKinematicsMethods
from utils import load_all_data


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
    all_data, trials = load_all_data(participants,
                    "/mnt/shared/Projet_hand_bike_markerless/process_data",
                                     trials
                                    )
    n_window = 14
    for participant in participants:
        for trial in all_data[participant].keys():
            markers = all_data[participant][trial]["markers_depth_interpolated"]
            marker_to_process = np.zeros((3, markers.shape[1], markers.shape[2]))
            # markers_depth_filtered = np.zeros((3, markers.shape[1], markers.shape[2]))
            # for i in range(3):
            #     markers_depth_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(markers[i, :, :],
            #                                                                                 4, 120, 4)
            # markers = markers_depth_filtered

            marker_to_process_ma = np.zeros((3, markers.shape[1], markers.shape[2]))
            marker_to_process_ma_rt = np.zeros((3, markers.shape[1], markers.shape[2]))
            markers_process = [RealTimeProcessing(120, n_window), RealTimeProcessing(120, n_window), RealTimeProcessing(120, n_window)]
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
            marker_to_process_ma_rt = np.zeros((3, markers.shape[1], markers.shape[2]))

            for i in range(markers.shape[2]):
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
                if i < n_window:
                    marker_to_process[:, :, i] = 0
                else:
                    # for j in range(3):
                    #     marker_to_process[j, :, i] = OfflineProcessing().butter_lowpass_filter(
                    #         markers[j, :, i-n_window:i+1], 4, 120)[:, -1]
                    for j in range(3):
                        marker_to_process_ma[j, :, i] = np.mean(markers[j, :, i-n_window:i+1], axis=1)

            for i in range(marker_to_process.shape[1]):
                plt.subplot(4, marker_to_process.shape[1]//4 + 1, i+1)
                for j in range(3):
                    # plt.plot(marker_to_process[j, i, :], "r", alpha=0.5)
                    plt.plot(markers[j, i, :], "g", alpha=0.5)
                    # plt.plot(markers_depth_filtered[j, i, :], "b", alpha=0.5)
                    plt.plot(marker_to_process_ma[j, i, :], "y", alpha=0.5)
                    plt.plot(marker_to_process_ma_rt[j, i, :], "c", alpha=0.5)
            plt.show()


