from biosiglive import load
from biosiglive import OfflineProcessing, RealTimeProcessing, MskFunctions, InverseKinematicsMethods
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


data = load("ik.bio")
ik_function = MskFunctions(
    model="data_files\P4_session2\model_depth_scaled.bioMod", data_buffer_size=data["markers"].shape[2]
)
ordered_markers = data["markers"]
q, q_dot, qddot = ik_function.compute_inverse_kinematics(
    ordered_markers, InverseKinematicsMethods.BiorbdKalman, kalman_freq=100
)
q_proc = OfflineProcessing().butter_lowpass_filter(q, 6, 60, 4)
# q_dot_fd = np.gradient(q_proc, axis=1)
# qddot_fd = np.gradient(q_dot_fd, axis=1)
q_dot_df = np.zeros_like(q)
q_ddot_df = np.zeros_like(q)
q_dot_proc = OfflineProcessing().butter_lowpass_filter(q_dot, 6, 60, 4)
qddot_proc = OfflineProcessing().butter_lowpass_filter(qddot, 6, 60, 4)
tau = np.zeros_like(q)
tau_proc_b = np.zeros_like(q)
tau_proc_a = np.zeros_like(q)
tau_fd = np.zeros_like(q)
tex_q_proc_live = np.zeros_like(q)
test_q_proc_extra = np.zeros_like(q)
for i in range(q.shape[1]):
    if i > 0:
        q_dot_df[:, i] = q_proc[:, i] - q_proc[:, i - 1]
        q_ddot_df[:, i] = q_dot_df[:, i] - q_dot_df[:, i - 1]
        tau_fd[:, i] = ik_function.model.InverseDynamics(q_proc[:, i], q_dot_df[:, i], q_ddot_df[:, i]).to_array()

    tau[:, i] = ik_function.model.InverseDynamics(q_proc[:, i], q_dot_proc[:, i], qddot_proc[:, i]).to_array()
    tau_proc_b[:, i] = ik_function.model.InverseDynamics(q[:, i], q_dot[:, i], qddot[:, i]).to_array()

tau_proc_a = OfflineProcessing().butter_lowpass_filter(tau, 6, 60, 4)

plt.figure("q")
for i in range(q.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.plot(q[i, :], label="kalman")
    plt.plot(data["q"][i, :], label="live")
    plt.plot(q_proc[i, :], label="butter")
plt.legend()
plt.figure("q_dot")
for i in range(q_dot.shape[0]):
    plt.subplot(4, 4, i + 1)
    # plt.plot(q_dot[i, :])
    # plt.plot(q_dot_proc[i, :], label="butter")
    plt.plot(q_dot_df[i, :], label="fd")
    plt.plot(data["q_dot_df"][i, 10:], label="live df")
    # plt.plot(data["q_dot"][i, :], label="live")

plt.figure("qddot")
for i in range(qddot.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.plot(qddot[i, :])
    plt.plot(qddot_proc[i, :], label="butter")
    plt.plot(data["qddot"][i, :], label="live")
    plt.plot(q_ddot_df[i, :], label="fd")

plt.figure("tau")
for i in range(tau.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.plot(tau[i, :])
    plt.plot(tau_proc_b[i, :], label="before")
    plt.plot(tau_proc_a[i, :], label="after")
    plt.plot(tau_fd[i, :], label="fd")
plt.legend()
plt.show()


markers = data["markers_positions"]
markers_processed = data["markers_pos_filtered"]
proc_data = OfflineProcessing().butter_lowpass_filter(markers, 6, 60, 4)
proc_ma = np.zeros_like(markers)
for i in range(3):
    proc_ma[i, :, :] = OfflineProcessing().process_generic_signal(
        markers[i, :, :],
        moving_average=True,
        band_pass_filter=False,
        centering=False,
        absolute_value=False,
        moving_average_window=10,
    )
from time import time

rt_proc = np.zeros_like(markers)
rt_proc_method_x = RealTimeProcessing(60, 30)
rt_proc_method_y = RealTimeProcessing(60, 30)
rt_proc_method_z = RealTimeProcessing(60, 30)
for i in range(markers.shape[2]):
    if i > 0:
        new_value_mat = np.concatenate((new_value_mat, markers[:, :, i][:, :, np.newaxis]), axis=2)
    else:
        new_value_mat = markers[:, :, i][:, :, np.newaxis]
    tic = time()
    for j in range(3):
        # rt_proc[j, :, i] = OfflineProcessing().process_generic_signal(new_value_mat[j, :, -60:],
        #                                        moving_average=True,
        #                                        band_pass_filter=False,
        #                                        centering=False,
        #                                        absolute_value=False,
        #                                        moving_average_window=8,
        #                                        )[:, -4]
        if j == 0:
            rt_proc_tmp = rt_proc_method_x.process_emg(
                markers[j, :, i][:, np.newaxis],
                moving_average=True,
                band_pass_filter=False,
                centering=False,
                absolute_value=False,
                moving_average_window=5,
            )[:, -1]
        elif j == 1:
            rt_proc_tmp = rt_proc_method_y.process_emg(
                markers[j, :, i][:, np.newaxis],
                moving_average=True,
                band_pass_filter=False,
                centering=False,
                absolute_value=False,
                moving_average_window=5,
            )[:, -1]
        elif j == 2:
            rt_proc_tmp = rt_proc_method_z.process_emg(
                markers[j, :, i][:, np.newaxis],
                moving_average=True,
                band_pass_filter=False,
                centering=False,
                absolute_value=False,
                moving_average_window=5,
            )[:, -1]
        if i > 60:
            rt_proc[j, :, i] = rt_proc_tmp
        else:
            rt_proc[j, :, i] = markers[j, :, i]
    print("time: {}".format(time() - tic))
plt.figure()
for j in range(markers.shape[1] - 10):
    plt.subplot(1, 3, j + 1)
    for i in range(3):
        plt.plot(markers[i, j, :], "r")
        plt.plot(proc_data[i, j, :], "b")
        plt.plot(proc_ma[i, j, :], "orange")
        plt.plot(rt_proc[i, j, :], "--")
        plt.plot(markers_processed[i, j, :], "g", alpha=0.5)
plt.legend(["raw", "butter", "ma", "rt_ma", "live processed"])
plt.show()
