from processing_data.data_processing_helper import calculate_euler_error, fill_and_interpolate
from processing_data.file_io import load_results
import numpy as np
import matplotlib.pyplot as plt
from biosiglive import load, save
import scipy


def compute_error(q_ref, q_to_compare, to_vector=False):
    sequence = [[ None, None, None], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1, 2], [2], [1]]
    angle_euler_ref = np.zeros((3, q_ref.shape[1]))
    angle_euler_to_compare = np.zeros((3, q_to_compare.shape[1]))
    all_errors = []
    error_to_evaluate = np.zeros_like(q_ref)

    if to_vector:
        for i in range(q_ref.shape[0]):
            error_to_evaluate[i, :]= q_ref[i, :] - q_to_compare[i, :]
    else:
        count = 0
        for i in range(len(sequence)):
            if None in sequence[i]:
                error_to_evaluate[count:count + len(sequence[i]), :] = (q_ref[count:count + len(sequence[i]), :] - q_to_compare[count:count + len(sequence[i]), :]) * 1000
                count += len(sequence[i])
                continue
            angle_euler_ref[sequence[i]] = q_ref[count:count + len(sequence[i]), :]
            angle_euler_to_compare[sequence[i]] = q_to_compare[count:count + len(sequence[i]), :]
            for j in range(q_ref.shape[1]):
                error_tmp = calculate_euler_error(angle_euler_ref[:, j], angle_euler_to_compare[:, j])
                error_to_evaluate[count:count + len(sequence[i]), j] = error_tmp[sequence[i]]
            count += len(sequence[i])
    error_to_evaluate = np.degrees(error_to_evaluate) if to_vector else error_to_evaluate
    rmse = np.sqrt(np.median(np.square(error_to_evaluate), axis=1))
    std = np.std(error_to_evaluate, axis=1)
    return error_to_evaluate, rmse, std

def divide_by_cycle(q_ref, error_to_evaluate, n_cycle=None):
    find_peak = scipy.signal.find_peaks(q_ref[-4, :], height=0.01, distance=100)
    find_peak = find_peak[0][:n_cycle] if n_cycle is not None else find_peak[0]
    #plt.plot(q_ref[-2, :])
    #plt.plot(find_peak[0], q_ref[-2, find_peak[0]], "x")
    nb_peak = len(find_peak)
    cycle_error = np.zeros((nb_peak , error_to_evaluate.shape[0], 100))
    for i in range(nb_peak-1):
        cycle_tmp = error_to_evaluate[:, find_peak[i]:find_peak[i+1]]
        cycle_error[i, ...] = fill_and_interpolate(cycle_tmp, 100, fill=False)
    rmse_cycle = np.sqrt(np.median(np.square(cycle_error), axis=0))
    std_cycle = np.std(cycle_error, axis=0)
    return cycle_error, rmse_cycle, std_cycle

def plot_cycle_error(rmse_cycle, std_cycle, index=None):
    name =("cycle_error" if index is None else f"cycle_error_{index}"   )
    plt.figure(name)
    for i in range(rmse_cycle.shape[0]):
        plt.subplot(int(np.ceil(rmse_cycle.shape[0]/4)), 4, i+1)
        plt.plot(rmse_cycle[i, :], label=f"cycle {i+1}")
        plt.fill_between(np.arange(100), rmse_cycle[i, :] - std_cycle[i, :], rmse_cycle[i, :] + std_cycle[i, :], alpha=0.2)


if __name__ == "__main__":
    participants = [f"P{i}" for i in range(9, 17)]
    to_vector = True
    reload_data = True
    if reload_data:
        all_data, trials = load_results(
            participants,
            "/mnt/shared/Projet_hand_bike_markerless/process_data",
            file_name="_with_technical_marker.bio",
            recompute_cycles=False,
            to_exclude=["live_filt"],
        )
        save(all_data, "_all_data_tmp.bio", safe=False)
    else:
        all_data = load("_all_data_tmp.bio")
    ref_key = ["vicon"]
    to_compare = ["dlc_1"]
    all_data_tmp = all_data.copy()
    participants = [f"P{i}" for i in range(9, 17)]
    nb_q = all_data_tmp[participants[0]][list(all_data_tmp[participants[0]].keys())[0]]["vicon"]["q"].shape[0]
    for patient in all_data_tmp.keys():
        for trial in all_data_tmp[patient].keys():
            plt.figure(f"q_{trial}_{patient}")
            for i in range(nb_q):
                plt.subplot(int(np.ceil(nb_q/4)), 4, i+1)
                plt.plot(np.degrees(all_data_tmp[patient][trial]["vicon"]["q"][i, :]), label="vicon")
                plt.plot(np.degrees(all_data_tmp[patient][trial]["dlc_1"]["q"][i, :]), label="dlc_1")
                plt.legend()
            plt.show()
    all_rmse = np.zeros((len(participants), nb_q, 100))
    all_std = np.zeros((len(participants), nb_q, 100))
    for i, participant in enumerate(participants):
        trials_rmse = np.zeros((4, nb_q, 100))
        trials_std = np.zeros((4, nb_q, 100))
        for t, trial in enumerate(all_data[participant].keys()):
            data_tmp = all_data[participant][trial]
            q_ref = data_tmp[ref_key[0]]["q"]
            q_to_compare = data_tmp[to_compare[0]]["q"]

            error_to_evaluate, rmse, std = compute_error(q_ref, q_to_compare, to_vector=to_vector)
            cycle_error, trials_rmse[t, ...], trials_std[t, ...] = divide_by_cycle(q_ref, error_to_evaluate, n_cycle=None)
        all_rmse[i, ...] = np.median(trials_rmse, axis=0)
        all_std[i, ...] = np.median(trials_std, axis=0)
        plot_cycle_error(all_rmse[i, :], all_std[i, :], i)

    all_rmse_mean = np.median(all_rmse, axis=0)
    all_std_mean = np.median(all_std, axis=0)
    plot_cycle_error(all_rmse_mean, all_std_mean)

    plt.show()





