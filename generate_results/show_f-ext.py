import matplotlib.pyplot as plt
from utils_old import load_all_data
from scipy.signal import find_peaks
import numpy as np


if __name__ == "__main__":
    participants = ["P9"]  # , "P11", "P12", "P13"]#, "P14", "P15", "P16"]
    trials = [["gear_10"]] * len(participants)
    all_data, trials = load_all_data(participants, "/mnt/shared/Projet_hand_bike_markerless/process_data", trials)
    n_peaks = 10
    for part in all_data.keys():
        for f, file in enumerate(all_data[part].keys()):
            sensix_data = all_data[part][file]["sensix_data_interpolated"]
            peaks, _ = find_peaks(sensix_data["crank_angle"][0, :])
            peaks = [peak for peak in peaks if sensix_data["crank_angle"][0, peak] > 6]
            plt.figure("crank angle")
            data_to_plot = sensix_data["crank_angle"][0, peaks[0] : peaks[n_peaks]]
            x = np.linspace(0, 2 * np.pi, data_to_plot.shape[0])

            plt.plot(data_to_plot, label=part)

            plt.figure("forceX")
            data_to_plot = sensix_data["RFX"][0, peaks[0] : peaks[n_peaks]]
            plt.plot(x, data_to_plot, label=part)
            plt.figure("forceY")
            data_to_plot = sensix_data["RFY"][0, peaks[0] : peaks[n_peaks]]
            plt.plot(x, data_to_plot, label=part)
            plt.figure("forceZ")
            data_to_plot = sensix_data["RFZ"][0, peaks[0] : peaks[n_peaks]]
            plt.plot(x, data_to_plot, label=part)
            plt.figure("pedal angle")
            data_to_plot = sensix_data["right_pedal_angle"][0, peaks[0] : peaks[n_peaks]]
            plt.plot(x, data_to_plot, label=part)
    plt.legend(participants)
    plt.show()
