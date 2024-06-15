import numpy as np

from utils import *

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    #trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    #trials[-1] = ["gear_10"]
    all_data, _ = load_results(participants,
                            "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            )
    init_data, _ = load_all_data(participants,
                              "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            )
    all_colors = []
    time_ik = []
    time_id = []
    time_so = []
    std_ik = []
    std_id = []
    std_so = []
    time_tracking = []
    std_tracking = []
    for p, part in enumerate(all_data.keys()):
        for f, file in enumerate(all_data[part].keys()):
            time_ik.append(np.mean(all_data[part][file]["depth"]["time"]["ik"][1:]))
            time_id.append(np.mean(all_data[part][file]["depth"]["time"]["id"][1:]))
            time_so.append(np.mean(all_data[part][file]["depth"]["time"]["so"][1:]))
            std_ik.append(np.std(all_data[part][file]["depth"]["time"]["ik"][1:]))
            std_id.append(np.std(all_data[part][file]["depth"]["time"]["id"][1:]))
            std_so.append(np.std(all_data[part][file]["depth"]["time"]["so"][1:]))
        for f, file in enumerate(init_data[part].keys()):
            time_tracking.append(np.mean(init_data[part][file]["process_time_depth"][1:]))
            std_tracking.append(np.std(init_data[part][file]["process_time_depth"][1:]))

    print("time_ik : ", np.mean(time_ik) * 1000, "+/-", np.mean(std_ik)*1000, "ms")
    print("time_id : ", np.mean(time_id) * 1000, "+/-", np.mean(std_id)*1000, "ms")
    print("time_so : ", np.mean(time_so) * 1000, "+/-", np.mean(std_so)*1000, "ms")
    print("time_tracking : ", np.mean(time_tracking) * 1000, "+/-", np.mean(std_tracking)*1000, "ms")
    print("total time : ", (np.mean(time_ik) + np.mean(time_id) + np.mean(time_so) + np.mean(time_tracking)) * 1000,
          "+/-", (np.mean(std_ik) + np.mean(std_id) + np.mean(std_so)+ np.mean(std_tracking))*1000, "ms")

