import numpy as np
from processing_data.file_io import get_all_file, load_all_data

# from utils_old import *
prefix = "/mnt/shared"


if __name__ == "__main__":
    # trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    # trials[-1] = ["gear_10"]
    # all_data, _ = load_results(
    #     participants,
    #     "/mnt/shared/Projet_hand_bike_markerless/process_data",
    #     file_name="kalman_proc",
    #     recompute_cycles=False,
    # )
    # init_data, _ = load_all_data(
    #     participants,
    #     "/mnt/shared/Projet_hand_bike_markerless/process_data",
    # )
    participants = [f"P{i}" for i in range(9, 17)]
    processed_data_path = prefix + "/Projet_hand_bike_markerless/process_data"
    file_name = "kalman_proc_new.bio"
    all_files, mapped_part = get_all_file(participants, processed_data_path, to_include=[file_name], is_dir=False)
    all_data, participants = load_all_data(
        mapped_part, all_files, name_to_load=f"_{file_name[:-4]}_tmp.bio", reload=True
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
    for part, file in zip(participants, all_data):
        time_ik.append(np.mean(file["depth"]["time"]["ik"][1:]))
        time_id.append(np.mean(file["depth"]["time"]["id"][1:]))
        time_so.append(np.mean(file["depth"]["time"]["so"][1:]))
        std_ik.append(np.std(file["depth"]["time"]["ik"][1:]))
        std_id.append(np.std(file["depth"]["time"]["id"][1:]))
        std_so.append(np.std(file["depth"]["time"]["so"][1:]))
        # for f, file in enumerate(init_data[part].keys()):
        #     time_tracking.append(np.mean(init_data[part][file]["process_time_depth"][1:]))
        #     std_tracking.append(np.std(init_data[part][file]["process_time_depth"][1:]))

    print("time_ik : ", np.mean(time_ik) * 1000, "+/-", np.mean(std_ik) * 1000, "ms")
    print("time_id : ", np.mean(time_id) * 1000, "+/-", np.mean(std_id) * 1000, "ms")
    print("time_so : ", np.mean(time_so) * 1000, "+/-", np.mean(std_so) * 1000, "ms")
    # print("time_tracking : ", np.mean(time_tracking) * 1000, "+/-", np.mean(std_tracking) * 1000, "ms")
    print(
        "total time : ",
        (np.mean(time_ik) + np.mean(time_id) + np.mean(time_so) + np.mean(time_tracking)) * 1000,
        "+/-",
        (np.mean(std_ik) + np.mean(std_id) + np.mean(std_so) + np.mean(std_tracking)) * 1000,
        "ms",
    )
