import os
from biosiglive import load, save
import matplotlib.pyplot as plt
import numpy as np
from processing_data.file_io import get_all_file

prefix = "/mnt/shared" if os.name == "posix" else "Q:"


if __name__ == "__main__":
    file_name = (
        f"marker_pos_multi_proc_3_crops_normal_500_down_b1_ribs_and_cluster_1_with_model_pp_full_technical_marker.bio"
    )
    # file_name = "marker_pos_multi_proc_3_crops_normal_filtered.bio"
    participants = [f"P{i}" for i in range(12, 13)]
    trials = ["gear_5", "gear_10", "gear_15", "gear_20"]
    data_files = f"{prefix}/Projet_hand_bike_markerless/RGBD"
    # data_files = "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files"
    files, parts = get_all_file(participants, data_files, trial_names=trials, to_include="gear")
    good_trials = ["gear_5"]
    reference_data = load([file for file in files if good_trials[0] in file][0] + f"/{file_name}")
    marker_to_replace = "xiph"
    for file in files:
        if good_trials[0] in file:
            continue
        data = load(file + f"/{file_name}")
        marker_tmp = data["markers_in_meters"]
        marker_tmp[:, 1, :] = np.repeat(reference_data["markers_in_meters"][:, 1, 0:1], marker_tmp.shape[2], axis=1)
        data["markers_in_meters"] = marker_tmp
        save(data, file + f"/{file_name}", safe=False)
