import numpy as np

from utils_old import *

if __name__ == "__main__":
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    data_files = "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files"

    time_tracking_2D = []
    std_tracking_2D = []
    time_tracking_3D = []
    std_tracking_3D = []
    for p, part in enumerate(participants):
        files = os.listdir(f"{data_files}{os.sep}{part}")
        files = [file for file in files if os.path.isdir(f"{data_files}{os.sep}{part}{os.sep}" + file)]
        final_files = files if not trials else []
        if trials:
            for trial in trials[p]:
                for file in files:
                    if trial in file and "project" not in file:
                        final_files.append(file)
        files = final_files
        for f, file in enumerate(files):
            data = load(f"{data_files}{os.sep}{part}{os.sep}{file}{os.sep}marker_pos_multi_proc_3_crops_speed.bio")
            time_tracking_2D.append(np.mean(data["time_2D"][1:]))
            std_tracking_2D.append(np.std(data["time_2D"][1:]))
            time_tracking_3D.append(np.mean(data["time_3D"][1:]))
            std_tracking_3D.append(np.std(data["time_3D"][1:]))
    print(
        "time_tracking 2D : ",
        np.round(np.mean(time_tracking_2D) * 1000, 2),
        "+/-",
        np.round(np.mean(std_tracking_2D) * 1000, 2),
        "ms",
    )
    print(
        "time_tracking 3D : ",
        np.round(np.mean(time_tracking_3D) * 1000, 2),
        "+/-",
        np.round(np.mean(std_tracking_3D) * 1000, 2),
        "ms",
    )
