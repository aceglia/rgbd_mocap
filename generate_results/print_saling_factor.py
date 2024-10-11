import numpy as np
import json
import glob
import os


if __name__ == "__main__":
    dir = "D:\Documents\Programmation\pose_estimation\data_collection_mesurement"
    tmp_scaling_1 = "Q:\Projet_hand_bike_markerless\RGBD\scaling_tool_minimal_1.xml"
    tmp_scaling_2 = "Q:\Projet_hand_bike_markerless\RGBD\scaling_tool_minimal_2.xml"
    data_path = "Q:\Projet_hand_bike_markerless/RGBD"
    parts = [f"P{i}" for i in range(9, 17)]
    for part in parts:
        all_files = os.listdir(rf"{data_path}\{part}")
        all_files = [file for file in all_files if "gear_10" in file]
        for file in all_files:
            trc_path = f"{data_path}\{part}\{file}\{file}_dlc_ribs_and_cluster.trc"
            if not os.path.exists(trc_path):
                continue
            measurement_file = dir + "/measurements_" + part + ".json"
            with open(measurement_file) as f:
                data = json.load(f)
            mass = data["weight"] * 0.628
            scling_factor = data["ster_C7"] / 124.7
            print(f"part: {part}, file: {file}, mass: {mass}, scaling factor: {scling_factor}")
            # open as txt
            with open(tmp_scaling_1, "r") as f:
                scaling_1 = f.read()
            scaling_1 = scaling_1.replace("mass_to_replace", str(mass))
            scaling_1 = scaling_1.replace("ster_c7", str(scling_factor))
            scaling_1 = scaling_1.replace("path_to_replace", str(trc_path))
            scaling_1 = scaling_1.replace("path_2_replace", str(trc_path))
            with open(tmp_scaling_2, "r") as f:
                scaling_2 = f.read()
            scaling_2 = scaling_2.replace("mass_to_replace", str(mass))
            scaling_2 = scaling_2.replace("path_to_replace", str(trc_path))

            with open(f"{data_path}\{part}\{file}\scaling_tool_minimal_1.xml", "w") as f:
                f.write(scaling_1)
            with open(f"{data_path}\{part}\{file}\scaling_tool_minimal_2.xml", "w") as f:
                f.write(scaling_2)
