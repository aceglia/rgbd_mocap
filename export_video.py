import shutil
import os

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "16"]
    trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    data_files = "Q:\Projet_hand_bike_markerless\RGBD"
    processed_path = "Q:\Projet_hand_bike_markerless\process_data"
    if not os.path.isdir(processed_path + os.sep + "videos"):
        os.mkdir(processed_path + os.sep + "videos")
    for p, part in enumerate(participants):
        files = os.listdir(f"{data_files}{os.sep}{part}")
        files = [file for file in files if
                 "gear" in file and os.path.isdir(f"{data_files}{os.sep}{part}{os.sep}" + file)
                 ]
        final_files = files if not trials else []
        if trials:
            for trial in trials[p]:
                for file in files:
                    if trial in file:
                        final_files.append(file)
                        break
        files = final_files
        for f, file in enumerate(files):
            print(f"working on participant {part} for trial {file[:7]}")
            if os.path.isfile(f"{data_files}{os.sep}{part}{os.sep}{file}{os.sep}images_processed.avi"):
                src = f"{data_files}{os.sep}{part}{os.sep}{file}{os.sep}images_processed.avi"
                dist = processed_path + os.sep + "videos" + os.sep + part + "_" + file[:7] + "_images_processed.avi"
                shutil.copy2(src, dist)