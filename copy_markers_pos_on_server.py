import os
import shutil
import glob
from pathlib import Path

if __name__ == '__main__':
    participant = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    image_path = fr"D:\Documents\Programmation\pose_estimation\data_files"
    # image_path = fr"D:\Documents\Programmation\pose_estimation\config_camera_files"

    server_to_copy = "Q:\Projet_hand_bike_markerless\RGBD"
    # server_to_copy = "Q:\Projet_hand_bike_markerless\RGBD\config_camera_files"
    for part in participant:
        all_files = os.listdir(image_path + os.sep + part + os.sep )
        all_files = [file for file in all_files if "anato" in file or "gear" in file and os.path.isdir(image_path + os.sep + part + os.sep + file)]
        # all_files = glob.glob(image_path + os.sep + os.sep + "config**")
        for dir in all_files:
            # if not part in dir:
            #     continue
            src_file = image_path + os.sep + part + os.sep + dir + os.sep + "tracking_config.json"
            # src_file = dir
            # src_file = glob.glob(image_path + os.sep + part + os.sep + dir + os.sep + "kinematic**")
            # if src_file != []:
            #     src_file = src_file[0]
            # else:
            #     continue
            dist_file = server_to_copy + os.sep + part + os.sep + dir + os.sep + Path(src_file).stem + Path(src_file).suffix
            # dist_file = server_to_copy + os.sep + Path(src_file).stem + Path(src_file).suffix

            if os.path.isfile(src_file):
                if os.path.isfile(dist_file):
                    os.remove(dist_file)
                shutil.copy2(src_file, dist_file)
