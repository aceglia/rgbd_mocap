import os
import shutil
import glob
from pathlib import Path

if __name__ == '__main__':
    participant = ["P12", "P13"]#, "P12", "P13", "P14", "P15", "P16"]
    # participant = ["P14", "P15", "P16"]
    image_path = fr"D:\Documents\Programmation\pose_estimation\data_files"
    # image_path = fr"D:\Documents\Programmation\pose_estimation\config_camera_files"

    image_path = "Q:\Projet_hand_bike_markerless\RGBD"
    local = "F:\markerless_project"
    server_to_copy = ("Q:\Projet_hand_bike_markerless\RGBD")

    for part in participant:
        model_copied = False
        if not os.path.exists(local + os.sep + part):
            os.makedirs(local + os.sep + part)
        all_files_to_copy = []
        all_biomod = glob.glob(image_path + os.sep + part + os.sep + "**.bioMod")
        all_files_to_copy += all_biomod
        # all_files = glob.glob(image_path + os.sep + part + os.sep + "**_new_seth.bioMod")
        all_files = os.listdir(image_path + os.sep + part + os.sep)
        for f in all_files_to_copy:
            src_file = f
            dist_file = local + os.sep + part + os.sep + Path(src_file).stem + Path(src_file).suffix
            if os.path.isfile(src_file):
                if os.path.isfile(dist_file):
                    os.remove(dist_file)
                print(f"Copying {src_file} to {dist_file}")
                shutil.copy2(src_file, dist_file)
        # all_files = [file for file in all_files if "gear" in file and os.path.isdir(image_path + os.sep + part + os.sep + file)]
        # all_files = glob.glob(image_path + os.sep + os.sep + "config**")
        for dir in all_files:
            all_files_to_copy = []
            all_bio = glob.glob(image_path + os.sep + part + os.sep + dir + os.sep + "**.bio")
            all_files_to_copy += all_bio
            all_json = glob.glob(image_path + os.sep + part + os.sep + dir + os.sep + "**.json")
            all_files_to_copy += all_json
            if len(all_bio) == 0 or len(all_json) == 0:
                print(f"Missing files in {dir}")
                continue
            if not os.path.isdir(local + os.sep + part + os.sep + dir):
                os.makedirs(local + os.sep + part + os.sep + dir)
            # src_file = dir
            # if not part in dir:
            #     continue
            # src_file = image_path + os.sep + part + os.sep + dir + os.sep + "tracking_config_gui_3_crops_1er_gap_bis.json"
            # src_file = dir
            # src_file = glob.glob(image_path + os.sep + part + os.sep + dir + os.sep + "kinematic**")
            # if src_file != []:
            #     src_file = src_file[0]
            # else:
            #     continue
            # dist_file = server_to_copy + os.sep + part + os.sep + Path(src_file).stem + Path(src_file).suffix

            # dist_file = server_to_copy + os.sep + part + os.sep + dir + os.sep + Path(src_file).stem + Path(src_file).suffix
            # dist_file_bis = server_to_copy + os.sep + part + os.sep + dir + os.sep + "backup_" + Path(src_file).stem + Path(src_file).suffix

            # dist_file = server_to_copy + os.sep + Path(src_file).stem + Path(src_file).suffix
            for f in all_files_to_copy:
                src_file = f
                dist_file = local + os.sep + part + os.sep + dir + os.sep+ Path(src_file).stem + Path(src_file).suffix
                if os.path.isfile(src_file):
                    if os.path.isfile(dist_file):
                        os.remove(dist_file)
                    print(f"Copying {src_file} to {dist_file}")
                    shutil.copy2(src_file, dist_file)
                    # shutil.copy2(src_file, dist_file_bis)

                else:
                    print(f"File {src_file} does not exist")
                    continue
