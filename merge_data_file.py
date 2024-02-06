from biosiglive import save, load
import os


if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    for participant in participants:
        image_path = fr"D:\Documents\Programmation\pose_estimation\data_files\{participant}"
        files = os.listdir(f"{image_path}")
        files = [file for file in files if
                 "gear" in file and os.path.isdir(f"{image_path}{os.sep}" + file)
                 ]
        for file in files:
            print("processing file : ", file)
            if not os.path.isfile(f"{image_path}{os.sep}" + file + "/markers_pos.bio"):
                continue
            file = f"{image_path}{os.sep}" + file
            load_markers = load(file + "/markers_pos.bio")
            if os.path.isfile(file + "/markers_pos_merged.bio"):
                os.remove(file + "/markers_pos_merged.bio")
            save(load_markers, file + "/markers_pos_merged.bio")