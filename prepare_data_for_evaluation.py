import cv2
import csv
import numpy as np
from biosiglive import load
import glob
import json
from rgbd_mocap.utils import draw_markers
import os
from pathlib import Path


def add_to_csv(image_path, markers_pos, markers_names, csv_path):
    file = csv_path
    # write headers if file does not exist
    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scorer", "", ""] + ["Automatic labeling"] * 2 * len(markers_names))
            bodyparts = []
            x_y = []
            for i in range(len(markers_names)):
                bodyparts.append((markers_names[i]))
                bodyparts.append((markers_names[i]))
                x_y.append("x")
                x_y.append("y")
            writer.writerow(["bodyparts", "", ""] + bodyparts)
            writer.writerow(["coords", "", ""] + x_y)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        markers_pos_list = []
        for i in range(markers_pos.shape[1]):
            markers_pos_list.append(markers_pos[0, i])
            markers_pos_list.append(markers_pos[1, i])
        writer.writerow(["labeled-data", "data_test"] + [Path(image_path).stem + ".png"] + markers_pos_list)


def start_idx_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data["start_frame"]


if __name__ == "__main__":
    # load_images path
    participants = ["P3_session2", "P4_session2", "P2_session2"]  # "P4_session2",
    participants = ["P9", "P10", "P11", "P13", "P14", "P16"]
    participants = ["P11"]
    main_path = "Q:\Projet_hand_bike_markerless\RGBD"
    evaluating_path = r"Q:\Projet_hand_bike_markerless\evaluate_data_set"
    if not os.path.exists(evaluating_path):
        os.makedirs(evaluating_path)
    for participant in participants:
        files = os.listdir(f"{main_path}{os.sep}{participant}")
        files = [file for file in files if "gear_10" in file]
        for file in files:
            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            if not os.path.isfile(path + "/tracking_config_gui.json"):
                continue
            all_depth_files = glob.glob(path + "/depth*.png")
            idx = []
            for f in all_depth_files:
                idx.append(int(f.split("\\")[-1].split("_")[-1].removesuffix(".png")))
            idx.sort()
            frame_width = 848
            frame_height = 480
            out = cv2.VideoWriter(
                evaluating_path + f"/{participant}_{file[:7]}_data_depth_3D.avi", cv2.VideoWriter_fourcc("M", "J", "P3", "G"), 60, (frame_width, frame_height)
            )
            count = 0
            for i in idx:
                depth = cv2.imread(path + f"/depth_{i}.png", cv2.IMREAD_ANYDEPTH)
                depth = np.where(
                    (depth > 1.4 / (0.0010000000474974513)) | (depth <= 0),
                    0,
                    depth,
                )
                depth = cv2.convertScaleAbs(depth, alpha=(255.0 / np.max(depth)))
                # draw all contours
                depth_3d = np.dstack((depth, depth, depth))
                out.write(depth_3d)
                count += 1
                if count > 500:
                    break
            out.release()

