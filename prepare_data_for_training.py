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
    participants = ["P4"]
    main_path = "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files"
    for participant in participants:
        files = os.listdir(f"{main_path}{os.sep}{participant}")
        files = [file for file in files if file[:7] == "gear_20"]
        for file in files:
            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            if not os.path.isfile(path + "/tracking_config.json"):
                continue
            all_color_files = glob.glob(path + "/color*.png")
            idx = []
            for file in all_color_files:
                idx.append(int(file.split("\\")[-1].split("_")[-1].removesuffix(".png")))
            idx.sort()
            all_color_files = [path + f"/color_{i}.png" for i in idx]
            all_depth_files = [path + f"/depth_{i}.png" for i in idx]
            start_frame = start_idx_from_json(rf"{path}/tracking_config.json")
            idx = idx[start_frame:]
            markers_file = path + "/markers_kalman.bio"
            markers_data = load(markers_file, merge=False, number_of_line=len(idx))
            is_visible = []
            idx_final = []
            markers_final = []
            reliability_idx = []
            nb_file = 55
            for i in range(len(markers_data)):
                if False in markers_data[i]["occlusions"]:
                    continue
                else:
                    is_visible.append(markers_data[i]["occlusions"])
                    idx_final.append(idx[i])
                    reliability_idx.append(markers_data[i]["reliability_idx"])
                    markers_final.append(markers_data[i]["markers_in_pixel"][:, :, 0])

            downsample = int(len(idx_final) / nb_file)
            idx_final = idx_final[::downsample]
            markers_final = markers_final[::downsample]
            is_visible = is_visible[::downsample]
            reliability_idx = reliability_idx[::downsample]
            all_max = [
                np.max(cv2.imread(path + f"/depth_{idx_final[i]}.png", cv2.IMREAD_ANYDEPTH))
                for i in range(len(idx_final))
            ]
            frame_width = 848
            frame_height = 480
            out = cv2.VideoWriter(
                "data_depth_3D.avi", cv2.VideoWriter_fourcc("M", "J", "P3", "G"), 60, (frame_width, frame_height)
            )
            for i in range(len(idx)):
                color = cv2.imread(path + f"/color_{idx[i]}.png")
                color = cv2.rotate(color, cv2.ROTATE_180)
                depth = cv2.imread(path + f"/depth_{idx[i]}.png", cv2.IMREAD_ANYDEPTH)
                depth = cv2.rotate(depth, cv2.ROTATE_180)
                depth = np.where(
                    (depth > 1.4 / (0.0010000000474974513)) | (depth <= 0),
                    0,
                    depth,
                )

                # depth = cv2.convertScaleAbs(depth, alpha=(255.0 / np.median(all_max)))
                depth = cv2.convertScaleAbs(depth, alpha=(255.0 / np.max(depth)))
                # draw all contours
                depth_3d = np.dstack((depth, depth, depth))
                # training_path = "Z:\Projet_hand_bike_markerless\RGBD\Training_data"
                training_path = "Training_data_depth_3d"
                image_path = training_path + rf"/{participant}_depth_3d_{idx_final[i]}.png"
                # cv2.imwrite(image_path, depth_3d)
                # out.write(depth_3d)

                depth_3d = draw_markers(
                    depth_3d,
                    markers_pos=(markers_final[i][:-1]),
                    markers_names=markers_data[i]["markers_names"],
                    is_visible=is_visible[i],
                    scaling_factor=0.5,
                    markers_reliability_index=reliability_idx[i],
                )
                csv_path = training_path + "/CollectedData_Ame.csv"
                # add_to_csv(image_path, markers_final[i][:-1], markers_data[i]["markers_names"], csv_path)
                cv2.namedWindow("color", cv2.WINDOW_NORMAL)
                cv2.imshow("color", depth_3d)
                # cv2.namedWindow("uint8", cv2.WINDOW_NORMAL)
                # cv2.imshow("uint8", outputImg8U)
                # cv2.waitKey(10000)
                cv2.waitKey(10)
