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
    participants = ["P9", "P10", "P11", "P13", "P14", "P16"]
    participants = ["P10"]
    main_path = "data_files"
    # training_path = r"Q:\Projet_hand_bike_markerless\training_data_set"
    # if not os.path.exists(training_path):
    #     os.makedirs(training_path)
    for participant in participants:
        files = os.listdir(f"{main_path}{os.sep}{participant}")
        files = [file for file in files if "gear_10" in file]
        for file in files:
            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            if not os.path.isfile(path + "/tracking_config_gui_3_crops.json"):
                continue
            all_color_files = glob.glob(path + "/color*.png")
            markers_file = path + "/marker_pos_multi_proc_3_crops_pp.bio"
            markers_data = load(markers_file, merge=True)
            nb_file = 55
            downsample = int(len(markers_data["frame_idx"]) / nb_file)
            for key in markers_data.keys():
                if isinstance(markers_data[key], list):
                    markers_data[key] = markers_data[key][::downsample]
                if isinstance(markers_data[key], np.ndarray):
                    markers_data[key] = markers_data[key][..., ::downsample]

            frame_width = 848
            frame_height = 480
            # out = cv2.VideoWriter(
            #     "data_depth_3D.avi", cv2.VideoWriter_fourcc("M", "J", "P3", "G"), 60, (frame_width, frame_height)
            # )
            all_depth = np.ndarray((len(markers_data["frame_idx"]), frame_height, frame_width))
            for i in range(len(markers_data["frame_idx"])):
                color = cv2.imread(path + f'/color_{markers_data["frame_idx"][i]}.png')
                # color = cv2.rotate(color, cv2.ROTATE_180)
                depth = cv2.imread(path + f'/depth_{markers_data["frame_idx"][i]}.png', cv2.IMREAD_ANYDEPTH)
                # depth = cv2.rotate(depth, cv2.ROTATE_180)
                # depth = np.where(
                #     (depth > 1.4 / (0.0010000000474974513)) | (depth <= 0),
                #     0,
                #     depth,
                # )
                #     all_depth[i, :, :] = depth
                # max_depth = np.max(all_depth)
                # min_depth = np.min(all_depth[all_depth > 0]) - 50
                # for i in range(len(idx_final)):
                #     depth = all_depth[i, : ,:]
                # normalize image to 0-255 using min and max values
                min_depth = np.min(depth[depth > 0])
                max_depth = np.max(depth)
                normalize_depth = (depth - min_depth) / (max_depth - min_depth)
                normalize_depth[depth == 0] = 0
                normalize_depth = normalize_depth * 255
                depth = normalize_depth.astype(np.uint8)

                # depth = cv2.convertScaleAbs(depth, alpha=(255.0 / np.max(depth)))
                # draw all contours
                depth_3d = np.dstack((depth, depth, depth))
                # training_path = "Z:\Projet_hand_bike_markerless\RGBD\Training_data"
                # training_path = "Training_data_depth_3d"
                # image_path = training_path + rf"/{participant}_{file[:7]}_depth_3d_{idx_final[i]}.png"
                # cv2.imwrite(image_path, depth_3d)
                # out.write(depth_3d)

                # depth_3d = draw_markers(
                #     depth_3d,
                #     markers_pos=(markers_final[i][:-1]),
                #     markers_names=markers_data[i]["markers_names"],
                #     is_visible=is_visible[i],
                #     scaling_factor=0.5,
                #     # markers_reliability_index=reliability_idx[i],
                # )
                # csv_path = training_path + "/CollectedData_Ame.csv"
                # add_to_csv(image_path, markers_final[i][:-1], markers_data[i]["markers_names"], csv_path)
                cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
                cv2.imshow("depth", depth_3d)
                cv2.namedWindow("color", cv2.WINDOW_NORMAL)
                cv2.imshow("color", color)
                # # cv2.namedWindow("uint8", cv2.WINDOW_NORMAL)
                # # cv2.imshow("uint8", outputImg8U)
                # # cv2.waitKey(10000)
                cv2.waitKey(0)
