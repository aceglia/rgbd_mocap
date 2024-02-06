import numpy as np
import pandas as pd
from biosiglive import load
import matplotlib.pyplot as plt
import glob
import cv2
from rgbd_mocap.RgbdImages import RgbdImages
from rgbd_mocap.utils import check_and_attribute_depth


def eval_rmse(pt, pt_process, pixel=False):
    idx = 2 if pixel else 3
    error = []
    for i in range(pt.shape[1]):
        if np.mean(pt[:3, i, :]) == 0 or np.mean(pt_process[:3, i, :]) == 0:
            error.append(0)
        else:
            pt_wt_nan_idx = np.argwhere(np.isfinite(pt[:3, i, :]))[:, 1]
            pt_process_wt_nan_idx = np.argwhere(np.isfinite(pt_process[:3, i, :]))[:, 1]
            all_idx = np.intersect1d(pt_wt_nan_idx, pt_process_wt_nan_idx)
            error.append(np.sqrt(np.mean((pt[:idx, i, all_idx] - pt_process[:idx, i, all_idx]) ** 2, axis=0)).mean())
    return np.array(error)


def attrib_depth(data, depth_image):
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            pos = data[:, i, j]
            marker_depth, _ = check_and_attribute_depth(
                pos, depth_image[j], depth_scale=0.0010000000474974513
            )
            data[2, i, j] = marker_depth
    return data
def load_depth_images(path):
    all_depth_files = glob.glob(path + "/depth*.png")
    all_depth_files = [path + f"/depth_{i}.png" for i in range(len(all_depth_files))]
    depth_images = [
       cv2.rotate(cv2.imread(file, cv2.IMREAD_ANYDEPTH), cv2.ROTATE_180)
        for file in all_depth_files[100:300]
    ]
    return depth_images

if __name__ == '__main__':
    camera_config = "Q:\Projet_hand_bike_markerless\RGBD\config_camera_files\config_camera_P1.json"
    markers_label_path = "Q:\Projet_hand_bike_markerless\RGBD\P1\gear_5_28-07-2023_16_11_45"
    markers_dlc_path = "Q:\Projet_hand_bike_markerless\RGBD\P1"
    data_dlc = pd.read_csv(markers_dlc_path + "\P1_testDLC_mobnet_35_template_for_labelOct2shuffle1_100000.csv")
    mat_dlc = np.zeros((3, 13, 200))
    for i in range(2, 202):
        mat_dlc[0, :, i-2] = data_dlc.values[i, 1::3]
        mat_dlc[1, :, i-2] = data_dlc.values[i, 2::3]
    mat_dlc = attrib_depth(mat_dlc, depth_image=load_depth_images(markers_label_path))
    mat_dlc[:2, :, :] = mat_dlc[:2, :, :].astype("int")
    mat_dlc_in_pixel = mat_dlc.copy()
    mat_dlc_in_meters = mat_dlc.copy()
    data_markers = load(markers_label_path + "\markers_pos.bio", merge=True)
    markers_in_meters = data_markers["markers_in_meters"][:, :, :200]
    markers_in_pixel = data_markers["markers_in_pixel"][:, :, :200]
    camera = RgbdImages(conf_file=camera_config)
    for i in range(mat_dlc_in_meters.shape[2]):
        mat_dlc_in_meters[:, :, i], _, _, _ = camera.get_global_markers_pos_in_meter(mat_dlc_in_meters[:, :, i])
    rmse = eval_rmse(mat_dlc_in_meters*1000, markers_in_meters*1000, pixel=False)
    print(f"rmse in meters: {np.mean(rmse)}, {np.std(rmse)}")
    rmse = eval_rmse(mat_dlc_in_pixel, markers_in_pixel, pixel=True)
    print(f"rmse in pixels: {np.mean(rmse)}, {np.std(rmse)}")

    plt.figure("markers_depth")
    for i in range(mat_dlc.shape[1]):
        plt.subplot(4, 4, i + 1)
        for j in range(3):
            plt.plot(mat_dlc_in_pixel[j, i, :], c='b')
            plt.plot(markers_in_pixel[j, i, :], c='g')
    plt.legend(["dlc", "label"])

    plt.figure("markers_depth_in_meters")
    for i in range(mat_dlc.shape[1]):
        plt.subplot(4, 4, i + 1)
        for j in range(3):
            plt.plot(mat_dlc_in_meters[j, i, :], c='b')
            plt.plot(markers_in_meters[j, i, :], c='g')
    plt.legend(["dlc", "label"])
    plt.show()
