import os

import numpy as np

from pose_est.marker_class import MarkerSet
from pose_est.RgbdImages import RgbdImages
from pose_est.utils import *
import matplotlib.pyplot as plt
from biosiglive import save
import pyrealsense2 as rs
import cv2

if __name__ == "__main__":
    with_camera = False
    suffix = "11-07-2023_09_16_59"

    if not with_camera:
        images_dir = f"data_{suffix}"
        # image_file = r"D:\Documents\Programmation\vision\image_camera_trial_1_800.bio.gzip"
        camera = RgbdImages(conf_file=f"config_camera_{suffix}.json", images_dir=images_dir)
        # camera = RgbdImages(conf_file=r"config_camera_mod.json", merged_images=image_file)
    else:
        camera = RgbdImages()
        camera.init_camera(
            ColorResolution.R_848x480,
            DepthResolution.R_848x480,
            FrameRate.FPS_60,
            FrameRate.FPS_60,
            align=True,
        )
    camera.clipping_color = 20
    camera.is_frame_aligned = False
    markers_wand = MarkerSet(marker_set_name="wand", marker_names=["wand_1", "wand_2", "wand_3", "wand_4"], image_idx=0)
    camera.add_marker_set([markers_wand])
    camera.initialize_tracking(
        tracking_conf_file=f"tracking_conf_{suffix}.json",
        crop_frame=True,
        mask_parameters=True,
        label_first_frame=True,
        build_kinematic_model=False,
        model_name=f"kinematic_model_calib_{suffix}.bioMod",
        marker_sets=[markers_wand],
        method=DetectionMethod.CV2Blobs,
        rotation_angle=Rotation.ROTATE_180,
    )
    mask_params = camera.mask_params
    fig = plt.figure()
    import time
    count = 0
    # import open3d as o3d
    camera.frame_idx = 0
    while True:
        color_cropped, depth_cropped = camera.get_frames(
            aligned=False,
            detect_blobs=True,
            label_markers=True,
            bounds_from_marker_pos=False,
            method=DetectionMethod.CV2Blobs,
            filter_with_kalman=True,
            adjust_with_blobs=True,
            fit_model=False,
            rotation_angle=Rotation.ROTATE_180,
        )
        if not isinstance(color_cropped, list):
            color_cropped = [color_cropped]
            depth_cropped = [depth_cropped]
        for i in range(len(color_cropped)):
            cv2.namedWindow("cropped_final_" + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow("cropped_final_" + str(i), color_cropped[i])

        color = camera.color_frame.copy()
        depth_image = camera.depth_frame.copy()
        markers_pos, markers_names, occlusions, reliability_idx = camera.get_global_markers_pos()
        markers_in_meters, _, _, _ = camera.get_global_markers_pos_in_meter(markers_pos)

        dic = {"markers_in_meters": markers_in_meters[:, :, np.newaxis], "markers_names": markers_names}
        save(dic, f"markers_{suffix}.bio")

        color = draw_markers(
            color,
            markers_pos=markers_pos,
            markers_names=markers_names,
            is_visible=occlusions,
            scaling_factor=0.5,
            markers_reliability_index=reliability_idx,
        )
        # from biosiglive import MskFunctions, InverseKinematicsMethods
        cv2.namedWindow("color", cv2.WINDOW_NORMAL)
        cv2.imshow("color", color)
        cv2.waitKey(100)