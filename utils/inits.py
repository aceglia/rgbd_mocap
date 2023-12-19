from rgbd_mocap.RgbdImages import RgbdImages
from rgbd_mocap.utils import *

import os
import shutil


def init_RgbdImages(with_camera, files_paths):
    if with_camera:
        camera = init_camera()
    else:
        set_image_dir_and_config_file(files_paths, **files_paths)
        camera = init_no_camera(**files_paths)

    camera.markers_to_exclude_for_ik = ["M1", "M2", "M3"]
    camera.ik_method = "kalman"  # "kalman" or "least_squares"
    camera.clipping_color = 20
    camera.is_frame_aligned = False

    return camera


def init_camera():
    camera = RgbdImages()
    camera.init_camera(
        ColorResolution.R_848x480,
        DepthResolution.R_848x480,
        FrameRate.FPS_60,
        FrameRate.FPS_60,
        align=True,
    )

    return camera


def set_image_dir_and_config_file(files_path, participant, data_files, trial, suffix, path_to_project, file, **kwargs):
    if participant:
        images_dir = f"{data_files}{os.sep}{participant}{os.sep}{trial}_{suffix}"
        config_file = f"{path_to_project}config_camera_files{os.sep}config_camera_{participant}.json"
    else:
        images_dir = f"{file}"
        config_file = f"{path_to_project}config_camera_files{os.sep}config_camera_{suffix}.json"

    files_path["images_dir"] = images_dir
    files_path["config_file"] = config_file


def init_no_camera(start_idx, images_dir, config_file, **kwargs):
    print("working on : ", images_dir)
    camera = RgbdImages(
        conf_file=config_file,
        images_dir=images_dir,
        start_index=start_idx,
        # stop_index=10,
        downsampled=1,
        load_all_dir=False,
    )

    return camera


def init_tracking_conf(images_dir, tracking_files, **kwargs):
    print(images_dir)
    if os.path.isfile(rf"{images_dir}{os.sep}tracking_config.json"):
        tracking_conf = {"crop": False, "mask": False, "label": False, "build_kinematic_model": True}
    else:
        if len(tracking_files) == 0:
            tracking_conf = {"crop": True, "mask": True, "label": True, "build_kinematic_model": True}
        else:
            shutil.copy(tracking_files[0], rf"{images_dir}{os.sep}t" + f"racking_config.json")
            tracking_conf = {"crop": False, "mask": False, "label": False, "build_kinematic_model": True}

    return tracking_conf
