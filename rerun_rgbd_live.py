#!/usr/bin/env python3
"""A minimal example of streaming frames live from an Intel RealSense depth sensor."""
from __future__ import annotations

import argparse

import numpy as np
# import pyrealsense2 as rs
import rerun as rr  # pip install rerun-sdk
from utils import load_data_from_dlc, _convert_string
import biorbd
import biosiglive
from rgbd_mocap.camera.camera_converter import CameraConverter
from pyorerun import BiorbdModel, MultiPhaseRerun, PhaseRerun, DisplayModelOptions


def reorder_markers(markers, model, names):
    model_marker_names = [_convert_string(model.markerNames()[i].to_string()) for i in range(model.nbMarkers())]
    assert len(model_marker_names) == len(names)
    assert len(model_marker_names) == markers.shape[1]
    count = 0
    reordered_markers = np.zeros((markers.shape[0], len(model_marker_names), markers.shape[2]))
    for i in range(len(names)):
        if names[i] == "elb":
            names[i] = "elbow"
        if _convert_string(names[i]) in model_marker_names:
            reordered_markers[:, model_marker_names.index(_convert_string(names[i])),
            :] = markers[:, count, :]
            count += 1
    return reordered_markers


def run_realsense(num_frames: int | None, trial=None, part=None) -> None:
    # Visualize the data as RDF
    rr.log("animation_phase_0", rr.ViewCoordinates.RDF, timeless=True)
    camera_conf_file = f"config_camera_files\config_camera_P9.json"
    # rgbd = RgbdImages(path_to_camera_config_file)
    converter = CameraConverter()
    converter.set_intrinsics(camera_conf_file)
    converter.set_extrinsics(camera_conf_file)
    depth_intr = converter.depth
    rgb_intr = converter.depth
    converter.depth_to_color = np.eye(4)
    trans = converter.depth_to_color[:3, 3]
    rot = converter.depth_to_color[:3, :3]
    # Open the pipe
    # pipe = rs.pipeline()
    # profile = pipe.start()

    # We don't log the depth exstrinsics. We treat the "animation_phase_0" space as being at
    # the origin of the depth sensor so that "realsense/depth" = Identity

    # Get and log depth intrinsics
    # depth_profile = profile.get_stream(rs.stream.depth)
    # depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()

    rr.log(
        "animation_phase_0/depth/image",
        rr.Pinhole(
            resolution=[depth_intr.width, depth_intr.height],
            focal_length=[depth_intr.fx, depth_intr.fy],
            principal_point=[depth_intr.ppx, depth_intr.ppy],
        ),
        timeless=True,
    )

    # Get and log color extrinsics
    # rgb_profile = profile.get_stream(rs.stream.color)

    # rgb_from_depth = depth_profile.get_extrinsics_to(rgb_profile)
    rr.log(
        "animation_phase_0/rgb",
        rr.Transform3D(
            translation=trans,
            mat3x3=rot,
            from_parent=True,
        ),
        timeless=True,
    )

    # Get and log color intrinsics
    # rgb_intr = rgb_profile.as_video_stream_profile().get_intrinsics()

    rr.log(
        "animation_phase_0/rgb/image",
        rr.Pinhole(
            resolution=[rgb_intr.width, rgb_intr.height],
            focal_length=[rgb_intr.fx, rgb_intr.fy],
            principal_point=[rgb_intr.ppx, rgb_intr.ppy],
        ),
        timeless=True,
    )

    # model_path = "D:\Documents\Programmation\pose_estimation\data_files\P9\model_scaled_depth.bioMod"
    # model_path = "model_tmp_test_pyorerun.bioMod"
    model_path = f"Q:\Projet_hand_bike_markerless\RGBD\{part}/model_scaled_depth_new_seth.bioMod"
    display_option = DisplayModelOptions()
    display_option.mesh_color = (77, 77, 255)
    biorbd_model = BiorbdModel(model_path, display_option)
    q = np.zeros((biorbd_model.model.nbQ()))
    phase_rerun = PhaseRerun(timeless=True, window=None, name=None)
    phase_rerun.add_animated_model(biorbd_model, q, timeless=True)
    phase_rerun.update_animated_model(q)
    import os
    import glob
    import cv2
    image_path = r"data_files"
    main_path = "F:\markerless_project"
    main_path = "Q:\Projet_hand_bike_markerless\RGBD"
    # participants = ["P9"]#, "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    # for participant in participants:
    files = os.listdir(f"{main_path}{os.sep}{part}")
    files = [file for file in files if trial in file and "less" not in file and "more" not in file]
    for file in files:
        path = f"{main_path}{os.sep}{part}{os.sep}{file}"
        image_path_tmp = f"{image_path}{os.sep}{part}{os.sep}{file}"
        labeled_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_pp.bio"
        dlc_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_normal_alone_pp.bio"
        if not os.path.isdir(path) or not os.path.exists(labeled_data_path) or not os.path.exists(dlc_data_path):
            continue
        data_dlc, data_labeling = load_data_from_dlc(labeled_data_path, dlc_data_path, part, file)
        from biosiglive import MskFunctions, InverseKinematicsMethods, OfflineProcessing
        msk = MskFunctions(biorbd_model.model, data_buffer_size=1)
        reorder_marker_from_source = reorder_markers(data_labeling["markers_in_meters"][:, :-3, :],
                                                     biorbd_model.model,
                                                     data_dlc["markers_names"][:-3])
        new_markers_dlc_filtered = np.zeros((3, reorder_marker_from_source.shape[1], reorder_marker_from_source.shape[2]))
        for i in range(3):
            new_markers_dlc_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(
                reorder_marker_from_source[i, :, :],
                2, 60, 2)
        idx = data_dlc["frame_idx"]
        # all_color_files = glob.glob(path + "/color*.png")
        all_depth_files = glob.glob(image_path_tmp + "/depth*.png")
        if len(all_depth_files) == 0:
            continue
        # Read frames in a loop
        frame_nr = 0

        q, _ = msk.compute_inverse_kinematics(new_markers_dlc_filtered[:, :, 0:1], method=InverseKinematicsMethods.BiorbdLeastSquare,
                                              )
        # q[:6, :] = np.zeros_like(q[:6, :])
        initial_state = [q, np.zeros_like(q), np.zeros_like(q)]
        msk = MskFunctions(biorbd_model.model, data_buffer_size=new_markers_dlc_filtered.shape[2])
        q, _ = msk.compute_inverse_kinematics(new_markers_dlc_filtered[:, :, :],
                                                method=InverseKinematicsMethods.BiorbdKalman,
                                                # initial_state=initial_state
                                              )
        try:
            while True:
                if num_frames and frame_nr >= num_frames:
                    break
                try:
                    q[:, frame_nr]
                except:
                    break

                rr.set_time_sequence("frame_nr", frame_nr)
                frame_nr += 1
                try:
                    depth_image = cv2.imread(image_path_tmp + f"\depth_{idx[frame_nr]}.png", cv2.IMREAD_ANYDEPTH)
                    depth_image = np.where(
                        (depth_image > 1.2 / (0.0010000000474974513)) | (depth_image <= 0.2 / (0.0010000000474974513)),
                        0,
                        depth_image,
                    )
                    color_image = cv2.cvtColor(cv2.imread(image_path_tmp + f"\color_{idx[frame_nr]}.png"), cv2.COLOR_BGR2RGB)
                except:
                    print(f"frame {idx[frame_nr]} not found")
                    continue
                phase_rerun.update_animated_model(q[:, frame_nr])

                # frames = pipe.wait_for_frames()
                # for f in frames:
                    # Log the depth frame
                    # depth_frame = frames.get_depth_frame()
                    # depth_units = depth_frame.get_units()
                    # depth_image = np.asanyarray(depth_frame.get_data())
                rr.log("animation_phase_0/depth/image", rr.DepthImage(depth_image, meter=1 / converter.depth_scale))

                    # Log the color frame
                    # color_frame = frames.get_color_frame()
                # color_image = np.asanyarray(color_frame.get_data())
                rr.log("animation_phase_0/rgb/image", rr.Image(color_image))
                # rr.log("animation_phase_0/tracked", rr.Points3D(tracked_markers[..., frame_nr].T, colors=(255, 0, 0), radii=0.01))
                rr.log("animation_phase_0/fk", rr.Points3D(new_markers_dlc_filtered[..., frame_nr].T, colors=(0, 255, 0), radii=0.01))
                # rr.log("animation_phase_0/ik", rr.Points3D(tracked_markers_dlc[..., frame_nr].T, colors=(0, 125, 255), radii=0.01))

        finally:
            # pipe.stop()
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Streams frames from a connected realsense depth sensor.")
    parser.add_argument("--num-frames", type=int, default=None, help="The number of frames to log")

    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "rerun_example_live_depth_sensor")

    run_realsense(args.num_frames, "gear_10", "P16")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()