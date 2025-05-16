#!/usr/bin/env python3
"""A minimal example of streaming frames live from an Intel RealSense depth sensor."""
from __future__ import annotations

import json
import pyorerun
import rerun as rr
import argparse
import numpy as np

from biosiglive import load
from rgbd_mocap.camera.camera_converter import CameraConverter
import os
import glob
import cv2
import csv

def q_from_mot(trc_file):
    rows = []
    with open(trc_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r, row in enumerate(reader):
            if r >= 11:
                rows.append([float(x) for x in row[1:]])
            if r == 10:
                headers = row[1:]
    mot = np.array(rows).T
    q = np.zeros_like(mot)
    q[:3, :] = mot[3:6, :]
    q[3:, :] = np.radians(np.concatenate([mot[:3, :], mot[6:, :]], axis=0))
    return q


def run_realsense(num_frames: int | None, trial=None, part=None) -> None:
    # Visualize the data as RDF
    camera_conf_file = f"D:\Documents\Programmation\pose_estimation\generate_results\camera_config.json"
    model_path = "D:\Documents\Programmation\osim_to_biomod\example\Models\Model_Pose2Sim_scaled.osim"
    # rgbd = RgbdImages(path_to_camera_config_file)
    display_option = pyorerun.DisplayModelOptions()
    display_option.mesh_path = "Geometry_cleaned"
    model = pyorerun.ModelUpdater.from_file(model_path, options=display_option)
    import csv
    rows = None
    count = 0
    file_name = "20250422_162024"
    markers_file = fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons\{file_name}.bio"
    marker_data = load(markers_file)
    markers_names = marker_data["key_point_reduced_names"]
    connections = marker_data["key_points_connections"]
    markers_3d = marker_data["key_points_filtered"]
    mot_file = fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons\ik_mot.mot"
    q = pyorerun.OsimTimeSeries(mot_file, model_path).q_in_radian


    # rr.init("world", spawn=True)
    rr.log("", rr.ViewCoordinates.RDF, static=True
        # timeless=True,
        )
    model.model.options.transparent_mesh = False
    converter = CameraConverter()
    converter.set_intrinsics(camera_conf_file)
    converter.set_extrinsics(camera_conf_file)
    depth_intr = converter.depth
    rgb_intr = converter.depth
    converter.depth_to_color = np.eye(4)
    trans = converter.depth_to_color[:3, 3]
    rot = converter.depth_to_color[:3, :3]

    rr.log(
        "depth/image",
        rr.Pinhole(
            resolution=[depth_intr.width, depth_intr.height],
            focal_length=[depth_intr.fx, depth_intr.fy],
            principal_point=[depth_intr.ppx, depth_intr.ppy],
        ),
        static=True,
    )

    rr.log(
        "rgb",
        rr.Transform3D(
            translation=trans,
            mat3x3=rot,
            from_parent=True,
        ),
        static=True,
    )

    rr.log(
        "rgb/image",
        rr.Pinhole(
            resolution=[rgb_intr.width, rgb_intr.height],
            focal_length=[rgb_intr.fx, rgb_intr.fy],
            principal_point=[rgb_intr.ppx, rgb_intr.ppy],
        ),
        static=True,

    )
    color_path_tmp = fr"F:\CIME_LOC\tmp_videos\{file_name}"
    depth_path_tmp = fr"F:\CIME_LOC\tmp_videos\{file_name}depth"
    key_points_file = fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons"
    all_key_points_files = glob.glob(key_points_file + "/*.json")
    idxs = [int(os.path.basename(file).split("_")[1].split(".")[0]) for file in all_key_points_files]
    idxs = sorted(idxs)
    rr.log(
        "/",
        rr.AnnotationContext(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=1, label="Person"),
                keypoint_annotations=[rr.AnnotationInfo(id=i, label=str(i)) for i in range(25)],
                keypoint_connections=connections,
            )
        ),
        static=True,
    )
    frame_nr = 0
    display_option = pyorerun.DisplayModelOptions()
    display_option.mesh_color = (77, 77, 255)
    while True:
        if frame_nr >= q.shape[1] - 1:
            break
        rr.set_time_sequence("frame_nr", frame_nr)
        depth_image = cv2.imread(depth_path_tmp + f"\depth_{idxs[frame_nr]}.png", cv2.IMREAD_ANYDEPTH)
        with open(os.path.join(key_points_file, f"color_{idxs[frame_nr]}_keypoints.json"), 'r') as f:
            keypoints = json.load(f)
        keypoints = keypoints["people"][0]["pose_keypoints_2d"]
        keypoints = np.array(keypoints).reshape((-1, 3))
        depth_image = np.where(
            (depth_image > 2.2 / converter.depth_scale) | (depth_image <= 1.5 / converter.depth_scale),
            0,
            depth_image,
        )
        color_image = cv2.cvtColor(
            cv2.imread(color_path_tmp + f"\color_{idxs[frame_nr]}.png"), cv2.COLOR_BGR2RGB
        )
        # q, _, _ = msk.compute_inverse_kinematics(keypoints_3d[:, :, None],
        #     method=InverseKinematicsMethods.BiorbdLeastSquare,
        # )
        # except:
        #     print(f"frame {idxs[frame_nr]} not found")
        #     continue
        rr.log("depth/image", rr.DepthImage(depth_image, meter=1 / converter.depth_scale))
        rr.log("rgb/image", rr.Image(color_image))
        rr.log("rgb/image/2d_keypoints", rr.Points2D(keypoints[:, :2],
                                                           # colors=(0, 125, 255),
                                                           radii=4,
                                                           keypoint_ids=list(range(25)), class_ids=1, show_labels=False))
        rr.log("keypoints", rr.Points3D(markers_3d[:, :, frame_nr].T, colors=(0, 125, 255), radii=0.01,
                                              keypoint_ids=list(range(25)),
                                                class_ids=1, show_labels=False))
        # rr.log("world/keypoints markers", rr.Points3D(markers_model[:, :, frame_nr].T, colors=(125, 0, 255),
        #                                       radii=0.01,
        #                                       ))
        # phase_rerun.update_animated_model(q[:, frame_nr])
        model.to_rerun(q[:, frame_nr])
        frame_nr += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Streams frames from a connected realsense depth sensor.")
    parser.add_argument("--num-frames", type=int, default=None, help="The number of frames to log")
    #
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "rerun_example_live_depth_sensor")

    run_realsense(
        args.num_frames,
        "gear_10", "P11")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
