import datetime

import cv2
import numpy as np

# from biosiglive import save

try:
    import pyrealsense2 as rs
except ImportError:
    pass
import glob
import os
from typing import Union
from biosiglive import load, InverseKinematicsMethods, MskFunctions
import time
from markers.marker_set import MarkerSet, Marker
from processing.process_image import ProcessImage
from rgbd_mocap.utils import *
from camera.camera import Camera, CameraConverter
from kinematic_model.kin_model_check import KinematicModelChecker
from processing.config import config
from tracking.test_tracking import print_blobs

class RgbdImages:
    def __init__(
        self,
        conf_file: str = None,
    ):
        """
        Initialize the camera and the images parameters

        Parameters:
        -----------
        merged_images: str
            Path to the file containing the color and depth images. The file must be a .bio or a .bio.gzip file.
            The keys must be "color_images" and "depth_images". If the file is a .bio.gzip file it must have been
            compressed using the compress function from the biosiglive package with merge to False.
        color_images: np.ndarray or str
            The color images. If the images are stored in a file, the path to the file must be given.
             The file must be a .bio or a .bio.gzip file. The keys must be "color_images".
            If the file is a .bio.gzip file it must have been compressed using the compress function from
            the biosiglive package with merge to False.
        depth_images: np.ndarray or str
            The depth images. If the images are stored in a file, the path to the file must be given.
            The file must be a .bio or a .bio.gzip file. The keys must be "depth_images".
            If the file is a .bio.gzip file it must have been compressed using the compress function from
            the biosiglive package with merge to False.
        conf_file: str
            Path to the json file containing the camera configuration which was used to record the images.
            Only used if the images comes from a file and not a live stream.

        """
        self.frame = None
        self.conf_file = conf_file
        self.pipeline = None

        # Camera
        self.camera: Camera = None
        self.converter: CameraConverter = None

        self.frame_idx = 0

        self.config = None
        self.is_camera_init = False

        self.marker_sets = []

        self.ik_method = "least_squares"
        self.markers_to_exclude_for_ik = []

        self.process_image: ProcessImage = None
        self.kinematic_model_checker: KinematicModelChecker = None

    def get_frames(
        self,
        fit_model=False,
        save_data=False,
    ):
        """
        Get the color and depth frames

        Parameters:
        -----------
        cropped: bool
            If True, the frames will be cropped according to the cropping area selected by the user.

        Returns:
        --------
        color_frame: np.ndarray
            Color frame
        depth_frame: np.ndarray
            Depth frame
        """
        if self.process_image.index > self.config['end_index']:
            return False

        self.process_image.process_next_image()
        print(self.process_image.computation_time)

        if self.kinematic_model_checker:
            pos = self.kinematic_model_checker.fit_kinematics_model(self.frame_idx)

            # marker_pos = []
            # for i in range(len(self.process_image.marker_sets)):
            #     marker_pos.append([pos[0][i][0], pos[1][i][0], pos[2][i][0]])

            # marker_pos = np.array(marker_pos, dtype=int)
            print(pos)
            self.process_image.frames.color = print_blobs(self.process_image.frames.color, pos)

            cv2.imshow('test model pos', self.process_image.frames.color)

        for marker_set in self.marker_sets:
            for marker in marker_set:
                if marker.is_visible:
                    marker.set_reliability(0.5)
                if marker.is_depth_visible:
                    marker.set_reliability(0.5)

        if save_data:
            markers_pos, markers_names, occlusions, reliability_idx = self.get_global_markers_pos()
            markers_in_meters, _, _, _ = self.get_global_markers_pos_in_meter(markers_pos)
            dic = {
                "markers_in_meters": markers_in_meters[:, :, np.newaxis],
                "markers_in_pixel": markers_pos[:, :, np.newaxis],
                "markers_names": markers_names,
                "occlusions": occlusions,
                "reliability_idx": reliability_idx,
                "time_to_process": self.time_to_get_frame,
                "camera_frame_idx": self.camera_frame_numbers[self.frame_idx],
                "frame_idx": self.frame_idx,

            }
            save(dic, self.image_dir + os.sep + "markers_pos.bio", add_data=True)

        return True

    def _run_tapir_tracker(self):
        if not self.is_tapir_init:
            self.is_tapir_init = True
            checkpoint_path = (
                r"D:\Documents\Programmation\pose_estimation\pose_est\tapnet\checkpoints\causal_tapir_checkpoint.npy"
            )
            ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
            params, state = ckpt_state["params"], ckpt_state["state"]
            online_init = hk.transform_with_state(build_online_model_init)
            online_init_apply = jax.jit(online_init.apply)

            online_predict = hk.transform_with_state(build_online_model_predict)
            online_predict_apply = jax.jit(online_predict.apply)

            rng = jax.random.PRNGKey(42)
            online_init_apply = functools.partial(online_init_apply, params=params, state=state, rng=rng)
            self.online_predict_apply = functools.partial(online_predict_apply, params=params, state=state, rng=rng)
            if self.color_frame is None:
                self.color_frame = self.color_images[0]
                self.depth_frame = self.depth_images[0]

            self.video_empty_tapir = np.ndarray(
                (1, self.color_frame.shape[0], self.color_frame.shape[1], self.color_frame.shape[2]), dtype=np.uint8
            )
            self.height, self.width = self.video_empty_tapir.shape[1:3]
            self.video_empty_tapir[0] = self.color_frame
            self.resize_height_tapir, self.resize_width_tapir = 256, 256
            frames = media.resize_video(self.video_empty_tapir, (self.resize_height_tapir, self.resize_width_tapir))
            select_points, _, _, _ = self.get_global_markers_pos()
            select_points = select_points[:2].T

            # select_points = self._prepare_data_optical_flow(0)
            query_points = convert_select_points_to_query_points(0, select_points)
            query_points = transforms.convert_grid_coordinates(
                query_points,
                (1, self.height, self.width),
                (1, self.resize_height_tapir, self.resize_width_tapir),
                coordinate_format="tyx",
            )

            self.query_features, _ = online_init_apply(
                frames=preprocess_frames(frames[None, None, 0]), query_points=query_points[None]
            )
            self.causal_state = construct_initial_causal_state(
                query_points.shape[0], len(self.query_features.resolutions) - 1
            )
        self.video_empty_tapir[0] = self.color_frame
        frames = media.resize_video(self.video_empty_tapir, (self.resize_height_tapir, self.resize_width_tapir))

        # for i in tqdm(range(3)):
        (prediction, self.causal_state), _ = self.online_predict_apply(
            frames=preprocess_frames(frames[None, None, 0]),
            query_features=self.query_features,
            causal_context=self.causal_state,
        )
        prediction = prediction["tracks"][0]
        prediction = transforms.convert_grid_coordinates(
            prediction, (self.resize_width_tapir, self.resize_height_tapir), (self.width, self.height)
        )
        return prediction

    def _init_tapir_tracker(self):
        checkpoint_path = (
            r"D:\Documents\Programmation\pose_estimation\pose_est\tapnet\checkpoints\causal_tapir_checkpoint.npy"
        )
        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        params, state = ckpt_state["params"], ckpt_state["state"]
        self.online_init = hk.transform_with_state(build_online_model_init)
        self.online_init_apply = jax.jit(self.online_init.apply)

        self.online_predict = hk.transform_with_state(build_online_model_predict)
        self.online_predict_apply = jax.jit(self.online_predict.apply)

        rng = jax.random.PRNGKey(42)
        self.online_init_apply = functools.partial(self.online_init_apply, params=params, state=state, rng=rng)
        self.online_predict_apply = functools.partial(self.online_predict_apply, params=params, state=state, rng=rng)
        if self.color_frame is None:
            self.color_frame = self.color_images[0]
            self.depth_frame = self.depth_images[0]

        self.video_empty_tapir = np.ndarray(
            (1, self.color_frame.shape[0], self.color_frame.shape[1], self.color_frame.shape[2]), dtype=np.uint8
        )
        self.height, self.width = self.video_empty_tapir.shape[1:3]
        self.video_empty_tapir[0] = self.color_frame
        self.resize_height_tapir, self.resize_width_tapir = 256, 256
        frames = media.resize_video(self.video_empty_tapir, (self.resize_height_tapir, self.resize_width_tapir))
        select_points, _, _, _ = self.get_global_markers_pos()
        select_points = select_points[:2].T

        # select_points = self._prepare_data_optical_flow(0)
        query_points = convert_select_points_to_query_points(0, select_points)
        query_points = transforms.convert_grid_coordinates(
            query_points,
            (1, self.height, self.width),
            (1, self.resize_height_tapir, self.resize_width_tapir),
            coordinate_format="tyx",
        )

        self.query_features, _ = self.online_init_apply(
            frames=preprocess_frames(frames[None, None, 0]), query_points=query_points[None]
        )
        self.causal_state = construct_initial_causal_state(
            query_points.shape[0], len(self.query_features.resolutions) - 1
        )

    def initialize_tracking(
        self,
        config_dict,
        model_name: str = None,
        with_tapir=False,
        build_kinematic_model=True,
    ):
        self.config = config_dict

        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
        self.use_tapir = False

        tracking_options = {
            "naive": False,
            "kalman": True,
            "optical_flow": True,
        }

        ProcessImage.SHOW_IMAGE = True
        ProcessImage.ROTATION = -1
        self.process_image = ProcessImage(config_dict, tracking_options, shared=False)

        self.process_image.process_next_image()

        if with_tapir:
            self.use_tapir = True
            self.is_tapir_init = False
            print(
                "WARNING: "
                "You have selected with Tapir. Please be aware that it could take a lot of GPU or CPU ressources."
                "If you have a GPU, please make sure that you have installed the GPU requirements to be in realtime."
                "If not just be aware that the program will work slowly."
            )

        # build_kinematic_model = False
        if build_kinematic_model:
            path_to_camera_config_file = "/home/user/KaelFacon/Project/rgbd_mocap/config_camera_files/config_camera_P4_session2.json"
            self.converter = CameraConverter()

            self.converter.set_intrinsics(path_to_camera_config_file, self.process_image.frames.depth)
            path = config['directory']
            model_name = f"{path}kinematic_model_{dt_string}.bioMod"
            # model_name = f"kinematic_model_{dt_string}.bioMod" if not model_name else model_name
            self.kinematic_model_checker = KinematicModelChecker(self.process_image.frames,
                                                                 self.process_image.marker_sets,
                                                                 converter=self.converter,
                                                                 model_name=model_name)


def main():
    rgbd = RgbdImages(None)
    rgbd.initialize_tracking(config)

    while rgbd.get_frames():
        continue


if __name__ == '__main__':
    main()
