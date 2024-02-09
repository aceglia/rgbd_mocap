import datetime
import os
import cv2
import time

from biosiglive import save
try:
    import pyrealsense2 as rs
except ImportError:
    pass

import numpy as np
from .processing.process_image import ProcessImage
from .utils import _save_video
from .camera.camera import Camera, CameraConverter
from .kinematic_model_checker.kin_model_check import KinematicModelChecker
from .processing.config import load_json
from .crop.crop import DepthCheck


class RgbdImages:
    def __init__(
        self,
        camera_conf_file: str = None,
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
        self.video_object = None
        self.frame = None
        self.camera_conf_file = camera_conf_file
        self.pipeline = None

        # Camera
        self.camera: Camera = None
        self.converter = CameraConverter()
        self.converter.set_intrinsics(self.camera_conf_file)
        if self.converter.depth_scale != DepthCheck.DEPTH_SCALE:
            DepthCheck.set_depth_scale(self.converter.depth_scale)

        self.iter = 0

        self.is_camera_init = False

        self.marker_sets = []

        self.ik_method = "least_squares"
        self.markers_to_exclude_for_ik = []
        self.video_object = None

        self.process_image: ProcessImage = None
        self.kinematic_model_checker: KinematicModelChecker = None

        self.tracking_config = None
        self.from_model_tracker = None

        self.build_kinematic_model = False
        self.kin_marker_sets = None
        self.model_name = None

    def _get_all_markers(self):
        markers_pos = []
        markers_name = []
        markers_visibility = []

        for crop in self.process_image.crops:
            marker_set = crop.marker_set
            markers_pos += marker_set.get_markers_global_pos_3d()
            markers_name += marker_set.get_markers_names()
            markers_visibility += marker_set.get_markers_occlusion()
        return markers_pos, markers_name, markers_visibility

    def _get_global_markers_pos_in_meter(self):
        markers_pos, markers_name, markers_visibility = self._get_all_markers()
        return self.converter.get_markers_pos_in_meter(markers_pos), markers_name, markers_visibility

    def get_frames(
        self,
        fit_model=False,
        save_data=False,
        show_image=False,
        file_path=None,
        save_video=False,
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
        save_dir = self.tracking_config['directory']
        file_path = file_path if file_path else save_dir + os.sep + "markers_pos_test.bio"
        ProcessImage.SHOW_IMAGE = show_image
        if self.process_image.index > self.tracking_config['end_index']:
            return False

        self.process_image.process_next_image()
        fit_model_time = 0
        if fit_model:
            tic = time.time()
            if not self.kinematic_model_checker:
                self.kinematic_model_checker = KinematicModelChecker(self.process_image.frames,
                                                                     self.process_image.marker_sets,
                                                                     converter=self.converter,
                                                                     model_name=self.model_name,
                                                                     build_model=self.build_kinematic_model,
                                                                     kin_marker_set=self.kin_marker_sets)
            self.kinematic_model_checker.ik_method = "kalman"
            self.kinematic_model_checker.fit_kinematics_model(self.process_image)
            fit_model_time = time.time() - tic

        for marker_set in self.marker_sets:
            for marker in marker_set:
                if marker.is_visible:
                    marker.set_reliability(0.5)
                if marker.is_depth_visible:
                    marker.set_reliability(0.5)
        markers_pos, markers_names, occlusions = self._get_all_markers()
        print(occlusions)
        if save_data:
            markers_pos, markers_names, occlusions = self._get_all_markers()
            markers_in_meter = self.converter.get_markers_pos_in_meter(markers_pos)
            print(occlusions)
            dic = {
                "markers_in_meters": markers_in_meter[:, :, np.newaxis],
                "markers_in_pixel": np.array(markers_pos).T[:, :, np.newaxis],
                "markers_names": markers_names,
                "occlusions": occlusions,
                "time_to_process": self.process_image.computation_time + fit_model_time,
                "iteration": self.iter,
                "frame_idx": self.process_image.index,

            }
            save(dic, file_path, add_data=True)
        if save_video:
            video_path = save_dir + os.sep + "images_processed.avi"
            if self.iter == 0 and os.path.isfile(video_path):
                os.remove(video_path)
            color_for_video = self.process_image.get_processed_image()
            cv2.putText(
                color_for_video,
                f"Frame : {self.process_image.index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            self.video_object = _save_video(color_for_video,
                                            (color_for_video.shape[1], color_for_video.shape[0]),
                                            video_path,
                                            self.converter.color.fps,
                                            self.video_object)
        self.iter += 1
        return True

    def initialize_tracking(
        self,
        tracking_config_dict=None,
        model_name: str = None,
        build_kinematic_model=False,
        multi_processing=False,
        kin_marker_set=None,
        use_kalman=True,
        use_optical_flow=True,
    ):
        self.tracking_config = load_json(tracking_config_dict)
        self.tracking_config["depth_scale"] = self.converter.depth_scale
        self.iter = 0
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")

        tracking_options = {
            "naive": not use_kalman and not use_optical_flow,
            "kalman": use_kalman,
            "optical_flow": use_optical_flow,
        }

        self.process_image = ProcessImage(self.tracking_config, tracking_options, multi_processing=multi_processing)
        self.model_name = self.tracking_config['directory'] + os.sep + model_name if model_name else None
        self.build_kinematic_model = build_kinematic_model
        if build_kinematic_model:
            if kin_marker_set is None:
                raise ValueError("Please provide a set of markers to build the kinematic model")
            self.kin_marker_sets = kin_marker_set
            path = self.tracking_config['directory']
            self.model_name = f"{path}/kinematic_model_{dt_string}.bioMod" if not self.model_name else self.model_name

    def set_marker_to_exclude(self, markers):
        self.markers_to_exclude_for_ik = markers

