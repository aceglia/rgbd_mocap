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
from .crops.crop import DepthCheck
from .processing.process_image import print_marker_sets


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
        self.from_dlc = None
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
        self.quasi_static_markers = []
        self.quasi_static_bounds = []
        self.video_object = None

        self.process_image: ProcessImage = None
        self.kinematic_model_checker: KinematicModelChecker = None

        self.tracking_config = None
        self.from_model_tracker = None

        self.build_kinematic_model = False
        self.kin_marker_sets = None
        self.model_name = None
        self.static_markers = None

    def set_static_markers(self, markers):
        if not isinstance(markers, list):
            markers = [markers]
        self.static_markers = markers

    def set_quasi_static_markers(self, markers, bounds=(-15, 15), x_bounds=None, y_bounds=None):
        # if not isinstance(bounds, list):
        #     bounds = [bounds]
        # if len(bounds) != len(markers):
        #     bounds * len(markers)
        if bounds and not x_bounds and not y_bounds:
            x_bounds = y_bounds = bounds
        if not bounds and not x_bounds and not y_bounds:
            raise ValueError("Please provide bounds")
        # if x_bounds
        # assert len(x_bounds) == len(markers), "The number of bounds should be equal to the number of markers"
        if not isinstance(markers, list):
            markers = [markers]
        self.quasi_static_markers = markers
        markers_bounds = []
        for m in range(len(markers)):
            markers_bounds.append([x_bounds[m], y_bounds[m]])
        self.quasi_static_bounds = markers_bounds

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
        video_name=None,
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
        if self.static_markers and not self.process_image.static_markers:
            raise RuntimeError("Static marker should be set before initialization.")
        save_dir = self.tracking_config['directory']
        file_path = file_path if file_path else save_dir + os.sep + "markers_pos_test.bio"
        if not fit_model:
            ProcessImage.SHOW_IMAGE = show_image
        if not self.process_image.process_next_image(process_while_loading=True):
            return False
        fit_model_time = 0
        process_image = None
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
                self.kinematic_model_checker.markers_to_exclude = self.markers_to_exclude_for_ik
            self.kinematic_model_checker.fit_kinematics_model(self.process_image)
            fit_model_time = time.time() - tic
            process_image = None
            if show_image:
                im = cv2.cvtColor(self.process_image.frames.color, cv2.COLOR_GRAY2RGB)
                process_image = print_marker_sets(im, self.kinematic_model_checker.marker_sets)
                for marker_set in self.kinematic_model_checker.marker_sets:
                    for marker in marker_set:
                        if marker.is_bounded:
                            x_offset, y_offset = marker.crop_offset
                            process_image = cv2.rectangle(process_image, (marker.x_bounds.min + x_offset, marker.y_bounds.min + y_offset),
                                          (marker.x_bounds.max + x_offset, marker.y_bounds.max + y_offset), (255, 0, 0),
                                                          1)

                cv2.namedWindow('Main image :', cv2.WINDOW_NORMAL)
                cv2.putText(
                    process_image,
                    f"Frame : {self.process_image.index}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow('Main image :', process_image)
                if cv2.waitKey(1) == ord('q'):
                    return False

        for marker_set in self.marker_sets:
            for marker in marker_set:
                if marker.get_visibility():
                    marker.set_reliability(0.5)
                if marker.is_depth_visible:
                    marker.set_reliability(0.5)
        if save_data:
            if self.iter == 0 and os.path.isfile(file_path):
                os.remove(file_path)
            markers_pos, markers_names, occlusions = self._get_all_markers()
            markers_in_meter = self.converter.get_markers_pos_in_meter(markers_pos)

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
            video_name = video_name if video_name else "images_processed"
            video_path = save_dir + os.sep + video_name + ".avi"
            if self.iter == 0 and os.path.isfile(video_path):
                os.remove(video_path)

            if process_image is None:
                im = cv2.cvtColor(self.process_image.frames.color, cv2.COLOR_GRAY2RGB)
                process_image = print_marker_sets(im, self.kinematic_model_checker.marker_sets)
                for marker_set in self.kinematic_model_checker.marker_sets:
                    for marker in marker_set:
                        if marker.is_bounded:
                            x_offset, y_offset = marker.crop_offset
                            process_image = cv2.rectangle(process_image, (
                            marker.x_bounds.min + x_offset, marker.y_bounds.min + y_offset),
                                                          (marker.x_bounds.max + x_offset,
                                                           marker.y_bounds.max + y_offset), (255, 0, 0),
                                                          1)
            cv2.putText(
                process_image,
                f"Frame : {self.process_image.index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            self.video_object = _save_video(process_image,
                                            (process_image.shape[1], process_image.shape[0]),
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
        use_optical_flow=False,
        images_path=None,
        static_markers=None,
        from_dlc=False,
        dlc_model_path=None,
        marker_names=None,
        processor=None
    ):
        self.from_dlc = from_dlc
        self.static_markers = static_markers if static_markers else self.static_markers
        self.tracking_config = {} if not tracking_config_dict else load_json(tracking_config_dict)
        if from_dlc:
            multi_processing = False
            if not images_path:
                raise ValueError("Please provide the path to the images when using DLC interface.")
            if use_optical_flow:
                raise ValueError("Optical flow is not available when using DLC interface.")
            if not dlc_model_path:
                raise ValueError("Please provide the path to the model when using DLC interface.")

        if images_path:
            self.tracking_config["directory"] = images_path
        elif not tracking_config_dict:
            raise ValueError("Please provide the path to the images or the tracking config file.")
        if marker_names is None and tracking_config_dict is None:
            raise ValueError("Please provide the marker names or the tracking config file with marker names.")

        self.tracking_config["depth_scale"] = self.converter.depth_scale
        self.iter = 0
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")

        tracking_options = {
            "kalman": use_kalman,
            "optical_flow": use_optical_flow,
        }

        self.process_image = ProcessImage(self.tracking_config, tracking_options, self.static_markers,
                                          multi_processing=multi_processing,
                                          bounded_markers=[self.quasi_static_markers, self.quasi_static_bounds],
                                          from_dlc=self.from_dlc,
                                          dlc_model_path=dlc_model_path,
                                          processor=processor,
                                          marker_names=marker_names)
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

