import numpy as np
import datetime
try:
    import pyrealsense2 as rs
except ImportError:
    pass
import json
import glob
import os
from typing import Union
from biosiglive import load, InverseKinematicsMethods, MskFunctions
import cv2
from .enums import *
import time
from .marker_class import MarkerSet
from .utils import *

try:
    import biorbd
    is_biorbd_package = True
except ImportError:
    is_biorbd_package = False

from .model_creation import (
    BiomechanicalModel,
    C3dData,
    Marker,
    Segment,
    SegmentReal,
    SegmentCoordinateSystem,
    Translations,
    Rotations,
    Axis,
    Mesh
)


class RgbdImages:
    def __init__(
        self,
        merged_images: str = None,
        images_dir = None,
        color_images: Union[np.ndarray, str] = None,
        depth_images: Union[np.ndarray, str] = None,
        conf_file: str = None,
        start_index: int = 0,
        stop_index: int = None,
        downsampled: int = 1,
        load_all_dir: bool = False,
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

        # Camera intrinsics
        self.depth_fx_fy = None
        self.depth_ppx_ppy = None
        self.color_fx_fy = None
        self.intrinsics_color_mat = None
        self.intrinsics_depth_mat = None
        self.color_ppx_ppy = None
        self.color_model = None
        self.depth_model = None
        self.color_dist_coeffs = None
        self.depth_dist_coeffs = None
        self.depth_scale = None
        self.frame_idx = 0
        self.optical_flow_params = None
        self.blobs = []
        self.downsampled = downsampled
        self.kinematics_functions = None
        self.kinematic_model = None
        self.is_kinematic_model = False
        self.tracking_file_loaded = False
        self.time_to_get_frame = -1

        # Camera extrinsic
        self.depth_to_color = None

        # Cropping
        self.cropping = False
        self.is_cropped = None
        self.start_crop = None
        self.color_cropped = None
        self.depth_cropped = None
        self.last_color_cropped = None
        self.last_depth_cropped = None
        self.end_crop = None
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0

        # clipping
        self.is_frame_clipped = False
        self.clipping_distance_in_meters = []

        self.conf_data = None
        self.is_camera_init = False
        self.is_frame_aligned = False
        self.align = None
        self.color_images = None
        self.depth_images = None
        self.color_frame = None
        self.last_color_frame = None
        self.last_depth_frame = None
        self.depth_frame = None
        self.upper_bound = []
        self.lower_bound = []
        self.marker_sets = []
        self.is_tracking_init = False
        self.first_frame_markers = None
        self.mask_params = None
        self.rotation_angle = None
        self.is_frame_rotate = None
        self.start_index = start_index
        self.clipping_color = 0
        self.ik_method = "least_squares"
        self.markers_to_exclude_for_ik = []

        if self.conf_file:
            self.conf_data = get_conf_data(self.conf_file)
            self._set_images_params()

        if isinstance(color_images, str):
            self.color_images = self._load(color_images, "color")
        if isinstance(depth_images, str):
            self.depth_images = self._load(depth_images, "depth")
        if isinstance(color_images, np.ndarray):
            if color_images.ndim != 3:
                raise ValueError("The color images must be a 3D array of size (height, width, 3)")
            self.color_images = color_images
        if isinstance(depth_images, np.ndarray):
            if depth_images.ndim != 2:
                raise ValueError("The depth images must be a 2D array of size (height, width)")
            self.depth_images = depth_images
        if isinstance(merged_images, str):
            self.color_images, self.depth_images = self._load(merged_images, "both")
            self.color_images = self.color_images[:200]
            self.depth_images = self.depth_images[:200]
        self.load_all_dir = load_all_dir
        self.stop_index = stop_index
        self.color_images, self.depth_images = self._load_from_dir(images_dir, self.start_index, self.stop_index,
                                                                   self.downsampled,
                                                                   load_all_dir)

    @staticmethod
    def _load(path, data_type: str = "both") -> np.ndarray or tuple:
        if path.endswith(".bio.gzip"):
            data = load(path, merge=False, number_of_line=50)
        elif path.endswith(".bio"):
            data = load(path, merge=False)
        else:
            raise ValueError("The file must be a .bio or a .bio.gzip file.")
        if data_type == "both":
            data_end = [], []
        else:
            data_end = []
        for dic in data[0]:
            if data_type in ["color", "depth"]:
                for key in dic.keys():
                    if data_type in key:
                        data_end.append(dic[key])

            elif data_type == "both":
                for key in dic.keys():
                    if "color" in key:
                        data_end[0].append(dic[key])
                    elif "depth" in key:
                        data_end[1].append(dic[key])
            else:
                raise ValueError("The data type must be 'color', 'depth' or 'both'.")
        return data_end

    def _load_from_dir(self, path=None, start_index=None, end_index=None, down_sampling=None,
                       load_all_dir=False,
                       all_color_files=None,
                       all_depth_files=None) -> np.ndarray or tuple:

        if path:
            all_color_files_tmp = glob.glob(path + "/color*.png")
            idx = []
            for file in all_color_files_tmp:
                idx.append(int(file.split("\\")[-1].split("_")[-1].removesuffix(".png")))
            idx.sort()
        if all_color_files is None:
            self.all_color_files = [path + f"/color_{i}.png" for i in idx]
        if all_depth_files is None:
            self.all_depth_files = [path + f"/depth_{i}.png" for i in idx]

        start_index = 0 if not start_index else int(start_index)
        self.stop_index = len(self.all_color_files) if not end_index else int(end_index)
        if load_all_dir:
            end_index_tmp = self.stop_index
        else:
            end_index_tmp = start_index + 1
        down_sampling_factor = 1 if not down_sampling else int(down_sampling)
        color_images = [cv2.imread(file) for file in self.all_color_files[start_index:end_index_tmp:down_sampling_factor]]
        depth_images = [cv2.imread(file, cv2.IMREAD_ANYDEPTH) for file in self.all_depth_files[start_index:end_index_tmp:down_sampling_factor]]
        return color_images, depth_images

    def init_camera(self, color_size: Union[tuple, ColorResolution], depth_size: Union[tuple, DepthResolution],
                    color_fps: Union[int, FrameRate], depth_fps: Union[int, FrameRate], align: bool = False):
        """
        Initialize the camera and the images parameters

        Parameters:
        -----------
        color_size: tuple
            Size of the color image (width, height)
        depth_size: tuple
            Size of the depth image (width, height)
        color_fps: int
            Frame per second of the color image
        depth_fps: int
            Frame per second of the depth image
        align: bool
            If True, the depth and color images will be aligned.
            This can slow down the process.
        """
        if isinstance(depth_size, DepthResolution):
            depth_size = depth_size.value
        if isinstance(color_size, ColorResolution):
            color_size = color_size.value
        if isinstance(color_fps, FrameRate):
            color_fps = color_fps.value
        if isinstance(depth_fps, FrameRate):
            depth_fps = depth_fps.value
        if self.color_images or self.depth_images:
            raise ValueError("The camera can't be initialized if the images are loaded from a file.")
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                break
        if not found_rgb:
            print("The program requires Depth camera with Color sensor")
            exit(0)
        config.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, depth_fps)
        config.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, color_fps)
        self.pipeline.start(config)

        if align:
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            self.is_frame_aligned = True

        set_conf_file_from_camera(self.pipeline, device)
        self.conf_data = get_conf_data("camera_conf.json")
        self._set_images_params()
        self.is_camera_init = True

    def _set_intrinsics_from_file(self):
        self.depth_fx_fy = self.conf_data["depth_fx_fy"]
        self.depth_ppx_ppy = self.conf_data["depth_ppx_ppy"]
        self.color_fx_fy = self.conf_data["color_fx_fy"]
        self.color_ppx_ppy = self.conf_data["color_ppx_ppy"]

        self.intrinsics_depth_mat = np.array(
            [
                [self.depth_fx_fy[0], 0, self.depth_ppx_ppy[0]],
                [0, self.depth_fx_fy[1], self.depth_ppx_ppy[0]],
                [0, 0, 1],
            ],
            dtype=float,
        )
        self.intrinsics_color_mat = np.array(
            [
                [self.color_fx_fy[0], 0, self.color_ppx_ppy[0]],
                [0, self.color_fx_fy[1], self.color_ppx_ppy[0]],
                [0, 0, 1],
            ],
            dtype=float,
        )

    def _set_extrinsic_from_file(self):
        self.depth_to_color = np.eye(4)
        self.depth_to_color[:3, :3] = self.conf_data["depth_to_color_rot"]
        self.depth_to_color[:3, 3] = self.conf_data["depth_to_color_trans"]

    def _set_depth_scale_from_file(self):
        self.depth_scale = self.conf_data["depth_scale"]

    def _set_images_params(self):
        self._set_intrinsics_from_file()
        self._set_extrinsic_from_file()
        self._set_depth_scale_from_file()

    def _crop_frames(self, color, depth):
        color_cropped = []
        depth_cropped = []
        i = -1
        for i in range(len(self.start_crop[0])):
            # self._adapt_cropping(color, i)
            color_cropped.append(
                color[self.start_crop[1][i] : self.end_crop[1][i], self.start_crop[0][i] : self.end_crop[0][i], :]
            )
            depth_cropped.append(
                depth[self.start_crop[1][i] : self.end_crop[1][i], self.start_crop[0][i] : self.end_crop[0][i]]
            )
        if i == -1:
            return [color], [depth]
        return color_cropped, depth_cropped

    def _clip_frames(self, color, depth):
        color_clipped = []
        depth_clipped = []
        for i in range(len(color)):
            clipping_distance = self.clipping_distance_in_meters[i] / self.depth_scale
            depth_clipped.append(np.where((depth[i] > clipping_distance), -1, depth[i]))
            depth_image_3d = np.dstack((depth_clipped[i], depth_clipped[i], depth_clipped[i]))
            color_clipped.append(np.where((depth_image_3d <= 0), self.clipping_color, color[i]))

        return color_clipped, depth_clipped

    def _align_images(self, color, depth):
        return cv2.rgbd.registerDepth(
            self.intrinsics_depth_mat,
            self.intrinsics_color_mat,
            None,
            self.depth_to_color,
            depth,
            (color.shape[1], color.shape[0]),
            False,
        )

    def _get_frame_from_source(self):
        if self.is_camera_init:
            frames = self.pipeline.wait_for_frames()
            if self.is_frame_aligned:
                frames = self.align.process(frames)
            self.depth_frame = frames.get_depth_frame()
            self.color_frame = frames.get_color_frame()
            if not self.depth_frame or not self.color_frame:
                return None, None
            self.depth_frame = np.asanyarray(self.depth_frame.get_data())
            self.color_frame = np.asanyarray(self.color_frame.get_data())
        elif self.color_images and self.depth_images:
            if self.load_all_dir:
                if self.frame_idx == len(self.all_color_files) - 1:
                    self.frame_idx = 0
                    print("starting over...")
                self.color_frame, self.depth_frame = self.color_images[self.frame_idx], self.depth_images[self.frame_idx]
            else:
                if self.frame_idx == self.stop_index - 1:
                    self.frame_idx = 0
                    print("starting over...")
                self.color_frame, self.depth_frame = self._load_from_dir(start_index=self.start_index + self.frame_idx,
                                                                         all_color_files=self.all_color_files,
                                                                         all_depth_files=self.all_depth_files
                                                                         )
                self.color_frame = self.color_frame[0]
                self.depth_frame = self.depth_frame[0]

            if self.is_frame_aligned:
                self.depth_frame = self._align_images(self.color_frame, self.depth_frame)
        else:
            raise ValueError("Camera is not initialized and images are not loaded from a file.")
        if self.rotation_angle:
            # self.is_frame_rotate = True
            self.color_frame, self.depth_frame = rotate_frame(self.color_frame, self.depth_frame, self.rotation_angle)
        self.color_cropped, self.depth_cropped = self._crop_frames(self.color_frame, self.depth_frame)
        if self.is_frame_clipped:
            _, self.depth_cropped = self._clip_frames(self.color_cropped, self.depth_cropped)
        if self.first_frame_markers and self.frame_idx == 0:
            for i in range(len(self.first_frame_markers)):
                self._distribute_pos_markers(i)

        # self.frame_idx += 1

    def _distribute_pos_markers_tapir(self, pos_markers):
        pos_markers = np.array(pos_markers[:, 0, :].T, dtype=int)
        count = 0
        for i in range(len(self.marker_sets)):
            for m in range(self.marker_sets[i].nb_markers):
                self.marker_sets[i].markers[m].pos[:2] = self.express_in_local([pos_markers[0, count],
                                                                            pos_markers[1, count]],
                                                                           [self.start_crop[0][i], self.start_crop[1][i]])
                count += 1
            self.marker_sets[i].set_global_markers_pos(self.marker_sets[i].get_markers_pos(),
                                                       [self.start_crop[0][i], self.start_crop[1][i]])

    def _partial_get_frame(self, detect_blobs, color_list, i, depth, kwargs, label_markers, filter_with_kalman,
                            adjust_with_blobs, fit_model):
        if detect_blobs:
            # if bounds_from_marker_pos:
            #     if np.all(self.marker_sets[i].get_markers_pos() == 0):
            #         markers_values = self.marker_sets[i].get_markers_pos()
            #     else:
            #         markers_values = np.concatenate((self.marker_sets[i].get_markers_filtered_pos(),
            #                             self.marker_sets[i].get_markers_pos()),
            #                             axis=1)
            #     color_list[i], (x_min, x_max, y_min, y_max) = bounding_rect(
            #         color_list[i], markers_values,
            #         color=(0, 255, 0), delta=20,
            #     )
            # else:
            x_min, x_max, y_min, y_max = 0, color_list[i].shape[1], 0, color_list[i].shape[0]
            # self._adapt_cropping(color, x_min, x_max, y_min, y_max, i)
            self.blobs.append(
                get_blobs(
                    color_list[i],
                    params=self.mask_params[i],
                    return_centers=True,
                    image_bounds=(x_min, x_max, y_min, y_max),
                    depth=depth[i], clipping_color=self.clipping_color, depth_scale=self.depth_scale,
                    **kwargs,
                )
            )

        if label_markers:
            old_markers_pos, markers_visible_names = self._prepare_data_optical_flow(i)
            if len(old_markers_pos) != 0:
                error_threshold = 10
                current_color = background_remover(self.color_cropped[i],
                                                   self.depth_cropped[i],
                                                   1.9,
                                                   self.depth_scale,
                                                   100)

                previous_color = background_remover(self.last_color_cropped[i],
                                                    self.last_depth_cropped[i],
                                                    1.9,
                                                    self.depth_scale,
                                                    100)
                self._run_optical_flow(
                    i,
                    current_color,
                    previous_color,
                    old_markers_pos,
                    filter_with_kalman,
                    adjust_with_blobs,
                    markers_visible_names,
                    error_threshold,
                    use_tapir=self.use_tapir,
                    # markers_pos_tapir=markers_pos_tapir,
                )
            if not fit_model:
                color_list[i] = draw_markers(
                    self.color_cropped[i],
                    # markers_filtered_pos=self.marker_sets[i].get_markers_filtered_pos(),
                    markers_pos=self.marker_sets[i].get_markers_pos(),
                    markers_names=self.marker_sets[i].marker_names,
                    is_visible=self.marker_sets[i].get_markers_occlusion(),
                    scaling_factor=0.5,
                    markers_reliability_index=self.marker_sets[i].get_markers_reliability_index(self.frame_idx),
                )
                if len(self.blobs) != 0:
                    if len(self.blobs[i]) != 0:
                        color_list[i] = draw_blobs(color_list[i], self.blobs[i])

    def get_frames(
        self,
        aligned: bool = False,
        detect_blobs=False,
        label_markers=False,
        bounds_from_marker_pos=False,
        adjust_with_blobs=False,
        filter_with_kalman=False,
        fit_model=False,
        model_name=None,
        rotation_angle: Rotation = None,
        **kwargs,
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
        tic = time.time()
        self.rotation_angle = rotation_angle
        if adjust_with_blobs and not label_markers:
            raise ValueError("You need to label markers before adjusting them with blobs.")
        if adjust_with_blobs and not detect_blobs:
            raise ValueError("You need to detect blobs before adjusting markers with them.")
        if filter_with_kalman and not label_markers:
            raise ValueError("You need to label markers before filtering them with Kalman.")
        self.is_frame_aligned = aligned
        if self.color_frame is not None and self.depth_cropped is not None:
            self.last_color_frame = self.color_frame.copy()
            self.last_depth_frame = self.depth_frame.copy()
            self.last_depth_cropped = self.depth_cropped.copy()
            self.last_color_cropped = self.color_cropped.copy()
        _time_to_rotate = time.time() - tic

        self._get_frame_from_source()
        tic = time.time()
        if self.last_color_frame is None:
            self.last_color_frame = self.color_frame.copy()
            self.last_depth_frame = self.depth_frame.copy()
            self.last_depth_cropped = self.depth_cropped.copy()
            self.last_color_cropped = self.color_cropped.copy()
        self.blobs = []
        color_list = []
        depth = self.depth_cropped
        if self.use_tapir and label_markers:
            markers_pos_tapir = self._run_tapir_tracker()
            self._distribute_pos_markers_tapir(markers_pos_tapir)
        for i, color in enumerate(self.color_cropped):
            color_list.append(color.copy())
            self._partial_get_frame(detect_blobs, color_list, i, depth, kwargs, label_markers, filter_with_kalman,
                            adjust_with_blobs, fit_model)
            # if detect_blobs:
            #     # if bounds_from_marker_pos:
            #     #     if np.all(self.marker_sets[i].get_markers_pos() == 0):
            #     #         markers_values = self.marker_sets[i].get_markers_pos()
            #     #     else:
            #     #         markers_values = np.concatenate((self.marker_sets[i].get_markers_filtered_pos(),
            #     #                             self.marker_sets[i].get_markers_pos()),
            #     #                             axis=1)
            #     #     color_list[i], (x_min, x_max, y_min, y_max) = bounding_rect(
            #     #         color_list[i], markers_values,
            #     #         color=(0, 255, 0), delta=20,
            #     #     )
            #     # else:
            #     x_min, x_max, y_min, y_max = 0, color_list[i].shape[1], 0, color_list[i].shape[0]
            #     # self._adapt_cropping(color, x_min, x_max, y_min, y_max, i)
            #     self.blobs.append(
            #         get_blobs(
            #             color_list[i],
            #             params=self.mask_params[i],
            #             return_centers=True,
            #             image_bounds=(x_min, x_max, y_min, y_max),
            #             depth=depth[i], clipping_color=self.clipping_color, depth_scale=self.depth_scale,
            #             **kwargs,
            #         )
            #     )
            #
            # if label_markers:
            #     old_markers_pos, markers_visible_names = self._prepare_data_optical_flow(i)
            #     if len(old_markers_pos) != 0:
            #         error_threshold = 10
            #         current_color = background_remover(self.color_cropped[i],
            #                                            self.depth_cropped[i],
            #                                            1.9,
            #                                            self.depth_scale,
            #                                            100)
            #
            #         previous_color = background_remover(self.last_color_cropped[i],
            #                                             self.last_depth_cropped[i],
            #                                             1.9,
            #                                             self.depth_scale,
            #                                             100)
            #         self._run_optical_flow(
            #                                i,
            #                                current_color,
            #                                previous_color,
            #                                old_markers_pos,
            #                                filter_with_kalman,
            #                                adjust_with_blobs,
            #                                markers_visible_names,
            #                                error_threshold,
            #                                use_tapir=self.use_tapir,
            #                                # markers_pos_tapir=markers_pos_tapir,
            #                                )
            #     if not fit_model:
            #         color_list[i] = draw_markers(
            #             self.color_cropped[i],
            #             # markers_filtered_pos=self.marker_sets[i].get_markers_filtered_pos(),
            #             markers_pos=self.marker_sets[i].get_markers_pos(),
            #             markers_names=self.marker_sets[i].marker_names,
            #             is_visible=self.marker_sets[i].get_markers_occlusion(),
            #             scaling_factor=0.5,
            #             markers_reliability_index=self.marker_sets[i].get_markers_reliability_index(self.frame_idx),
            #         )
            #         if len(self.blobs) != 0:
            #             if len(self.blobs[i]) != 0:
            #                 color_list[i] = draw_blobs(color_list[i], self.blobs[i])
        if fit_model:
            if not label_markers:
                raise ValueError("You need to label markers before fitting the model.")
            color_list = self._fit_kinematics_model(model_name, color_list)
        for marker_set in self.marker_sets:
            for marker in marker_set.markers:
                if marker.is_visible:
                    marker.reliability_index += 0.5
                if marker.is_depth_visible:
                    marker.reliability_index += 0.5
        self.frame_idx += 1
        self.time_to_get_frame = (time.time() - tic) + _time_to_rotate
        return color_list, depth

    def _run_tapir_tracker(self):
        if not self.is_tapir_init:
            self.is_tapir_init = True
            checkpoint_path = r'D:\Documents\Programmation\pose_estimation\pose_est\tapnet\checkpoints\causal_tapir_checkpoint.npy'
            ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
            params, state = ckpt_state['params'], ckpt_state['state']
            online_init = hk.transform_with_state(build_online_model_init)
            online_init_apply = jax.jit(online_init.apply)

            online_predict = hk.transform_with_state(build_online_model_predict)
            online_predict_apply = jax.jit(online_predict.apply)

            rng = jax.random.PRNGKey(42)
            online_init_apply = functools.partial(
                online_init_apply, params=params, state=state, rng=rng
            )
            self.online_predict_apply = functools.partial(
                online_predict_apply, params=params, state=state, rng=rng
            )
            if self.color_frame is None:
                self.color_frame = self.color_images[0]
                self.depth_frame = self.depth_images[0]

            self.video_empty_tapir = np.ndarray((1, self.color_frame.shape[0], self.color_frame.shape[1], self.color_frame.shape[2]),
                               dtype=np.uint8)
            self.height, self.width = self.video_empty_tapir.shape[1:3]
            self.video_empty_tapir[0] = self.color_frame
            self.resize_height_tapir, self.resize_width_tapir = 256, 256
            frames = media.resize_video(self.video_empty_tapir, (self.resize_height_tapir, self.resize_width_tapir))
            select_points, _, _, _ = self.get_global_markers_pos()
            select_points = select_points[:2].T

            # select_points = self._prepare_data_optical_flow(0)
            query_points = convert_select_points_to_query_points(0, select_points)
            query_points = transforms.convert_grid_coordinates(
                query_points, (1, self.height, self.width),
                (1, self.resize_height_tapir, self.resize_width_tapir), coordinate_format='tyx')

            self.query_features, _ = online_init_apply(frames=preprocess_frames(frames[None, None, 0]),
                                                  query_points=query_points[None])
            self.causal_state = construct_initial_causal_state(query_points.shape[0], len(self.query_features.resolutions) - 1)
        self.video_empty_tapir[0] = self.color_frame
        frames = media.resize_video(self.video_empty_tapir, (self.resize_height_tapir, self.resize_width_tapir))

        # for i in tqdm(range(3)):
        (prediction, self.causal_state), _ = self.online_predict_apply(
            frames=preprocess_frames(frames[None, None, 0]),
            query_features=self.query_features,
            causal_context=self.causal_state,
        )
        prediction = prediction["tracks"][0]
        prediction = transforms.convert_grid_coordinates(prediction, (self.resize_width_tapir, self.resize_height_tapir)
                                                         , (self.width, self.height))
        return prediction

    def _fit_kinematics_model(self, model_name, color_list):
        if not model_name:
            raise ValueError("You need to specify a model name to fit the model.")
        if not os.path.isfile(model_name):
            raise ValueError("The model file does not exist. Please initialize the model creation before.")
        if self.kinematics_functions is None or self.frame_idx == 0:
            self.kinematics_functions = MskFunctions(model_name, 1)
        markers, names, is_visible, _ = self.get_global_markers_pos()
        markers, _, _, _ = self.get_global_markers_pos_in_meter(markers)
        final_markers = np.full((markers.shape[0], markers.shape[1], 1), np.nan)
        for m in range(final_markers.shape[1]):
            # if is_visible[m]:
            if names[m] not in self.markers_to_exclude_for_ik:
                final_markers[:, m, 0] = markers[:, m]
        _method = InverseKinematicsMethods.BiorbdLeastSquare if self.ik_method == "least_squares" else InverseKinematicsMethods.BiorbdKalman
        q, _ = self.kinematics_functions.compute_inverse_kinematics(
            final_markers,
            _method,
            kalman_freq=100
        )
        markers = self.kinematics_functions.compute_direct_kinematics(q)
        list_nb_markers = []
        for i in range(len(self.marker_sets)):
            list_nb_markers.append(len(self.marker_sets[i].markers))
        count = 0
        idx = 0
        markers_kalman = []
        dist_list = []
        _in_pixel = self.express_in_pixel(markers[:, :, 0])
        for i in range(markers.shape[1]):
            markers_local = np.array(self.express_in_local(
                _in_pixel[:, i], [self.start_crop[0][idx], self.start_crop[1][idx]]))
            markers_kalman.append(markers_local)
            blob_center, is_visible_tmp, dist = find_closest_blob(markers_local, self.blobs[idx],
                                                            delta=8, return_distance=True)
            dist_list.append(dist)
            threshold = 30
            self.marker_sets[idx].markers[count].is_visible = is_visible_tmp
            if is_visible_tmp:
                self.marker_sets[idx].markers[count].correct_from_kalman(blob_center)
                self.marker_sets[idx].markers[count].pos[:2] = blob_center

            else:
                self.marker_sets[idx].markers[count].correct_from_kalman(markers_local)
                self.marker_sets[idx].markers[count].pos[:2] = markers_local

            if self.marker_sets[idx].markers[count].pos[0] and int(
                    self.marker_sets[idx].markers[count].pos[0]) not in list(
                    range(0, self.color_cropped[idx].shape[1] - 1)):
                self.marker_sets[idx].markers[count].pos[0] = self.color_cropped[idx].shape[1] - 1 if \
                self.marker_sets[idx].markers[count].pos[0] > self.color_cropped[idx].shape[1] - 1 else 0
            if self.marker_sets[idx].markers[count].pos[1] and int(
                    self.marker_sets[idx].markers[count].pos[1]) not in list(
                    range(0, self.color_cropped[idx].shape[0] - 1)):
                self.marker_sets[idx].markers[count].pos[1] = self.color_cropped[idx].shape[0] - 1 if \
                self.marker_sets[idx].markers[count].pos[1] > self.color_cropped[idx].shape[0] - 1 else 0

            marker_depth, is_depth_visible = check_and_attribute_depth(self.marker_sets[idx].markers[count].pos[:2],
                                                                       self.depth_cropped[idx],
                                                                       depth_scale=self.depth_scale)
            if is_depth_visible:
                self.marker_sets[idx].markers[count].pos[2] = marker_depth
                self.marker_sets[idx].markers[count].is_depth_visible = True
            else:
                self.marker_sets[idx].markers[count].pos[2] = markers[2, i, 0]
                self.marker_sets[idx].markers[count].is_depth_visible = False

            count += 1
            if count == list_nb_markers[idx]:
                color_list[idx] = draw_markers(
                    # self.color_cropped[idx],
                    self.color_cropped[idx],
                    markers_pos=self.marker_sets[idx].get_markers_pos(),
                    markers_names=self.marker_sets[idx].marker_names,
                    is_visible=self.marker_sets[idx].get_markers_occlusion(),
                    scaling_factor=0.5,
                    circle_scaling_factor=10,
                    # markers_reliability_index=self.marker_sets[idx].get_markers_reliability_index(self.frame_idx),
                    markers_reliability_index=np.array(dist_list).astype(int),

                )

                # color_list[idx] = draw_markers(
                #     color_list[idx],
                #     # markers_filtered_pos=self.marker_sets[idx].get_markers_filtered_pos(),
                #     markers_pos=np.array(markers_kalman).T,
                #     # markers_names=self.marker_sets[idx].marker_names,
                #     is_visible=self.marker_sets[idx].get_markers_occlusion(),
                #     circle_scaling_factor=3,
                #     thickness=-1,
                #     # markers_reliability_index=self.marker_sets[idx].get_markers_reliability_index(self.frame_idx),
                # )
                if len(self.blobs[idx]) != 0:
                    color_list[idx] = draw_blobs(color_list[idx], self.blobs[idx])
                markers_kalman = []
                count = 0
                idx += 1
                if idx == len(self.color_cropped):
                    break
        return color_list

    def _prepare_data_optical_flow(self, idx):
        markers_pos_optical_flow = []
        marker_names = []
        for marker in self.marker_sets[idx].markers:
            # if marker.is_visible:
            markers_pos_optical_flow.append(list(marker.pos[:2]))
            marker_names.append(marker.name)
        return markers_pos_optical_flow, marker_names

    def _run_optical_flow(self,
                          idx, color, prev_color, prev_pos, kalman_filter, blob_detector, markers_visible_names,
                          error_threshold=10, use_tapir=False):
        if not use_tapir:
            prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (9, 9), 0)
            color_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            color_gray = cv2.GaussianBlur(color_gray, (9, 9), 0)
            if isinstance(prev_pos, list):
                prev_pos = np.array(prev_pos, dtype=np.float32)
            if isinstance(prev_pos, np.ndarray):
                prev_pos = prev_pos.astype(np.float32)
            new_markers_pos_optical_flow, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                color_gray,
                prev_pos,
                None,
                **self.optical_flow_params,
            )
        else:
            new_markers_pos_optical_flow = []
        count = 0
        markers_pos_list = []
        last_pos = []
        for m, marker in enumerate(self.marker_sets[idx].markers):
            last_pos.append(marker.pos[:2])
            if marker.name in markers_visible_names:
                if not use_tapir:
                    if st[count] == 1 and err[count] < error_threshold:
                        marker.pos[:2] = new_markers_pos_optical_flow[count]
                        marker.is_visible = False
                    # else:
                    #     marker.pos = [None, None, None]
                else:
                    new_markers_pos_optical_flow.append(marker.pos[:2])
                if kalman_filter:
                    marker.predict_from_kalman()
                    # marker.correct_from_kalman(new_markers_pos_optical_flow[count])
                if blob_detector:
                    center = np.array((marker.pos[:2])).astype(int)
                    blob_center, marker.is_visible = find_closest_blob(center, self.blobs[idx], delta=8)
                    # check distance between blob_center and marker.pos
                    # dist = math.sqrt((marker.pos[0] - blob_center[0]) ** 2 + (marker.pos[1] - blob_center[1]) ** 2)
                    # if dist > 20:
                    #     blob_center = last_pos[-1]
                    #     marker.correct_from_kalman(blob_center)
                    if marker.is_visible:
                        if kalman_filter:
                            marker.correct_from_kalman(blob_center)
                        else:
                            marker.pos[:2] = blob_center
                    else:
                        if kalman_filter:
                            marker.correct_from_kalman(new_markers_pos_optical_flow[count])
                count += 1
            markers_pos_list.append(marker.pos[:2])
            # for j, marker_pos in enumerate(marker.pos[:2]):
            #     j_bis = 0 if j == 1 else 1
            #     if marker_pos is not None and marker_pos > self.color_cropped[idx].shape[j_bis] - 1:
            #         marker.pos[j] = self.color_cropped[idx].shape[j_bis] - 1
            if marker.pos[0] and int(marker.pos[0]) not in list(range(0, self.color_cropped[idx].shape[1]-1)):
                marker.pos[0] = self.color_cropped[idx].shape[1] - 1 if marker.pos[0] > self.color_cropped[idx].shape[1] - 1 else 0
            if marker.pos[1] and int(marker.pos[1]) not in list(range(0, self.color_cropped[idx].shape[0]-1)):
                marker.pos[1] = self.color_cropped[idx].shape[0] - 1 if marker.pos[1] > self.color_cropped[idx].shape[0] - 1 else 0

            if marker.pos[0] is not None:
                marker_depth, marker.is_depth_visible = check_and_attribute_depth(marker.pos[:2],
                                                         self.depth_cropped[idx],
                                                         depth_scale=self.depth_scale)
                if marker.is_depth_visible:
                    marker.pos[2] = marker_depth

            marker.set_global_pos(marker.pos, [self.start_crop[0][idx], self.start_crop[1][idx]])

    @staticmethod
    def _apply_transformation(markers, rt, t, sub):
        result_points = np.zeros(markers.shape)
        for i in range(markers.shape[0]):
            result_points[i, :] = np.dot(rt, markers[i, :] - sub) + t
        return result_points.T

    @staticmethod
    def _minimal_marker_model(marker_set, idxs):
        model_markers = np.array([marker for marker in marker_set.marker_set_model]).T
        markers_model = []
        for idx in idxs:
            markers_model.append(model_markers[idx, :2])
        return markers_model

    def _adapt_cropping(self, color, x_min, x_max, y_min, y_max, idx):
        delta = 10
        if x_min < delta:
            self.start_crop[0][idx] = self.start_crop[0][idx] - delta
        if color.shape[1] - x_max < delta:
            self.end_crop[0][idx] = self.end_crop[0][idx] + (delta - (color.shape[1] - x_max))
        if y_min < delta:
            self.start_crop[1][idx] = self.start_crop[1][idx] - delta
        if color.shape[0] - y_max < delta:
            self.end_crop[1][idx] = self.end_crop[1][idx] + (delta- (color.shape[0] - y_max))

    @staticmethod
    def express_in_global(points, start_crop):
        return [points[0] + start_crop[0], points[1] + start_crop[1]]

    @staticmethod
    def express_in_local(points, start_crop):
        return [points[0] - start_crop[0], points[1] - start_crop[1]]

    def express_in_pixel(self, marker_pos_in_meters):
        _intrinsics = rs.intrinsics()
        _intrinsics.width = self.depth_frame.shape[1]
        _intrinsics.height = self.depth_frame.shape[0]
        _intrinsics.ppx = self.depth_ppx_ppy[0]
        _intrinsics.ppy = self.depth_ppx_ppy[1]
        _intrinsics.fx = self.depth_fx_fy[0]
        _intrinsics.fy = self.depth_fx_fy[1]
        _intrinsics.model = rs.distortion.inverse_brown_conrady
        _intrinsics.coeffs = self.conf_data["dist_coeffs_color"]

        markers_in_pixels = np.zeros_like(marker_pos_in_meters)
        for m in range(marker_pos_in_meters.shape[1]):
            markers_in_pixels[:2, m] = rs.rs2_project_point_to_pixel(_intrinsics,
                                                                    [marker_pos_in_meters[0, m],
                                                                       marker_pos_in_meters[1, m],
                                                                      marker_pos_in_meters[2, m]])
        return markers_in_pixels

    def get_global_markers_pos(self):
        markers_pos = None
        markers_names = []
        occlusions = []
        reliability = []
        for i, marker_set in enumerate(self.marker_sets):
            if markers_pos is not None:
                markers_pos = np.append(markers_pos, marker_set.get_markers_global_pos(), axis=1)
            else:
                markers_pos = marker_set.get_markers_global_pos()
            markers_names = markers_names + marker_set.get_markers_names()
            occlusions = occlusions + marker_set.get_markers_occlusion()
            reliability = reliability + marker_set.get_markers_reliability_index(self.frame_idx)
        return markers_pos, markers_names, occlusions, reliability

    def get_merged_global_markers_pos(self):
        markers_pos = None
        markers_names = []
        occlusions = []
        reliability = []

        for i, marker_set in enumerate(self.marker_sets):
            occlusion_tmp = marker_set.get_markers_occlusion()
            marker_pos_temp = check_filtered_or_true_pos(
                marker_set.get_markers_global_pos(),
                marker_set.get_markers_global_filtered_pos(),
                occlusion_tmp)
            if markers_pos is not None:
                markers_pos = np.append(markers_pos, marker_pos_temp, axis=1)
            else:
                markers_pos = marker_pos_temp
            markers_names = markers_names + marker_set.get_markers_names()
            occlusions = occlusions + occlusion_tmp
            reliability = reliability + marker_set.get_markers_reliability_index(self.frame_idx)

        return markers_pos, markers_names, occlusions, reliability

    def get_merged_local_markers_pos(self):
        markers_pos = None
        markers_names = []
        occlusions = []
        reliability = []

        for i, marker_set in enumerate(self.marker_sets):
            occlusion_tmp = marker_set.get_markers_occlusion()
            marker_pos_temp = check_filtered_or_true_pos(
                marker_set.get_markers_pos(),
                marker_set.get_markers_filtered_pos(),
                occlusion_tmp)
            if markers_pos is not None:
                markers_pos = np.append(markers_pos, marker_pos_temp, axis=1)
            else:
                markers_pos = marker_pos_temp
            markers_names = markers_names + marker_set.get_markers_names()
            occlusions = occlusions + occlusion_tmp
            reliability = reliability + marker_set.get_markers_reliability_index(self.frame_idx)

        return markers_pos, markers_names, occlusions, reliability

    def get_merged_global_markers_pos_in_meter(self, marker_pos_in_pixel=None):
        if marker_pos_in_pixel is None:
            marker_pos_in_pixel, markers_names, occlusions, reliability = self.get_merged_global_markers_pos()
        else:
            markers_names, occlusions = None, None

        if self.is_camera_init:
            _intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        else:
            _intrinsics = rs.intrinsics()
            _intrinsics.width = self.depth_frame.shape[1]
            _intrinsics.height = self.depth_frame.shape[0]
            _intrinsics.ppx = self.depth_ppx_ppy[0]
            _intrinsics.ppy = self.depth_ppx_ppy[1]
            _intrinsics.fx = self.depth_fx_fy[0]
            _intrinsics.fy = self.depth_fx_fy[1]
            _intrinsics.model = rs.distortion.inverse_brown_conrady
            _intrinsics.coeffs = self.conf_data["dist_coeffs_color"]
            # _intrinsics.model = rs.distortion.none

        markers_in_meters = np.zeros_like(marker_pos_in_pixel)
        for m in range(marker_pos_in_pixel.shape[1]):
            markers_in_meters[:, m] = rs.rs2_deproject_pixel_to_point(_intrinsics,
                                                                      [marker_pos_in_pixel[0, m],
                                                                       marker_pos_in_pixel[1, m]],
                                                                      marker_pos_in_pixel[2, m])

        return markers_in_meters, markers_names, occlusions

    def get_global_markers_pos_in_meter(self, marker_pos_in_pixel=None):
        if marker_pos_in_pixel is None:
            marker_pos_in_pixel, markers_names, occlusions, reliability = self.get_global_markers_pos()
        else:
            markers_names, occlusions, reliability = None, None, 0

        if self.is_camera_init:
            _intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        else:
            _intrinsics = rs.intrinsics()
            _intrinsics.width = self.depth_frame.shape[1]
            _intrinsics.height = self.depth_frame.shape[0]
            _intrinsics.ppx = self.depth_ppx_ppy[0]
            _intrinsics.ppy = self.depth_ppx_ppy[1]
            _intrinsics.fx = self.depth_fx_fy[0]
            _intrinsics.fy = self.depth_fx_fy[1]
            _intrinsics.model = rs.distortion.none

        markers_in_meters = np.zeros_like(marker_pos_in_pixel)
        for m in range(marker_pos_in_pixel.shape[1]):
            markers_in_meters[:, m] = rs.rs2_deproject_pixel_to_point(_intrinsics,
                                                                      [marker_pos_in_pixel[0, m],
                                                                       marker_pos_in_pixel[1, m]],
                                                                      marker_pos_in_pixel[2, m])

        return markers_in_meters, markers_names, occlusions, reliability

    def get_global_filtered_markers_pos(self):
        markers_pos = None
        markers_names = []
        occlusions = []
        reliability = []
        for i, marker_set in enumerate(self.marker_sets):
            if markers_pos is not None:
                markers_pos = np.append(markers_pos, marker_set.get_markers_global_filtered_pos(), axis=1)
            else:
                markers_pos = marker_set.get_markers_global_filtered_pos()
            markers_names = markers_names + marker_set.get_markers_names()
            reliability = reliability + marker_set.get_markers_reliability_index(self.frame_idx)
            occlusions = occlusions + marker_set.get_markers_occlusion()
        return markers_pos, markers_names, occlusions, reliability

    def get_local_filtered_markers_pos(self):
        markers_pos = None
        markers_names = []
        occlusions = []
        reliability = []
        for i, marker_set in enumerate(self.marker_sets):
            if markers_pos is not None:
                markers_pos = np.append(markers_pos, marker_set.get_markers_filtered_pos(), axis=1)
            else:
                markers_pos = marker_set.get_markers_filtered_pos()
            markers_names = markers_names + marker_set.get_markers_names()
            occlusions = occlusions + marker_set.get_markers_occlusion()
            reliability = reliability + marker_set.get_markers_reliability_index(self.frame_idx)
        return markers_pos, markers_names, occlusions, reliability

    def select_cropping(self):
        self.is_cropped = False
        self.start_crop = [[], []]
        self.end_crop = [[], []]
        color, _ = self.get_frames(self.is_frame_aligned, rotation_angle=self.rotation_angle)
        while True:
            cv2.namedWindow("select area, use c to continue to an other cropping or q to end.", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("select area, use c to continue to an other cropping or q to end.", self._mouse_crop)
            self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
            while True:
                c = color[0].copy()
                # color, _ = self.get_frames(self.is_frame_aligned)
                if not self.cropping:
                    cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)
                    if (self.x_start + self.y_start + self.x_end + self.y_end) > 0:
                        cv2.rectangle(c, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                        cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)

                elif self.cropping:
                    cv2.rectangle(c, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                    cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)
                # cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord("c"):
                    cv2.destroyAllWindows()
                    self.frame_idx = 0
                    break

                elif cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    self.frame_idx = 0
                    self.is_cropped = True
                    return self.start_crop, self.end_crop

    def set_cropping_area(self, start_crop: list, end_crop: list):
        self.start_crop = start_crop
        self.end_crop = end_crop
        self.is_cropped = True

    def _mouse_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
            self.start_crop[0].append(x)
            self.start_crop[1].append(y)
            self.cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                if self.x_end != x and self.y_end != y:
                    self.x_end, self.y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            self.x_end, self.y_end = x, y
            self.end_crop[0].append(x)
            self.end_crop[1].append(y)
            self.cropping = False  # cropping is finished

    def select_mask(self, method: DetectionMethod = DetectionMethod.CV2Blobs, filter: Union[str, list] = "hsv"):
        self.mask_params = self._find_bounds_color(method=method,
                                                   filter=filter,
                                                   depth_scale=self.depth_scale,
                                                   is_aligned=self.is_frame_aligned,
                                                   is_cropped=self.is_cropped)

    def _find_bounds_color(self, method, filter, depth_scale=1, is_cropped=True, is_aligned=True):
        """
        Find the bounds of the image
        """
        def nothing(x):
            pass
        self.is_cropped = is_cropped
        self.is_frame_aligned = is_aligned
        mask_params = []
        for i in range(len(self.start_crop[0])):
            if method == DetectionMethod.CV2Contours:
                cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
                cv2.createTrackbar("min threshold", "Trackbars", 30, 255, nothing)
                cv2.createTrackbar("max threshold", "Trackbars", 111, 255, nothing)
                cv2.createTrackbar("min area", "Trackbars", 5, 500, nothing)
                cv2.createTrackbar("max area", "Trackbars", 500, 1000, nothing)
                cv2.createTrackbar("n_step", "Trackbars", 5, 8, nothing)
                cv2.createTrackbar("clipping distance in meters", "Trackbars", 110, 800, nothing)
                cv2.createTrackbar("clahe clip limit", "Trackbars", 2, 40, nothing)
                cv2.createTrackbar("clahe tile Grid Size", "Trackbars", 8, 40, nothing)

            elif method == DetectionMethod.CV2Blobs:
                default_values = {"min_area": 3, "min_threshold": 150, "max_threshold": 255,
                                  "clahe_clip_limit": 1, "clahe_autre": 3, "circularity": 1,
                                  "convexity": 1, "n_step": 1, "clipping_distance_in_meters": 140,
                                  "use_contour": 0, "use_bg_remover": 0}
                if self.tracking_file_loaded:
                    default_values = self.mask_params[i]
                    default_values["clipping_distance_in_meters"] = int(default_values["clipping_distance_in_meters"] * 100)
                    default_values["convexity"] = int(default_values["convexity"] * 100)
                    default_values["circularity"] = int(default_values["circularity"] * 100)
                cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
                cv2.createTrackbar("min area", "Trackbars", default_values["min_area"], 255, nothing)
                # cv2.createTrackbar("max area", "Trackbars", 500, 5000, nothing)
                # cv2.createTrackbar("min threshold V", "Trackbars", 30, 255, nothing)
                # cv2.createTrackbar("max threshold V", "Trackbars", 111, 255, nothing)
                cv2.createTrackbar("min threshold", "Trackbars", default_values["min_threshold"], 255, nothing)
                cv2.createTrackbar("max threshold", "Trackbars", default_values["max_threshold"], 255, nothing)
                cv2.createTrackbar("clahe clip limit", "Trackbars", default_values["clahe_clip_limit"], 40, nothing)
                cv2.createTrackbar("clahe tile grid size", "Trackbars", default_values["clahe_autre"], 40, nothing)
                cv2.createTrackbar("circularity", "Trackbars", default_values["circularity"], 100, nothing)
                cv2.createTrackbar("convexity", "Trackbars", default_values["convexity"], 100, nothing)
                # cv2.createTrackbar("n_step", "Trackbars", 1, 8, nothing)
                # cv2.createTrackbar("alpha", "Trackbars", 0, 200, nothing)
                # cv2.createTrackbar("beta", "Trackbars", -200, 200, nothing)
                # cv2.createTrackbar("hist or hsv", "Trackbars", 0, 1, nothing)
                # cv2.createTrackbar("blob color", "Trackbars", 255, 255, nothing)
                cv2.createTrackbar("clipping distance in meters", "Trackbars",
                                   default_values["clipping_distance_in_meters"], 800, nothing)
                cv2.createTrackbar("use contour", "Trackbars", default_values["use_contour"], 1, nothing)
                cv2.createTrackbar("use bg remover", "Trackbars", default_values["use_bg_remover"], 1, nothing)

            elif method == DetectionMethod.SCIKITBlobs:
                cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
                cv2.createTrackbar("min area", "Trackbars", 1, 255, nothing)
                cv2.createTrackbar("max area", "Trackbars", 500, 5000, nothing)
                cv2.createTrackbar("color", "Trackbars", 255, 255, nothing)
                cv2.createTrackbar("min threshold", "Trackbars", 30, 255, nothing)
                cv2.createTrackbar("max threshold", "Trackbars", 111, 255, nothing)
            else:
                raise ValueError("Method not supported")
            while True:
                color_frame, depth = self.get_frames(self.is_frame_aligned, rotation_angle=self.rotation_angle)
                color_frame = color_frame[i]
                color_frame_init = color_frame.copy()
                depth = depth[i]
                if method == DetectionMethod.CV2Contours:
                    min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
                    min_area = cv2.getTrackbarPos("min area", "Trackbars")
                    max_area = cv2.getTrackbarPos("max area", "Trackbars")
                    max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
                    n_step = cv2.getTrackbarPos("n_step", "Trackbars")
                    clahe_clip_limit = cv2.getTrackbarPos("clahe clip limit", "Trackbars")
                    clahe_autre = cv2.getTrackbarPos("clahe autre", "Trackbars")
                    clipping_distance = cv2.getTrackbarPos("clipping distance in meters", "Trackbars") / 100
                    params = {
                        "min_threshold": min_threshold,
                        "max_threshold": max_threshold,
                        "clipping_distance_in_meters": clipping_distance,
                        "min_area": min_area,
                        "max_area": max_area,
                        "n_step": n_step,
                        "hist_hsv": 0,
                        "clahe_clip_limit": clahe_clip_limit,
                        "clahe_autre": clahe_autre,
                    }
                    # depth_image_3d = np.dstack((depth, depth, depth))
                    # color_frame = np.where(
                    #     (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= 0),
                    #     self.clipping_color,
                    #     color_frame_init,
                    # )
                    im_from, contours = get_blobs(
                        color_frame, method=method, params=params, return_image=True, return_centers=True, depth=depth
                        , clipping_color=self.clipping_color
                    )
                    draw_blobs(color_frame, contours)

                elif method == DetectionMethod.CV2Blobs:
                    # hist_hsv = cv2.getTrackbarPos("hist or hsv", "Trackbars")
                    min_area = cv2.getTrackbarPos("min area", "Trackbars")
                    # max_area = cv2.getTrackbarPos("max area", "Trackbars")
                    # min_threshold_v = cv2.getTrackbarPos("min threshold V", "Trackbars")
                    # max_threshold_v = cv2.getTrackbarPos("max threshold V", "Trackbars")
                    min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
                    max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
                    clahe_clip_limit = cv2.getTrackbarPos("clahe clip limit", "Trackbars")
                    clahe_autre = cv2.getTrackbarPos("clahe tile grid size", "Trackbars")
                    # alpha = cv2.getTrackbarPos("alpha", "Trackbars") / 100
                    # beta = cv2.getTrackbarPos("beta", "Trackbars")
                    circularity = cv2.getTrackbarPos("circularity", "Trackbars") / 100
                    convexity = cv2.getTrackbarPos("convexity", "Trackbars") / 100
                    use_contour = cv2.getTrackbarPos("use contour", "Trackbars") == 0
                    use_bg_remover = cv2.getTrackbarPos("use bg remover", "Trackbars") == 0
                    # blob_color = cv2.getTrackbarPos("blob color", "Trackbars")
                    # n_step = cv2.getTrackbarPos("n_step", "Trackbars")

                    clipping_distance = cv2.getTrackbarPos("clipping distance in meters", "Trackbars") / 100
                    if min_area == 0:
                        min_area = 1
                    # if max_area == 0:
                    #     max_area = 1
                    # if max_area < min_area:
                    #     max_area = min_area + 1
                    if min_threshold == 0:
                        min_threshold = 1
                    if max_threshold == 0:
                        max_threshold = 1
                    # if max_threshold < min_threshold:
                    #     max_threshold = min_threshold + 1
                    # if min_threshold_v == 0:
                    #     min_threshold_v = 1
                    # if max_threshold_v == 0:
                    #     max_threshold_v = 1
                    # if max_threshold_v < min_threshold_v:
                    #     max_threshold_v = min_threshold_v + 1
                    # if n_step == 0:
                    #     n_step = 1
                    # if n_step > 8:
                    #     n_step = 8
                    if circularity == 0:
                        circularity = 0.1
                    if convexity == 0:
                        convexity = 0.1
                    if clahe_clip_limit == 0:
                        clahe_clip_limit = 1
                    if clahe_autre == 0:
                        clahe_autre = 1

                    # Set up the detector parameters

                    params = {
                        # "hist_hsv": hist_hsv,
                        "min_area": min_area,
                        # "max_area": max_area,
                        "min_threshold": min_threshold,
                        "max_threshold": max_threshold,
                        "convexity": convexity,
                        "circularity": circularity,
                        # "alpha": alpha,
                        # "beta": beta,
                        # "n_step": n_step,
                        # "min_threshold_v": min_threshold_v,
                        # "max_threshold_v": max_threshold_v,
                        "clipping_distance_in_meters": clipping_distance,
                        "clahe_clip_limit": clahe_clip_limit,
                        "clahe_autre": clahe_autre,
                        "blob_color": 255,
                        "use_contour": use_contour,
                        "use_bg_remover": use_bg_remover,
                    }
                    # depth_image_3d = np.dstack((depth, depth, depth))
                    # color_frame = np.where(
                    #     (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= 0),
                    #     self.clipping_color,
                    #     color_frame_init,
                    # )
                    im_from, contours = get_blobs(
                        color_frame, method=method, params=params, return_image=True, return_centers=True,depth=depth
                        , clipping_color=self.clipping_color, depth_scale=depth_scale,
                        )
                    draw_blobs(color_frame, contours)

                elif method == DetectionMethod.SCIKITBlobs:
                    from skimage.feature import blob_log
                    from math import sqrt

                    min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
                    max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
                    params = {
                        "max_sigma": 30,
                        "min_sigma": 2,
                        "threshold": 0.3,
                        "min_threshold": min_threshold,
                        "max_threshold": max_threshold,
                    }
                    im_from, blobs = get_blobs(color_frame, method, params, return_image=True,
                                               n_blob_steps=5, blob_threshold_step=10,)
                    result = color_frame.copy()
                    for blob in blobs:
                        y, x, area = blob
                        result = cv2.circle(result, (int(x), int(y)), int(area), (0, 0, 255), 1)
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.imshow("Result", color_frame)
                cv2.namedWindow("im_from", cv2.WINDOW_NORMAL)
                cv2.imshow("im_from", im_from)
                # key = cv2.waitKey(100)
                key = cv2.waitKey(200) & 0xFF
                if key & 0xFF == ord(" "):
                    color_frame, depth = self.get_frames(self.is_frame_aligned, rotation_angle=self.rotation_angle)
                    depth = depth[i]
                    color_frame = color_frame[i]
                    color_frame_init = color_frame.copy()
                # if self.frame_idx > len(self.color_images) - 1:
                #     self.frame_idx = 0
                # else:
                #     self.frame_idx += 1

                if key & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    mask_params.append(params)
                    self.frame_idx = 0
                    break
        return mask_params

    def load_tracking_conf_file(self, tracking_conf_file: str):
        with open(tracking_conf_file, "r") as f:
            try:
                conf = json.load(f)
            except json.decoder.JSONDecodeError:
                # in cas of empty file
                conf = {}
        for key in conf.keys():
            self.__dict__[key] = conf[key]
        if self.start_crop and self.end_crop:
            self.is_cropped = True
        if self.mask_params:
            self.is_frame_clipped = True
            for params in self.mask_params:
                self.clipping_distance_in_meters.append(params["clipping_distance_in_meters"])
        if self.first_frame_markers:
            for i in range(len(self.first_frame_markers)):
                self._distribute_pos_markers(i)
        self.tracking_file_loaded = True

    def _distribute_pos_markers(self, i: int):
        """
        Distribute the markers in a dictionary of markers
        :param pos_markers_dic: dictionary of markers
        :return: list of markers
        """
        markers = np.zeros((3, len(self.first_frame_markers[i].keys())))
        occlusion = []
        depth_occlusion = []
        c = 0
        for key in self.first_frame_markers[i].keys():
            markers[:, c] = np.array(self.first_frame_markers[i][key][0], dtype=float)
            occlusion.append(self.first_frame_markers[i][key][1])
            depth_occlusion.append(self.first_frame_markers[i][key][2])
            c += 1
        self.marker_sets[i].set_markers_pos(markers)
        self.marker_sets[i].init_kalman(markers)
        # self.marker_sets[i].init_filtered_pos(markers)
        self.marker_sets[i].marker_set_model = markers
        if self.is_cropped:
            self.marker_sets[i].set_global_markers_pos(markers, [self.start_crop[0][i], self.start_crop[1][i]])

        self.marker_sets[i].set_markers_occlusion(occlusion)
        self.marker_sets[i].set_markers_depth_occlusion(depth_occlusion)

    def add_marker_set(self, marker_set: Union[list[MarkerSet], MarkerSet]):
        if not isinstance(marker_set, list):
            marker_set = [marker_set]
        for m_set in marker_set:
            if not isinstance(m_set, MarkerSet):
                raise TypeError("The marker set must be of type MarkerSet.")
            self.marker_sets.append(m_set)
        image_idx = []
        for marker_set in self.marker_sets:
            if marker_set.image_idx not in image_idx:
                image_idx.append(marker_set.image_idx)
            else:
                raise ValueError("The image index of the marker set must be unique.")

    def label_first_frame(self, method: DetectionMethod = DetectionMethod.CV2Contours):
        color_frame, depth_frame = self.get_frames(self.is_frame_aligned, rotation_angle=self.rotation_angle)
        if not isinstance(color_frame, list):
            color_frame = [color_frame]
        dic = []
        if len(self.marker_sets) != len(color_frame):
            raise ValueError("The number of marker sets and the number of frames are not equal.")
        idx = 0
        for i in range(len(color_frame)):
            # depth_image_3d = np.dstack((depth_frame[i], depth_frame[i], depth_frame[i]))
            # color_blobs = np.where(
            #     (depth_image_3d > self.mask_params[i]["clipping_distance_in_meters"] / self.depth_scale) | (
            #                 depth_image_3d <= 0),
            #     self.clipping_color,
            #     color_frame[i],
            # )
            blobs = get_blobs(color_frame[i], method=method, params=self.mask_params[i], return_centers=True,
            depth=depth_frame[i], clipping_color=self.clipping_color, depth_scale=self.depth_scale)

            for idx in range(len(self.marker_sets)):
                if i == self.marker_sets[idx].image_idx:
                    break
            marker_names = self.marker_sets[idx].marker_names
            x_tmp, y_tmp = [], []

            def click_event(event, x, y, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:
                    x_tmp.append(x)
                    y_tmp.append(y)
                    cv2.circle(color_frame[i], (x, y), 1, (255, 0, 0), -1)
                    cv2.putText(
                        color_frame[i],
                        marker_names[len(x_tmp) - 1],
                        (x + 2, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )
                    cv2.imshow(f"select markers on that order: {marker_names} by click on the image.", color_frame[i])
                if event == cv2.EVENT_RBUTTONDOWN:
                    x_tmp.pop()
                    y_tmp.pop()
                    color_frame[i] = color_frame[i].copy()
                    for x, y in zip(x_tmp, y_tmp):
                        cv2.circle(color_frame[i], (x, y), 1, (255, 0, 0), -1)
                        cv2.putText(
                            color_frame[i],
                            marker_names[len(x_tmp) - 1],
                            (x + 2, y + 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                        )

                    cv2.imshow(f"select markers on that order: {marker_names} by click on the image.", color_frame[i])

            cv2.namedWindow(
                f"select markers on that order: {self.marker_sets[idx].marker_names} by click on the image.",
                cv2.WINDOW_NORMAL,
            )
            cv2.setMouseCallback(
                f"select markers on that order: {self.marker_sets[idx].marker_names} by click on the image.",
                click_event,
            )

            while len(x_tmp) < self.marker_sets[idx].nb_markers:
                if method == DetectionMethod.SCIKITBlobs:
                    for blob in blobs:
                        cv2.circle(color_frame[i], (int(blob[1]), int(blob[0])), int(blob[2]), (0, 0, 255), 1)
                elif method == DetectionMethod.CV2Contours or method == DetectionMethod.CV2Blobs:
                    if blobs:
                        draw_blobs(color_frame[i], blobs, (255, 0, 0))
                cv2.imshow(
                    f"select markers on that order: {self.marker_sets[idx].marker_names} by click on the image.",
                    color_frame[i],
                )
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
            m = 0
            dic.append({})
            for x, y in zip(x_tmp, y_tmp):
                self.marker_sets[idx].markers[m].pos[:2],\
                self.marker_sets[idx].markers[m].is_visible = find_closest_blob(
                    [x, y], blobs, delta=2
                )
                if not self.marker_sets[idx].markers[m].is_visible:
                    self.marker_sets[idx].markers[m].pos[:2] = [x, y]
                self.marker_sets[idx].markers[m].pos[2:],\
                self.marker_sets[idx].markers[m].is_depth_visible = check_and_attribute_depth(
                    self.marker_sets[idx].markers[m].pos[:2], depth_frame[i], self.depth_scale)
                dic[-1][marker_names[m]] = [
                    self.marker_sets[idx].markers[m].pos.tolist(),
                    self.marker_sets[idx].markers[m].is_visible,
                    self.marker_sets[idx].markers[m].is_depth_visible,
                ]
                m += 1
        return dic

    def _create_kinematic_model(self, model_name: str = 'kinematic_model.bioMod', marker_sets: list = None):
        if self.color_frame is None:
            self.color_frame = self.color_images[0]
            self.depth_frame = self.depth_images[0]

        marker_pos_in_meter, names, _, _ = self.get_global_markers_pos_in_meter()
        create_c3d_file(marker_pos_in_meter[:, :, np.newaxis], names, '_tmp_markers_data.c3d')
        kinematic_model = BiomechanicalModel()
        self.model_name = model_name
        for i, marker_set in enumerate(marker_sets):
            if i == 0:
                origin = marker_set.markers[0].name
                second_marker = marker_set.markers[1].name
                third_marker = marker_set.markers[2].name
                kinematic_model[marker_set.name] = Segment(
                    name=marker_set.name,
                    # parent_name='ground',
                    translations=Translations.XYZ,
                    rotations=Rotations.XYZ,
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin=origin,
                        first_axis=Axis(name=Axis.Name.X, start=origin, end=second_marker),
                        second_axis=Axis(name=Axis.Name.Y, start=origin, end=third_marker),
                        axis_to_keep=Axis.Name.X,
                    ),
                    mesh=Mesh(tuple([m.name for m in marker_set.markers])),

                )
                for m in marker_set.markers:
                    kinematic_model[marker_set.name].add_marker(Marker(m.name))
            else:
                if marker_set.nb_markers <= 2:
                    raise ValueError('number of markers in marker set must be greater than 1')
                # origin, first_axis, second_axis = build_axis(marker_set)
                origin = marker_sets[i - 1].markers[-1].name
                # origin = marker_set.markers[0].name

                second_marker = marker_set.markers[0].name
                third_marker = marker_set.markers[1].name
                kinematic_model[marker_set.name] = Segment(
                        name=marker_set.name,
                        rotations=Rotations.XYZ,
                        # translations=Translations.XYZ,
                        parent_name=marker_sets[i - 1].name,
                        segment_coordinate_system=SegmentCoordinateSystem(
                            origin=origin,
                            first_axis=Axis(name=Axis.Name.X, start=origin, end=second_marker),
                            second_axis=Axis(name=Axis.Name.Y, start=origin, end=third_marker),
                            axis_to_keep=Axis.Name.X,
                        ),
                        mesh=Mesh(tuple([m.name for m in marker_set.markers])),
                )
                for m in marker_set.markers:
                    kinematic_model[marker_set.name].add_marker(Marker(m.name))
        kinematic_model.write(model_name, C3dData('_tmp_markers_data.c3d'))
        # read txt file
        with open(model_name, 'r') as file:
            data = file.read()
        kalman = MskFunctions(model_name, 1)
        q, _ = kalman.compute_inverse_kinematics(marker_pos_in_meter[:, :, np.newaxis],
                                                 InverseKinematicsMethods.BiorbdKalman)

        # replace the target string
        data = data.replace('shoulder\n\tRT -0.000 0.000 -0.000 xyz 0.000 0.000 0.000',
                            f'shoulder\n\tRT {q[3, 0]} {q[4, 0]} {q[5, 0]} xyz {q[0, 0]} {q[1, 0]} {q[2, 0]}')
        with open(model_name, 'w') as file:
            file.write(data)
        kalman = MskFunctions(model_name, 1)
        q, _ = kalman.compute_inverse_kinematics(marker_pos_in_meter[:, :, np.newaxis],
                                                 InverseKinematicsMethods.BiorbdKalman)
        # import bioviz
        # b = bioviz.Viz(model_name)
        # b.load_movement(np.repeat(q, 2, axis=1))
        # b.load_movement(q)
        # b.load_experimental_markers(np.repeat(marker_pos_in_meter[:, :, np.newaxis], 2, axis=2))
        # b.load_experimental_markers(marker_pos_in_meter[:, :, np.newaxis])

        # b.exec()
        os.remove('_tmp_markers_data.c3d')

    def _init_tapir_tracker(self):
        checkpoint_path = r'D:\Documents\Programmation\pose_estimation\pose_est\tapnet\checkpoints\causal_tapir_checkpoint.npy'
        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        params, state = ckpt_state['params'], ckpt_state['state']
        self.online_init = hk.transform_with_state(build_online_model_init)
        self.online_init_apply = jax.jit(self.online_init.apply)

        self.online_predict = hk.transform_with_state(build_online_model_predict)
        self.online_predict_apply = jax.jit(self.online_predict.apply)

        rng = jax.random.PRNGKey(42)
        self.online_init_apply = functools.partial(
            self.online_init_apply, params=params, state=state, rng=rng
        )
        self.online_predict_apply = functools.partial(
            self.online_predict_apply, params=params, state=state, rng=rng
        )
        if self.color_frame is None:
            self.color_frame = self.color_images[0]
            self.depth_frame = self.depth_images[0]

        self.video_empty_tapir = np.ndarray((1, self.color_frame.shape[0], self.color_frame.shape[1], self.color_frame.shape[2]),
                           dtype=np.uint8)
        self.height, self.width = self.video_empty_tapir.shape[1:3]
        self.video_empty_tapir[0] = self.color_frame
        self.resize_height_tapir, self.resize_width_tapir = 256, 256
        frames = media.resize_video(self.video_empty_tapir, (self.resize_height_tapir, self.resize_width_tapir))
        select_points, _, _, _ = self.get_global_markers_pos()
        select_points = select_points[:2].T

        # select_points = self._prepare_data_optical_flow(0)
        query_points = convert_select_points_to_query_points(0, select_points)
        query_points = transforms.convert_grid_coordinates(
            query_points, (1, self.height, self.width),
            (1, self.resize_height_tapir, self.resize_width_tapir), coordinate_format='tyx')

        self.query_features, _ = self.online_init_apply(frames=preprocess_frames(frames[None, None, 0]),
                                              query_points=query_points[None])
        self.causal_state = construct_initial_causal_state(query_points.shape[0], len(self.query_features.resolutions) - 1)

    def initialize_tracking(
        self, tracking_conf_file: str = None,
            crop_frame=False, mask_parameters=False, label_first_frame=False,
            build_kinematic_model=False,
            model_name: str = None,
            marker_sets: list = None,
            rotation_angle: Rotation = None,
            with_tapir=False,
            **kwargs
    ):
        self.rotation_angle = rotation_angle
        now = datetime.datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M")
        self.use_tapir = False
        if tracking_conf_file:
            if os.path.isfile(tracking_conf_file):
                self.load_tracking_conf_file(tracking_conf_file)
            else:
                open(tracking_conf_file, "w").close()

        if crop_frame:
            self.start_crop, self.end_crop = self.select_cropping()

        if mask_parameters:
            self.is_frame_clipped = False
            self.select_mask(**kwargs)

        if label_first_frame:
            self.first_frame_markers = self.label_first_frame(**kwargs)

            self.is_tracking_init = True

        if label_first_frame + mask_parameters + crop_frame > 0:
            dic = {
                "start_crop": self.start_crop,
                "end_crop": self.end_crop,
                "mask_params": self.mask_params,
                "first_frame_markers": self.first_frame_markers,
                "start_frame": self.start_index,
            }
            if not tracking_conf_file:
                tracking_conf_file = f"tracking_conf_{dt_string}.json"
            with open(tracking_conf_file, "w") as f:
                json.dump(dic, f, indent=4)

        if tracking_conf_file:
            if os.path.isfile(tracking_conf_file):
                self.load_tracking_conf_file(tracking_conf_file)
            else:
                open(tracking_conf_file, "w").close()

        if with_tapir:
            self.use_tapir = True
            self.is_tapir_init = False

            print("WARNING: "
                  "You have selected with Tapir. Please be aware that it could take a lot of GPU or CPU ressources."
                  "If you have a GPU, please make sure that you have installed the GPU requirements to be in realtime."
                  "If not just be aware that the program will work slowly.")
            # self._init_tapir_tracker()
        self.optical_flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        if build_kinematic_model:
            if not self.first_frame_markers:
                raise ValueError("You need to initialize tracking first")
            if not is_biorbd_package:
                raise ImportError("biorbd package is not installed")
            model_name = f"kinematic_model_{dt_string}.bioMod" if not model_name else model_name
            self._create_kinematic_model(model_name=model_name, marker_sets=marker_sets)
            self.is_kinematic_model = True


