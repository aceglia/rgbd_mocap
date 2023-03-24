import numpy as np
import pyrealsense2 as rs
import json
from typing import Union
from biosiglive import load
import cv2
from .enums import *
from .marker_class import MarkerSet
from .utils import *


class RgbdImages:
    def __init__(self, merged_images: str = None, color_images: Union[np.ndarray, str] = None, depth_images: Union[np.ndarray, str] = None, conf_file: str = None):
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

        # Camera extrinsic
        self.depth_to_color = None

        # Cropping
        self.cropping = False
        self.is_cropped = None
        self.start_crop = None
        self.color_cropped = None
        self.depth_cropped = None
        self.end_crop = None
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0

        # clipping
        self.is_clipped = False
        self.clipping_distance = None

        self.conf_data = None
        self.is_camera_init = False
        self.is_frame_aligned = False
        self.align = None
        self.color_images = None
        self.depth_images = None
        self.upper_bound = []
        self.lower_bound = []
        self.marker_sets = []

        if self.conf_file:
            self.conf_data = self.get_conf_data(self.conf_file)
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

    @staticmethod
    def _load(path, data_type: str = "both") -> np.ndarray or tuple:
        if path.endswith(".bio.gzip"):
            data = load(path, merge=False)
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

    def init_camera(self, color_size: tuple, depth_size: tuple, color_fps: int, depth_fps: int, align: bool = False):
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
        if self.color_images or self.depth_images:
            raise ValueError("The camera can't be initialized if the images are loaded from a file.")
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
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

        self._set_conf_file_from_camera(self.pipeline, device)
        self.conf_data = self.get_conf_data("camera_conf.json")
        self._set_images_params()
        self.is_camera_init = True

    def _set_intrinsics_from_file(self):
        self.depth_fx_fy = self.conf_data['depth_fx_fy']
        self.depth_ppx_ppy = self.conf_data['depth_ppx_ppy']
        self.color_fx_fy = self.conf_data['color_fx_fy']
        self.color_ppx_ppy = self.conf_data['color_ppx_ppy']

        self.intrinsics_depth_mat = np.array([[self.depth_fx_fy[0], 0, self.depth_ppx_ppy[0]],
                                     [0, self.depth_fx_fy[1], self.depth_ppx_ppy[0]],
                                     [0, 0, 1]], dtype=float)
        self.intrinsics_color_mat = np.array([[self.color_fx_fy[0], 0, self.color_ppx_ppy[0]],
                                     [0, self.color_fx_fy[1], self.color_ppx_ppy[0]],
                                     [0, 0, 1]], dtype=float)

    def _set_extrinsic_from_file(self):
        self.depth_to_color = np.eye(4)
        self.depth_to_color[:3, :3] = self.conf_data['depth_to_color_rot']
        self.depth_to_color[:3, 3] = self.conf_data['depth_to_color_trans']

    def _set_depth_scale_from_file(self):
        self.depth_scale = self.conf_data['depth_scale']

    def _set_images_params(self):
        self._set_intrinsics_from_file()
        self._set_extrinsic_from_file()
        self._set_depth_scale_from_file()

    @staticmethod
    def get_conf_data(conf_file):
        with open(conf_file, 'r') as infile:
            data = json.load(infile)
        return data

    @staticmethod
    def _set_conf_file_from_camera(pipeline, device):
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        d_profile = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
        d_intr = d_profile.get_intrinsics()
        scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        c_profile = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
        c_intr = c_profile.get_intrinsics()
        deth_to_color = d_profile.get_extrinsics_to(c_profile)
        r = np.array(deth_to_color.rotation).reshape(3, 3)
        t = np.array(deth_to_color.translation)
        dic = {"camera_name": device_product_line,
               'depth_scale': scale,
               'depth_fx_fy': [d_intr.fx, d_intr.fy],
               'depth_ppx_ppy': [d_intr.ppx, d_intr.ppy],
               'color_fx_fy': [c_intr.fx, c_intr.fy],
               'color_ppx_ppy': [c_intr.ppx, c_intr.ppy],
               'depth_to_color_trans': t.tolist(),
               'depth_to_color_rot': r.tolist(),
               "model_color": c_intr.model.name,
               "model_depth": d_intr.model.name,
               "dist_coeffs_color": c_intr.coeffs,
               "dist_coeffs_depth": d_intr.coeffs,
               "size_color": [c_intr.width, c_intr.height],
               "size_depth": [d_intr.width, d_intr.height],
               "color_rate": c_profile.fps(),
               "depth_rate": d_profile.fps()
               }

        with open('camera_conf.json', 'w') as outfile:
            json.dump(dic, outfile, indent=4)

    def _get_frame_from_file(self, cropped: bool = False):
        if self.frame_idx >= len(self.color_images):
            self.frame_idx = 0
        color, depth = self.color_images[self.frame_idx], self.depth_images[self.frame_idx]
        if self.is_frame_aligned:
            depth = self._align_images(color, depth)
        if self.clipping_distance:
            clipping_distance = self.clipping_distance / self.depth_scale
            depth_image_3d = np.dstack(
                (depth, depth, depth))  # depth image is 1 channel, color is 3 channels
            color = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 155, color)
        if cropped:
            color, depth = self._crop_frames(color, depth)
        self.frame_idx += 1
        return color, depth

    def _crop_frames(self, color, depth):
        color_cropped = []
        depth_cropped = []
        for i in range(len(self.start_crop[0])):
            color_cropped.append(color[self.start_crop[1][i]:self.end_crop[1][i], self.start_crop[0][i]:self.end_crop[0][i], :])
            depth_cropped.append(depth[self.start_crop[1][i]:self.end_crop[1][i], self.start_crop[0][i]:self.end_crop[0][i]])
        return color_cropped, depth_cropped

    def _align_images(self, color, depth):
        return cv2.rgbd.registerDepth(self.intrinsics_depth_mat,
                                       self.intrinsics_color_mat,
                                       None,
                                      self.depth_to_color, depth,
                                       (color.shape[1], color.shape[0]),
                                      False)

    def _get_frame_from_camera(self, cropped: bool = False):
        frames = self.pipeline.wait_for_frames()
        if self.is_frame_aligned:
            frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())
        if self.clipping_distance:
            clipping_distance = self.clipping_distance / self.depth_scale
            depth_image_3d = np.dstack((depth, depth, depth))
            color = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 255, color)
        if cropped:
            color, depth = self._crop_frames(color, depth)
        return color, depth

    def get_frames(self, clipping_distance: float = None, cropped: bool = False, aligned: bool = False):
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
        if clipping_distance:
            self.clipping_distance = clipping_distance
        else:
            self.clipping_distance = 10
        if cropped and not self.is_cropped:
            raise ValueError("The cropping area is not selected yet."
                             " Please select one using the select_cropping method.")
        self.is_frame_aligned = aligned
        if self.is_camera_init:
            return self._get_frame_from_camera(cropped)
        elif self.color_images and self.depth_images:
            return self._get_frame_from_file(cropped)
        else:
            raise ValueError("Camera is not initialized and images are not loaded from a file.")

    def select_cropping(self):
        self.start_crop = [[], []]
        self.end_crop = [[], []]
        while True:
            cv2.namedWindow("select area, use c to continue to an other cropping or q to end.", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("select area, use c to continue to an other cropping or q to end.", self._mouse_crop)
            color, _ = self.get_frames(self.clipping_distance, False, self.is_frame_aligned)
            self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
            while True:
                color, _ = self.get_frames(self.clipping_distance, False, self.is_frame_aligned)
                c = color.copy()
                if not self.cropping:
                    cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)
                    if (self.x_start + self.y_start + self.x_end + self.y_end) > 0:
                        cv2.rectangle(c, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                        cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)

                elif self.cropping:
                    cv2.rectangle(c, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                    cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    cv2.destroyAllWindows()
                    self.frame_idx = 0
                    break
                elif cv2.waitKey(1) & 0xFF == ord('q'):
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

    def select_mask(self, method: DetectionMethod = DetectionMethod.CV2Blobs,
                    filter: Union[str, list] = "hsv"):
        color_frame, _ = self.get_frames(self.clipping_distance, self.is_cropped, self.is_frame_aligned)
        if not isinstance(color_frame, list):
            color_frame = [color_frame]
        self.mask_params = []
        for i in range(len(color_frame)):
            self.mask_params.append(find_bounds_color(color_frame[i], method, filter))

    def load_tracking_conf_file(self, tracking_conf_file: str):
        with open(tracking_conf_file, "r") as f:
            conf = json.load(f)
        self.start_crop, self.end_crop = conf["start_crop"], conf["end_crop"]
        if self.start_crop and self.end_crop:
            self.is_cropped = True
        self.mask_params = conf["mask_params"]
        self.first_frame_markers = conf["first_frame_markers"]

    def add_marker_set(self, marker_set: MarkerSet):
        self.marker_sets.append(marker_set)

    def label_first_frame(self, method: DetectionMethod = DetectionMethod.CV2Contours):
        marker_set = None
        color_frame, _ = self.get_frames(self.clipping_distance, self.is_cropped, self.is_frame_aligned)
        if not isinstance(color_frame, list):
            color_frame = [color_frame]
        color_frame = [color_frame[0]]
        dic = [{}] * len(color_frame)
        if len(self.marker_sets) != len(color_frame):
            raise ValueError("The number of marker sets and the number of frames are not equal.")
        idx = 0
        for i in range(len(color_frame)):
            blobs = get_blobs(color_frame[i], method, self.mask_params[i])
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
                    cv2.putText(color_frame[i], marker_names[len(x_tmp)-1], (x + 2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
                    cv2.imshow(f"select markers on that order: {marker_names} by click on the image.", color_frame[i])
                if event == cv2.EVENT_RBUTTONDOWN:
                    x_tmp.pop()
                    y_tmp.pop()
                    color_frame[i] = color_frame[i].copy()
                    for x, y in zip(x_tmp, y_tmp):
                        cv2.circle(color_frame[i], (x, y), 1, (255, 0, 0), -1)
                        cv2.putText(color_frame[i], marker_names[len(x_tmp)-1], (x + 2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 0, 0), 1)

                    cv2.imshow(f"select markers on that order: {marker_names} by click on the image.", color_frame[i])
            cv2.namedWindow(f"select markers on that order: {self.marker_sets[idx].marker_names} by click on the image.", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(f"select markers on that order: {self.marker_sets[idx].marker_names} by click on the image.", click_event)

            while len(x_tmp) < self.marker_sets[idx].nb_markers:
                if method == DetectionMethod.SCIKITBlobs:
                    for blob in blobs:
                        cv2.circle(color_frame[i], (int(blob[1]), int(blob[0])), int(blob[2]), (0, 0, 255), 1)
                elif method == DetectionMethod.CV2Contours:
                    for c, contour in enumerate(blobs):
                        if cv2.contourArea(contour) > 5 and cv2.contourArea(contour) < 50:
                            M = cv2.moments(contour)
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(color_frame[i], (cx, cy), 5, (255, 0, 0), 1)
                cv2.imshow(f"select markers on that order: {self.marker_sets[idx].marker_names} by click on the image.", color_frame[i])
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
            for x, y in zip(x_tmp, y_tmp):
                self.marker_sets[idx].pos[:, idx, :] = np.array([x, y])[:, np.newaxis]
            for m in range(self.marker_sets[idx].nb_markers):
                dic[i][marker_names[m]] = (x_tmp[m], y_tmp[m])
        return dic

    def initialize_tracking(self,
                            tracking_conf_file: str = None,
                            crop_frame=False,
                            mask_parameters=False,
                            label_first_frame=False,
                            clipping_distance_in_meters=None,
                            **kwargs
                            ):
        if tracking_conf_file:
            self.load_tracking_conf_file(tracking_conf_file)

        if clipping_distance_in_meters:
            self.clipping_distance = clipping_distance_in_meters
            self.is_clipped = True

        if crop_frame:
            self.start_crop, self.end_crop = self.select_cropping()

        if mask_parameters:
            self.select_mask(**kwargs)

        if label_first_frame:
            self.first_frame_markers = self.label_first_frame(**kwargs)

        self.is_tracking_init = True
        dic = {"start_crop": self.start_crop,
               "end_crop": self.end_crop,
               "mask_params": self.mask_params,
               "first_frame_markers": self.first_frame_markers
               }
        with open("tracking_conf.json", "w") as f:
            json.dump(dic, f, indent=4)
