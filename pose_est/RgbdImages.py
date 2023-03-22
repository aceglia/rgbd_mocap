import numpy as np
import pyrealsense2 as rs
import json
from typing import Union
from biosiglive import load
import cv2


class RgbdImages:
    def __init__(self, merged_images: str, color_images: Union[np.ndarray, str] = None, depth_images: Union[np.ndarray, str] = None, conf_file: str = None):
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

        self.conf_data = None
        self.is_camera_init = False
        self.is_frame_aligned = False
        self.align = None

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
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
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
        config.enable_stream(rs.stream.depth, depth_size[0], depth_size[0], rs.format.z16, depth_fps)
        config.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, color_fps)
        pipeline.start(config)

        if align:
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            self.is_frame_aligned = True

        self._set_conf_file_from_camera(pipeline, device)
        self.conf_data = self.get_conf_data(self.conf_file)
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
            json.dump(dic, outfile)

    def _get_frame_from_file(self, cropped: bool = False):
        if self.frame_idx >= len(self.color_images):
            self.frame_idx = 0
        color, depth = self.color_images[self.frame_idx], self.depth_images[self.frame_idx]
        if self.is_frame_aligned:
            depth = self._align_images(color, depth)
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
        if cropped:
            color, depth = self._crop_frames(color, depth)
        return color, depth

    def get_frames(self, cropped: bool = False):
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
        if cropped and not self.is_cropped:
            raise ValueError("The cropping area is not selected yet."
                             " Please select one using the select_cropping method.")
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
            color, _ = self.get_frames()
            self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
            while True:
                color, _ = self.get_frames()
                c = color.copy()
                if not self.cropping:
                    cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)
                    if (self.x_start + self.y_start + self.x_end + self.y_end) > 0:
                        cv2.rectangle(c, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                        cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)

                elif self.cropping:
                    cv2.rectangle(c, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                    cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)

                if cv2.waitKey(10) & 0xFF == ord('c'):
                    cv2.destroyAllWindows()
                    self.frame_idx = 0
                    break
                elif cv2.waitKey(10) & 0xFF == ord('q'):
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
