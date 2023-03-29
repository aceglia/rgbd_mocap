import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    pass
import json
from typing import Union
from biosiglive import load
import cv2
from .enums import *
from .marker_class import MarkerSet
from .utils import *


class RgbdImages:
    def __init__(
        self,
        merged_images: str = None,
        color_images: Union[np.ndarray, str] = None,
        depth_images: Union[np.ndarray, str] = None,
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
        self.blobs = []

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
        self.is_frame_clipped = False
        self.clipping_distance_in_meters = []

        self.conf_data = None
        self.is_camera_init = False
        self.is_frame_aligned = False
        self.align = None
        self.color_images = None
        self.depth_images = None
        self.color_frame = None
        self.depth_frame = None
        self.upper_bound = []
        self.lower_bound = []
        self.marker_sets = []
        self.is_tracking_init = False
        self.first_frame_markers = None

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

    def _get_frame_from_file(self):
        if self.frame_idx >= len(self.color_images):
            self.frame_idx = 0
            if self.first_frame_markers:
                for i in range(len(self.first_frame_markers)):
                    markers_pos, occlusion = distribute_pos_markers(self.first_frame_markers[i])
                    self.marker_sets[i].set_markers_pos(markers_pos)
                    self.marker_sets[i].init_kalman(markers_pos)
                    self.marker_sets[i].set_markers_occlusion(occlusion)
        self.color_frame, self.depth_frame = self.color_images[self.frame_idx], self.depth_images[self.frame_idx]
        if self.is_frame_aligned:
            self.depth_frame = self._align_images(self.color_frame, self.depth_frame)
        self.color_cropped, self.depth_cropped = self._crop_frames(self.color_frame, self.depth_frame)
        if self.is_frame_clipped:
            self.color_cropped = self._clip_frames(self.color_cropped, self.depth_cropped)
        self.frame_idx += 1

    def _crop_frames(self, color, depth):
        color_cropped = []
        depth_cropped = []
        for i in range(len(self.start_crop[0])):
            # self._adapt_cropping(color, i)
            color_cropped.append(
                color[self.start_crop[1][i] : self.end_crop[1][i], self.start_crop[0][i] : self.end_crop[0][i], :]
            )
            depth_cropped.append(
                depth[self.start_crop[1][i] : self.end_crop[1][i], self.start_crop[0][i] : self.end_crop[0][i]]
            )
        return color_cropped, depth_cropped

    def _adapt_cropping(self, color, idx):
        delta = 10
        color_cropped = color[
            self.start_crop[1][idx] : self.end_crop[1][idx], self.start_crop[0][idx] : self.end_crop[0][idx], :
        ]
        color_cropped, (x_min, x_max, y_min, y_max) = bounding_rect(
            color_cropped, self.marker_sets[idx].get_markers_pos(), color=(0, 0, 255), delta=delta
        )
        if x_min < delta:
            self.start_crop[1][idx] = self.start_crop[1][idx] - delta
        if color_cropped.shape[1] - x_max < delta:
            self.end_crop[1][idx] = self.end_crop[1][idx] + (color_cropped.shape[1] - x_max)
        if y_min < delta:
            self.start_crop[0][idx] = self.start_crop[0][idx] - delta
        if color_cropped.shape[0] - y_max < delta:
            self.end_crop[0][idx] = self.end_crop[0][idx] + (color_cropped.shape[0] - y_max)

    def _clip_frames(self, color, depth):
        color_clipped = []
        for i in range(len(color)):
            clipping_distance = self.clipping_distance_in_meters[i] / self.depth_scale
            depth_image_3d = np.dstack((depth[i], depth[i], depth[i]))
            color_clipped.append(np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 155, color[i]))
        return color_clipped

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

    def _get_frame_from_camera(self):
        frames = self.pipeline.wait_for_frames()
        if self.is_frame_aligned:
            frames = self.align.process(frames)
        self.depth_frame = frames.get_depth_frame()
        self.color_frame = frames.get_color_frame()
        if not self.depth_frame or not self.color_frame:
            return None, None
        self.depth_frame = np.asanyarray(self.depth_frame.get_data())
        self.color_frame = np.asanyarray(self.color_frame.get_data())
        if self.is_frame_clipped:
            self.color_frame = self._clip_frames(self.color_frame, self.depth_frame)
        self.color_cropped, self.depth_cropped = self._crop_frames(self.color_frame, self.depth_frame)

    def get_frames(
        self,
        aligned: bool = False,
        detect_blobs=False,
        label_markers=False,
        bounds_from_marker_pos=False,
        filter_markers=True,
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
        self.is_frame_aligned = aligned

        if self.is_camera_init:
            self._get_frame_from_camera()
        elif self.color_images and self.depth_images:
            self._get_frame_from_file()
        else:
            raise ValueError("Camera is not initialized and images are not loaded from a file.")
        self.blobs = []
        color_list = []
        depth = self.depth_cropped
        for i, color in enumerate(self.color_cropped):
            if detect_blobs:
                if bounds_from_marker_pos:
                    if np.all(self.marker_sets[i].get_markers_filtered_pos() == 0):
                        markers_values = self.marker_sets[i].get_markers_pos()
                    else:
                        markers_values = np.concatenate((self.marker_sets[i].get_markers_filtered_pos(),
                                            self.marker_sets[i].get_markers_pos()),
                                           axis=1)
                    color, (x_min, x_max, y_min, y_max) = bounding_rect(
                        color, markers_values,
                        color=(0, 255, 0), delta=10,
                    )
                else:
                    x_min, x_max, y_min, y_max = 0, color.shape[1], 0, color.shape[0]
                self.blobs.append(
                    get_blobs(
                        color,
                        params=self.mask_params[i],
                        return_centers=True,
                        image_bounds=(x_min, x_max, y_min, y_max),
                        **kwargs,
                    )
                )
                if len(self.blobs[i]) != 0:
                    color = draw_blobs(color, self.blobs[i])
            if label_markers:
                if not detect_blobs:
                    raise ValueError("You need to detect blobs before labeling them.")
                if filter_markers:
                    for marker in self.marker_sets[i].markers:
                        marker.predict_from_kalman()
                        blob_center, marker.is_visible = find_closest_blob(marker.filtered_pos, self.blobs[i])
                        if marker.is_visible:
                            marker.correct_from_kalman(blob_center)
                        else:
                            marker.pos = [None, None]
                        marker.set_global_filtered_pos(marker.filtered_pos, [self.start_crop[0][i], self.start_crop[1][i]])
                        marker.set_global_pos(marker.pos, [self.start_crop[0][i], self.start_crop[1][i]])

                color = draw_markers(
                    color,
                    markers_filtered_pos=self.marker_sets[i].get_markers_filtered_pos(),
                    markers_pos=self.marker_sets[i].get_markers_pos(),
                    markers_names=self.marker_sets[i].marker_names,
                    is_visible=self.marker_sets[i].get_markers_occlusion(),
                )
            color_list.append(color)
        return color_list, depth

    def get_global_markers_pos(self):
        markers_pos = None
        markers_names = []
        occlusions = []
        for i, marker_set in enumerate(self.marker_sets):
            if markers_pos is not None:
                markers_pos = np.append(markers_pos, marker_set.get_markers_global_pos(), axis=1)
            else:
                markers_pos = marker_set.get_markers_global_pos()
            markers_names = markers_names + marker_set.get_markers_names()
            occlusions = occlusions + marker_set.get_markers_occlusion()
        return markers_pos, markers_names, occlusions

    def get_merged_global_markers_pos(self):
        markers_pos = None
        markers_names = []
        occlusions = []
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
        return markers_pos, markers_names, occlusions

    def get_markers_depth(self, markers_pos=None):
        if markers_pos is None:
            markers_pos, _, _ = self.get_merged_global_markers_pos()
        markers_depth = []
        for i in range(markers_pos.shape[1]):
            markers_depth.append(self.depth_frame[markers_pos[0][i], markers_pos[1][i]] / self.depth_scale)
        return markers_depth

    def get_global_filtered_markers_pos(self):
        markers_pos = None
        markers_names = []
        for i, marker_set in enumerate(self.marker_sets):
            if markers_pos is not None:
                markers_pos = np.append(markers_pos, marker_set.get_markers_global_filtered_pos(), axis=1)
            else:
                markers_pos = marker_set.get_markers_global_filtered_pos()
            markers_names = markers_names + marker_set.get_markers_names()
        return markers_pos, markers_names

    def select_cropping(self):
        self.start_crop = [[], []]
        self.end_crop = [[], []]
        while True:
            cv2.namedWindow("select area, use c to continue to an other cropping or q to end.", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("select area, use c to continue to an other cropping or q to end.", self._mouse_crop)
            color, _ = self.get_frames(self.is_frame_aligned)
            self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
            while True:
                color, _ = self.get_frames(self.is_frame_aligned)
                c = color.copy()
                if not self.cropping:
                    cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)
                    if (self.x_start + self.y_start + self.x_end + self.y_end) > 0:
                        cv2.rectangle(c, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                        cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)

                elif self.cropping:
                    cv2.rectangle(c, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                    cv2.imshow("select area, use c to continue to an other cropping or q to end.", c)

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
        self.mask_params = self._find_bounds_color(method=method, filter=filter, depth_scale=self.depth_scale)

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
                cv2.createTrackbar("clipping distance in meters", "Trackbars", 14, 80, nothing)

            elif method == DetectionMethod.CV2Blobs:
                cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
                cv2.createTrackbar("min area", "Trackbars", 1, 255, nothing)
                cv2.createTrackbar("max area", "Trackbars", 500, 5000, nothing)
                cv2.createTrackbar("color", "Trackbars", 255, 255, nothing)
                cv2.createTrackbar("min threshold", "Trackbars", 30, 255, nothing)
                cv2.createTrackbar("max threshold", "Trackbars", 111, 255, nothing)

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
                color_frame, depth = self.get_frames(self.is_frame_aligned)
                color_frame = color_frame[i]
                depth = depth[i]
                color_frame_init = color_frame.copy()
                if method == DetectionMethod.CV2Contours:
                    min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
                    max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
                    clipping_distance = cv2.getTrackbarPos("clipping distance in meters", "Trackbars") / 10
                    params = {
                        "min_threshold": min_threshold,
                        "max_threshold": max_threshold,
                        "clipping_distance_in_meters": clipping_distance,
                    }
                    depth_image_3d = np.dstack((depth, depth, depth))
                    color_frame = np.where(
                        (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= 0),
                        155,
                        color_frame_init,
                    )
                    im_from, contours = get_blobs(
                        color_frame, method=method, params=params, return_image=True, return_centers=True
                    )
                    draw_blobs(color_frame, contours)

                elif method == DetectionMethod.CV2Blobs:
                    min_area = cv2.getTrackbarPos("min area", "Trackbars")
                    max_area = cv2.getTrackbarPos("max area", "Trackbars")
                    color = cv2.getTrackbarPos("color", "Trackbars")
                    min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
                    max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
                    if min_area == 0:
                        min_area = 1
                    if max_area == 0:
                        max_area = 1
                    if max_area < min_area:
                        max_area = min_area + 1
                    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
                    # Set up the detector parameters
                    params = cv2.SimpleBlobDetector_Params()
                    # Filter by color
                    params.filterByColor = True
                    params.blobColor = color
                    params.filterByArea = True
                    params.minArea = min_area
                    params.maxArea = max_area
                    # Create the detector object

                    detector = cv2.SimpleBlobDetector_create(params)
                    im_from = hsv[:, :, 2]
                    im_from = cv2.GaussianBlur(im_from, (5, 5), 0)
                    im_from = cv2.inRange(im_from, min_threshold, max_threshold, im_from)
                    result_mask = cv2.bitwise_and(color_frame, color_frame, mask=im_from)

                    keypoints = detector.detect(im_from)

                    result = cv2.drawKeypoints(
                        color_frame, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                    )
                    params = {
                        "min_area": min_area,
                        "max_area": max_area,
                        "color": color,
                        "min_threshold": min_threshold,
                    }

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
                    im_from, blobs = get_blobs(color_frame, method, params, return_image=True)
                    result = color_frame.copy()
                    for blob in blobs:
                        y, x, area = blob
                        result = cv2.circle(result, (int(x), int(y)), int(area), (0, 0, 255), 1)
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.imshow("Result", color_frame)
                cv2.namedWindow("im_from", cv2.WINDOW_NORMAL)
                cv2.imshow("im_from", im_from)
                key = cv2.waitKey(100)
                self.frame_idx += 1
                if key & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    mask_params.append(params)
                    self.frame_idx = 0
                    break
        return mask_params

    def load_tracking_conf_file(self, tracking_conf_file: str):
        with open(tracking_conf_file, "r") as f:
            conf = json.load(f)
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
                markers_pos, occlusion = distribute_pos_markers(self.first_frame_markers[i])
                self.marker_sets[i].set_markers_pos(markers_pos)
                self.marker_sets[i].init_kalman(markers_pos)
                self.marker_sets[i].set_markers_occlusion(occlusion)

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
        color_frame, _ = self.get_frames(self.is_frame_aligned)
        if not isinstance(color_frame, list):
            color_frame = [color_frame]
        dic = []
        if len(self.marker_sets) != len(color_frame):
            raise ValueError("The number of marker sets and the number of frames are not equal.")
        idx = 0
        for i in range(len(color_frame)):
            blobs = get_blobs(color_frame[i], method=method, params=self.mask_params[i], return_centers=True)
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
                elif method == DetectionMethod.CV2Contours:
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
                self.marker_sets[idx].markers[m].pos, self.marker_sets[idx].markers[m].is_visible = find_closest_blob(
                    [x, y], blobs
                )
                dic[-1][marker_names[m]] = [
                    self.marker_sets[idx].markers[m].pos,
                    self.marker_sets[idx].markers[m].is_visible,
                ]
                m += 1
        return dic

    def initialize_tracking(
        self, tracking_conf_file: str = None, crop_frame=False, mask_parameters=False, label_first_frame=False, **kwargs
    ):
        if tracking_conf_file:
            self.load_tracking_conf_file(tracking_conf_file)

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
            }
            with open("tracking_conf.json", "w") as f:
                json.dump(dic, f, indent=4)
