import datetime
import numpy as np
import time
# from biosiglive import save

try:
    import pyrealsense2 as rs
except ImportError:
    pass
import os
from .processing.process_image import ProcessImage
from .utils import *
from .camera.camera import Camera, CameraConverter
from .kinematic_model_checker.kin_model_check import KinematicModelChecker
from .processing.config import load_json
from .tracking.utils import print_blobs
from .crop.crop import DepthCheck
from .tracking.tracking_markers import Tracker

def plot_3D(color, depth_image, pos, camera):
    import open3d as o3d
    pt_cloud = o3d.geometry.PointCloud()
    pt_cloud.points = o3d.utility.Vector3dVector(pos[:, :, 0].T)
    pt_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    sphere_list = []
    for pt in pt_cloud.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(pt)
        sphere.paint_uniform_color([0.8, 0.2, 0.2])  # Set color of spheres to red
        sphere_list.append(sphere)

    # Create line sets for x-, y-, and z- axes
    lineset = o3d.geometry.LineSet()

    # x-axis (red)
    lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
    lineset.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0]]))
    lineset.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))

    # y-axis (green)
    lineset2 = o3d.geometry.LineSet()
    lineset2.lines = o3d.utility.Vector2iVector([[0, 1]])
    lineset2.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 1, 0]]))
    lineset2.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))

    # z-axis (blue)
    lineset3 = o3d.geometry.LineSet()
    lineset3.lines = o3d.utility.Vector2iVector([[0, 1]])
    lineset3.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1]]))
    lineset3.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]]))

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width=depth_image.shape[0], height=depth_image.shape[1], fx=camera.depth.fx,
                              fy=camera.depth.fy,
                              cx=camera.depth.ppx, cy=camera.depth.ppy)
    tic = time.time()
    depth_3d_image = o3d.geometry.Image(depth_image)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color_3d = o3d.geometry.Image(color)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_3d, depth_3d_image,
                                                                          convert_rgb_to_intensity=False)
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd1, lineset, lineset2, lineset3] + sphere_list)
    # o3d.visualization.draw_geometries([pcd1, lineset, lineset2, lineset3])


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
        self.frame = None
        self.camera_conf_file = camera_conf_file
        self.pipeline = None

        # Camera
        self.camera: Camera = None
        self.converter = CameraConverter()
        self.converter.set_intrinsics(self.camera_conf_file)
        if self.converter.depth_scale != DepthCheck.DEPTH_SCALE:
            DepthCheck.set_depth_scale(self.converter.depth_scale)
        self.frame_idx = 0

        self.is_camera_init = False

        self.marker_sets = []

        self.ik_method = "least_squares"
        self.markers_to_exclude_for_ik = []

        self.process_image: ProcessImage = None
        self.kinematic_model_checker: KinematicModelChecker = None

        self.tracking_config = None
        self.from_model_tracker = None

        self.build_kinematic_model = False
        self.kin_marker_sets = None
        self.model_name = None

    def get_frames(
        self,
        fit_model=False,
        save_data=False,
        show_markers=False,
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
        ProcessImage.SHOW_IMAGE = show_markers
        if self.process_image.index > self.tracking_config['end_index']:
            return False

        self.process_image.process_next_image()
        if fit_model:
            if not self.kinematic_model_checker:
                self.kinematic_model_checker = KinematicModelChecker(self.process_image.frames,
                                                                     self.process_image.marker_sets,
                                                                     converter=self.converter,
                                                                     model_name=self.model_name,
                                                                     build_model=self.build_kinematic_model,
                                                                     kin_marker_set=self.kin_marker_sets)
            self.kinematic_model_checker.ik_method = "kalman"
            self.kinematic_model_checker.fit_kinematics_model(self.process_image)

        for marker_set in self.marker_sets:
            for marker in marker_set:
                if marker.is_visible:
                    marker.set_reliability(0.5)
                if marker.is_depth_visible:
                    marker.set_reliability(0.5)
        self.iter += 1

        if save_data:
            markers_pos, markers_names, occlusions, reliability_idx = self.get_global_markers_pos()
            markers_in_meters, _, _, _ = self.get_global_markers_pos_in_meter(markers_pos)
            dic = {
                "markers_in_meters": markers_in_meters[:, :, np.newaxis],
                "markers_in_pixel": markers_pos[:, :, np.newaxis],
                "markers_names": markers_names,
                "occlusions": occlusions,
                "reliability_idx": reliability_idx,
                "time_to_process": self.process_image.computation_time,
                "frame_idx": self.frame_idx,

            }
            save(dic, self.image_dir + os.sep + "markers_pos.bio", add_data=True)

        return True

    def initialize_tracking(
        self,
        tracking_config_dict=None,
        model_name: str = None,
        build_kinematic_model=False,
        multi_processing=False,
        kin_marker_set=None,
    ):
        self.tracking_config = load_json(tracking_config_dict)
        self.tracking_config["depth_scale"] = self.converter.depth_scale
        self.iter = 0
        self.all_markers = []

        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")

        tracking_options = {
            "naive": False,
            "kalman": True,
            "optical_flow": True,
        }

        # ProcessImage.SHOW_IMAGE = True
        # ProcessImage.ROTATION = None
        self.process_image = ProcessImage(self.tracking_config, tracking_options, multi_processing=multi_processing)
        # build_kinematic_model = False
        self.model_name = self.tracking_config['directory'] + os.sep + model_name if model_name else None
        self.build_kinematic_model = build_kinematic_model
        if build_kinematic_model:
            if kin_marker_set is None:
                raise ValueError("Please provide a set of markers to build the kinematic model")
            self.kin_marker_sets = kin_marker_set
            path = self.tracking_config['directory']
            self.model_name = f"{path}/kinematic_model_{dt_string}.bioMod" if not self.model_name else self.model_name
            # model_name = f"kinematic_model_{dt_string}.bioMod" if not model_name else model_name

    def set_marker_to_exclude(self, markers):
        self.markers_to_exclude_for_ik = markers

