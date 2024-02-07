import datetime

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
from .tracking.test_tracking import print_blobs
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
        if self.process_image.index > self.tracking_config['end_index']:
            return False

        self.process_image.process_next_image()

        if self.kinematic_model_checker:
            pos = self.kinematic_model_checker.fit_kinematics_model(self.frame_idx)
            # self.kinematic_model_checker.set_all_markers_pos(pos)

            print(pos)
            self.process_image.frames.color = print_blobs(self.process_image.frames.color, pos)

            cv2.imshow('test model pos', self.process_image.frames.color)
            cv2.waitKey(0)

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
    ):
        self.tracking_config = load_json(tracking_config_dict)
        self.tracking_config["depth_scale"] = self.converter.depth_scale

        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")

        tracking_options = {
            "naive": False,
            "kalman": True,
            "optical_flow": True,
        }

        ProcessImage.SHOW_IMAGE = False
        ProcessImage.ROTATION = None
        self.process_image = ProcessImage(self.tracking_config, tracking_options, multi_processing=True)

        # build_kinematic_model = False
        if build_kinematic_model:
            path = self.tracking_config['directory']
            model_name = f"{path}kinematic_model_{dt_string}.bioMod" if not model_name else model_name
            # model_name = f"kinematic_model_{dt_string}.bioMod" if not model_name else model_name
            self.kinematic_model_checker = KinematicModelChecker(self.process_image.frames,
                                                                 self.process_image.marker_sets,
                                                                 converter=self.converter,
                                                                 model_name=model_name)
