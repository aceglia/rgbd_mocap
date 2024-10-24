from ..enums import ColorResolution, DepthResolution, FrameRate
from typing import Union
from ..utils import get_conf_data
import numpy as np
import json
from ..camera.camera_converter import CameraConverter

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportWarning("Cannot use camera: Import of the library pyrealsense2 failed")
    pass


class Camera:
    config_file_path = "camera_conf.json"

    def __init__(
        self,
        color_size: Union[tuple, ColorResolution],
        depth_size: Union[tuple, DepthResolution],
        color_fps: Union[int, FrameRate],
        depth_fps: Union[int, FrameRate],
        align: bool = False,
    ):
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
        # Intialize attributes
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()

        # Test if the rgb channel exist
        self.find_rgb()

        # Set depth and rgb channels
        self.config.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, depth_fps)
        self.config.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, color_fps)
        self.pipeline.start(self.config)

        # Align if needed the depth and color streams
        self.align = None
        if align:
            align_to = rs.stream.color
            self.align = rs.align(align_to)

        # Set the CameraConverter with its intrinsics
        self.camera_intrinsics = CameraConverter(use_camera=True)
        self.camera_intrinsics.set_intrinsics(self.pipeline)

        # Write the camera config file in the file at 'Camera.config_file_path'
        self.write_config_file()
        self.conf_data = get_conf_data(Camera.config_file_path)

        # Set Extrinsic
        self._set_extrinsic_from_file()

        # Set scale
        self._set_scale()

    def find_rgb(self):
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                return

        raise AssertionError("The program requires Depth camera with Color sensor")

    def _set_extrinsic_from_file(self):
        self.depth_to_color = np.eye(4)
        self.depth_to_color[:3, :3] = self.conf_data["depth_to_color_rot"]
        self.depth_to_color[:3, 3] = self.conf_data["depth_to_color_trans"]

    def _set_scale(self):
        self.scale = self.conf_data["depth_scale"]

    def config_to_dict(self):
        device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        d_profile = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
        d_intr = d_profile.get_intrinsics()
        scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        c_profile = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
        c_intr = c_profile.get_intrinsics()
        depth_to_color = d_profile.get_extrinsics_to(c_profile)
        r = np.array(depth_to_color.rotation).reshape(3, 3)
        t = np.array(depth_to_color.translation)
        dic = {
            "camera_name": device_product_line,
            "depth_scale": scale,
            "depth_fx_fy": [d_intr.fx, d_intr.fy],
            "depth_ppx_ppy": [d_intr.ppx, d_intr.ppy],
            "color_fx_fy": [c_intr.fx, c_intr.fy],
            "color_ppx_ppy": [c_intr.ppx, c_intr.ppy],
            "depth_to_color_trans": t.tolist(),
            "depth_to_color_rot": r.tolist(),
            "model_color": c_intr.model.name,
            "model_depth": d_intr.model.name,
            "dist_coeffs_color": c_intr.coeffs,
            "dist_coeffs_depth": d_intr.coeffs,
            "size_color": [c_intr.width, c_intr.height],
            "size_depth": [d_intr.width, d_intr.height],
            "color_rate": c_profile.fps(),
            "depth_rate": d_profile.fps(),
        }

        return dic

    def write_config_file(self):
        dic = self.config_to_dict()
        with open(Camera.config_file_path, "w") as outfile:
            json.dump(dic, outfile, indent=4)


if __name__ == "__main__":
    camera = Camera(ColorResolution.R_480x270, DepthResolution.R_480x270, FrameRate.FPS_60, FrameRate.FPS_60, True)
