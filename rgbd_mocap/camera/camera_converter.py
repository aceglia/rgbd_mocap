import numpy as np
import json

try:
    import pyrealsense2 as rs

    rs_package = True
except ImportError:
    rs_package = False
    pass
    # print ImportWarning("Cannot use camera: Import of the library pyrealsense2 failed")


# from rgbd_mocap.RgbdImages import RgbdImages


class CameraIntrinsics:
    def __init__(self):
        self.fps = None
        self.width = None
        self.height = None
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None
        self.intrinsics_mat = None
        self.model = None
        self.dist_coefficients = None

    def set_intrinsics_from_file(self, fx_fy, ppx_ppy, dist_coefficients, size, fps):
        self.height = size[1]
        self.width = size[0]

        self.fx = fx_fy[0]
        self.fy = fx_fy[1]

        self.ppx = ppx_ppy[0]
        self.ppy = ppx_ppy[1]

        self.dist_coefficients = dist_coefficients
        self.model = rs.distortion.inverse_brown_conrady if rs_package else None

        self.fps = fps

        self._set_intrinsics_mat()

    def set_intrinsics(self, intrinsics):
        self.width = intrinsics.width
        self.height = intrinsics.height
        self.ppx = intrinsics.ppx
        self.ppy = intrinsics.ppy
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.dist_coefficients = intrinsics.coeffs
        self.model = intrinsics.model

        self._set_intrinsics_mat()

    def _set_intrinsics_mat(self):
        self.intrinsics_mat = np.array(
            [
                [self.fx, 0, self.ppx],
                [0, self.fy, self.ppx],
                [0, 0, 1],
            ],
            dtype=float,
        )

    def get_intrinsics(self, model=None):
        _intrinsics = rs.intrinsics()
        _intrinsics.width = self.width
        _intrinsics.height = self.height
        _intrinsics.ppx = self.ppx
        _intrinsics.ppy = self.ppy
        _intrinsics.fx = self.fx
        _intrinsics.fy = self.fy
        _intrinsics.coeffs = self.dist_coefficients
        _intrinsics.model = self.model
        if model:
            _intrinsics.model = model

        return _intrinsics


class CameraConverter:
    """
    CameraConverter class init the camera intrinsics to
    calculate the position of the markers in pixel
    via method 'express_in_pixel' or in meters
    via method 'get_markers_pos_in_meters'.
    """

    def __init__(self, use_camera: bool = False, model=None):
        """
        Init the Camera and its intrinsics. You can determine
        if you are using a camera to get the intrinsics via
        the 'use_camera parameter'. As well, you can change the
        default method of image distortion with 'model' parameter.

        Parameters
        ----------
        use_camera: bool
            True if you get the intrinsics via a connected camera.
            False if you get the intrinsics via configuration files.
        model: rs.intrinsics.property
            Model for distortion to apply on the image for the computation.
        """
        # Camera intrinsics
        self.depth = CameraIntrinsics()
        self.color = CameraIntrinsics()
        self.model = model
        self.set_intrinsics = self._set_intrinsics_from_file if not use_camera else self._set_intrinsics_from_pipeline
        self.set_extrinsics = self._set_extrinsic_from_file
        # Camera extrinsic
        self.depth_to_color = None
        self.conf_data_dic = None
        self.depth_scale = None

    def _set_intrinsics_from_file(self, conf_data: dict):
        """
        Private method.
        Set the Camera intrinsics from file and frame.

        Parameters
        ----------
        conf_data: dict
            Dictionary containing the values to init the intrinsics of the camera.
        """
        conf_data = load_json(conf_data)
        self.conf_data_dic = conf_data
        self.depth_scale = conf_data["depth_scale"]
        self.depth.set_intrinsics_from_file(conf_data["depth_fx_fy"],
                                            conf_data["depth_ppx_ppy"],
                                            conf_data["dist_coeffs_color"],
                                            conf_data["size_depth"],
                                            conf_data["depth_rate"])
        self.color.set_intrinsics_from_file(conf_data["color_fx_fy"],
                                            conf_data["color_ppx_ppy"],
                                            conf_data["dist_coeffs_color"],
                                            conf_data["size_color"],
                                            conf_data["color_rate"])

    def _set_extrinsic_from_file(self, conf_data=None):
        conf_data = self.conf_data_dic if not conf_data else load_json(conf_data)
        self.depth_to_color = np.eye(4)
        self.depth_to_color[:3, :3] = conf_data["depth_to_color_rot"]
        self.depth_to_color[:3, 3] = conf_data["depth_to_color_trans"]

    def _set_intrinsics_from_pipeline(self, pipeline):
        """
        Private method.
        Set the Camera intrinsics from pipeline.

        Parameters
        ----------
        pipeline: Any
            Pipeline linked to the connected camera.
        """
        _intrinsics = (
            pipeline.get_active_profile()
            .get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        self.depth.set_intrinsics(_intrinsics)
        self.color.set_intrinsics(_intrinsics)

    def get_marker_pos_in_pixel(self, marker_pos_in_meters: np.array):
        """
        Get the intrinsics and compute the markers positions
        in meters to get the markers positions in pixel.\

        Parameters
        ----------
        marker_pos_in_meters: np.array
            Markers positions in meters.

        Returns
        -------
        np.array
        """
        _intrinsics = self.depth.get_intrinsics(self.model)
        markers = []

        for i in range(len(marker_pos_in_meters)):
            computed_pos = rs.rs2_project_point_to_pixel(
                _intrinsics,
                np.array([marker_pos_in_meters[i][0],
                          marker_pos_in_meters[i][1],
                          marker_pos_in_meters[i][2]], dtype=np.float32),
            )

            markers.append(computed_pos)
        markers[0][0] = np.array(markers[0][0]).clip(0, self.depth.width)
        markers[0][1] = np.array(markers[0][1]).clip(0, self.depth.height)
        markers[0][0] = 0 if np.isnan(markers[0][0]) else markers[0][0]
        markers[0][1] = 0 if np.isnan(markers[0][1]) else markers[0][1]

        return np.array(markers, dtype=np.int64)

    def get_markers_pos_in_meter(self, marker_pos_in_pixel: np.array):
        """
        Get the intrinsics and compute the markers positions
        in pixels to get the markers positions in meters.
        If both parameters are set then the given
        'marker_pos_in_pixel' override the one get from
        the method.

        Parameters
        ----------
        marker_pos_in_pixel: np.array
            Markers positions in meters.

        Returns
        -------
        np.array
        """

        _intrinsics = self.depth.get_intrinsics(self.model)
        # markers_in_meters = self._compute_markers(_intrinsics, marker_pos_in_pixel, rs.rs2_deproject_pixel_to_point)

        markers = [[], [], []]

        for i, pos in enumerate(marker_pos_in_pixel):
            computed_pos = rs.rs2_deproject_pixel_to_point(
                _intrinsics, np.array([pos[0], pos[1]], dtype=np.float32),
                float(pos[2])
            )

            markers[0].append(computed_pos[0])
            markers[1].append(computed_pos[1])
            markers[2].append(computed_pos[2])

        return np.array(markers)

        # return markers_in_meters

    @staticmethod
    def _compute_markers(intrinsics, marker_pos, method, ):
        """
        Private method.
        Compute the markers positions with the given method and intrinsics.
        For positions in meters to pixels use rs.rs2_project_point_to_pixel
        For positions in pixels to meters use rs.rs2_deproject_pixel_to_point

        Parameters
        ----------
        intrinsics: rs.intrinsics
            Camera intrinsics.
        marker_pos:
            Markers positions.
        method:
            Method to compute markers positions.

        Returns
        -------
        np.array
        """
        markers = [[], [], []]

        for i, pos in enumerate(marker_pos):
            computed_pos = method(
                intrinsics, np.array([pos[0], pos[1]], dtype=np.float32),
                float(pos[2])
            )

            markers[0].append(computed_pos[0])
            markers[1].append(computed_pos[1])
            markers[2].append(computed_pos[2])

        return np.array(markers)


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)
