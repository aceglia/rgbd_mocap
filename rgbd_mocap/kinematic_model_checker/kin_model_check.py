import cv2
import numpy as np
import os

from biosiglive import InverseKinematicsMethods, MskFunctions
from ..utils import create_c3d_file
from ..model_creation import (
    BiomechanicalModel,
    C3dData,
    Marker,
    Segment,
    SegmentCoordinateSystem,
    Translations,
    Rotations,
    Axis,
    Mesh,
)
from ..frames.frames import Frames
from ..markers.marker_set import MarkerSet
from ..camera.camera_converter import CameraConverter
from ..tracking.utils import set_marker_pos
from ..processing.multiprocess_handler import MultiProcessHandler


class KinematicModelChecker:
    def __init__(self, frames: Frames,
                 marker_sets: list[MarkerSet],
                 converter: CameraConverter,
                 model_name: str = "kinematic_model.bioMod",
                 markers_to_exclude=[],
                 build_model=True,
                 kin_marker_set=None,
                 ):

        self.frames = frames
        self.marker_sets = marker_sets
        self.kin_marker_sets = kin_marker_set
        self.converter = converter
        self.show_model = False

        # Set Model name if valid
        self.model_name = model_name
        if not self.model_name:
            raise ValueError("You need to specify a model name to fit the model.")

        # Create Model
        if build_model:
            self._create_kinematic_model()

        # Set Kinematic functions
        self.kinematics_functions = MskFunctions(self.model_name, 1)

        # Set Markers to exclude
        self.markers_to_exclude = markers_to_exclude

        self.ik_method = 'least_square'

    # utils
    def _get_all_markers(self):
        markers_pos = []
        markers_name = []
        markers_visibility = []

        for marker_set in self.marker_sets:
            markers_pos += marker_set.get_markers_global_pos_3d()
            markers_name += marker_set.get_markers_names()
            markers_visibility += marker_set.get_markers_occlusion()

        return markers_pos, markers_name, markers_visibility

    def set_all_markers_pos(self, positions_list: list):
        start = 0
        end = 0

        for marker_set in self.marker_sets:
            end += len(marker_set.markers)

            marker_set.set_markers_pos(positions_list[start:end])

            start = end

    def _get_global_markers_pos_in_meter(self):
        markers_pos, markers_name, markers_visibility = self._get_all_markers()
        return self.converter.get_markers_pos_in_meter(markers_pos), markers_name, markers_visibility

    def _create_kinematic_model(self):
        # Get markers pos in meters and names
        marker_pos_in_meter, names, _ = self._get_global_markers_pos_in_meter()
        # Create C3D
        create_c3d_file(marker_pos_in_meter[:, :, np.newaxis], names, "_tmp_markers_data.c3d")

        # Init Kinematic Model
        kinematic_model = BiomechanicalModel()

        for i, marker_set in enumerate(self.kin_marker_sets):
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
                    raise ValueError("number of markers in marker set must be greater than 1")
                # origin, first_axis, second_axis = build_axis(marker_set)
                origin = self.kin_marker_sets[i - 1].markers[-1].name
                # origin = marker_set.markers[0].name

                second_marker = marker_set.markers[0].name
                third_marker = marker_set.markers[1].name
                kinematic_model[marker_set.name] = Segment(
                    name=marker_set.name,
                    rotations=Rotations.XYZ,
                    # translations=Translations.XYZ,
                    parent_name=self.kin_marker_sets[i - 1].name,
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
        kinematic_model.write(self.model_name, C3dData("_tmp_markers_data.c3d"))
        # read txt file
        with open(self.model_name, "r") as file:
            data = file.read()
        kalman = MskFunctions(self.model_name, 1)
        q, _ = kalman.compute_inverse_kinematics(
            marker_pos_in_meter[:, :, np.newaxis], InverseKinematicsMethods.BiorbdKalman
        )
        # replace the target string
        data = data.replace(
            f"{self.kin_marker_sets[0].name}\n\tRT -0.000 0.000 -0.000 xyz 0.000 0.000 0.000",
            f"{self.kin_marker_sets[0].name}\n\tRT {q[3, 0]} {q[4, 0]} {q[5, 0]} xyz {q[0, 0]} {q[1, 0]} {q[2, 0]}",
        )
        with open(self.model_name, "w") as file:
            file.write(data)
        os.remove("_tmp_markers_data.c3d")

    def fit_kinematics_model(self, process_image):
        crops = process_image.crops
        handler = process_image.process_handler
        if isinstance(handler, MultiProcessHandler):
            blobs = []
            while True:
                try:
                    blobs.append(handler.queue_blobs.get_nowait())
                except Exception as e:
                    break

            assert len(blobs) == len(crops)
            for b, blob in enumerate(blobs):
                crops[blob[0]].tracker.blobs = blob[1]
        i = process_image.index
        if not os.path.isfile(self.model_name):
            raise ValueError("The model file does not exist. Please initialize the model creation before.")

        # Get Markers Information
        markers, names, is_visible = self._get_global_markers_pos_in_meter()
        # Final pos for the markers ?
        final_markers = np.full((markers.shape[0], markers.shape[1], 1), np.nan)

        for m in range(final_markers.shape[1]):
            if names[m] not in self.markers_to_exclude:
                final_markers[:, m, 0] = markers[:, m]

        _method = (
            InverseKinematicsMethods.BiorbdLeastSquare
            if self.ik_method == "least_squares"
            else InverseKinematicsMethods.BiorbdKalman
        )

        q, _ = self.kinematics_functions.compute_inverse_kinematics(final_markers, _method, kalman_freq=100)
        markers = self.kinematics_functions.compute_direct_kinematics(q)
        start = 0
        for m, marker_set in enumerate(self.marker_sets):
            _in_local = []
            end = start + marker_set.nb_markers
            markers_local = markers[:, start:end, 0]
            for i in range(marker_set.nb_markers):
                crops[m].tracker.estimated_positions[i] = []
                marker_in_pixel = self.converter.get_marker_pos_in_pixel(markers_local[:, i][np.newaxis, :])[0, :]
                marker_in_local = marker_in_pixel - marker_set.markers[0].crop_offset
                _in_local.append(marker_in_local)
                crops[m].tracker._get_blob_near_position(marker_in_local, i)
            crops[m].tracker.merge_positions()
            crops[m].tracker.check_tracking()
            crops[m].tracker.check_bounds(crops[m].frame)
            positions = crops[m].tracker.positions
            from rgbd_mocap.tracking.position import Position
            for p, pos in enumerate(positions):
                positions[p] = Position(_in_local[p], visibility=False) if pos == () else pos
            crops[m].attribute_depth_from_position(positions)
            set_marker_pos(crops[m].marker_set, positions)
            start += marker_set.nb_markers

        # count = 0
        # idx = 0
        # markers_kalman = []
        # dist_list = []
        # _in_pixel = self.converter.get_marker_pos_in_pixel(markers[:, :, 0])
        # nb_past_markers = 0
        # all_markers_local = []
        #
        # for i in range(markers.shape[1]):
        #     # if the marker is static then skip
        #     if self.marker_sets[idx].markers[count].is_static:
        #         count += 1
        #         if count == list_nb_markers[idx]:
        #             # if len(self.blobs[idx]) != 0:
        #             #     color_list[idx] = draw_blobs(color_list[idx], self.blobs[idx])
        #             markers_kalman = []
        #             count = 0
        #             nb_past_markers += self.marker_sets[idx].nb_markers
        #             idx += 1
        #             if idx == len(self.color_cropped):
        #                 break
        #         continue
        #
        #     # Express back in pixel in local
        #     markers_local = np.array(
        #         self.express_in_local(_in_pixel[:, i], [self.start_crop[0][idx], self.start_crop[1][idx]])
        #     )
        #     markers_kalman.append(markers_local)
        #
        #     # Find the closest blobs in local
        #     blob_center, is_visible_tmp, dist = find_closest_blob(
        #         markers_local, self.blobs[idx], delta=8, return_distance=True
        #     )
        #
        #     # Get the distances
        #     dist_list.append(dist)
        #
        #     # Set visibility
        #     self.marker_sets[idx].markers[count].is_visible = is_visible_tmp
        #
        #     all_markers_local.append(markers_local)
