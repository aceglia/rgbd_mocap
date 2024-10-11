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
from ..frames.shared_frames import SharedFrames
from ..markers.marker_set import MarkerSet
from ..camera.camera_converter import CameraConverter
from ..tracking.utils import set_marker_pos
from ..processing.multiprocess_handler import MultiProcessHandler
from ..tracking.position import Position
from rgbd_mocap.utils import find_closest_blob




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
        self.order_idx = self._get_reordered_index()
        if build_model:
            self._create_kinematic_model()

        # Set Kinematic functions
        self.kinematics_functions = MskFunctions(self.model_name, 1)

        # Set Markers to exclude
        self.markers_to_exclude = markers_to_exclude

        self.ik_method = 'kalman'

        self.last_q = None

    def _get_reordered_index(self):
        kin_markers_names = sum([kin_mark.get_markers_names() for kin_mark in self.kin_marker_sets], [])
        init_markers_names = sum([kin_mark.get_markers_names() for kin_mark in self.marker_sets], [])

        self.kin_to_init = []
        self.init_to_kin = []
        for i in range(len(init_markers_names)):
            if init_markers_names[i] in kin_markers_names:
                self.kin_to_init.append(kin_markers_names.index(init_markers_names[i]))
            if kin_markers_names[i] in init_markers_names:
                self.init_to_kin.append(init_markers_names.index(kin_markers_names[i]))

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

    def _get_global_markers_pos_in_meter(self):
        markers_pos, markers_name, markers_visibility = self._get_all_markers()
        return self.converter.get_markers_pos_in_meter(markers_pos), markers_name, markers_visibility

    def _create_kinematic_model(self):
        # Get markers pos in meters and names
        marker_pos_in_meter, names, _ = self._get_global_markers_pos_in_meter()
        names = [names[i] for i in self.init_to_kin]
        marker_pos_in_meter = marker_pos_in_meter[:, self.init_to_kin]
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
                    translations=marker_set.translations,
                    rotations=marker_set.rotations,
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
                    translations=marker_set.translations,
                    rotations=marker_set.rotations,
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
        q, _, _ = kalman.compute_inverse_kinematics(
            marker_pos_in_meter[:, :, np.newaxis], InverseKinematicsMethods.BiorbdKalman, kalman_freq=60
        )
        # replace the target string
        data = data.replace(
            f"{self.kin_marker_sets[0].name}\n\tRT -0.000 0.000 -0.000 xyz 0.000 0.000 0.000",
            f"{self.kin_marker_sets[0].name}\n\tRT {q[3, 0]} {q[4, 0]} {q[5, 0]} xyz {q[0, 0]} {q[1, 0]} {q[2, 0]}",
        )
        with open(self.model_name, "w") as file:
            file.write(data)
        os.remove("_tmp_markers_data.c3d")

    def _set_previous_estimation(self, crops):
        if self.frames.get_index() == 0:
            return crops
        for m, marker_set in enumerate(self.marker_sets):
            crops[m].tracker.frame = crops[m].frame
            for i in range(marker_set.nb_markers):
                crops[m].tracker.estimated_positions[i] = [Position(marker_set.markers[i].pos,
                                                                    marker_set.markers[i].get_visibility())]
        return crops

    def _set_markers(self, markers, crops):
        start = 0
        markers_in_pixel = []
        if isinstance(self.frames, SharedFrames):
            crops = self._set_previous_estimation(crops)

        for m, marker_set in enumerate(self.marker_sets):
            _in_local = []
            end = start + marker_set.nb_markers
            markers_local = markers[:, start:end, 0]
            for i in range(marker_set.nb_markers):
                if marker_set.markers[i].is_static:
                    crops[m].tracker.estimated_positions[i] = [Position(marker_set.markers[i].pos, False)]
                    _in_local.append(marker_set.markers[i].pos)
                    continue
                # crops[m].tracker.estimated_positions[i] = []
                marker_in_pixel = self.converter.get_marker_pos_in_pixel(markers_local[:, i][np.newaxis, :])[0, :]
                markers_in_pixel.append(marker_in_pixel)
                marker_in_local = marker_in_pixel - marker_set.markers[0].crop_offset
                _in_local.append(marker_in_local)
                if marker_set.markers[i].name in marker_set.markers_from_dlc:
                    if marker_set.markers[i].name not in marker_set.dlc_enhance_markers:
                        crops[m].tracker.estimated_positions[i].append(Position(marker_in_local, True))
                    elif len(crops[m].tracker.blobs) > 0 and marker_set.markers[i].name in marker_set.dlc_enhance_markers:
                        position, visible = find_closest_blob(marker_in_local, crops[m].tracker.blobs, delta=10)
                        crops[m].tracker.estimated_positions[i].append(Position(position, visible))
                else:
                     crops[m].tracker.estimated_positions[i].append(Position(marker_in_local, True))
                # else:
                #     crops[m].tracker.estimated_positions[i].append(None)
            crops[m].tracker.merge_positions()
            for p, pos in enumerate(crops[m].tracker.positions):
                if pos == ():
                    crops[m].tracker.positions[p] = Position(_in_local[p], False)
            crops[m].tracker.check_tracking()
            crops[m].tracker.check_bounds(crops[m].frame)
            crops[m].attribute_depth_from_position(crops[m].tracker.positions)
            set_marker_pos(crops[m].marker_set, crops[m].tracker.positions)
            start += marker_set.nb_markers
        return markers_in_pixel

    def _check_last_q(self, q):
        if self.last_q is None:
            self.last_q = q
            return q

        final_q = q.copy()
        for i, q_tmp in enumerate(q):
            if float(abs(q_tmp - self.last_q[i, 0])) > 0.7:
                print(f"q{i} is too high: {q_tmp} - {self.last_q[i, 0]}")
                final_q[i, 0] = self.last_q[i, 0]
        self.last_q = q
        return final_q

    def fit_kinematics_model(self, process_image):
        crops = process_image.crops

        handler = process_image.process_handler

        if isinstance(handler, MultiProcessHandler):
            blobs = []
            while len(blobs) != len(crops):
                try:
                    blobs.append(handler.queue_blobs.get_nowait())
                except Exception as e:
                    pass
            assert len(blobs) == len(crops)
            for b, blob in enumerate(blobs):
                crops[blob[0]].tracker.blobs = blob[1]

        if not os.path.isfile(self.model_name):
            raise ValueError("The model file does not exist. Please initialize the model creation before.")

        # Get Markers Information

        markers, names, is_visible = self._get_global_markers_pos_in_meter()
        names = [names[i] for i in self.init_to_kin]
        markers = markers[:, self.init_to_kin]
        markers_for_ik = np.full((markers.shape[0], markers.shape[1], 1), np.nan)
        for m in range(markers_for_ik.shape[1]):
            if names[m] not in self.markers_to_exclude:
                markers_for_ik[:, m, 0] = markers[:, m]

        _method = (
            InverseKinematicsMethods.BiorbdLeastSquare
            if self.ik_method == "least_squares"
            else InverseKinematicsMethods.BiorbdKalman
        )
        q, _, _ = self.kinematics_functions.compute_inverse_kinematics(markers_for_ik, _method, kalman_freq=60)
        # q = self._check_last_q(q)
        markers = self.kinematics_functions.compute_direct_kinematics(q)
        markers = markers[:, self.kin_to_init]
        self._set_markers(markers, crops)
        return q
