import cv2
import math
import numpy as np
from rgbd_mocap.utils import find_closest_blob, check_and_attribute_depth
from rgbd_mocap.marker_class import MarkerSet, Marker
from position import Position
from optical_flow import OpticalFlow


class Tracking:
    def __init__(self, frame, marker_set: MarkerSet, optical_flow=True, kalman=True):
        self.marker_set = marker_set

        self.optical_flow = None
        if optical_flow:
            self.optical_flow = OpticalFlow(frame, marker_set.get_markers_pos_2d)

        self.kalman = kalman
        self.blobs = None
        self.positions = []
        # self.depth = depth

    def _get_blob_near_position(self, position, delta=8):
        self.positions.append(position)

        position, visible = find_closest_blob(position, self.blobs, delta=delta)
        if visible:
            self.positions[-1] = position

        return visible

    @staticmethod
    def _merge_positions(positions: list[Position]):
        if not positions:
            return ()

        final_position = (0, 0)
        visibility_count = 0
        for position in positions:
            if position.visibility:  # Does not take account of the Optical_flow estimation if it's not visible
                final_position += position.position
                visibility_count += 1

        if visibility_count == 0:
            # By default, get the last position estimated (via optical flow if used)
            final_position = positions[-1].position
        else:
            final_position //= visibility_count

        return Position(final_position, visibility_count > 0)

    def _track_marker(self, marker: tuple[int, Marker]):
        index, marker = marker
        if marker.is_static:
            return marker.pos[:2]

        positions = []
        # If the marker is visible search for the closest blob
        # if marker.is_visible:
        #    self._get_blob_near_position(marker.pos[:2], positions)

        # if we use Kalman then search the closest blob to the prediction
        if self.kalman:
            prediction = marker.predict_from_kalman()

            if self._get_blob_near_position(prediction):
                marker.correct_from_kalman(positions[-1])

        # If we use optical flow get the estimation, if the flow has been found
        # and the level of error is below the threshold then take the estimation
        # Then search for the closest blob, if found update the position estimated
        if self.optical_flow:
            estimation, st, err = self.optical_flow[index]

            threshold = 10
            if st == 1 and err < threshold:
                self._get_blob_near_position(estimation)

        return self._merge_positions(positions)

    def _correct_overlapping(self, i, j):
        dist_i = self.positions[i].distance_from_marker(self.marker_set[i])
        dist_j = self.positions[j].distance_from_marker(self.marker_set[j])

        if dist_i < dist_j:
            self.positions[j] = self.marker_set[j].pos
            # Check j closest blobs
            # If closest blobs still the same (pos i) then keep last pos

        else:
            self.positions[i] = self.marker_set[i].pos
            # Check i closest blobs
            # If closest blobs still the same (pos j) then keep last pos

    # Handle Overlapping
    def check_tracking(self):
        nb_pos = len(self.positions)

        for i in range(nb_pos):
            for j in range(i, nb_pos):
                if self.positions[i] == self.positions[j]:
                    self._correct_overlapping(i, j)

    # Check new_positions are within the crop
    def check_bounds(self, frame):
        max_x = frame.shape[1] - 1
        max_y = frame.shape[0] - 1
        # min_d, max_d = self.depth

        for position in self.positions:
            position.check_bounds(max_x, max_y)

        # TODO check depth and set if visible

    def track(self, frame, blobs):
        self.blobs = blobs

        # Track the next position for all markers
        self.positions = []
        if self.optical_flow:
            self.optical_flow.get_optical_flow_pos(frame)

        for marker in enumerate(self.marker_set.markers):
            self.positions.append(self._track_marker(marker))

        self.check_tracking()
        self.check_bounds(frame)

        # self.set_new_positions()
