from typing import List, Tuple

from rgbd_mocap.utils import find_closest_blob
from rgbd_mocap.markers.marker_set import MarkerSet, Marker
from rgbd_mocap.tracking.position import Position
from rgbd_mocap.tracking.optical_flow import OpticalFlow
from rgbd_mocap.tracking.kalman import KalmanSet
from rgbd_mocap.frames.crop_frames import CropFrames


class Tracker:
    DELTA = 10

    def __init__(self, frame: CropFrames, marker_set: MarkerSet, naive=False, optical_flow=True, kalman=True, **kwargs):
        self.marker_set = marker_set

        # Tracking method
        self.optical_flow = None
        if optical_flow:
            self.optical_flow = OpticalFlow(frame.color, marker_set.get_markers_pos_2d())

        self.kalman = None
        if kalman:
            self.kalman = KalmanSet(marker_set)

        self.naive = naive

        # Blobs and positions for tracking in each frame
        self.blobs = None
        self.positions: List[Position] = []
        self.estimated_positions: List[List[Position]] = [None] * len(marker_set.markers)

    def _get_blob_near_position(self, position, index):
        self.estimated_positions[index].append(Position(position, False))

        position, visible = find_closest_blob(position, self.blobs, delta=Tracker.DELTA)
        # if visible:
        self.estimated_positions[index].append(Position(position, visible))
            # self.estimated_positions[index][-1].set(position, visible)

        return visible

    @staticmethod
    def _merge_positions(positions: List[Position]):
        if not positions:
            return ()

        final_position = (0, 0)
        visibility_count = 0
        for position in positions:
            if position.visibility:  # Does not take account of the estimations if it's not visible
                final_position += position.position
                visibility_count += 1

        if visibility_count == 0:
            # By default, get the last position estimated (via optical flow if used)
            # final_position = positions[-1].position
            return ()
        else:
            final_position //= visibility_count

        return Position(final_position, visibility_count > 0)

    def merge_positions(self):
        self.positions = []

        for positions in self.estimated_positions:
            self.positions.append(self._merge_positions(positions))

    def _track_marker(self, marker: Tuple[int, Marker]):
        index, marker = marker
        if marker.is_static:
            return marker.pos[:2]

        self.estimated_positions[index]: List[Position] = []
        # If the marker is visible search for the closest blob
        if self.naive:
            self._get_blob_near_position(marker.pos[:2], index)

        # if we use Kalman then search the closest blob to the prediction
        if self.kalman:
            # prediction = marker.predict_from_kalman()
            prediction = self.kalman[index].predict()

            self._get_blob_near_position(prediction, index)
            # marker.correct_from_kalman(self.estimated_positions[index][-1].position)

        # If we use optical flow get the estimation, if the flow has been found
        # and the level of error is below the threshold then take the estimation
        # Then search for the closest blob, if found update the position estimated
        if self.optical_flow:
            estimation, st, err = self.optical_flow[index]

            threshold = 10
            if st == 1 and err < threshold:
                self._get_blob_near_position(estimation, index)

        # return self._merge_positions(self.estimated_positions[index])

    def _correct_overlapping(self, i, j):
        dist_i = self.positions[i].distance_from_marker(self.marker_set[i])
        dist_j = self.positions[j].distance_from_marker(self.marker_set[j])

        if dist_i < dist_j:
            self.positions[j].position = self.marker_set[j].pos
            # Check j closest blobs
            # If closest blobs still the same (pos i) then keep last pos

        else:
            self.positions[i].position = self.marker_set[i].pos
            # Check i closest blobs
            # If closest blobs still the same (pos j) then keep last pos

    # Handle Overlapping
    def check_tracking(self):
        nb_pos = len(self.positions)

        for i in range(nb_pos):
            for j in range(i + 1, nb_pos):
                if self.positions[i] == self.positions[j] and self.positions[i] != ():
                    self._correct_overlapping(i, j)

    # Check new_positions are within the crop
    def check_bounds(self, frame: CropFrames):
        max_x = frame.width - 1
        max_y = frame.height - 1

        for i in range(len(self.positions)):
            if self.positions[i] != ():
                self.positions[i].check_bounds(max_x, max_y)

    def track(self, frame: CropFrames, blobs):
        self.blobs = blobs

        # Correct trajectories from last iteration
        self.correct()

        # Track the next position for all markers
        if self.optical_flow:
            self.optical_flow.get_optical_flow_pos(frame.color)

        for marker in enumerate(self.marker_set.markers):
            self._track_marker(marker)
        self.merge_positions()

        self.check_tracking()
        self.check_bounds(frame)

        return self.positions, self.estimated_positions

    def correct(self):
        if self.kalman:
            self.kalman.correct()

        if self.optical_flow:
            self.optical_flow.set_positions([marker.pos[:2] for marker in self.marker_set])
