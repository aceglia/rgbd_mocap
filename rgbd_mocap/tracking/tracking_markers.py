from typing import List, Tuple

import cv2
import numpy as np

from ..utils import find_closest_blob
from ..markers.marker_set import MarkerSet, Marker
from ..tracking.position import Position
from ..tracking.optical_flow import OpticalFlow
from ..tracking.kalman import KalmanSet
from ..frames.crop_frames import CropFrames


class Tracker:
    DELTA = 10

    def __init__(
        self,
        frame: CropFrames,
        marker_set: MarkerSet,
        naive=False,
        optical_flow=True,
        kalman=True,
        depth_range=None,
        from_dlc=False,
        dlc_live=None,
        ignore_all_checks=False,
        **kwargs,
    ):
        self.marker_set = marker_set
        self.from_dlc = from_dlc
        self.ignore_all_checks = ignore_all_checks
        self.dlc_live = dlc_live
        if self.from_dlc and dlc_live is None:
            raise ValueError("DLC live must be provided if from_dlc is True")

        # Tracking method
        self.optical_flow = None
        if optical_flow:
            depth_clipped = np.where((frame.depth > depth_range[1]), -1, frame.depth)
            self.optical_flow = OpticalFlow(frame.color, depth_clipped, marker_set)

        self.kalman = None
        if kalman:
            self.kalman = KalmanSet(marker_set)

        self.naive = naive
        self.frame = None
        self.depth = None

        # Blobs and positions for tracking in each frame
        self.blobs = None
        self.positions: List[Position] = []
        self.estimated_positions: List[List[Position]] = [None] * len(marker_set.markers)

    def get_blob_near_position(self, estimated_position, index):
        size = self.frame.color.shape
        if not (0 <= estimated_position[0] < size[1] and 0 <= estimated_position[1] < size[0]):
            return False
        position, visible = find_closest_blob(estimated_position, self.blobs, delta=Tracker.DELTA)
        self.estimated_positions[index].append(Position(position, visible))
        return visible

    @staticmethod
    def _merge_positions(positions: List[Position], ignore_all_checks=False):
        if not positions:
            return ()
        if ignore_all_checks:
            return Position(np.array([pos.position for pos in positions]).mean(axis=0).astype(int), True)
        # return Position(positions[-1].position, True)
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

    def merge_positions(self, ignore_all_checks=False):
        self.positions = []

        for positions in self.estimated_positions:
            self.positions.append(self._merge_positions(positions, ignore_all_checks))

    def _track_marker(self, marker: Tuple[int, Marker], idx_dlc: bool = False):
        index, marker = marker
        self.estimated_positions[index]: List[Position] = []
        if marker.is_static:
            return self.estimated_positions[index].append(Position(marker.pos[:2], False))

        # If the marker is visible search for the closest blob
        if self.naive:
            self.get_blob_near_position(marker.pos[:2], index)

        # if we use Kalman then search the closest blob to the prediction
        if self.kalman:
            prediction = self.kalman.kalman_filters[index].predict()
            # self.get_blob_near_position(prediction, index)
            if len(self.blobs) > 0:
                position, visible = find_closest_blob(prediction, self.blobs, delta=Tracker.DELTA)
                self.estimated_positions[index].append(Position(position, visible))
                if not visible:
                    self.kalman.kalman_filters[index].correct(self.marker_set.markers[index].pos[:2])
            else:
                self.estimated_positions[index].append(Position(prediction, True))

        # If we use optical flow get the estimation, if the flow has been found
        # and the level of error is below the threshold then take the estimation
        # Then search for the closest blob, if found update the position estimated
        if self.optical_flow:
            estimation, st, err = self.optical_flow[index]
            if self.from_dlc and marker.name in self.marker_set.markers_from_dlc:
                if marker.name in self.marker_set.dlc_enhance_markers:
                    if len(self.blobs) > 0:
                        self.get_blob_near_position(estimation, index)
                    else:
                        self.estimated_positions[index].append(Position(estimation, True))
                else:
                    self.estimated_positions[index].append(Position(estimation, False))
            else:
                self.get_blob_near_position(estimation, index)

        if self.from_dlc:
            # if marker.name in self.marker_set.markers_from_dlc:
            # if marker.name in self.marker_set.dlc_enhance_markers:
            if not marker.from_dlc:
                pos, likelihood, dlc_visible = [0, 0], 0, False
            else:
                pos, likelihood = self.dlc_live[idx_dlc]
                dlc_visible = likelihood > self.dlc_live.p_cutoff
            # else:
            #     pos, likelihood, dlc_visible = [0, 0], 0, False
            self.estimated_positions[index].append(Position(pos, dlc_visible))
            self.likelihood.append(likelihood)

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
                if self.positions[i] == () or self.positions[j] == ():
                    continue
                if tuple(self.positions[i].position) == tuple(self.positions[j].position):
                    self._correct_overlapping(i, j)

    # Check new_positions are within the crops
    def check_bounds(self, frame: CropFrames = None, size: [int, int] = None):
        max_x, max_y, min_x, min_y = 0, 0, 0, 0
        if frame is not None:
            max_x = frame.width - 1
            max_y = frame.height - 1
        if size is not None:
            max_x, max_y = size[0] - 1, size[1] - 1
        if not frame and not size:
            raise ValueError("Either frame or size must be given")
        if frame and size and (frame.width, frame.height) != size:
            raise ValueError("Frame and size must have the same dimensions")

        for i in range(len(self.positions)):
            if self.marker_set.markers[i].is_bounded:
                [max_x_tmp, max_y_tmp, min_x_tmp, min_y_tmp] = self.marker_set.markers[i].get_bounds()
            else:
                max_x_tmp, max_y_tmp, min_x_tmp, min_y_tmp = max_x, max_y, min_x, min_y
            if self.positions[i] != ():
                self.positions[i].check_bounds(max_x=max_x_tmp, max_y=max_y_tmp, min_x=min_x_tmp, min_y=min_y_tmp)

    def track(self, frame: CropFrames, depth, blobs):
        self.blobs = blobs
        self.frame = frame
        self.depth = depth

        # if self.kalman:
        #    self.kalman.correct()
        self.correct()

        if self.optical_flow:
            self.optical_flow.set_positions(self.marker_set)

        if self.from_dlc:
            self.dlc_live.get_pose()
            if not self.ignore_all_checks:
                self.dlc_live.check_from_last()

        # Track the next position for all markers
        if self.optical_flow:
            self.optical_flow.get_optical_flow_pos(frame.color, self.depth, self.marker_set)

        # depth_to_show = self.dlc_live.depth_image.copy()
        # for p in range(poses.shape[0]):
        #     cv2.circle(depth_to_show, poses[p, :2].astype(int), 5, (255,0,0), -1)
        #
        # cv2.imshow("depth", depth_to_show)
        # cv2.waitKey(0)

        self.likelihood = []
        count = 0
        idx_dlc = None
        for m, marker in enumerate(self.marker_set.markers):
            from_dlc = marker.name in self.marker_set.markers_from_dlc
            if from_dlc:
                idx_dlc = count
                count += 1
            self._track_marker((m, marker), idx_dlc=idx_dlc)

        self.merge_positions(self.ignore_all_checks)
        if not self.ignore_all_checks:
            self.check_tracking()
            self.check_bounds(frame)

        # if self.kalman:
        #     self.kalman.correct()

        return self.positions, self.estimated_positions

    def correct(self):
        if self.kalman:
            self.kalman.correct()

        if self.optical_flow:
            self.optical_flow.set_positions(self.marker_set)
