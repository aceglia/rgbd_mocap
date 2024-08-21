import numpy as np
from typing import List

from ..frames.crop_frames import Frames, CropFrames
from ..frames.shared_frames import SharedFrames
from ..markers.marker_set import MarkerSet
from ..tracking.tracking_markers import Tracker, Position
from ..filters.filter import Filter
from ..tracking.dlc_live import DlcLive


def get_pixels(array, x, y, delta):
    pos = -1
    pos_2d = [x, y]
    delta = 1
    while pos <= 0 and delta < 15:
        d = array[pos_2d[0] - delta: pos_2d[0] + delta, pos_2d[1] - delta: pos_2d[1] + delta]
        if len(d[d > 0]) > 0:
            pos = np.median(d[d > 0])
        delta += 1
    return pos


class DepthCheck:
    DELTA = 10
    DEPTH_SCALE = 0.001

    @staticmethod
    def _check_bounds(pos, depth_image):
        if pos[1] >= depth_image.shape[0] or pos[0] >= depth_image.shape[1]:
            return [depth_image.shape[1] - 1, depth_image.shape[0] - 1]
        return pos

    @staticmethod
    def check(pos, depth_image, depth_min, depth_max):
        if DepthCheck.DEPTH_SCALE is None:
            raise ValueError("Please provide the depth scale to have the markers depth in meters."
                             "You can set it with the DepthCheck.set_depth_scale method.")
        depth, visibility = 0, True

        pos = DepthCheck._check_bounds(pos, depth_image)

        if depth_image[pos[1], pos[0]] > 0:
            depth = depth_image[pos[1], pos[0]]

        else:
            visibility = False
            depth = get_pixels(depth_image,
                             x=pos[1], y=pos[0],
                             delta=DepthCheck.DELTA)

        if depth_min <= depth <= depth_max:
            return depth * DepthCheck.DEPTH_SCALE, visibility

        return -1, False

    @staticmethod
    def set_depth_scale(scale):
        DepthCheck.DEPTH_SCALE = scale


class Crop:
    def __init__(self, area, frame: Frames, marker_set: MarkerSet, filter_option, method, from_dlc=False,
                 dlc_model=None, dlc_processor=None, ignore_all_checks=False):
        if isinstance(frame, SharedFrames):
            frame.color = np.frombuffer(frame.color_array, dtype=np.uint8).reshape((frame.width, frame.height))
            frame.depth = np.frombuffer(frame.depth_array, dtype=np.int32).reshape((frame.width, frame.height))
            for marker in marker_set:
                marker.pos = np.frombuffer(marker.raw_array_pos, dtype=np.int32)

        # Marker
        self.marker_set = marker_set
        self.from_dlc = from_dlc
        self.ignore_all_checks=ignore_all_checks
        self.dlc_live = None
        # Image
        self.frame = CropFrames(area, frame)

        self.depth_min = filter_option['distance_in_centimeters'][0] / 100 / DepthCheck.DEPTH_SCALE
        self.depth_max = filter_option['distance_in_centimeters'][1] / 100 / DepthCheck.DEPTH_SCALE
        self.tracking_option = method
        # Image computing
        self.filter = Filter(filter_option)
        if from_dlc:
            self.dlc_live = DlcLive(dlc_model, dlc_processor, depth_scale=DepthCheck.DEPTH_SCALE)
            first_markers_position = self.dlc_live.get_pose(self.frame.depth, min_depth=200, bg_remover_threshold=1.2)[:, :2]
            self.set_marker_pos(first_markers_position)
            self.attribute_depth()
        self.tracker = Tracker(
            self.frame, marker_set, depth_range=[self.depth_min, self.depth_max], from_dlc=from_dlc,
            dlc_live=self.dlc_live, ignore_all_checks=ignore_all_checks, **method
        )

    def attribute_depth_from_position(self, positions: list[Position]):
        assert len(positions) == len(self.marker_set.markers)
        for i in range(len(positions)):
            if self.marker_set[i].is_static and self.marker_set[i].get_depth() != -1:
                continue
            if isinstance(positions[i], Position):
                depth, visibility = DepthCheck.check(positions[i].position,
                                                     self.frame.depth,
                                                     self.depth_min,
                                                     self.depth_max)
                if self.ignore_all_checks:
                    self.marker_set[i].set_depth(depth)
                    self.marker_set[i].set_depth_visibility(visibility)
                    continue

                if self.marker_set[i].get_depth() == -1 or depth != -1 and abs(self.marker_set[i].get_depth() - depth) < 0.08:
                    self.marker_set[i].set_depth(depth)
                self.marker_set[i].set_depth_visibility(visibility)
            else:
                self.marker_set[i].set_depth_visibility(False)

    def re_init(self, marker_set, method):
        self.tracker = Tracker( self.frame, marker_set, depth_range=[self.depth_min, self.depth_max],
                                from_dlc=self.from_dlc,
            dlc_live=self.dlc_live, **method)

    def attribute_depth(self):
        self.attribute_depth_from_position([Position(marker.pos, True) for marker in self.marker_set])

    def set_marker_pos(self, positions: List[Position], dlc_enhance_markers=()):
        assert len(self.marker_set.markers_from_dlc) == len(positions)
        if isinstance(positions, np.ndarray):
            positions = positions.astype(int)
        count = 0
        for i, mark in enumerate(self.marker_set.markers):
            # if mark.name not in self.marker_set.markers_from_dlc and mark.name not in self.marker_set.dlc_enhance_markers:
            if not mark.from_dlc and mark.name not in self.marker_set.dlc_enhance_markers:
                if mark.name in self.marker_set.markers_from_dlc:
                    count += 1
                continue
        # for i in range(len(positions)):
            if self.marker_set[i].is_bounded and self.marker_set[i].x_bounds.min == 0 and self.marker_set[i].y_bounds.min == 0:
                current_max = self.marker_set[i].x_bounds.max
                self.marker_set[i].x_bounds.set(positions[count][0] - current_max, positions[count][0] + current_max)
                current_max = self.marker_set[i].y_bounds.max
                self.marker_set[i].y_bounds.set(positions[count][1] - current_max, positions[count][1] + current_max)
            self.marker_set[i].set_pos(positions[count].astype(int))
            self.marker_set[i].set_visibility(True)
            count += 1

    @staticmethod
    def draw_blobs(frame, blobs, color=(255, 0, 0), scale=5):
        import cv2
        if blobs is not None:
            for blob in blobs:
                frame = cv2.circle(frame, (int(blob[0]), int(blob[1])), scale, color, 1)
        return frame

    def track_markers(self):

        # Get updated frame
        self.frame.update_image()

        # Get Blobs
        get_blobs = False
        if not self.from_dlc:
            get_blobs = True
        if self.tracking_option["optical_flow"]:
            get_blobs = True
        blobs = [] if not get_blobs else self.filter.get_blobs(self.frame)
        # if self.from_dlc and self.tracking_option["optical_flow"]:
        #     self.filter.apply_filters(self.frame)
        if self.from_dlc:
            self.dlc_live.update_depth_frame(self.filter.filtered_depth)
        # Get tracking positions
        positions, estimate_positions = self.tracker.track(self.frame, self.filter.filtered_depth, blobs)

        # Set depth for the new positions
        self.attribute_depth_from_position(positions)

        return blobs, positions, estimate_positions
