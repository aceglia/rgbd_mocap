import time

import numpy as np

from frames.crop_frames import Frames, CropFrames
from markers.marker_set import MarkerSet, Marker
from tracking.tracking_markers import Tracker, Position
from filter.filter import Filter


def get_pixels(array, x, y, delta):
    return array[x - delta: x + delta, y - delta: y + delta]


class DepthCheck:
    DELTA = 8

    @staticmethod
    def check(pos, depth_image):
        if depth_image[pos[1], pos[0]] > 0:
            return depth_image[pos[1], pos[0]], True

        depth = np.median(get_pixels(depth_image,
                                     x=pos[1], y=pos[0],
                                     delta=DepthCheck.DELTA))

        if not np.isfinite(depth):
            depth = -1

        return depth, False


class Crop:
    def __init__(self, area, frame: Frames, marker_set: MarkerSet, filter_option, method):
        # Image
        self.frame = CropFrames(area, frame)

        # Marker
        self.marker_set = marker_set

        # Image computing
        self.filter = Filter(filter_option)
        self.tracker = Tracker(self.frame, marker_set, **method)

    def attribute_depth_from_position(self, positions: list[Position]):
        assert len(positions) == len(self.marker_set.markers)

        for i in range(len(positions)):
            if isinstance(positions[i], Position):
                depth, visibility = DepthCheck.check(positions[i].position, self.frame.depth)
                self.marker_set[i].set_depth(depth, visibility)
            else:
                self.marker_set[i].set_depth_visibility(False)

    def attribute_depth(self):
        for marker in self.marker_set:
            depth, visibility = DepthCheck.check(marker.pos[:2], self.frame.depth)

            marker.set_depth(depth, visibility)

    def track_markers(self):
        # Get updated frame
        self.frame.update_image()

        # Get Blobs
        blobs = self.filter.get_blobs(self.frame)

        # Get tracking positions
        positions, estimate_positions = self.tracker.track(self.frame, blobs)

        # Set depth for the new positions
        self.attribute_depth_from_position(positions)

        return blobs, positions, estimate_positions
