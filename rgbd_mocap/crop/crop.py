import numpy as np

from ..frames.crop_frames import Frames, CropFrames
from ..markers.marker_set import MarkerSet
from ..tracking.tracking_markers import Tracker, Position
from ..filter.filter import Filter


def get_pixels(array, x, y, delta):
    return array[x - delta: x + delta, y - delta: y + delta]


class DepthCheck:
    DELTA = 8

    @staticmethod
    def check(pos, depth_image, depth_min, depth_max):
        depth, visibility = 0, True

        if depth_image[pos[1], pos[0]] > 0:
            depth = depth_image[pos[1], pos[0]]

        else:
            visibility = False
            n_d = get_pixels(depth_image,
                             x=pos[1], y=pos[0],
                             delta=DepthCheck.DELTA)
            n_d = n_d[n_d != -1]
            n_d = n_d[n_d != 0]

            if n_d is not []:
                depth = np.median(n_d)

            else:
                return -1, False

        if depth_min <= depth <= depth_max:
            return depth, visibility

        return -1, False


class Crop:
    def __init__(self, area, frame: Frames, marker_set: MarkerSet, filter_option, method):
        # Image
        self.frame = CropFrames(area, frame)

        # Marker
        self.marker_set = marker_set

        self.depth_min = filter_option['distance_in_centimeters'][0] * 10
        self.depth_max = filter_option['distance_in_centimeters'][1] * 10

        # Image computing
        self.filter = Filter(filter_option)
        self.tracker = Tracker(self.frame, marker_set, **method)

    def attribute_depth_from_position(self, positions: list[Position]):
        assert len(positions) == len(self.marker_set.markers)

        for i in range(len(positions)):
            if isinstance(positions[i], Position):
                depth, visibility = DepthCheck.check(positions[i].position,
                                                     self.frame.depth,
                                                     self.depth_min,
                                                     self.depth_max)
                if depth != -1:
                    self.marker_set[i].set_depth(depth)
                self.marker_set[i].set_depth_visibility(visibility)
            else:
                self.marker_set[i].set_depth_visibility(False)

    def re_init(self, marker_set, method):
        self.tracker = Tracker(self.frame, marker_set, **method)

    def attribute_depth(self):
        self.attribute_depth_from_position([marker.pos for marker in self.marker_set])

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
