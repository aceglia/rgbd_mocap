import numpy as np

from ..frames.crop_frames import Frames, CropFrames
from ..frames.shared_frames import SharedFrames
from ..markers.marker_set import MarkerSet
from ..tracking.tracking_markers import Tracker, Position
from ..filter.filter import Filter


def get_pixels(array, x, y, delta):
    return array[x - delta: x + delta, y - delta: y + delta]


class DepthCheck:
    DELTA = 8
    DEPTH_SCALE = None

    @staticmethod
    def _check_bounds(pos, depth_image):
        if pos[1] >= depth_image.shape[0] or pos[0] >= depth_image.shape[1]:
            return [depth_image.shape[0] - 1, depth_image.shape[1] - 1]
        return pos

    @staticmethod
    def check(pos, depth_image, depth_min, depth_max):
        if DepthCheck.DEPTH_SCALE is None:
            raise ValueError("Please provide the depth scale to have the markers depth in meters."
                             "You can set it with the DepthCheck.set_depth_scale method.")
        depth, visibility = 0, True

        pos = DepthCheck._check_bounds(pos, depth_image)
        try:
            depth = depth_image[pos[1], pos[0]]
        except:
            print(depth_image.shape, pos)
            return -1, False

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
            return depth * DepthCheck.DEPTH_SCALE, visibility

        return -1, False

    @staticmethod
    def set_depth_scale(scale):
        DepthCheck.DEPTH_SCALE = scale


class Crop:
    def __init__(self, area, frame: Frames, marker_set: MarkerSet, filter_option, method):
        if isinstance(frame, SharedFrames):
            frame.color = np.frombuffer(frame.color_array, dtype=np.uint8).reshape((frame.width, frame.height, 3))
            frame.depth = np.frombuffer(frame.depth_array, dtype=np.int32).reshape((frame.width, frame.height))
            for marker in marker_set:
                marker.pos = np.frombuffer(marker.raw_array_pos, dtype=np.int32)
        # Image
        self.frame = CropFrames(area, frame)

        # Marker
        self.marker_set = marker_set

        self.depth_min = filter_option['distance_in_centimeters'][0] * 10
        self.depth_max = filter_option['distance_in_centimeters'][1] * 10
        self.tracking_option = method
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
