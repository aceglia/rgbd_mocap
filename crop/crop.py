import numpy as np

from frames.crop_frames import Frames, CropFrames
from rgbd_mocap.marker_class import MarkerSet
from tracking.tracking_markers import Tracker
from filter.filter import Filter


class Crop:
    def __init__(self, area, frame: Frames, marker_set: MarkerSet, filter_options):
        # Image
        self.crop = CropFrames(area, frame)

        # Marker
        self.marker_set = marker_set

        # Image computing
        self.filter = Filter(filter_options)
        self.tracker = Tracker(self.crop, marker_set)

    def track_markers(self):
        # TODO get blobs with filter

        blobs = self.filter.get_blobs(self.crop)

        positions, estimate_positions = self.tracker.track(self.crop, blobs)

        set_marker_pos(marker_set, positions)
        tracker.optical_flow.set_positions([marker.pos[:2] for marker in marker_set])

        # img = print_blobs(img, blobs, size=5)
        img = print_estimated_positions(img, estimate_positions)
        img = print_marker(img, marker_set)

        # TODO check depth


