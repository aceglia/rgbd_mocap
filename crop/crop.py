import time

import numpy as np

from frames.crop_frames import Frames, CropFrames
from rgbd_mocap.marker_class import MarkerSet
from tracking.tracking_markers import Tracker
from filter.filter import Filter


class Crop:
    def __init__(self, area, frame: Frames, marker_set: MarkerSet, options):
        # Image
        self.frame = CropFrames(area, frame)

        # Marker
        self.marker_set = marker_set

        # Image computing
        self.filter = Filter(options['filter_options'])
        self.tracker = Tracker(self.frame, marker_set, **options['tracking_options'])

    def _check_depth(self):
        pass

    def track_markers(self):
        tik = time.time()
        blobs = self.filter.get_blobs(self.frame)

        positions, estimate_positions = self.tracker.track(self.frame, blobs)
        print(time.time() - tik)

        # set_marker_pos(marker_set, positions)

        # img = print_blobs(img, blobs, size=5)
        # img = print_estimated_positions(img, estimate_positions)
        # img = print_marker(img, marker_set)

        # print(positions[-1].__str__(),
        #       [pos.__str__() for pos in estimate_positions[-1]])
        return blobs, positions, estimate_positions

        # TODO check depth


