import numpy as np

from frames.crop_frames import Frames, CropFrames
from rgbd_mocap.marker_class import MarkerSet
from tracking.tracking_markers import Tracker


class Crop:
    def __init__(self, area, frame: Frames, marker_set: MarkerSet, image_filter):
        # Image
        self.crop = CropFrames(area, frame)

        # Marker
        self.marker_set = marker_set

        # Image computing
        self.filter = image_filter
        self.tracker = Tracker(self.crop, marker_set)


