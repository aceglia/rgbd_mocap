import time

import numpy as np
import cv2
from rgbd_mocap.GUI.Video_editing.video_edit import VideoEdit
from rgbd_mocap.GUI.Video_editing.image_options import ImageOptions
from ...filters.filter import Filter
from ...frames.frames import Frames
# from video_edit_linker import VideoEditLinker


class VideoFilters:
    def __init__(self, parent):
        self.parent = parent
        self.image_options = ImageOptions(video_filter=self, parent=parent)
        self.filter = Filter(self.image_options)

    ###### Update functions #############################
    def update_params(self):
        ### Update the blob detector params
        if self.image_options.show_params["blob_option"]:
            self.filter.blobs_param.minThreshold = self.image_options["white_range"][0]
            self.filter.blobs_param.maxThreshold = self.image_options["white_range"][1]
            self.filter.blobs_param.minArea = self.image_options["blob_area"][0]
            self.filter.blobs_param.maxArea = self.image_options["blob_area"][1]
            self.filter.blobs_param.minCircularity = self.image_options["circularity"] / 100
            self.filter.blobs_param.minConvexity = self.image_options["convexity"] / 100
            self.filter.blobs_param.minDistBetweenBlobs = self.image_options['distance_between_blobs']

        ### Update the clahe filters
        if self.image_options.show_params["clahe_option"]:
            self.filter.clahe.setClipLimit(self.image_options["clahe_clip_limit"])
            self.filter.clahe.setTilesGridSize((self.image_options["clahe_grid_size"],
                                         self.image_options["clahe_grid_size"]))

    ##### Main functions ####################################
    def update(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """
        self.update_params()
        if self.parent.ve.color_frame is not None:
            self.parent.ve.filtered_frame = self.parent.ve.color_frame.copy()
            self.parent.ve.apply_mask([])
            color = self.parent.ve.color_frame.copy()
            color[self.parent.ve.filtered_frame == 0] = 0
            frame = Frames(color, self.parent.ve.depth_frame.copy())
            self.parent.ve.blobs = self.filter.get_blobs(frame)
            self.parent.ve.filtered_frame = self.filter.filtered_frame.copy()
            self.parent.ve.apply_blend(self.image_options["blend"])
            self.parent.ve.apply_blob_detect()
            self.parent.ve.update()
