import numpy as np
import cv2


class Frames:
    def __init__(self, color_frame, depth_frame, index, downsample_ratio=1):
        self.unscaled_depth = None
        self.color = None if color_frame is None else np.copy(color_frame)
        self.depth = np.copy(depth_frame) if depth_frame is not None else None
        self.downsample_ratio = downsample_ratio
        self.width = int(color_frame.shape[0] * downsample_ratio)
        self.height = int(color_frame.shape[1] * downsample_ratio)
        self.index = index

    def _shape_error(self, got, expected):
        raise ValueError(f'{self}: Given array has a wrong shape, got "{got.shape}" expected "{expected.shape}".')

    def _check_frame(self, old, new):
        # if self.downsample_ratio != 1:
        #     new = cv2.resize(new, (self.height, self.width))
        if old.shape != new.shape:
            self._shape_error(old, new)
        return new

    def check_color_and_depth(self, color_frame, depth_frame):
        color = None
        depth = None
        if color_frame is not None:
            color = self._check_frame(self.color, color_frame)
        if depth_frame is not None:
            depth = self._check_frame(self.depth, depth_frame)
        return color, depth

    def set_images(self, color_frame, depth_frame, index):
        self.unscaled_depth = depth_frame.copy() if depth_frame is not None else None
        color_frame, depth_frame = self.check_color_and_depth(color_frame, depth_frame)
        self.color = None if self.color is None else np.copy(color_frame)
        self.depth = np.copy(depth_frame) if depth_frame is not None else None
        self.index = index

    def get_images(self):
        return self.color, self.depth

    def get_index(self):
        return self.index

    @staticmethod
    def _get_crop(image, area):
        if image is None:
            return None
        return image[area[1] : area[3], area[0] : area[2]]

    def get_crop(self, area, downsample_ratio=1):
        color, depth = self._get_crop(self.color, area), self._get_crop(self.depth, area)
        if downsample_ratio != 1:
            w = int(depth.shape[0] * downsample_ratio) if depth is not None else color.shape[0]
            h = int(depth.shape[1] * downsample_ratio) if depth is not None else color.shape[1]
            if self.downsample_ratio != 1:
                color = cv2.resize(color, (h, w)) if color is not None else None
                depth = cv2.resize(depth, (h, w)) if depth is not None else None
        return color, depth
