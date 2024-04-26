import numpy as np


class Frames:
    def __init__(self, color_frame, depth_frame, index):
        # if color_frame is None or depth_frame is None:
        #     raise ValueError(f'{self}: color_frame and depth frame should be init.')

        self.width = depth_frame.shape[0]
        self.height = depth_frame.shape[1]

        self.color = None if color_frame is None else np.copy(color_frame)
        self.depth = np.copy(depth_frame)
        self.index = index

    def _shape_error(self, got, expected):
        raise ValueError(
            f'{self}: Given array has a wrong shape, got "{got.shape}" expected "{expected.shape}".')

    def _check_frame(self, old, new):
        if old.shape != new.shape:
            self._shape_error(old, new)

    def check_color_and_depth(self, color_frame, depth_frame):
        if color_frame is not None:
            self._check_frame(self.color, color_frame)
        self._check_frame(self.depth, depth_frame)

    def set_images(self, color_frame, depth_frame, index):
        self.check_color_and_depth(color_frame, depth_frame)
        self.color = None if self.color is None else np.copy(color_frame)
        self.depth = np.copy(depth_frame)
        self.index = index

    def get_images(self):
        return self.color, self.depth

    def get_index(self):
        return self.index

    @staticmethod
    def _get_crop(image, area):
        if image is None:
            return None
        return image[area[1]: area[3], area[0]:area[2]]

    def get_crop(self, area):
        return self._get_crop(self.color, area), self._get_crop(self.depth, area)
