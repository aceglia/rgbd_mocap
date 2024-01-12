import numpy as np
from multiprocessing import RawArray


class Frames:
    def __init__(self, color_frame, depth_frame):
        if color_frame is None or depth_frame is None:
            raise ValueError(f'{self}: color_frame and depth frame should be init.')

        self.width = color_frame.shape[0]
        self.height = color_frame.shape[1]

        self.color = np.copy(color_frame)
        self.depth = np.copy(depth_frame)

    def _shape_error(self, got, expected):
        raise ValueError(
            f'{self}: Given array has a wrong shape, got "{got.shape}" expected "{expected.shape}".')

    def _check_frame(self, old, new):
        if old.shape != new.shape:
            self._shape_error(old, new)

    def check_color_and_depth(self, color_frame, depth_frame):
        self._check_frame(self.color, color_frame)
        self._check_frame(self.depth, depth_frame)

    def set_images(self, color_frame, depth_frame):
        self.check_color_and_depth(color_frame, depth_frame)

        self.color = np.copy(color_frame)
        self.depth = np.copy(depth_frame)

    def get_images(self):
        return self.color, self.depth


class SharedFrames(Frames):
    def __init__(self, color_frame, depth_frame):
        super().__init__(color_frame, depth_frame)

        color_array = RawArray('c', self.width * self.height * 3)  # 'c' -> value between 0-255
        depth_array = RawArray('i', self.width * self.height)  # 'i' -> int32

        self.color = np.frombuffer(color_array, dtype=np.uint8).reshape((self.width, self.height, 3))
        self.depth = np.frombuffer(depth_array, dtype=np.int32).reshape((self.width, self.height))

        np.copyto(self.color, color_frame)
        np.copyto(self.depth, depth_frame)

    def set_images(self, color_frame, depth_frame):
        self.check_color_and_depth(color_frame, depth_frame)

        np.copyto(self.color, color_frame)
        np.copyto(self.depth, depth_frame)