from ..frames.frames import Frames, np
from multiprocessing import RawArray, RawValue


class SharedFrames(Frames):
    def __init__(self, color_frame, depth_frame, index=None):
        super().__init__(color_frame, depth_frame, index)

        self.color_array = RawArray('c', self.width * self.height)  # 'c' -> value between 0-255
        self.depth_array = RawArray('i', self.width * self.height)  # 'i' -> int32
        self.index = RawValue('i', 0)

        self.color = np.frombuffer(self.color_array, dtype=np.uint8).reshape((self.width, self.height))
        self.depth = np.frombuffer(self.depth_array, dtype=np.int32).reshape((self.width, self.height))

        np.copyto(self.color, color_frame)
        np.copyto(self.depth, depth_frame)
        self.index.value = index

    def set_images(self, color_frame, depth_frame, index=None):
        self.check_color_and_depth(color_frame, depth_frame)

        np.copyto(self.color, color_frame)
        np.copyto(self.depth, depth_frame)
        self.index.value = index

    def get_index(self):
        return self.index.value
