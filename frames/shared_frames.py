from frames.frames import Frames, np
from multiprocessing import RawArray


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
