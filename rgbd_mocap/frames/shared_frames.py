from ..frames.frames import Frames, np
from multiprocessing import RawArray, RawValue
import cv2


class SharedFrames(Frames):
    def __init__(self, color_frame, depth_frame, index=None, downsample_ratio=None):
        super().__init__(color_frame, depth_frame, index, downsample_ratio)
        self.color_array = RawArray("c", self.width * self.height) if color_frame is not None else None
        self.depth_array = RawArray("i", self.width * self.height) if depth_frame is not None else None
        self.index = RawValue("i", 0)

        self.color = (
            np.frombuffer(self.color_array, dtype=np.uint8).reshape((self.width, self.height))
            if self.color_array is not None
            else None
        )
        self.depth = (
            np.frombuffer(self.depth_array, dtype=np.int32).reshape((self.width, self.height))
            if self.depth_array is not None
            else None
        )

        if self.downsample_ratio != 1:
            self.color = cv2.resize(self.color, (self.width, self.height)) if self.color is not None else None
            self.depth = cv2.resize(self.depth, (self.width, self.height)) if self.depth is not None else None
        
        if self.color is not None:
            np.copyto(self.color, color_frame)

        if self.depth is not None:
            np.copyto(self.depth, depth_frame)
        self.index.value = index

    def set_images(self, color_frame, depth_frame, index=None):
        color_frame, depth_frame = self.check_color_and_depth(color_frame, depth_frame)

        if self.color is not None:
            np.copyto(self.color, color_frame)

        if self.depth is not None:
            np.copyto(self.depth, depth_frame)
            
        self.index.value = index

    def get_index(self):
        return self.index.value
