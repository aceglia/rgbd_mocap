import numpy as np
import cv2


class Marker:
    def __init__(self, name, is_static=False):
        self.name = name

        # Position arrays
        self.pos = np.zeros((3,), dtype=np.int32)
        self.last_pos = self.pos.copy()
        self.crop_offset = np.zeros((2,), dtype=np.int32)

        # Visibility and reliability
        self.is_visible = False
        self.is_depth_visible = False
        # self._reliability_index = 0
        self.reliability_index = 0
        # self.mean_reliability_index = 0

        ### Is static
        self.is_static = is_static

    # Getter #####
    def get_pos(self):
        return self.pos[:2]

    def get_depth(self):
        return self.pos[2]

    def get_reliability_index(self, frame_idx):
        return self.reliability_index / (frame_idx + 1)

    def get_visibility(self):
        return self.is_visible

    def get_depth_visibility(self):
        return self.is_depth_visible

    def get_global_pos(self):
        return self.pos[:2] + self.crop_offset

    def get_global_pos_3d(self):
        pos = self.get_global_pos()
        return pos[0], pos[1], self.pos[2]
        # return np.array([self.pos[:2] + self.crop_offset] + self.pos[2])

    # Setter #####
    def set_pos(self, position):
        assert len(position) >= 2

        self.pos[:2] = np.array(position[:2], dtype=np.int32)

    def set_pos_and_last(self, position):
        if position == ():
            return

        self.last_pos[:2] = self.pos[:2]
        self.pos[:2] = position

    def set_crop_offset(self, x, y):
        self.crop_offset = (x, y)

    def set_depth(self, depth, visibility=None):
        self.last_pos[2] = self.pos[2]
        self.pos[2] = depth

        if visibility is not None:
            self.set_depth_visibility(visibility)

    def set_reliability(self, reliability):
        self.reliability_index += reliability

    def set_visibility(self, visibility):
        self.is_visible = visibility

    def set_depth_visibility(self, visibility):
        self.is_depth_visible = visibility
