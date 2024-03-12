import numpy as np


class Bounds:
    def __init__(self, min, max):
        self.min = min if min >= 0 else 0
        self.max = max if max >= 0 else 0

    def set_min(self, min):
        self.min = min

    def set_max(self, max):
        self.max = max

    def set(self, min, max):
        self.set_min(min)
        self.set_max(max)

    def get(self):
        return self.min, self.max


class Marker:
    def __init__(self, name, is_static=False):
        self.x_bounds = None
        self.y_bounds = None
        self.name = name

        # Position arrays
        self.pos = np.zeros((2,), dtype=np.int32)
        self.depth = -1
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
        self.is_bounded = False

    # Getter #####
    def get_pos(self):
        return self.pos[:2]

    def get_depth(self):
        return self.depth

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
        return pos[0], pos[1], self.get_depth()
        # return np.array([self.pos[:2] + self.crop_offset] + self.pos[2])

    # Setter #####
    def set_pos(self, position):
        assert len(position) >= 2

        self.pos[:2] = position[:2]

    def set_pos_and_last(self, position):
        if position == ():
            return

        self.last_pos[:2] = self.pos[:2]
        self.pos[:2] = position

    def set_crop_offset(self, x, y):
        self.crop_offset = (x, y)

    def set_depth(self, depth, visibility=None):
        # self.last_pos[2] = self.pos[2]
        self.depth = depth

        if visibility is not None:
            self.set_depth_visibility(visibility)

    def set_bounds(self, bounds):
        self.x_bounds = Bounds(min=bounds[0][0] + self.pos[0], max=bounds[0][1] + self.pos[0])
        self.y_bounds = Bounds(min=bounds[1][0] + self.pos[1], max=bounds[1][1] + self.pos[1])
        self.is_bounded = True

    def get_bounds(self):
        """
        Get the bounds of the marker
        Returns
        -------
        list
            max_x, max_y, min_x, min_y
        """
        bounds = [None, None, None, None]
        if self.x_bounds is not None:
            bounds[2], bounds[0] = self.x_bounds.get()
        if self.y_bounds is not None:
            bounds[3], bounds[1] = self.y_bounds.get()
        return bounds

    def set_reliability(self, reliability):
        self.reliability_index += reliability

    def set_visibility(self, visibility):
        self.is_visible = visibility

    def set_depth_visibility(self, visibility):
        self.is_depth_visible = visibility
