import numpy as np
from multiprocessing import RawArray, RawValue
from ..markers.marker import Marker


c_int = 'i'
c_bool = 'c'
c_float = 'd'


class SharedMarker(Marker):
    def __init__(self, name, is_static=False):
        super().__init__(name, is_static)

        ### Shared Memory
        self.raw_array_pos = RawArray(c_int, 2)
        self.pos = np.frombuffer(self.raw_array_pos, dtype=np.int32)
        self.depth = RawValue(c_float, 0)
        self.last_pos = self.pos.copy()

        # Visibility and reliability
        self.is_visible = RawValue(c_bool, False)
        self.is_depth_visible = RawValue(c_bool, False)
        # self._reliability_index = RawValue(c_float, 0)
        self.reliability_index = RawValue(c_float, 0)
        # self.mean_reliability_index = RawValue(c_float, 0)

    # Override getters for shared data
    def get_reliability_index(self, frame_idx):
        return self.reliability_index.value / (frame_idx + 1)

    def get_visibility(self):
        return self.is_visible.value

    def get_depth_visibility(self):
        return self.is_depth_visible.value

    # Override setters for shared data
    def set_reliability(self, reliability):
        self.reliability_index.value += reliability

    def set_visibility(self, visibility):
        self.is_visible.value = visibility

    def set_depth_visibility(self, visibility):
        self.is_depth_visible.value = visibility
