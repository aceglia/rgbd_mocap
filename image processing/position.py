import numpy as np
from rgbd_mocap.marker_class import Marker


def check_bounds(i, max_bound, min_bound=0):
    if i < min_bound:
        return min_bound, False

    elif i > max_bound:
        return max_bound, False

    return i, True


class Position:
    def __init__(self, position, visibility):
        self.position = position
        self.visibility = visibility

    def __eq__(self, other):
        return self.position == other.position

    def get(self):
        return self.position

    def distance_from_marker(self, marker: Marker):
        return np.linalg.norm(self.position, marker.pos[:2])

    def check_bounds(self, max_x, max_y):
        self.position[0], visibility_x = check_bounds(self.position[0], max_x)
        self.position[1], visibility_y = check_bounds(self.position[1], max_y)

        self.visibility = visibility_x and visibility_y
