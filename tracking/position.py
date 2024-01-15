import math

import numpy as np
from rgbd_mocap.marker_class import Marker


def check_bounds(i, max_bound, min_bound=0):
    if i < min_bound:
        print(i)
        return min_bound, False

    elif i > max_bound:
        print(i)
        return max_bound, False

    return i, True


class Position:
    def __init__(self, position, visibility):
        self.position = np.array(position, dtype=np.int32)
        self.visibility = visibility

    def __eq__(self, other):
        if type(other) is not Position:
            return False

        return self.position[0] == other.position[0] and self.position[1] == other.position[1]

    def __str__(self):
        return f'{self.position}: Visible {self.visibility}'

    def get(self):
        return self.position, self.visibility

    def set(self, position, visibility):
        self.position = position
        self.visibility = visibility

    def distance_from_marker(self, marker: Marker):
        print(self.position, marker.pos[:2])

        dist = math.sqrt(((self.position[0] - marker.pos[0]) ** 2) *
                         ((self.position[1] - marker.pos[1]) ** 2))

        return dist

        # return np.linalg.norm(self.position, marker.pos[:2])

    def check_bounds(self, max_x, max_y):
        self.position[0], visibility_x = check_bounds(self.position[0], max_x)
        self.position[1], visibility_y = check_bounds(self.position[1], max_y)

        self.visibility = visibility_x and visibility_y
