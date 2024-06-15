import cv2
import numpy as np

from ..markers.marker_set import MarkerSet


class Kalman:
    def __init__(self, points, fps=60, n_states=4, n_measures=2):
        """
        points: Origin of the marker
        """
        # self.dt = 1/fps
        self.dt = 1/fps
        self.n_states = n_states
        self.n_measures = n_measures

        self.Measurement_array = []
        self.dt_array = []
        self.kalman: cv2.KalmanFilter = None
        self.last_predicted_pos = None
        self.last_corrected_pos = None

        self.init_kalman(points)

    def init_kalman(self, points):
        self.kalman = cv2.KalmanFilter(self.n_states, self.n_measures)
        self.kalman.transitionMatrix = np.eye(self.n_states, dtype=np.float32)
        self.kalman.measurementNoiseCov = np.eye(self.n_measures, dtype=np.float32) * 10
        # self.kalman.errorCovPost = 5 * np.eye(self.n_states, self.n_states, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(self.n_states, dtype=np.float32) * 2
        # self.kalman.errorCovPre = np.eye(self.n_states, self.n_states, dtype=np.float32) * 5

        # self.kalman.measurementMatrix = np.zeros((self.n_measures, self.n_states), dtype=np.float32)
        self.kalman.measurementMatrix = np.zeros((self.n_measures, self.n_states), dtype=np.float32)
        self.kalman.measurementMatrix[:self.n_measures, :self.n_measures] = np.eye((self.n_measures))
        self.kalman.transitionMatrix[:self.n_measures, self.n_measures:] = np.eye((self.n_measures)) * self.dt

        pos = np.array(points)
        input_points = np.float32(np.ndarray.flatten(points))
        input_points_list = []
        for n in range(self.n_states):
            if n < self.n_states/2:
                input_points_list.append(input_points[n])
            else:
                input_points_list.append(0)

        self.kalman.statePre = np.array(input_points_list, dtype=np.float32)
        self.kalman.statePost = np.array(input_points_list, dtype=np.float32)
        self.last_predicted_pos = self.predict()
        # self.kalman.predict()
        return self.kalman, pos

    def predict(self):
        self.last_predicted_pos = self.kalman.predict()[:self.n_measures].reshape(self.n_measures, )
        return self.last_predicted_pos

    def correct(self, points):
        self.last_corrected_pos = self.kalman.correct(np.array(points[:self.n_measures], dtype=np.float32))[:self.n_measures].reshape(self.n_measures, )
        return self.last_corrected_pos


class KalmanSet:
    def __init__(self, marker_set: MarkerSet):
        self.marker_set = marker_set
        self.kalman_filters: list[Kalman] = []

        for marker in marker_set.markers:
            k = Kalman(marker.get_pos())
            self.kalman_filters.append(k)

    def __getitem__(self, item):
        return self.kalman_filters[item]

    def __iter__(self):
        for f in self.kalman_filters:
            yield f

    def predict(self):
        return [kalman.predict() for kalman in self]

    def correct_from_positions(self, positions: list[tuple[2]]):
        [kalman.correct(positions[i]) for i, kalman in enumerate(self)]

    def correct(self):
        for m, marker in enumerate(self.marker_set.markers):
            if marker.is_visible:
                self.kalman_filters[m].correct(marker.pos[:2])

    def reinit_kalman(self):
        [kalman.init_kalman(self.marker_set[i].pos[:2]) for i, kalman in enumerate(self)]

