import cv2
import numpy as np

from ..markers.marker_set import MarkerSet


class Kalman:
    def __init__(self, points, fps=60, n_measures=2, n_diff=1, init=True, **kwargs):
        """
        points: Origin of the marker
        """
        # self.dt = 1/fps
        self.last_pos = None
        self.dt = 1/fps
        # self.dt = 1
        self.measurement_noise_factor = 10
        self.process_noise_factor = 2
        self.error_cov_post_factor = 0
        self.error_cov_pre_factor = 0
        self.n_diff = n_diff

        for key in kwargs:
            self.__dict__[key] = kwargs[key]

        self.n_measures = n_measures
        self.n_states = self.n_measures * (self.n_diff + 1)

        self.Measurement_array = []
        self.dt_array = []
        self.kalman: cv2.KalmanFilter = None
        self.last_predicted_pos = None
        self.last_corrected_pos = None
        self.initial_points = points
        if init:
            self.init_kalman(points)

    def init_kalman(self, points=None):
        points = self.initial_points if points is None else points
        self.kalman = cv2.KalmanFilter(self.n_states, self.n_measures, 0)
        self.kalman.measurementNoiseCov = np.eye(self.n_measures, dtype=np.float32) * float(self.measurement_noise_factor)
        self.kalman.errorCovPost = np.eye(self.n_states, self.n_states, dtype=np.float32) * float(self.error_cov_post_factor)
        self.kalman.processNoiseCov = np.eye(self.n_states, dtype=np.float32) * float(self.process_noise_factor)
        self.kalman.errorCovPre = np.eye(self.n_states, self.n_states, dtype=np.float32) * float(self.error_cov_pre_factor)

        self.kalman.measurementMatrix = np.zeros((self.n_measures, self.n_states), dtype=np.float32)
        self.kalman.measurementMatrix[:self.n_measures, :self.n_measures] = np.eye((self.n_measures))
        # F = np.zeros((self.n_states, self.n_states), np.float32)
        # for d in range(self.n_diff + 1):
        #     for i in range(self.n_states - (d * 2)):
        #         F[i, i + (2 * d)] = (self.dt ** d) / max(1, d)
        # A = np.eye(self.n_diff).astype(np.float32)
        # for i in range(self.n_diff):
        #     for j in range(i):
        #         A[i, j] = self.dt ** (i - j) / np.math.factorial(i - j)
        # self.kalman.measurementMatrix = A
        # self.kalman.measurementMatrix = np.array([
        #     [1, 0, self.dt, 0, 0.5 * self.dt ** 2, 0],
        #     [0, 1, 0, self.dt, 0, 0.5 * self.dt ** 2],
        #     [0, 0, 1, 0, self.dt, 0],
        #     [0, 0, 0, 1, 0, self.dt],
        #     [0, 0, 0, 0, 1, 0],
        #     [0, 0, 0, 0, 0, 1]
        # ], dtype=np.float32)
        self.kalman.transitionMatrix = self._get_transition_matrix(self.dt)

        pos = np.round(np.array(points), 3)
        input_points = np.float32(np.ndarray.flatten(points))
        input_points_list = []
        for n in range(self.n_states):
            if n < self.n_states/(self.n_diff + 1):
                input_points_list.append(input_points[n])
            else:
                input_points_list.append(0)

        self.kalman.statePre = np.array(input_points_list, dtype=np.float32)
        self.kalman.statePost = np.array(input_points_list, dtype=np.float32)
        self.last_predicted_pos = self.predict()
        return self.kalman, pos

    def predict(self):
        prediction = self.kalman.predict()
        self.last_predicted_pos = prediction[:self.n_measures].reshape(self.n_measures, )
        self.last_pos = prediction
        return self.last_predicted_pos

    def correct(self, points):
        points = np.round(np.array(points[:self.n_measures]), 8).astype(np.float32)
        self.last_pos = self.kalman.correct(points)
        self.last_corrected_pos = self.last_pos[:self.n_measures].reshape(self.n_measures, )
        return self.last_corrected_pos

    def _get_transition_matrix(self, dt):
        # A = np.eye(self.n_states, dtype=np.float32)
        # A[:self.n_measures, self.n_measures:self.n_measures * 2] = np.eye((self.n_measures)) * dt
        # if self.n_diff > 1:
        #     A[self.n_measures:self.n_measures * 2, self.n_measures * 2:] = np.eye((self.n_measures)) * dt ** 2
        A = np.eye(self.n_states, self.n_states, dtype=np.float32)
        A[:self.n_measures, self.n_measures:self.n_measures * 2] = np.eye((self.n_measures)) * dt
        if self.n_diff == 2:
            A[:self.n_measures, self.n_measures * 2:] = np.eye((self.n_measures)) * 0.5 * dt ** 2

        #A[:self.n_measures, self.n_measures:] = np.eye((self.n_measures)) * dt
        return A
    def get_future_pose(self, dt=None):
        if dt is None:
            dt = self.dt
        A = self._get_transition_matrix(dt)
        future_pose = np.dot(A, self.last_pos)
        future_pose = future_pose[:self.n_measures].reshape(self.n_measures, )
        return future_pose

    def set_params(self, measurement_noise_factor=None, process_noise_factor=None, error_cov_post_factor=None, error_cov_pre_factor=None):
        if measurement_noise_factor is not None:
            self.measurement_noise_factor = measurement_noise_factor
        if process_noise_factor is not None:
            self.process_noise_factor = process_noise_factor
        if error_cov_post_factor is not None:
            self.error_cov_post_factor = error_cov_post_factor
        if error_cov_pre_factor is not None:
            self.error_cov_pre_factor = error_cov_pre_factor

class KalmanSet:
    def __init__(self, marker_set: MarkerSet):
        self.marker_set = marker_set
        self.kalman_filters: list[Kalman] = []

        for marker in marker_set.markers:
            k = Kalman(marker.get_pos(), n_diff=2, init=False)
            k.set_params(measurement_noise_factor=1e-3, process_noise_factor=1e-1, error_cov_post_factor=0, error_cov_pre_factor=0)
            k.init_kalman()
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

