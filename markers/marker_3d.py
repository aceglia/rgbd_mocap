import numpy as np
import cv2


class Marker:
    def __init__(self, name):
        self.name = name
        self.pos = np.zeros((3,))
        self.filtered_pos = np.zeros((3,))
        self.global_filtered_pos = np.zeros((3,))
        self.global_pos = np.zeros((3,))
        self.is_visible = False
        self.is_depth_visible = False
        self._reliability_index = 0
        self.reliability_index = 0
        self.mean_reliability_index = 0
        self.dt = 0.001
        self.n_states = 6
        self.n_measures = 3
        self.kalman = None

    def predict_from_kalman(self):
        self.pos[:] = self.kalman.predict()[:3].reshape(
            3,
        )
        return self.pos[:3]

    def get_reliability_index(self, frame_idx):
        return self.reliability_index / (frame_idx + 1)

    def correct_from_kalman(self, points):
        self.pos[:] = np.array(points)[:, 0]

        _ = self.kalman.correct(np.array(points[:3], dtype=np.float32))[:3].reshape(
            3,
        )

    def init_kalman(self, points):
        self.kalman = cv2.KalmanFilter(self.n_states, self.n_measures)
        self.kalman.transitionMatrix = np.eye(self.n_states, dtype=np.float32)
        # self.kalman.processNoiseCov = np.eye(self.n_states, dtype = np.float32) * 1
        self.kalman.measurementNoiseCov = np.eye(self.n_measures, dtype=np.float32) * 0.005
        self.kalman.errorCovPost = 1.0 * np.eye(self.n_states, self.n_states, dtype=np.float32)
        self.kalman.measurementMatrix = np.zeros((self.n_measures, self.n_states), np.float32)
        self.Measurement_array = []
        self.dt_array = []

        for i in range(0, self.n_states, 6):
            self.Measurement_array.append(i)
            self.Measurement_array.append(i + 1)
            self.Measurement_array.append(i + 2)

        for i in range(0, self.n_states):
            if i not in self.Measurement_array:
                self.dt_array.append(i)

        self.kalman.transitionMatrix[0, 2] = self.dt
        self.kalman.transitionMatrix[1, 3] = self.dt
        self.kalman.transitionMatrix[2, 4] = self.dt

        for i in range(0, self.n_measures):
            self.kalman.measurementMatrix[i, self.Measurement_array[i]] = 1

        self.pos[:2] = np.array(points)
        input_points = np.float32(np.ndarray.flatten(points))
        self.kalman.statePre = np.array([input_points[0], input_points[1], input_points[2], 0, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([input_points[0], input_points[1], input_points[2], 0, 0, 0], dtype=np.float32)
        self.predict_from_kalman()

    def set_global_pos(self, local_pos, start_crop):
        if local_pos[0] is not None and local_pos[1] is not None:
            self.global_pos[0] = local_pos[0] + start_crop[0]
            self.global_pos[1] = local_pos[1] + start_crop[1]
            self.global_pos[2] = local_pos[2]
        else:
            self.global_pos = np.array([None, None, None])

    def set_global_filtered_pos(self, local_pos, start_crop):
        self.global_filtered_pos[0] = local_pos[0] + start_crop[0]
        self.global_filtered_pos[1] = local_pos[1] + start_crop[1]
        self.global_filtered_pos[2] = local_pos[2]
