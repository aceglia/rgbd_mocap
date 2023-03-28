import numpy as np
import cv2


class Marker:
    def __init__(self, name):
        self.name = name
        self.pos = np.zeros((2, 1), dtype=int)
        self.filtered_pos = np.zeros((2, 1), dtype=int)
        self.depth = np.zeros((1, 1))
        self.speed = np.zeros((2, 1))
        self.next_pos = np.zeros((2, 1), dtype=int)
        self.next_predicted_area = np.zeros((2, 1), dtype=int)
        self.is_visible = False
        self.dt = 1
        n_states = 4
        n_measures = 2

        self.kalman = cv2.KalmanFilter(n_states, n_measures)
        self.kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
        # kalman.processNoiseCov = np.eye(n_states, dtype = np.float32)*0.9
        self.kalman.measurementNoiseCov = np.eye(n_measures, dtype=np.float32) * 0.0005

        self.kalman.measurementMatrix = np.zeros((n_measures, n_states), np.float32)
        self.Measurement_array = []
        self.dt_array = []
        for i in range(0, n_states, 4):
            self.Measurement_array.append(i)
            self.Measurement_array.append(i + 1)

        for i in range(0, n_states):
            if i not in self.Measurement_array:
                self.dt_array.append(i)

        for i, j in zip(self.Measurement_array, self.dt_array):
            self.kalman.transitionMatrix[i, j] = self.dt
        for i in range(0, n_measures):
            self.kalman.measurementMatrix[i, self.Measurement_array[i]] = 1

    @staticmethod
    def compute_speed(pos, pos_old, dt=1):
        """
        Compute the speed of the markers
        """
        return (pos - pos_old) / dt

    @staticmethod
    def compute_next_position(speed, pos, dt=1):
        """
        Compute the next position of the markers
        """
        return pos + speed * dt

    def predict_from_kalman(self):
        self.filtered_pos = self.kalman.predict()[:2].astype(int)

    def correct_from_kalman(self, points):
        self.pos = np.array(points, dtype=int)
        self.filtered_pos = self.kalman.correct(np.array(points, dtype=np.float32))[:2].astype(int)

    def init_kalman(self, points):
        self.pos = np.array(points, dtype=int)
        input_points = np.float32(np.ndarray.flatten(points))
        self.kalman.statePost = np.array([input_points[0], input_points[1], 0, 0], dtype=np.float32)
        self.kalman.statePre = np.array([input_points[0], input_points[1], 0, 0], dtype=np.float32)


class MarkerSet:
    """
    This class is used to store the marker information
    """
    def __init__(self, marker_names: list[str], image_idx: int = 0, fps=30):
        """
        init markers class with number of markers, names and image index

        Parameters
        ----------
        marker_names : list
            list of names for the markers
        image_idx : list
            index of the image where the marker set is located
        """
        self.markers = []
        for marker_name in marker_names:
            marker = Marker(name=marker_name)
            marker.dt = 1 / fps
            self.markers.append(marker)

        self.nb_markers = len(marker_names)
        self.image_idx = image_idx
        self.marker_names = marker_names
        self.speed = np.zeros((2, self.nb_markers, 1))
        self.marker_set_model = None
        self.markers_idx_in_image = []
        self.estimated_area = []
        self.next_pos = np.zeros((2, self.nb_markers, 1), dtype=int)
        self.model = None

    def get_markers_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return np.array([marker.pos for marker in self.markers], dtype=int).T.reshape(2, self.nb_markers)

    def get_markers_filtered_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return np.array([marker.filtered_pos for marker in self.markers], dtype=int).T.reshape(2, self.nb_markers)

    def set_markers_pos(self, pos):
        """
        Set the position of the markers

        Parameters
        ----------
        pos : np.ndarray
            position of the markers
        """
        for m, marker in enumerate(self.markers):
            marker.pos = pos[:, m].astype(int)

    def set_filtered_markers_pos(self, pos):
        """
        Set the position of the markers

        Parameters
        ----------
        pos : np.ndarray
            position of the markers
        """
        for m, marker in enumerate(self.markers):
            marker.filtered_pos = pos[:, m].astype(int)

    def set_markers_occlusion(self, occlusions):
        """
        Set the position of the markers

        Parameters
        ----------
        occlusions: list
            occlusion of the markers

        """
        for m, marker in enumerate(self.markers):
            marker.is_visible = occlusions[m]

    def get_markers_occlusion(self):
        """
        Get the occlusion of the markers

        Returns
        -------
        np.ndarray
            occlusion of the markers
        """
        return np.array([marker.is_visible for marker in self.markers])

    def get_marker(self, name: str = None, idx: int = None):
        """
        Get a marker from the marker set

        Parameters
        ----------
        name : str
            name of the marker
        idx : int
            index of the marker

        Returns
        -------
        Marker
            marker object
        """
        if name and idx:
            raise ValueError("You can't use both name and idx")
        if not name and not idx:
            raise ValueError("You must use either name or idx")
        for m, marker in enumerate(self.markers):
            if name:
                if marker.name == name:
                    return marker
            elif idx:
                if m == idx:
                    return marker

    def update_speed(self):
        for m, mark in enumerate(self.markers):
            for i in range(2):
                mark.compute_speed(mark.pos[i, -1], mark.pos[i, -2])

    def init_kalman(self, points):
        for m, mark in enumerate(self.markers):
            mark.init_kalman(points[:, m])

