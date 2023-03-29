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
        self.dt = 1
        n_states = 6
        n_measures = 3

        self.kalman = cv2.KalmanFilter(n_states, n_measures)
        self.kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
        # kalman.processNoiseCov = np.eye(n_states, dtype = np.float32)*0.9
        self.kalman.measurementNoiseCov = np.eye(n_measures, dtype=np.float32) * 0.0005

        self.kalman.measurementMatrix = np.zeros((n_measures, n_states), np.float32)
        self.Measurement_array = []
        self.dt_array = []
        for i in range(0, n_states, 6):
            self.Measurement_array.append(i)
            self.Measurement_array.append(i + 1)
            self.Measurement_array.append(i + 2)

        for i in range(0, n_states):
            if i not in self.Measurement_array:
                self.dt_array.append(i)

        for i, j in zip(self.Measurement_array, self.dt_array):
            self.kalman.transitionMatrix[i, j] = self.dt
        for i in range(0, n_measures):
            self.kalman.measurementMatrix[i, self.Measurement_array[i]] = 1

    def predict_from_kalman(self):
        self.filtered_pos = self.kalman.predict()[:3]

    def correct_from_kalman(self, points):
        self.pos = np.array(points)
        self.filtered_pos = self.kalman.correct(np.array(points, dtype=np.float32))[:3]

    def init_kalman(self, points):
        self.pos = np.array(points)
        input_points = np.float32(np.ndarray.flatten(points))
        self.kalman.statePost = np.array([input_points[0], input_points[1], input_points[2], 0, 0, 0], dtype=np.float32)
        self.kalman.statePre = np.array([input_points[0], input_points[1], input_points[2], 0, 0, 0], dtype=np.float32)

    def set_global_pos(self, local_pos, start_crop):
        if local_pos[0] is not None and local_pos[1] is not None:
            self.global_pos[0] = local_pos[0] + start_crop[0]
            self.global_pos[1] = local_pos[1] + start_crop[1]
            if len(local_pos) == 3:
                self.global_pos[2] = local_pos[2]
            else:
                self.global_pos[2] = None
        else:
            self.global_pos = np.array([None, None, None])

    def set_global_filtered_pos(self, local_pos, start_crop):
        self.global_filtered_pos[0] = local_pos[0] + start_crop[0]
        self.global_filtered_pos[1] = local_pos[1] + start_crop[1]
        self.global_filtered_pos[2] = local_pos[2]


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
        self.marker_set_model = None
        self.markers_idx_in_image = []

    def get_markers_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return np.array([marker.pos for marker in self.markers]).T.reshape(3, self.nb_markers)

    def get_markers_filtered_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return np.array([marker.filtered_pos for marker in self.markers]).T.reshape(3, self.nb_markers)

    def get_markers_global_filtered_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return np.array([marker.global_filtered_pos for marker in self.markers]).T.reshape(3, self.nb_markers)

    def get_markers_global_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return np.array([marker.global_pos for marker in self.markers]).T.reshape(3, self.nb_markers)

    def get_markers_names(self):
        """
        Get the names of the markers

        Returns
        -------
        list
            names of the markers
        """
        return [marker.name for marker in self.markers]

    def set_markers_pos(self, pos):
        """
        Set the position of the markers

        Parameters
        ----------
        pos : np.ndarray
            position of the markers
        """
        for m, marker in enumerate(self.markers):
            if pos.shape[0] != 3:
                marker.pos[:2] = pos[:, m]
            else:
                marker.pos = pos[:, m]

    def set_filtered_markers_pos(self, pos):
        """
        Set the position of the markers

        Parameters
        ----------
        pos : np.ndarray
            position of the markers
        """
        for m, marker in enumerate(self.markers):
            marker.filtered_pos = pos[:, m]

    def set_global_filtered_markers_pos(self, pos, start_crop):
        """
        Set the position of the markers

        Parameters
        ----------
        pos : np.ndarray
            position of the markers
        """
        for m, marker in enumerate(self.markers):
            marker.set_global_filtered_pos(pos[:, m], start_crop)

    def set_global_markers_pos(self, pos, start_crop):
        """
        Set the position of the markers

        Parameters
        ----------
        pos : np.ndarray
            position of the markers
        """
        for m, marker in enumerate(self.markers):
            marker.set_global_pos(pos[:, m], start_crop)

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

    def set_markers_depth_occlusion(self, occlusions):
        """
        Set the position of the markers

        Parameters
        ----------
        occlusions: list
            occlusion of the markers

        """
        for m, marker in enumerate(self.markers):
            marker.is_depth_visible = occlusions[m]

    def get_markers_occlusion(self):
        """
        Get the occlusion of the markers

        Returns
        -------
        np.ndarray
            occlusion of the markers
        """
        return [marker.is_visible for marker in self.markers]

    def get_markers_depth_occlusion(self):
        """
        Get the occlusion of the markers

        Returns
        -------
        np.ndarray
            occlusion of the markers
        """
        return [marker.is_depth_visible for marker in self.markers]

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

    def init_kalman(self, points):
        for m, mark in enumerate(self.markers):
            mark.init_kalman(points[:, m])
