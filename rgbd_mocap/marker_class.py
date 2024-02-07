import numpy as np
import cv2
from multiprocessing import RawArray, RawValue

c_int = 'i'
c_bool = 'c'
c_float = 'd'


class Marker:
    def __init__(self, name):
        self.name = name

        ### Shared Memory
        # Position arrays
        self.pos = np.frombuffer(RawArray(c_int, 3), dtype=np.int32)
        #print(self.pos)
        self.last_pos = self.pos.copy()
        self.filtered_pos = np.frombuffer(RawArray(c_int, 3), dtype=np.int32)
        self.global_filtered_pos = np.frombuffer(RawArray(c_int, 3), dtype=np.int32)
        # self.global_pos = np.frombuffer(RawArray(c_int, 3), dtype=np.int32)
        self.global_pos = np.zeros((3,))

        # Visibility and reliability
        self.is_visible = RawValue(c_bool, False)
        self.is_depth_visible = RawValue(c_bool, False)
        self._reliability_index = RawValue(c_float, 0)
        self.reliability_index = RawValue(c_float, 0)
        self.mean_reliability_index = RawValue(c_float, 0)

        ### Kalman
        self.dt = 1
        self.n_states = 4
        self.n_measures = 2
        self.kalman = None

        ### Is static
        self.is_static = False

    def predict_from_kalman(self):
        predict = self.kalman.predict()[:2].reshape(2,)

        # self.pos[:2] = predict

        return predict

    def get_reliability_index(self, frame_idx):
        return self.reliability_index.value / (frame_idx + 1)

    def correct_from_kalman(self, points):
        self.pos[:2] = np.array(points)
        _ = self.kalman.correct(np.array(points[:2], dtype=np.float32))[:2].reshape(
            2,
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
        for i in range(0, self.n_states, 4):
            self.Measurement_array.append(i)
            self.Measurement_array.append(i + 1)

        for i in range(0, self.n_states):
            if i not in self.Measurement_array:
                self.dt_array.append(i)

        self.kalman.transitionMatrix[0, 2] = self.dt
        self.kalman.transitionMatrix[1, 3] = self.dt
        for i in range(0, self.n_measures):
            self.kalman.measurementMatrix[i, self.Measurement_array[i]] = 1
        self.pos = np.array(points)

        # input_points = np.float32(np.ndarray.flatten(points))
        input_points = points
        self.kalman.statePre = np.array([input_points[0], input_points[1], 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([input_points[0], input_points[1], 0, 0], dtype=np.float32)
        self.predict_from_kalman()

    def set_pos_2d(self, position):
        if position != ():
            self.pos[:2] = position

    def set_pos_and_last_2d(self, position):
        self.last_pos[:2] = self.pos[:2]
        self.pos[:2] = position

    def set_depth(self, depth, visibility=None):
        self.depth = depth

        if visibility is not None:
            self.is_depth_visible = visibility

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

    def get_shared_memory(self):
        return (self.name,
                self.pos,
                self.filtered_pos,
                self.global_filtered_pos,
                self.is_visible,
                self.is_depth_visible,
                self.reliability_index,
                self._reliability_index,
                self.mean_reliability_index)

    def set_shared_memory(self,
                          name,
                          pos,
                          filtered_pos,
                          global_filtered_pos,
                          is_visible,
                          is_depth_visible,
                          reliability_index,
                          _reliability_index,
                          mean_reliability_index):

        self.name = name
        self.pos = pos
        self.filtered_pos = filtered_pos
        self.global_filtered_pos = global_filtered_pos
        self.is_visible = is_visible
        self.is_depth_visible = is_depth_visible
        self.reliability_index = reliability_index
        self._reliability_index = _reliability_index
        self.mean_reliability_index = mean_reliability_index


class Marker3D:
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


class MarkerSet:
    """
    This class is used to store the marker information
    """

    def __init__(self, marker_set_name, marker_names: list[str], image_idx: int = 0, fps=30, marker_type="2d"):
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
        self.name = marker_set_name

        for marker_name in marker_names:
            if marker_type == "2d":
                marker = Marker(name=marker_name)
            elif marker_type == "3d" or marker_type == "3D":
                marker = Marker3D(name=marker_name)
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

    def get_markers_pos_2d(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return np.array([marker.pos[:2] for marker in self.markers]).T.reshape(self.nb_markers, 2)

    def get_markers_reliability_index(self, frame_idx):
        return list(np.array([marker.get_reliability_index(frame_idx) for marker in self.markers]).round(2))

    def get_marker_set_model_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return np.array([marker for marker in self.marker_set_model]).T.reshape(3, self.nb_markers)

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

    def init_kalman_from_pos(self, positions):
        assert len(positions) == len(self.markers)

        for i in range(len(positions)):
            self.markers[i].init_kalman(positions[i])

    def init_filtered_pos(self, points: np.ndarray) -> None:
        """
        Set the position of the markers
        Parameters
        ----------
        points: np.ndarray
            position of the markers

        """
        for m, mark in enumerate(self.markers):
            mark.filtered_pos = points[:, m]

    def get_markers_local_in_meters(self):
        pass

    def get_markers_global_in_meters(self):
        pass

    def get_shared_memories(self):
        shared_memories = []

        for marker in self.markers:
            shared_memories.append(marker.get_shared_memory())

        return shared_memories

    def add_marker(self, marker: Marker):
        self.markers.append(marker)
        self.nb_markers += 1
        self.marker_names.append(marker.name)

    def __str__(self):
        string = self.name + ' ['

        for marker in self.markers[:-1]:
            string += f'{marker.name}, '

        string += f'{self.markers[-1].name}]'
        return string

    @staticmethod
    def set_shared_memories(shared_memories):
        ms = MarkerSet('shared_marker_set', [])

        for memory in shared_memories:
            marker = Marker('New')
            marker.set_shared_memory(*memory)
            ms.add_marker(marker)

        return ms

    def __getitem__(self, item):
        return self.markers[item]

    def __iter__(self):
        for marker in self.markers:
            yield marker
