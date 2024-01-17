import numpy as np

from markers.marker import Marker
from markers.shared_marker import SharedMarker


class MarkerSet:
    """
    This class is used to store the marker information
    """

    def __init__(self, marker_set_name, marker_names: list[str], shared=False):
        """
        init markers class with number of markers, names and image index

        Parameters
        ----------
        marker_names : list
            list of names for the markers
        image_idx : list
            index of the image where the marker set is located
        """
        self.name = marker_set_name

        self.markers: list[Marker] = []
        for marker_name in marker_names:
            if shared:
                self.markers.append(SharedMarker(name=marker_name))

            else:
                self.markers.append(Marker(name=marker_name))

        self.nb_markers = len(marker_names)

    def get_markers_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return [marker.pos for marker in self]

    def get_markers_pos_2d(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return [marker.pos[:2] for marker in self]

    def get_markers_reliability_index(self, frame_idx):
        return [np.array([marker.get_reliability_index(frame_idx) for marker in self.markers]).round(2)]

    def get_markers_global_pos(self):
        """
        Get the position of the markers

        Returns
        -------
        np.ndarray
            position of the markers
        """
        return [marker.get_global_pos() for marker in self]

    def get_markers_names(self):
        """
        Get the names of the markers

        Returns
        -------
        list
            names of the markers
        """
        return [marker.name for marker in self.markers]

    def set_markers_pos(self, positions):
        """
        Set the position of the markers

        Parameters
        ----------
        pos : np.ndarray
            position of the markers
        """
        for i in range(len(self.markers)):
            self[i].set_pos(positions[i])

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

    def set_offset_pos(self, start_crop):
        """
        Set the position of the markers

        Parameters
        ----------
        start_crop : np.ndarray
            Starting position of the crop (Up-Left)

        """
        for marker in self:
            marker.set_crop_offset(start_crop[0], start_crop[1])

    def set_markers_occlusion(self, occlusions):
        """
        Set the position of the markers

        Parameters
        ----------
        occlusions: list
            occlusion of the markers

        """
        for i, marker in enumerate(self):
            marker.set_visibility(occlusions[i])

    def set_markers_depth_occlusion(self, occlusions):
        """
        Set the position of the markers

        Parameters
        ----------
        occlusions: list
            occlusion of the markers

        """
        for i, marker in enumerate(self.markers):
            marker.set_depth_visibility(occlusions[i])

    def get_markers_occlusion(self):
        """
        Get the occlusion of the markers

        Returns
        -------
        np.ndarray
            occlusion of the markers
        """
        return [marker.get_visibility() for marker in self.markers]

    def get_markers_depth_occlusion(self):
        """
        Get the occlusion of the markers

        Returns
        -------
        np.ndarray
            occlusion of the markers
        """
        return [marker.get_depth_visibility() for marker in self.markers]

    def get_marker_by_name(self, name: str,):
        """
        Get a marker from the marker set

        Parameters
        ----------
        name : str
            name of the marker

        Returns
        -------
        Marker
            marker object
        """
        for marker in self:
            if marker.name == name:
                return marker

        return None

    def get_markers_local_in_meters(self):
        pass

    def get_markers_global_in_meters(self):
        pass

    def add_marker(self, marker: Marker):
        self.markers.append(marker)
        self.nb_markers += 1

    def __str__(self):
        string = self.name + ' ['

        for marker in self.markers[:-1]:
            string += f'{marker.name}, '

        string += f'{self.markers[-1].name}]'
        return string

    def __getitem__(self, item):
        return self.markers[item]

    def __iter__(self):
        for marker in self.markers:
            yield marker
