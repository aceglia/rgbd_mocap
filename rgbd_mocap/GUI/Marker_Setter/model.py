import json
import os

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from rgbd_mocap.GUI.Marker_Setter.drag_marker import DragMarker
from rgbd_mocap.GUI.Utils.file_dialog import LoadDialog
from rgbd_mocap.GUI.Utils.popup import ErrorPopUp


class Model(QTreeWidget):
    """
    Contains a list of DragMarker that can be
    dragged over a DropImage to be set upon it.
    In a MarkerList the focused DragMarker will
    interact first. Only one DragMarker can be
    focused at a time. Clicking on the MarkerList
    will unfocus all DragMarkers.
    """

    def __init__(self, l: [str] = [], parent=None):
        """
        Initialize the MarkerList to contains all the DragMarker in list
        and set its parent.
        :param list: list containing the names of the DragMarker to be initialized and put in the MarkerList
        :type list: list[str]
        :param parent: QWidget parent container
        :type parent: QObject
        """
        super(Model, self).__init__(parent)
        self.setAcceptDrops(False)
        self.setMouseTracking(False)
        self.segments_list = []
        self.markers_list = []

        self.current_marker = None
        self.current_segment = None
        self.setHeaderLabels(["Model"])
        self.setColumnCount(1)

        ### Save and load
        self.kwargs = {}
        if "SNAP" in os.environ:
            self.kwargs["options"] = QFileDialog.DontUseNativeDialog

        ### Create Scroll Layout and container for the Dock Widget
        self.layout = QVBoxLayout(self)

    def _add_marker_to_list(self, marker: str, insert=False):
        """
        Initialize a DragMarker from string and insert it at the end of the MarkerList
        :param marker: Name of the marker to be init
        :type marker: str
        :return: None
        """
        marker_widget = QTreeWidgetItem(self, [marker])
        self.layout.addWidget(marker_widget, Qt.AlignTop)
        if insert:
            self.layout.insertWidget(len(self.list), marker_widget, 0, Qt.AlignTop)
        else:
            self.layout.addWidget(marker_widget, Qt.AlignTop)
        self.markers_list.append(marker_widget)

    def add_marker(self, marker: str):
        """
        Initialize a DragMarker from string and insert it at the end of the MarkerList
        :param marker: Name of the marker to be init
        :type marker: str
        :return: None
        """
        new_name = self._check_name(marker)
        self._add_marker_to_list(new_name, insert=True)
        self.set_focused_marker(self.markers_list[-1])

    def _get_all_names(self):
        return [marker.name for marker in self.list]

    def _check_name(self, name):
        if name in self._get_all_names():
            i = 1
            while name + str(i) in self._get_all_names():
                i += 1
            return name + str(i)
        return name

    def remove_marker(self):
        """
        Remove the currently selected marker.
        Set the focus to the last marker contained
        :return: None
        """
        if not len(self.list) or self.current_marker is None:
            return

        self.current_marker.unfocused()
        self.layout.removeWidget(self.current_marker)
        self.list.remove(self.current_marker)
        self.current_marker = None

        if len(self.list):
            self.set_focused_marker(self.list[0])

    def remove_all_marker(self):
        """
        Remove all the markers contained in the MarkerList
        :return: None
        """
        if self.current_marker:
            self.current_marker = None

        for _ in range(len(self.list)):
            self.layout.removeWidget(self.list.pop())

    def set_focused_marker(self, marker: DragMarker):
        """
        Set the focus to a DragMarker.
        :param marker: DragMarker taking the focus of the MarkerList
        :type marker: DragMarker
        :return: None
        """
        if self.current_marker:
            self.current_marker.unfocused()

        marker.focused()
        self.current_marker = marker

    def unfocused(self):
        """
        Unfocus the current DragMarker
        :return: None
        """
        if self.current_marker:
            self.current_marker.unfocused()
            self.current_marker = None

    def dropEvent(self, e):
        """
        Dropping a DragMarker back into the MarkerList
        will replace it at the end of the list and
        remove it from the DropImage it has been placed in.
        :param e: Drop event containing the dropped DragMarker
        :type e: QDropEvent
        :return: None
        """
        marker = e.source()

        if not isinstance(marker, DragMarker):
            return

        if marker not in self.list:
            marker.remove_from_image()

        e.accept()

    def mousePressEvent(self, a0):
        """
        Left-clicking anywhere on the MarkerList
        will unfocus the currently focused marker.
        :param a0: Mouse event
        :type a0: QMouseEvent
        :return: None
        """
        if a0.buttons() == Qt.MouseButton.LeftButton and self.current_marker is not None:
            self.unfocused()

    ### Load premade marker set
    def load_marker_set(self):
        LoadDialog(
            parent=self,
            caption="Load placement",
            filter="Save File (*.json);; Any(*)",
            load_method=self.load_marker_set_file,
        )

    def load_marker_set_file(self, file):
        with open(file, "r") as f:
            all_markers = json.load(f)
            try:
                placed_markers = all_markers["placed_markers"]
                unplaced_markers = all_markers["unplaced_markers"]

                for crop in placed_markers:
                    unplaced_markers.extend([marker["name"] for marker in crop["markers"]])

                for marker in unplaced_markers:
                    self.add_marker(marker)
            except TypeError and KeyError:
                ErrorPopUp("File could not be loaded, wrong format")

    def save_marker_set(self):
        pass

    def clear(self):
        self.list = []
        self.current_marker = None
