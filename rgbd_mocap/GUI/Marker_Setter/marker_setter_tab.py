import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2

from rgbd_mocap.GUI.Marker_Setter.marker_adder import MarkerAdder
from rgbd_mocap.GUI.Marker_Setter.display_marker_image import DisplayMarkerImage

# from Marker_Setter.drop_image import DropImage
from rgbd_mocap.GUI.Marker_Setter.drop_image_tab import DropImageTab


class DropImageButton(QWidget):
    """
    QWidget linked to a DropImage widget.
    A DropImageButton contains various buttons
    and check boxes that allow to interact with
    the linked DropImage.
    The 'Save markers' button allow to call the
    save_markers function of the DropImage.
    The 'Load markers' button allow to call the
    load_markers function of the DropImage.
    The 'Markers name' check box allow to set
    or unset the show_marker_name flag of the DropImage.
    The 'Remove placed markers' button allow to call the
    remove_markers function of the DropImage.
    """

    def __init__(self, dimage: DropImageTab, parent=None):
        """
        Initialize the DropImageButtons
        :param dimage: DropImage to link the buttons to.
        :type dimage:  DropImageTab
        :param parent: QObject container parent
        :type parent: MarkerSetter
        """
        super(DropImageButton, self).__init__(parent)
        self.dimage = dimage

        self.save_button = QPushButton("Save markers")
        self.save_button.pressed.connect(dimage.save_markers)

        self.load_button = QPushButton("Load markers")
        self.load_button.pressed.connect(dimage.load_markers)

        self.show_markers_name = QCheckBox("Markers name")
        self.show_markers_name.setChecked(True)
        self.show_markers_name.pressed.connect(self.show_markers_name_method)

        self.remove_markers = QPushButton("Remove placed markers")
        self.remove_markers.pressed.connect(self.remove_markers_method)

        layout = QHBoxLayout()
        layout.addWidget(self.remove_markers, Qt.AlignLeft)
        layout.addWidget(self.show_markers_name, Qt.AlignCenter)
        layout.addWidget(self.save_button, Qt.AlignRight)
        layout.addWidget(self.load_button, Qt.AlignRight)

        self.setLayout(layout)

    def show_markers_name_method(self):
        for tab in self.dimage.tabs:
            tab.set_marker_name(not self.show_markers_name.isChecked())

    def remove_markers_method(self):
        current = self.dimage[self.dimage.currentIndex()]
        if isinstance(current, DisplayMarkerImage):
            for tab in self.dimage.tabs:
                tab.remove_markers()

        else:
            current.remove_markers()


class MarkerSetter(QMainWindow):
    """
    The MarkerSetter class is a window containing
    and linking a MarkerAdder and a DropImage.
    The MarkerAdder is set on a DockWidget and
    therefore movable and dockable on the left
    and right of the MarkerSetter window.
    The MarkerSetter also link its DropImage
    to a DropImageButton.
    """

    def __init__(self, path, marker_set=[], crops=[]):
        """
        Initialize the MarkerSetter window
        :param marker_set: List of string containing the name of the DragMarker to be inited in the MarkerList
        :type marker_set: List[str]
        """
        super(MarkerSetter, self).__init__()
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        layout = QGridLayout()

        ### Create Dock Widget
        self.dock = QDockWidget()
        self.dock.DockWidgetClosable = False
        self.dock.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.dock.setFloating(False)
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.dock.setParent(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

        ### MarkerAdder ###
        self.marker_adder = MarkerAdder(l=marker_set, parent=self)
        self.dock.setWidget(self.marker_adder)

        ### DisplayImage ###
        self.display_image = DisplayMarkerImage()

        ### Tabs for DropImage ###
        self.drop_image_tab = DropImageTab(self.marker_adder, self.display_image, crops)

        # Load image #
        self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(path.replace("color", "depth"), cv2.IMREAD_ANYDEPTH)
        depth_3d = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        self.image = cv2.addWeighted(self.image, 0.5, depth_3d, 0.5, 0, self.image)
        self.drop_image_tab.set_image(self.image)

        ### Drop Image Buttons ####
        self.drop_image_button = DropImageButton(self.drop_image_tab)

        ### Set layout
        layout.addWidget(self.drop_image_tab, 0, 0, 1, 1)
        layout.addWidget(self.drop_image_button, 1, 0, 1, 1, Qt.AlignBottom)

        content = QWidget()
        content.setLayout(layout)
        self.setCentralWidget(content)

    def load_project_dict(self, parameters):
        self.drop_image_tab.load_project_dict(parameters)

    def to_dict(self):
        return self.drop_image_tab.markers_to_dict()

    def get_markers_from(self, tab_name):
        markers = []
        tab = self.drop_image_tab.get_tab(tab_name)

        if tab is None:
            return markers

        for marker in tab.markers:
            markers.append(marker.to_dict())

        return markers

    def get_unplaced_markers(self):
        return self.marker_adder.markers_to_dict()

    def reload(self, parameters):
        self.clear()
        self.drop_image_tab.load_project_dict(parameters)

    def clear(self):
        self.drop_image_tab.clear()
        self.marker_adder.clear()


if __name__ == "__main__":
    app = QApplication([])
    path = "D:\Documents\Programmation\pose_estimation\data_files\P14\gear_5_22-01-2024_16_15_16/color_997.png"
    l = []
    crops = [
        ("Hand", (191, 226, 308, 357)),
        ("Arm", (256, 270, 369, 392)),
        ("Back", (343, 139, 470, 364)),
    ]
    w = MarkerSetter(path, l, crops)
    w.show()

    app.exec_()
