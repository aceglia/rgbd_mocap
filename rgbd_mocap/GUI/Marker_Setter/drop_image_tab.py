import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import json
import os

from rgbd_mocap.GUI.Marker_Setter.marker_adder import MarkerAdder
from rgbd_mocap.GUI.Marker_Setter.display_marker_image import DisplayMarkerImage
from rgbd_mocap.GUI.Marker_Setter.drop_image import DropImage
from rgbd_mocap.GUI.Utils.file_dialog import SaveDialog, LoadDialog
from rgbd_mocap.GUI.Utils.popup import ErrorPopUp


class DropImageTab(QTabWidget):
    """
    A QTabWidget like Widget with default
    layout in tab when calling the addTab function.
    This Tab Widget also has its tab position set
    to North and all the tabs are closable except
    for the first one opened
    """

    def __init__(self, marker_adder, display_image, crops=[], parent=None):
        super(DropImageTab, self).__init__(parent)
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.currentChanged.connect(lambda index: self.change_window(index))

        ### Markers
        self.marker_adder = marker_adder

        ### List containing all added tabs
        self.tabs = []

        ### Set default tabs
        self.display_image = display_image
        self.add_tab(self.display_image, "Base image")
        for crop in crops:
            self.add_tab(DropImage(self.marker_adder, crop[1], crop[0]), crop[0])

    def __getitem__(self, item):
        return self.tabs[item]

    def get_tab(self, name):
        for tab in self:
            if tab.name == name:
                return tab

        return None

    def __iter__(self):
        for tab in self.tabs:
            yield tab

    def add_tab(self, widget, name=None):
        """
        This method is based on the default
        addTab method of the QWidgetTab.
        All the added tab via this method will not be closable
        :param widget: Widget to be added in the tab
        :type widget: QWidget
        :param name: Name of the new tab
        :type name: str
        :return: None
        """
        super(DropImageTab, self).addTab(widget, name)
        self.tabBar().setTabButton(self.count() - 1, QTabBar.RightSide, None)
        self.tabs.append(widget)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        """
        Update all its children to resize correctly
        """

        super(DropImageTab, self).resizeEvent(a0)
        for tab in self.tabs:
            tab.resizeEvent(a0)  # update_image allow resizing for VideoCropper & Cropped Video

    def set_image(self, image_color, depth_image=None):
        ### Set the new image color/depth to all tabs
        if depth_image is not None:
            depth_3d = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
            image_color = cv2.addWeighted(image_color, 0.5, depth_3d, 0.5, 0, image_color)
        for tab in self.tabs:
            tab.set_image(image_color)

        ### Update only the currently selected tab
        ### with filters to avoid unecessary lagging
        i = self.currentIndex()
        self.tabs[i].update_image()

    def removeTab(self, index):
        ### When a tab is remove delete the VideoEditLinker contained in
        ### and also remvoe it from the list self.tabs to avoid lags
        ### and unecessary RAM usage
        item = self.widget(index)
        self.tabs.remove(item)
        item.deleteLater()
        super(DropImageTab, self).removeTab(index)

    def change_window(self, index):
        if len(self.tabs) == 0:
            return

        ### Hide all the dock
        for tab in self.tabs:
            tab.hide()

        self.tabs[index].show()
        if isinstance(self.tabs[index], DisplayMarkerImage):
            all_markers = []
            for tab in self.tabs:
                if isinstance(tab, DropImage):
                    all_markers.extend(tab.markers)

            self.tabs[index].draw_markers(all_markers)

    def markers_to_dict(self):
        all_markers = {}
        crops = []
        for tab in self[1:]:
            crop_marker = {
                "name": tab.name,
                # 'area': tab.area,
                "markers": tab.markers_to_dict(),
            }
            crops.append(crop_marker)

        all_markers["crops"] = crops
        all_markers["markers"] = self.marker_adder.markers_to_dict()

        return all_markers

    def save_markers(self):
        SaveDialog(
            parent=self,
            caption="Save placement",
            filter="Save File (*.json)",
            suffix="json",
            save_method=self.save_markers_file,
        )

    def save_markers_file(self, file):
        with open(file, "w") as f:
            json.dump(self.markers_to_dict(), f, indent=2)

    def dict_to_markers(self, crop):
        for i in range(1, self.count()):
            if self.tabs[i].name == crop["name"] and self.tabs[i].markers == []:
                # self.tabs[i].area = crop['area']
                self.tabs[i].load_markers(crop["markers"])
                return True

        return False

    def load_markers(self):
        LoadDialog(
            parent=self,
            caption="Load placement",
            filter="Save File (*.json);; Any(*)",
            load_method=self.load_markers_file,
        )

        ### To update the tabs
        self.change_window(self.currentIndex())

    def load_markers_file(self, file):
        with open(file, "r") as f:
            try:
                project = json.load(f)
            except TypeError and KeyError:
                ErrorPopUp("File could not be loaded, wrong format")
            self.load_project_dict(project)

    def load_project_dict(self, project_dict):
        crops = project_dict["crops"]
        if "markers" not in project_dict.keys():
            project_dict["markers"] = []
        unplaced_markers = project_dict["markers"]

        for crop in crops:
            if not self.dict_to_markers(crop):
                continue
            # unplaced_markers.extend([marker['name'] for marker in crop['markers']])
        self.marker_adder.load_markers(unplaced_markers)
        self.change_window(self.currentIndex())

    def clear(self):
        for tab in self.tabs:
            tab.clear()
