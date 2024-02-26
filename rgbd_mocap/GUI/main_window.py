import csv
import json
import os
import re
import sys

import cv2
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from Marker_Setter.model_setter_tab import MarkerSetter
from video_crop_widget import CropWidget
from video_crop_window import CropWindow
from Utils.video_player import VideoControl
from Video_cropping.crop_video_tab import CropVideoTab
from Video_cropping.crop_video import VideoCropper
from Utils.error_popup import ErrorPopUp
from Utils.warning_popup import WarningPopUp
from Utils.file_dialog import SaveDialog, LoadDialog, LoadFolderDialog


class MainWindowMenuBar(QMenuBar):
    """
    Menu bar for the Crop window.
    Contains shortcut for loading
    folder of image, crops saved files
    and for saving actual crops.
    """

    def __init__(self, main_window):
        """
        Initialize a CropWindowMeuBar and link it
        to a CropWindow. The actions contained in
        the CropWindowMenuBar will be linked to the
        given CropWindow.
        :param crop_window: CropWindow to link with
        :type crop_window: CropWindow
        """
        super(MainWindowMenuBar, self).__init__(main_window)
        self.mw = main_window

        ### Init menu bar actions
        self.open_file_action = QAction('Load Directory', self)
        self.open_file_action.triggered.connect(main_window.vce.load_folder)

        self.quit_app_action = QAction('Quit', self)
        self.quit_app_action.triggered.connect(main_window.close)

        self.save_crops_action = QAction('Save Crops', self)
        self.save_crops_action.triggered.connect(main_window.save_crops)

        self.load_crops_action = QAction('Load Crops', self)
        self.load_crops_action.triggered.connect(main_window.load_crop)

        self.save_project_action = QAction('Save Project', self)
        self.save_project_action.triggered.connect(main_window.save_project)

        self.load_project_action = QAction('Load Project', self)
        self.load_project_action.triggered.connect(main_window.load_project)

        ### Init and fill menu bar
        file = self.addMenu('File')
        file.addAction(self.open_file_action)
        file.addSeparator()
        file.addAction(self.save_crops_action)
        file.addAction(self.load_crops_action)
        file.addSeparator()
        file.addAction(self.save_project_action)
        file.addAction(self.load_project_action)
        file.addSeparator()
        file.addAction(self.quit_app_action)


class SwitchWindowButton(QPushButton):
    def __init__(self, crop_window):
        super(SwitchWindowButton, self).__init__(crop_window)


class MainWindowButton(QWidget):
    def __init__(self, main_window):
        super(MainWindowButton, self).__init__(main_window)
        layout = QHBoxLayout()

        ## Buttons (Save/Quit)
        self.save_crops_button = QPushButton("Save")
        self.save_crops_button.clicked.connect(main_window.quick_save)
        self.save_crops_button.setMaximumWidth(100)

        self.quit_app_button = QPushButton("Quit")
        self.quit_app_button.clicked.connect(main_window.close)
        self.quit_app_button.setMaximumWidth(100)

        self.marker_set_button = QPushButton('Set Markers')
        self.marker_set_button.clicked.connect(main_window.marker_setting)
        self.marker_set_button.setMaximumWidth(100)

        layout.addWidget(self.marker_set_button, 1, Qt.AlignLeft)
        layout.addWidget(self.save_crops_button, 0, Qt.AlignRight)
        layout.addWidget(self.quit_app_button, 0, Qt.AlignRight)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    """
    A CropWindow containing a VideoTab and VideoPlayer.
    The user can load a folder of images, then create crops
    and edit the video with various parameters.

    """

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setGeometry(300, 200, 900, 600)
        self.setWindowTitle("RGBD MoCap")
        self.layout = QGridLayout()

        ### Video cropping and editing
        self.vce = CropWidget(self)

        ### Menu bar
        self.menu_bar = MainWindowMenuBar(self)
        self.setMenuBar(self.menu_bar)

        ### Marker Setter
        self.marker_setter_tab = None

        ### Project informations
        self.project_dict = {}

        ### Buttons
        self.buttons = MainWindowButton(self)

        ### Place in layout
        self.layout.addWidget(self.vce, 0, 0, 1, 1,)
        self.layout.addWidget(self.buttons, 1, 0, 1, 1, Qt.AlignBottom)

        self.central_video_widget = QWidget()
        self.central_video_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_video_widget)

    ### Saves and Loads
    def quick_save(self):
        if self.save_file is None:
            self.save_project()

        else:
            self.save_project_file(self.save_file)

    def load_crop(self):
        LoadDialog(parent=self,
                   caption='Load crops file',
                   filter='Save File (*.json);; Any(*)',
                   load_method=self.vce.video_cropper.load_crops_file)

    def save_project(self):
        SaveDialog(parent=self,
                   caption='Save Project',
                   filter='Save File (*json)',
                   suffix='json',
                   save_method=self.save_project_file)

    def save_project_file(self, file):
        self.save_file = file

        ### Write it
        with open(file, 'w') as file:
            json.dump(self.project_to_dict(), file, indent=2)

    def project_to_dict(self):
        parameters_vce = self.vce.to_dict()
        parameters_ms = self.marker_setter_tab.to_dict()
        mask_list = [None] * len(parameters_vce['crops'])
        for c, crop_vce in enumerate(parameters_vce['crops']):
            if crop_vce["filters"]["mask"] is not None:
                mask_list[c] = {"name": crop_vce["name"], "value": crop_vce["filters"]["mask"]}
                crop_vce["filters"]["mask"] = True

        for crop_ms in parameters_ms['crops']:
            for crop_vce in parameters_vce['crops']:
                if crop_vce['name'] == crop_ms['name'] and crop_vce['markers'] == []:
                    crop_vce['markers'] = crop_ms['markers']
                    break

            ### Way to unplace markers if crops not found rather than removing them
        parameters_vce['markers'] = parameters_ms['markers']
        parameters_vce['masks'] = mask_list
        return parameters_vce

    def save_crops(self):
        crops = []
        for tab in self.video_tab.tabs[1:]:
            tab_dict = tab.save_dic()

            # If marker_setter is placed then
            if self.marker_setter_tab is not None:
                tab_dict['markers'] = self.marker_setter_tab.get_markers_from(tab.name)

            crops.append(tab_dict)

        return crops

    def load_project(self):
        LoadDialog(parent=self,
                   caption='Load project file',
                   filter='Save File (*.json);; Any(*)',
                   load_method=self.load_project_file)

    def load_project_file(self, file):
        with open(file, 'r') as f:
            parameters = json.load(f)

            self.vce.load_project_dict(parameters)

            if self.marker_setter_tab is None:
                self.create_marker_setter()

            self.marker_setter_tab.load_project_dict(parameters)
            # self.marker_setter_tab.drop_image_tab.load_project_dict(parameters)
            self.set_vce()

    def set_vce(self):
        if self.marker_setter_tab is not None:
            self.marker_setter_tab.setParent(None)
            self.layout.addWidget(self.vce, 0, 0, 1, 1)

    def set_marker_setter(self):
        if self.marker_setter_tab is not None:
            self.vce.setParent(None)
            self.layout.addWidget(self.marker_setter_tab, 0, 0, 1, 1)

    def create_marker_setter(self):
        path = self.vce.get_first_frame()
        if path is None:
            ErrorPopUp('Cannot find first frame index')
            return

        crops = self.vce.get_crops()
        self.marker_setter_tab = MarkerSetter(path=path,
                                              marker_set=[],
                                              crops=crops, )

    def marker_setting(self):
        ### Warning Pop Up
        if not WarningPopUp('Pass to marker settings').res:
            return

        ### If no MarkerSetter then create a new MarkerSetter
        ### Else a MarkerSetter as already been loaded then reload it to apply eventual changes
        if self.marker_setter_tab is None:
            self.create_marker_setter()
        else:
            self.marker_setter_tab.reload(self.project_to_dict())

        self.set_marker_setter()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainWindow()
    # demo.vce.dir = "../../../../rgbd_mocap/data_files/P4_session2/gear_20_15-08-2023_10_52_14/"
    # demo.vce.load_images()
    demo.show()
    sys.exit(app.exec_())
