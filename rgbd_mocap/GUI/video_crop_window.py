import csv
import json
import os
import re
import sys
import time

import cv2
# import qtpy
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from Marker_Setter.model_setter_tab import MarkerSetter
from Utils.video_player import VideoControl
from Video_cropping.crop_video_tab import CropVideoTab
from Video_cropping.crop_video import VideoCropper
from Utils.popup import ErrorPopUp
from Utils.warning_popup import WarningPopUp
from Utils.file_dialog import SaveDialog, LoadDialog, LoadFolderDialog


class CropWindowButtons(QWidget):
    def __init__(self, video_crop):
        super(CropWindowButtons, self).__init__(video_crop)
        layout = QGridLayout(self)

        ### Buttons
        # Save
        self.save_crops_button = QPushButton("Save")
        self.save_crops_button.clicked.connect(video_crop.quick_save)
        self.save_crops_button.setMaximumWidth(100)

        # Quit
        self.quit_app_button = QPushButton("Quit")
        self.quit_app_button.clicked.connect(video_crop.close)
        self.quit_app_button.setMaximumWidth(100)

        # Set Markers
        self.marker_set_button = QPushButton('Set Markers')
        self.marker_set_button.clicked.connect(video_crop.marker_setting)
        self.marker_set_button.setMaximumWidth(200)
        self.marker_set_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(self.marker_set_button, 0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.save_crops_button, 1, 0, 1, 1, Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.quit_app_button, 1, 1, 1, 1, Qt.AlignmentFlag.AlignBottom)

        self.setLayout(layout)
        self.setMaximumSize(200, 100)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)


class CropWindowMenuBar(QMenuBar):
    """
    Menu bar for the Crop window.
    Contains shortcut for loading
    folder of image, crops saved files
    and for saving actual crops.
    """

    def __init__(self, crop_window):
        """
        Initialize a CropWindowMeuBar and link it
        to a CropWindow. The actions contained in
        the CropWindowMenuBar will be linked to the
        given CropWindow.
        :param crop_window: CropWindow to link with
        :type crop_window: CropWindow
        """
        super(CropWindowMenuBar, self).__init__(crop_window)
        self.cw = crop_window

        ### Init menu bar actions
        self.open_file_action = QAction('Load Directory', self)
        self.open_file_action.triggered.connect(self.cw.load_folder)

        self.quit_app_action = QAction('Quit', self)
        self.quit_app_action.triggered.connect(self.cw.close)

        self.save_crops_action = QAction('Save Crops', self)
        self.save_crops_action.triggered.connect(self.cw.save_crop)

        self.load_crops_action = QAction('Load Crops', self)
        self.load_crops_action.triggered.connect(self.cw.load_crop)

        self.save_project_action = QAction('Save Project', self)
        self.save_project_action.triggered.connect(self.cw.save_project)

        self.load_project_action = QAction('Load Project', self)
        self.load_project_action.triggered.connect(self.cw.load_project)

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


class CropWindow(QMainWindow):
    """
    A CropWindow containing a VideoTab and VideoPlayer.
    The user can load a folder of images, then create crops
    and edit the video with various parameters.

    """

    def __init__(self, parent=None):
        super(CropWindow, self).__init__(parent)
        self.setGeometry(300, 200, 900, 600)
        self.setWindowTitle("Cropping Window")
        self.video_layout = QGridLayout()

        self.setMenuBar(CropWindowMenuBar(self))

        ### Directory and index for image init
        self.dir = None
        self.index = []
        self.min_index = None
        self.max_index = None

        ### Save and load
        # self.save_file = None
        # self.kwargs = {}
        # if 'SNAP' in os.environ:
        #     self.kwargs['options'] = QFileDialog.DontUseNativeDialog

        ### Crop video tab
        self.video_tab = CropVideoTab(self)
        self.video_cropper = VideoCropper(name="Base image", video_tab=self.video_tab, video_crop_window=self)
        self.video_tab.addTab(self.video_cropper, self.video_cropper.name)

        ### Video Control
        self.video_player = VideoControl()
        self.video_player.setDisabled(True)
        self.video_player.slider_anim.valueChanged.connect(self.update_image)

        ### Marker Setter
        self.marker_setter_tab = None

        ### Video Crop Buttons
        self.buttons = CropWindowButtons(self)

        ### Place in layout
        self.video_layout.addWidget(self.video_tab, 0, 0, 1, 2)
        self.video_layout.addWidget(self.video_player, 1, 0, 2, 1)
        self.video_layout.addWidget(self.buttons, 1, 1, 1, 1)

        self.central_video_widget = QWidget()
        self.central_video_widget.setLayout(self.video_layout)
        self.setCentralWidget(self.central_video_widget)

    ### Image
    def load_images(self):
        self.index = []
        for file in os.listdir(self.dir):
            match_result = re.search("^color_([0-9]*)\.png$", file)
            if match_result:
                self.index.append(int(match_result.group(1)))
        self.index.sort()

        if len(self.index):
            self.min_index = self.index[0]
            self.max_index = self.index[-1]
            self.video_player.value = self.min_index
            self.video_player.adjust(self.min_index, self.max_index)

            self.update_image()
            return True

        else:
            ErrorPopUp("Could not load any file from this directory", self)
            return False

    def update_image(self):
        if self.index != [] and self.min_index is not None:
            color_path = self.dir + os.sep + f"color_{self.video_player.value}.png"
            depth_path = self.dir + os.sep + f"depth_{self.video_player.value}.png"
            if not os.path.isfile(color_path) or not os.path.isfile(depth_path):
                return
            try:
                image_color = cv2.imread(color_path)
                image_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            except:
                print('Error loading image number', self.video_player.value)
                return

            if image_color is None or image_depth is None:
                return
            # tik = time.time()
            image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
            cv2.putText(image_color, f"Frame {self.video_player.value}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            ### Check if the image has been well loaded
            self.video_tab.set_image(image_color, image_depth)

    ### Saves and Loads
    def load_folder(self):
        """ This function will load the user selected image
            and set it to label using the set_photo function
        """
        folder = LoadFolderDialog(parent=self,
                                  caption='Load directory')

        if folder.filename == '':
            return False

        self.dir = folder.filename
        if self.load_images():
            return True

        self.dir = None
        return False

    def quick_save(self):
        if self.save_file is None:
            self.save_project()

        else:
            self.save_project_file(self.save_file)

    def save_crop(self):
        SaveDialog(parent=self,
                   caption='Save crops file',
                   filter='Save File (*.csv)',
                   suffix='csv',
                   save_method=self.save_crop_file)

    def save_crop_file(self, file):
        with open(file, 'w') as file:
            for tab in self.video_tab.tabs[1:]:
                save_crop_area(tab.name,
                               tab.ve.area,
                               file)

    def load_crop(self):
        LoadDialog(parent=self,
                   caption='Load crops file',
                   filter='Save File (*.csv);; Any(*)',
                   load_method=self.video_cropper.load_crops_file)

    def save_project(self):
        SaveDialog(parent=self,
                   caption='Save Project',
                   filter='Save File (*json)',
                   suffix='json',
                   save_method=self.save_project_file)

    def save_project_file(self, file):
        # self.save_file = file

        ### Write it
        with open(file, 'w') as file:
            json.dump(self.project_to_dict(), file, indent=2)

    def project_to_dict(self):
        return {'directory': self.dir,
                'start_index': self.video_player.slider_anim.value()[0],
                'end_index': self.video_player.slider_anim.value()[2],
                'crops': self.save_crops(),
                'markers': self.save_unplaced_markers(),
                }

    def save_crops(self):
        crops = []
        for tab in self.video_tab.tabs[1:]:
            tab_dict = tab.save_dic()

            # If marker_setter is placed then
            if self.marker_setter_tab is not None:
                tab_dict['markers'] = self.marker_setter_tab.get_markers_from(tab.name)

            crops.append(tab_dict)

        return crops

    def save_unplaced_markers(self):
        if self.marker_setter_tab is None:
            return []

        return self.marker_setter_tab.get_unplaced_markers()

    def load_project(self):
        LoadDialog(parent=self,
                   caption='Load project file',
                   filter='Save File (*.json);; Any(*)',
                   load_method=self.load_project_file)

    def load_project_file(self, file):
        self.save_file = file
        with open(file, 'r') as f:
            parameters = json.load(f)
            try:
                self.dir = parameters['directory']
                self.load_images()
                self.video_player.slider_anim.setValue((parameters['start_index'],
                                                        parameters['start_index'],
                                                        parameters['end_index']))

                self.video_cropper.load_project(parameters)

                if self.marker_setter_tab is not None:
                    self.marker_setter_tab.drop_image_tab.load_project_dict(parameters)
            except TypeError and KeyError:
                ErrorPopUp('File could not be loaded, wrong format', self)

        if self.central_video_widget.isHidden():
            self.setCentralWidget(self.central_video_widget)

    def marker_setting(self):
        if not self.dir:
            ErrorPopUp('Nothing to place markers on !', self)
            return

        ### Warning Pop Up
        if not WarningPopUp('Pass to marker settings').res:
            return

        path = self.dir + os.sep + f"color_{self.video_player.slider_anim.value()[0]}.png"
        crops = self.video_tab.get_crops()
        self.marker_setter_tab = MarkerSetter(path=path,
                                              marker_set=[],
                                              crops=crops, )

        self.central_video_widget.setParent(None)
        self.setCentralWidget(self.marker_setter_tab)


def save_crop_area(name, area, file):
    writer = csv.writer(file)
    writer.writerow([name,
                     int(area[0]),
                     int(area[1]),
                     int(area[2]),
                     int(area[3])])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = CropWindow()
    # demo.dir = "/home/user/KaelFacon/Project/rgbd_mocap/data_files/P4_session2/gear_20_15-08-2023_10_52_14/"
    # demo.load_images()
    demo.show()
    sys.exit(app.exec_())
