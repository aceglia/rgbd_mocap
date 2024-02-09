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

from Marker_Setter.marker_setter_tab import MarkerSetter
from Utils.video_player import VideoControl
from Video_cropping.crop_video_tab import CropVideoTab
from Video_cropping.crop_video import VideoCropper
from Utils.error_popup import ErrorPopUp
from Utils.warning_popup import WarningPopUp
from Utils.file_dialog import SaveDialog, LoadDialog, LoadFolderDialog


class CropWidget(QMainWindow):
    """
    A CropWindow containing a VideoTab and VideoPlayer.
    The user can load a folder of images, then create crops
    and edit the video with various parameters.

    """

    def __init__(self, parent=None):
        super(CropWidget, self).__init__(parent)
        self.setGeometry(300, 200, 900, 600)
        self.setWindowTitle("Cropping Window")
        self.video_layout = QGridLayout()

        ### Directory and index for image init
        self.dir = None
        self.index = []
        self.min_index = None
        self.max_index = None

        ### Crop video tab
        self.video_tab = CropVideoTab(self)
        self.video_cropper = VideoCropper(name="Base image", video_tab=self.video_tab, video_crop_window=self)
        self.video_tab.addTab(self.video_cropper, self.video_cropper.name)

        ### Video Control
        self.video_player = VideoControl()
        self.video_player.setDisabled(True)
        self.video_player.slider_anim.valueChanged.connect(self.update_image)

        ### Place in layout
        self.video_layout.addWidget(self.video_tab, 0, 0, 1, 1)
        self.video_layout.addWidget(self.video_player, 1, 0, 1, 1)

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

            self.video_player.adjust(self.min_index, self.max_index)

            self.update_image()
            return True

        else:
            ErrorPopUp("Could not load any file from this directory", self)
            return False

    def update_image(self):
        if self.dir and self.min_index is not None:
            color_path = self.dir + os.sep + f"color_{self.video_player.value}.png"
            depth_path = self.dir + os.sep + f"depth_{self.video_player.value}.png"
            image_color = cv2.imread(color_path)
            if image_color is None:
                return
            tik = time.time()
            image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
            image_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            print('Time:', time.time() - tik)

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
        return self.load_images()

    def to_dict(self):
        return {'directory': self.dir,
                'start_index': self.video_player.slider_anim.value()[0],
                'end_index': self.video_player.slider_anim.value()[2],
                'crops': self.save_crops(),
                }

    def save_crops(self):
        crops = []
        for tab in self.video_tab.tabs[1:]:
            crops.append(tab.save_dic())

        return crops

    # def save_unplaced_markers(self):
    #     if self.marker_setter_tab is None:
    #         return []
    #
    #     return self.marker_setter_tab.get_unplaced_markers()

    def load_project(self):
        LoadDialog(parent=self,
                   caption='Load project file',
                   filter='Save File (*.json);; Any(*)',
                   load_method=self.load_project_file)

    def load_project_dict(self, parameters):
        self.dir = parameters['directory']
        self.load_images()
        self.video_player.slider_anim.setValue((parameters['start_index'],
                                                parameters['start_index'],
                                                parameters['end_index']))

        self.video_cropper.load_project(parameters)

    def get_first_frame(self):
        if self.dir is not None and self.video_tab is not None:
            return self.dir + os.sep + f"color_{self.video_player.slider_anim.value()[0]}.png"

        return None

    def get_crops(self):
        return self.video_tab.get_crops()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = CropWidget()
    # demo.dir = "/home/user/KaelFacon/Project/rgbd_mocap/data_files/P4_session2/gear_20_15-08-2023_10_52_14/"
    # demo.load_images()
    demo.show()
    sys.exit(app.exec_())
