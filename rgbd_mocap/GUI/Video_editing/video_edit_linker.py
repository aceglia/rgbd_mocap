import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from rgbd_mocap.GUI.Video_editing.video_filters import VideoFilters, VideoEdit, ImageOptions
from rgbd_mocap.GUI.Utils.popup import ErrorPopUp


class VideoEditLinker(QMainWindow):
    """
    A QWidget linking a VideoFilters to a VideoEdit.
    """
    def __init__(self, area, parent=None):
        """
        Initialize a VideoEditLinker and its
        VideoEdit and VideoFilters.
        Also link the Filters to the VideoEdit.
        :param parent: QObject container parent
        :type parent: QObject
        """
        super(VideoEditLinker, self).__init__(parent)
        self.name = 'Crop'

        self.ve = VideoEdit(area, parent=self)
        self.setCentralWidget(self.ve)
        self.ve.setAlignment(Qt.AlignCenter)

        self.select_area_button = QCheckBox(text='Select area')
        self.mask = None
        self.filters = VideoFilters(self)
        self.filters.image_options.setParent(self)

        self.dock = DockableImageOption(self.filters.image_options, 'Image Options', self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

        self.show()

    def set_image(self, color, depth):
        self.ve.set_image(color, depth)

        if self.mask is None:
            self.set_mask()

    def set_mask(self, mask=None):
        if mask is None:
            self.mask = np.ones(self.ve.color_frame.shape[:2], dtype=np.uint8)
        else:
            self.mask = np.ones_like(self.ve.color_frame)
            if max(mask[0]) < self.mask.shape[0] and max(mask[1]) < self.mask.shape[1]:
                self.mask[np.array(mask[0]), np.array(mask[1])] = 0
            else:
                ErrorPopUp('Mask from loaded parameters is not compatible with the current image shape.'
                           'Mask are modified to handle the new shape.')


    def update_image(self):
        self.ve.update()
        self.filters.update()

    def resizeEvent(self, a0):
        super(VideoEditLinker, self).resizeEvent(a0)
        self.filters.image_options.resizeEvent(a0)

    def update(self):
        super(VideoEditLinker, self).update()
        self.ve.update()

    def save_dic(self):
        dic = {
            'name': self.name,
            'area': self.ve.area,
            'filters': self.filters.image_options.to_dict(),
            'markers': []
        }

        return dic


class VerticalSeparator(QFrame):
    def __init__(self, width=1):
        """
        Create a horizontal separator.
        :param width: thickness of the separator (by default 1)
        :type width: int
        """
        super(VerticalSeparator, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setLineWidth(width)


class DockableWidget(QDockWidget):
    def __init__(self, widget, text='', parent=None):
        super(DockableWidget, self).__init__(parent=parent)

        ### Set default parameters for the DockableWidget
        self.DockWidgetClosable = False
        self.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.setFloating(False)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        ### Init scroll area containing the widget
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)

        self.setWidget(scroll)


class DockableImageOption(QDockWidget):
    def __init__(self, image_option: ImageOptions, text='', parent=None):
        super(DockableImageOption, self).__init__(parent=parent)

        ### Set default parameters for the DockableWidget
        self.DockWidgetClosable = False
        self.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.setFloating(False)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        ### Init layout
        layout = QVBoxLayout()

        ### Init scroll area containing the widget
        scroll = QScrollArea()
        scroll.setWidget(image_option)
        scroll.setWidgetResizable(True)

        ### Set in the layout
        layout.addWidget(scroll)
        layout.addWidget(image_option.buttons, 0, Qt.AlignBottom)

        content = QWidget()
        content.setLayout(layout)
        # image_option.buttons.setParent(content)
        self.setWidget(content)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()

    area = (100, 100, 400, 400)
    vt = VideoEditLinker(area, main_window)

    main_window.setCentralWidget(vt)

    path = "D:\Documents\Programmation\pose_estimation\data_files\P14\gear_5_22-01-2024_16_15_16"
    frame_color = cv2.flip(cv2.cvtColor(cv2.imread(path + "\color_997.png"), cv2.COLOR_BGR2RGB), -1)
    frame_depth = cv2.flip(cv2.imread(path + "\depth_997.png", cv2.IMREAD_ANYDEPTH), -1)

    vt.set_image(frame_color, frame_depth)
    # main_window.show()
    sys.exit(app.exec_())
