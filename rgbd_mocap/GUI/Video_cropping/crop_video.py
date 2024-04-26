import csv
import json
import sys
import cv2

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from rgbd_mocap.GUI.Video_editing.video_edit_linker import VideoEditLinker


class CropPopUp(QMessageBox):
    """
        Pop up showing the 'crop_q_pixmap' and
        two options :
        'Cancel' which return QMessageBox.Cancel
        'Ok' which return QMessageBox.Ok
    """

    def __init__(self, crop_q_pixmap, parent=None):
        super(CropPopUp, self).__init__(parent)
        self.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        self.focusPreviousChild()
        self.setIconPixmap(crop_q_pixmap)
        self.setWindowTitle('Do you want to keep this crops ?')

        self.res = self.exec_() == QMessageBox.Ok


class VideoCropper(QLabel):
    """
        This class allow the mouse tracking on an QLabel
        containing an image. The : mousePressEvent, mouseMoveEvent,
        mouseReleaseEvent are modified to set new functionalities.
        You can select a zone on the image to crops it and add it
        in another tab.
        This class also include the set_image and update_image functions
        which allows to respectively to set the image of the VideoCropper
        and to update it for resizing purpose
    """

    def __init__(self, name='Base Image', video_tab=None, video_crop_window=None):
        super(VideoCropper, self).__init__(video_tab)
        self.setMinimumSize(100, 100)

        self.name = name
        self.video_tab = video_tab
        self.video_crop_window = video_crop_window
        self.mouse_pressed = False
        self.image = None
        self.depth = None
        self.resized_image = None

        self.setText("Load directory")
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)

        ### Cropping selection zone
        self.current_q_rubber_band = None
        self.origin_q_point = None

    ### Add new Crop
    def new_crop(self, rect, name='Crop'):
        if self.video_tab is None:
            return

        vel = VideoEditLinker(rect, parent=self.video_tab)
        vel.set_image(self.image, self.depth)
        self.video_tab.add_crop(vel, name)

        return vel

    ### Overrided Method to select a zone to crops
    def mousePressEvent(self, eventQMouseEvent):
        """
            This function is enable only if the image is loaded
            and init the selection to be cropped
        """
        if self.image is not None:
            ### Init selection zone
            self.origin_q_point = eventQMouseEvent.pos()

            ### Need to try/except because of double click and quick selection that can cause RuntimeErrors
            try:
                self.current_q_rubber_band = QRubberBand(QRubberBand.Rectangle, self)
                self.current_q_rubber_band.setGeometry(QRect(self.origin_q_point, QSize()))
                self.current_q_rubber_band.show()
                self.mouse_pressed = True
            except RuntimeError:
                pass

        ### If image is not set and CropVideo is linked to a VideoCropWindow then load a folder
        else:
            if self.video_crop_window is not None and self.video_crop_window.load_folder():
                self.setText("")

    def mouseMoveEvent(self, eventQMouseEvent):
        """
            Update the selection size if inited
        """
        if self.mouse_pressed and self.current_q_rubber_band:
            ### Need to try/except because of double click and quick selection that can cause RuntimeErrors
            try:
                self.current_q_rubber_band.setGeometry(QRect(self.origin_q_point, eventQMouseEvent.pos()).normalized())
            except RuntimeError:
                pass

    def mouseReleaseEvent(self, eventQMouseEvent):
        """
            Does nothing if the selection has not been init.
            Else hide the selection and open a CropPopUp box.
        """
        # If the mouse has not been pressed before release the selected area does not exist
        if not self.mouse_pressed:
            return

        self.mouse_pressed = False
        ### Need to try/except because of double click and quick selection that can cause RuntimeErrors
        try:
            self.current_q_rubber_band.hide()
            current_q_rect: QRect = self.current_q_rubber_band.geometry()
            self.current_q_rubber_band.deleteLater()

            ### Adjust the current_q_rect to take in account marging while centering the image
            current_q_rect = self.adjust_rect(current_q_rect)

            # Invalid or too small crops area
            if (self.resized_image.size() == current_q_rect.size() or
                    current_q_rect.width() * current_q_rect.height() < 500):
                return

            # Pop Up to ask if you accept the crops, add it to a new tab (if accepted)
            if CropPopUp(self.resized_image.copy(current_q_rect)).res:
                self.new_crop(self.calculate_area(current_q_rect))

            self.current_q_rubber_band = None

        except RuntimeError:
            pass

    ### Utils to calculate cropped area
    def calculate_area(self, rect: QRect):
        base_width, resized_width = self.image.shape[1], self.resized_image.width()
        base_height, resized_height = self.image.shape[0], self.resized_image.height()

        start_x, start_y, end_x, end_y = rect.getCoords()

        start_x = start_x * base_width // resized_width
        start_y = start_y * base_height // resized_height
        end_x = end_x * base_width // resized_width
        end_y = end_y * base_height // resized_height

        return start_x, start_y, end_x, end_y

    def adjust_rect(self, rect):
        adjust = self.size() - self.resized_image.size()

        return (rect.adjusted(- adjust.width() // 2,
                              - adjust.height() // 2,
                              - adjust.width() // 2,
                              - adjust.height() // 2).
                intersected(self.resized_image.rect()))

    ### Load
    def load_crops_file(self, file):
        """
        Load from file name and position of crops
        then create VideoEditLinker corresponding
        and add it to the CropVideo.
        :param file: Path to the file
        :type file: str
        """
        if self.image is None:
            return

        with open(file, 'r') as file:
            parameters = json.load(file)
            for crop in parameters.keys():
                self.new_crop(parameters[crop], crop)

    def load_project(self, parameters):
        crops = parameters['crops']

        for crop in crops:
            ### Add new crops to the CropVideoTab
            vel = self.new_crop(crop['area'], crop['name'])

            value = None
            ### Set the filters to the newly added VideoEditLinker
            for n, mask in enumerate(parameters["masks"]):
                if mask is None:
                    continue
                if mask["name"] == crop['name']:
                    value = parameters["masks"][n]["value"]
                    break
            vel.filters.image_options.set_params(crop['filters'], value)

    ### Resizing and updating image
    def resizeEvent(self, a0):
        self.update_image()

    def update_image(self):
        """ This function will resize the image and save it
            into the 'resized_image' variable this function
            is only for display purposes.
        """
        if self.image is None:
            return
        format = QImage.Format_RGB888 if len(self.image.shape) == 3 else QImage.Format_Grayscale8
        image = QImage(self.image,
                       self.image.shape[1],
                       self.image.shape[0],
                       self.image.strides[0],
                       format
                       )

        self.resized_image = QPixmap.fromImage(image).scaled(self.size().width(),
                                                             self.size().height(),
                                                             Qt.KeepAspectRatio,
                                                             Qt.TransformationMode.SmoothTransformation)

        # If successful apply the image to the QLabel
        self.setPixmap(self.resized_image)

    def set_image(self, image, depth):
        """
        Set the current image of the VideoCropper with
        the given image param and update the image afterward
        """
        self.image = image
        self.depth = depth

        self.update_image()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    vc = VideoCropper()
    main_window.resizeEvent = (lambda a: vc.resizeEvent(a))
    main_window.setCentralWidget(vc)

    path = "D:\Documents\Programmation\pose_estimation\data_files\P9\gear_5_11-01-2024_16_59_32/"

    frame_color = cv2.imread(path + 'color_1372.png')
    frame_depth = cv2.imread(path + 'depth_1372.png', cv2.IMREAD_ANYDEPTH)
    import time
    time.sleep(1)
    time_flip = []
    time_rotate = []
    rot = 0
    for i in range(1000):
        tic = time.time()
        cv2.flip(frame_color, -1)
        time_flip.append(time.time() - tic)
        tic = time.time()
        cv2.rotate(frame_color, rot)
        time_rotate.append(time.time() - tic)

    print('Flip:', sum(time_flip) / len(time_flip))
    print('Rotate:', sum(time_rotate) / len(time_rotate))
    vc.set_image(frame_color, frame_depth)
    main_window.show()
    sys.exit(app.exec_())
