import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2


class MaskPopUp(QMessageBox):
    """
    Pop up showing the 'crop_q_pixmap' and
    two options :
    'Cancel' which return QMessageBox.Cancel
    'Ok' which return QMessageBox.Ok
    """

    def __init__(self, crop_q_pixmap, value, parent=None):
        super(MaskPopUp, self).__init__(parent)
        self.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        self.focusPreviousChild()
        self.setIconPixmap(crop_q_pixmap)
        if value:
            self.setWindowTitle("Remove mask")
        else:
            self.setWindowTitle("Place mask")

        self.res = self.exec_() == QMessageBox.Ok


class VideoEdit(QLabel):
    """
    A QLabel containing a color_frame and a depth_frame
    that can be updated. A filtered_frame that correspond
    to the color_image post application of the filters.
    """

    def __init__(self, area, parent=None):
        """
        Initialize a VideoEdit widget.
        :param parent: QObject container parent (to be linked with Filters a VideoEditLinker)
        :type parent: QObject | VideoEditLinker
        """
        super(VideoEdit, self).__init__(parent)
        self.setMinimumSize(100, 100)
        self.setMouseTracking(True)

        ### Image
        self.color_frame = None
        self.depth_frame = None
        self.filtered_frame = None
        self.resized_image: QPixmap = None
        self.start_x = 0
        self.start_y = 0

        ### Options
        self.area = area
        self.blobs = []
        self.current_q_rubber_band = None
        self.origin_q_point = None

    def apply_filter(self, filter):
        """
        Apply the filters given in parameters
        to the filtered frame.
        :param filter: A filters function to apply to the VideoEdit frame
        :type filter: function
        :return: None
        """
        if self.color_frame is None:
            return

        a = filter(self)
        self.filtered_frame = a

    def apply_mask(self, masks):
        """
        Merge all the masks given in parameters
        and apply the resulting mask to the
        filtered frame.
        :param masks: The masks to apply to the VideoFrame
        :type masks:
        :return: None
        """
        if self.color_frame is None or self.depth_frame is None:
            return

        if self.parent().select_area_button.isChecked() and self.parent().select_area_button.isEnabled():
            mask = self.parent().mask.copy()
        else:
            mask = np.ones(self.color_frame.shape[:2], dtype=np.uint8)

        for m in masks:
            mask[:] *= m[:]

        self.filtered_frame[mask == 0] = 0

    def apply_blend(self, blend):
        """
        Merge the filtered_frame with the base color_frame
        and blend them with blend intensity.
        blend=0: Base image
        blend=1: Filtered image
        :param blend: Intensity of the blend to apply
        :type blend: float [0.0, 1.0]
        :return: None
        """
        if self.color_frame is None:
            return

        if blend is None:
            self.filtered_frame = self.color_frame.copy()
            return

        blend /= 100
        img = (
            self.filtered_frame
            if len(self.filtered_frame.shape) == 3
            else cv2.cvtColor(self.filtered_frame, cv2.COLOR_GRAY2RGB)
        )
        self.filtered_frame = cv2.addWeighted(self.color_frame, 1 - blend, img, blend, 0)

    def apply_blob_detect(self, blob_size=3):
        """
        Display the blobs contained in the blobs
        list.
        :param blob_size: Size of the blob to display (default=3)
        :type blob_size: int
        :return: None
        """
        if self.color_frame is None:
            return

        for blob in self.blobs:
            self.filtered_frame = cv2.circle(self.filtered_frame, (blob[0], blob[1]), blob_size, (255, 0, 0), -1)
            # self.filtered_frame[blob[1] - blob_size:blob[1] + blob_size,
            #                     blob[0] - blob_size:blob[0] + blob_size] = [255, 0, 0]

    def mousePressEvent(self, eventQMouseEvent):
        """
        On a right-click start a selection zone to
        delete markers.
        :param eventQMouseEvent: Mouse event (containing position and type of the click)
        :type eventQMouseEvent: QMouseEvent
        :return: None
        """
        if not self.parent().select_area_button.isChecked() or not self.parent().select_area_button.isEnabled():
            return

        ### Select a zone to place mask
        try:
            self.origin_q_point = eventQMouseEvent.pos()
            self.current_q_rubber_band = QRubberBand(QRubberBand.Rectangle, self)
            self.current_q_rubber_band.setGeometry(QRect(self.origin_q_point, QSize()))
            self.current_q_rubber_band.show()
        except RuntimeError:
            return
        return

    def mouseMoveEvent(self, eventQMouseEvent):
        """
        On a right-click update the selection size if inited.
        On a left-click try to grab the closest DragMarker
        within range. If a DragMarker is found then proceed
        like a classic drag and drop.
        :param eventQMouseEvent: Mouse event (containing position and type of the click)
        :type eventQMouseEvent: QMouseEvent
        :return: None
        """
        ### Holding click update the selection zone
        if self.current_q_rubber_band is not None:
            try:
                self.current_q_rubber_band.setGeometry(QRect(self.origin_q_point, eventQMouseEvent.pos()).normalized())
            except RuntimeError:
                return

    def mouseReleaseEvent(self, ev):
        """
        On right-click release if a selection zone was inited
        then proceed to remove from the DropImage all
        the markers contained in the zone.
        On a left-click release place the currently selected
        DragMarker of the MarkerList on the DropImage.
        (equivalent to a drag & drop)
        :param ev: Mouse event (containing position and type of the click)
        :type ev: QMouseEvent
        :return: None
        """
        ### If a zone was selected place (l-click)/remove (r-click) mask
        if self.current_q_rubber_band is not None:
            try:
                rect = self.current_q_rubber_band.geometry()
                rect.adjust(
                    -self.start_x,
                    -self.start_y,
                    -self.start_x,
                    -self.start_y,
                )

                self.current_q_rubber_band.deleteLater()

                copy_rect = self.resized_image.copy(rect)
                # Invalid area
                if not rect.isValid() or copy_rect.size() == self.resized_image.size():
                    return

                click = ev.button() == Qt.MouseButton.RightButton
                ### Update the image with mask change
                if MaskPopUp(copy_rect, click, parent=self).res:
                    self.update_mask(rect.getCoords(), click)

                self.current_q_rubber_band = None

            except RuntimeError:
                return

    def update_mask(self, rect, value):
        ### Need to recalculate new coords with rescaling
        rect = (
            (rect[0] * self.color_frame.shape[1] // self.resized_image.width()) if rect[0] > 0 else 0,
            (rect[1] * self.color_frame.shape[0] // self.resized_image.height()) if rect[1] > 0 else 0,
            rect[2] * self.color_frame.shape[1] // self.resized_image.width(),
            rect[3] * self.color_frame.shape[0] // self.resized_image.height(),
        )

        self.parent().mask[rect[1] : rect[3], rect[0] : rect[2]] = int(value)
        self.parent().update_image()

    def resizeEvent(self, a0: QResizeEvent) -> None:
        """
        Resize the VideoEdit QLabel.
        :param a0: Resize event
        :type a0: QResizeEvent
        :return: None
        """
        self.update_size()

    def update_size(self):
        """
        Update the size of the current image base
        on the filtered_frame.
        :return: None
        """
        if self.filtered_frame is None:
            return
        format = QImage.Format_RGB888 if len(self.filtered_frame.shape) == 3 else QImage.Format_Grayscale8
        image = QImage(
            self.filtered_frame,
            self.filtered_frame.shape[1],
            self.filtered_frame.shape[0],
            self.filtered_frame.strides[0],
            format,
        )

        self.resized_image = QPixmap.fromImage(image).scaled(
            self.size().width(), self.size().height(), Qt.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        self.setPixmap(self.resized_image)
        # Update Margins
        self.start_x = (self.size().width() - self.resized_image.size().width()) // 2
        self.start_y = (self.size().height() - self.resized_image.size().height()) // 2

    def update(self) -> None:
        super(VideoEdit, self).update()
        self.update_size()

    def set_image(self, color, depth):
        """
        Set the color_frame and the depth_frame
        of the VideoEdit to color and depth.
        Reinit filtered_frame to color.
        :param color: New color frame
        :type color: Mat | ndarray[Any, dtype[generic]] | ndarray
        :param depth: New depth frame
        :type depth: Mat | ndarray[Any, dtype[generic]] | ndarray
        :return: None
        """
        if color is None or depth is None:
            return
        self.color_frame = color[self.area[1] : self.area[3], self.area[0] : self.area[2]]
        self.depth_frame = depth[self.area[1] : self.area[3], self.area[0] : self.area[2]]
        self.filtered_frame = self.color_frame.copy()
        self.update_size()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = QMainWindow()

    area = (100, 100, 400, 400)
    vt = VideoEdit(area, main_window)
    main_window.setCentralWidget(vt)

    path = "D:\Documents\Programmation\pose_estimation\data_files\P9\gear_5_11-01-2024_16_59_32/"

    frame_color = cv2.flip(cv2.cvtColor(cv2.imread(path + "color_1372.png"), cv2.COLOR_BGR2RGB), -1)
    frame_depth = cv2.flip(cv2.imread(path + "depth_1372.png", cv2.IMREAD_ANYDEPTH), -1)

    # vt.set_image(frame_color, frame_depth)
    vt.set_image(None, frame_depth)

    main_window.show()
    sys.exit(app.exec_())
