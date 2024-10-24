import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import csv
import cv2
from typing import List

from .drag_marker import DragMarker


class DisplayMarkerImage(QLabel):
    """
    A Qlabel widget that display a resizing image
    which accepts drop events with DragMarker.
    When a DragMarker is drop in the image it will
    be added to the markers contained in the image
    and will be printed on it. (with or without its name)
    Markers can be drag and drop inside the image by
    holding the left-click.
    A simple left-click on the image will add, if
    selected, the currently focused marker in the MarkerList.
    Holding the right click will create a selection
    zone, all the markers within it will be removed
    from the DropImage.
    """

    def __init__(self, marker_tab=None, name="Base Image"):
        """
        Initialize a DisplayMarkerImage
        :param marker_tab: DropImageTab to link with all the DropImage
        :type marker_tab: DropImageTab
        """
        super(DisplayMarkerImage, self).__init__(marker_tab)
        self.setMinimumSize(100, 100)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.name = name

        ### Image
        self.image = None
        self.marked_image = None

        ### Markers
        self.marker_tab = marker_tab
        self.markers: List[DragMarker] = []

        ### Options
        self.show_marker_name = True

    ### Marker modification
    def remove_markers(self):
        """
        Clear the marker list and redraw the image.
        :return: None
        """
        self.markers = []
        self.draw_markers(self.markers)

    def draw_markers(self, markers, color=(0, 255, 0)):
        """
        Draw the DragMarkers upon the DisplayMarkerImage.
        Taking in account the show_names flag.
        :param markers: list of markers to draw
        :type markers: list[DragMarker]
        :param color: The drawing color of the DragMarkers
        :type color: tuple[int]
        :return: None
        """
        if self.image is None:
            return

        ### copy the default image
        self.marked_image = self.image.copy()

        self.markers = markers
        ### For all markers draw the circle around their positions
        for marker in self.markers:
            y = int(marker.x + marker.area[1])
            x = int(marker.y + marker.area[0])
            cv2.circle(
                self.marked_image,
                (x, y),
                5,
                color,
                1,
            )

            ### If flag show_marker_name is set at True then display the DragMarker name above it
            if self.show_marker_name:
                shapes = self.marked_image.copy()
                if marker.text() == "ster":
                    delta = 25
                    delta_x = 5
                else:
                    delta = 15
                    delta_x = 10

                (w, h), _ = cv2.getTextSize(marker.text(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    shapes,
                    (x + delta_x, y + delta + 4),
                    (x + delta_x + 2 + w, y + delta - h),
                    color,
                    -1,
                )
                cv2.putText(
                    shapes,
                    marker.text(),
                    (x + delta_x, y + delta),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
                alpha = 0.8
                self.marked_image = cv2.addWeighted(shapes, alpha, self.marked_image, 1 - alpha, 0, self.marked_image)

        ### Update the image to show the difference
        self.update_image()

    def set_marker_name(self, show: bool):
        """
        Set the flag showing the marker names on the image.
        And redraw the markers accordingly.
        :param show: Boolean to set
        :type show: bool
        :return: None
        """
        self.show_marker_name = show
        self.draw_markers(self.markers)

    ### Image Display
    def resizeEvent(self, a0):
        """
        Update the currently displayed image
        size via 'update_image' method.
        """
        self.update_image()

    def update_image(self):
        """
        Update the currently displayed image.
        And resize it accordingly to its container
        size without distortion.
        :return: None
        """
        if self.image is None:
            return

        # From array to QPixmap
        format = QImage.Format_RGB888 if len(self.marked_image.shape) == 3 else QImage.Format_Grayscale8
        image = QImage(
            self.marked_image,
            self.marked_image.shape[1],
            self.marked_image.shape[0],
            self.marked_image.strides[0],
            format,
        )

        # Convert frame into image and apply the resize
        resized_image = QPixmap.fromImage(image).scaled(
            self.size().width(), self.size().height(), Qt.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        # If successful apply the image to the QLabel
        self.setPixmap(resized_image)

    def set_image(self, image):
        """
        Set the current image of the DisplayMarkerImage
        and update its display. Also enabling it.
        :param image: Image to set in the DisplayMarkerImage
        :type image: Mat | ndarray[Any, dtype[generic]] | ndarray
        :return: None
        """
        self.image = image
        self.marked_image = self.image.copy()
        self.update_image()
        self.setEnabled(True)
