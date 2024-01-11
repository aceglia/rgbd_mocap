import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import csv
import cv2
from typing import List

from Marker_Setter.drag_marker import DragMarker


class DropImage(QLabel):
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

    def __init__(self, marker_adder=None, area=None, name='Crop'):
        """
        Initialize a DropImage
        :param marker_adder: MarkerSetter to link with the MarkerAdder
        :type marker_adder: MarkerSetter
        """
        super(DropImage, self).__init__(marker_adder)
        self.setMinimumSize(100, 100)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setEnabled(False)
        self.setAlignment(Qt.AlignCenter)
        self.name = name

        ### Image
        self.image = None
        self.marked_image = None
        self.resized_image: QPixmap = None
        self.start_x = 0
        self.start_y = 0

        ### Markers
        self.marker_adder = marker_adder
        self.markers: List[DragMarker] = []

        ### Options
        self.current_q_rubber_band = None
        self.origin_q_point = None
        self.show_marker_name = True
        self.area = area

    ### Dragging Events
    def dragEnterEvent(self, e):
        """
        Accept dragging event
        :param e: Drag event
        :type e: QDragEnterEvent
        :return: None
        """
        e.accept()

    def dropEvent(self, e):
        """
        Drop an item onto the DropImage. If the
        item is a DragMarker then it will be added
        to the list of markers and will be displayed
        on the image.
        :param e: Drop event (containing position and source to the item)
        :type e: QDropEvent
        :return: None
        """
        pos = e.pos()
        marker = e.source()

        ### DragMarker Object ??
        if not isinstance(marker, DragMarker):
            return

        x = (pos.y() - self.start_y) * self.image.shape[0] // self.resized_image.height()
        y = (pos.x() - self.start_x) * self.image.shape[1] // self.resized_image.width()

        if marker in self.markers:
            marker.set_pos(x, y)
        else:
            marker.place_on_image(x, y, self.area)
            self.markers.append(marker)

        e.accept()
        self.draw_markers()

    ### Mouse Events
    def mousePressEvent(self, eventQMouseEvent):
        """
        On a right-click start a selection zone to
        delete markers.
        :param eventQMouseEvent: Mouse event (containing position and type of the click)
        :type eventQMouseEvent: QMouseEvent
        :return: None
        """
        ### (Right click) Select a zone to erase placed marker
        if eventQMouseEvent.buttons() == Qt.MouseButton.RightButton:
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
        ### Holding right click update the selection zone
        if eventQMouseEvent.buttons() == Qt.MouseButton.RightButton and self.current_q_rubber_band is not None:
            try:
                self.current_q_rubber_band.setGeometry(QRect(self.origin_q_point, eventQMouseEvent.pos()).normalized())
            except RuntimeError:
                return

        ### Drag and drop inside the DropImage
        if eventQMouseEvent.buttons() == Qt.MouseButton.LeftButton:
            marker = self.get_closest_marker(eventQMouseEvent.pos())
            if marker:
                self.markers.remove(marker)
                marker.remove_from_image()
                self.draw_markers()
                marker.mouseMoveEvent(eventQMouseEvent)

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
        ### If a zone was selected delete markers contained in
        if self.current_q_rubber_band is not None:
            try:
                rect = self.current_q_rubber_band.geometry()
                rect.adjust(- self.start_x, - self.start_y,
                            - self.start_x, - self.start_y, )

                markers_to_delete = self.get_marker_in(rect.getCoords())
                for marker in markers_to_delete:
                    self.markers.remove(marker)
                    marker.remove_from_image()

                self.current_q_rubber_band.deleteLater()
                self.current_q_rubber_band = None

                ### If no marker selected don't redraw the markers
                if len(markers_to_delete) == 0:
                    return

                ### If at least one marker has been deleted draw the markers and update the image
                self.draw_markers()
            except RuntimeError:
                return

        ### (Left Click) Place the currently selected marker
        else:
            ### Get the currently selected marker
            selected_marker = self.marker_adder.list_marker.current_marker
            if not selected_marker:
                return

            pos = ev.pos()
            ### If valid set its position and add it to the image
            x = (pos.y() - self.start_y) * self.image.shape[0] // self.resized_image.height()
            y = (pos.x() - self.start_x) * self.image.shape[1] // self.resized_image.width()

            ### If found check the validity of the given position
            if (self.image.shape[0] < x or x < 0 or
                    self.image.shape[1] < y or y < 0):
                return

            selected_marker.place_on_image(x, y, self.area)

            self.markers.append(selected_marker)
            self.draw_markers()

    ### Marker modification
    def get_marker_in(self, rect):
        """
        Get all the markers within the boundaries
        of a rectangle.
        :param rect: Set of point representing the rectangle to search in.
        :type rect: QRect
        :return: All the DragMarker contained in the rectangle
        :rtype: List[DragMarker]
        """
        ### Need to recalculate new coords with rescaling
        rect = (rect[0] * self.image.shape[1] // self.resized_image.width(),
                rect[1] * self.image.shape[0] // self.resized_image.height(),
                rect[2] * self.image.shape[1] // self.resized_image.width(),
                rect[3] * self.image.shape[0] // self.resized_image.height())

        selected_marker = []
        for marker in self.markers:
            ### Within the rectangle ?
            if (rect[0] < marker[1] < rect[2] and
                    rect[1] < marker[0] < rect[3]):
                selected_marker.append(marker)

        return selected_marker

    def get_closest_marker(self, pos):
        """
        Get the closest marker to a position.
        (Not exactly the first but priority
        to the first in the list of markers)
        :param pos: Set of coordinates (x, y)
        :type pos: QPoint
        :return: return the 'closest' DragMarker if found
        :rtype: DragMarker | None
        """
        pos = ((pos.x() - self.start_x) * self.image.shape[1] // self.resized_image.width(),
               (pos.y() - self.start_y) * self.image.shape[0] // self.resized_image.height(),)

        ### Size of the bounds to search within
        size = 5
        for marker in self.markers:
            if (pos[0] - size < marker[1] < pos[0] + size and
                    pos[1] - size < marker[0] < pos[1] + size):
                return marker

        return None

    def set_marker_name(self, show: bool):
        """
        Set the flag showing the marker names on the image.
        :param show: Boolean to set
        :type show: bool
        :return: None
        """
        self.show_marker_name = show
        self.draw_markers()

    def remove_markers(self):
        """
        Remove all the DragMarkers set upon the DropImage.
        :return: None
        """
        for _ in range(len(self.markers)):
            self.markers.pop().remove_from_image()
        self.draw_markers()

    ### Marker Saving
    def markers_to_dict(self):
        """
        Save the DragMarker placed upon the DropImage
        in a default cvs file named 'saved_markers.csv'.
        In the format {name},{x_position},{y_position}
        :return: None
        """
        markers = []
        for marker in self.markers:
            markers.append(marker.to_dict())

        return markers

    def load_markers(self, markers):
        """
        Load the DragMarker from a default cvs file
        named 'saved_markers.csv'.
        In the format {name},{x_position},{y_position}
        Initialize the DragMarkers and put them on the
        DropImage.
        :return: None
        """
        for marker in markers:
            new_marker = DragMarker(marker['name'],
                                    parent=self.marker_adder.list_marker)
            new_marker.set_pos(marker['pos'][0],
                               marker['pos'][1],
                               self.area)
            new_marker.hide()
            self.markers.append(new_marker)
        self.draw_markers()

    ### Image Display
    def draw_markers(self, color=(0, 255, 0)):
        """
        Draw the DragMarkers upon the DropImage.
        Taking in account the show_names flag.
        :param color: The drawing color of the DragMarkers
        :type color: tuple[int]
        :return: None
        """
        if self.image is None:
            return

        ### copy the default image
        self.marked_image = self.image.copy()

        ### For all markers draw the circle around their positions
        for marker in self.markers:
            x = marker[1]
            y = marker[0]
            cv2.circle(
                self.marked_image,
                (x, y),
                5,
                color,
                1,
            )

            ### If flag show_marker_name is set at True then display the DragMarker name above it
            if self.show_marker_name:
                cv2.putText(
                    self.marked_image,
                    marker.text(),
                    (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    1,
                )

        ### Update the image to show the difference
        self.update_image()

    def resizeEvent(self, a0):
        self.update_image()

        # Update Margins
        self.start_x = (self.size().width() - self.resized_image.size().width()) // 2
        self.start_y = (self.size().height() - self.resized_image.size().height()) // 2

    def update_image(self):
        """
        Update the currently displayed image.
        Whether it is for resize purpose or
        displaying the drawn markers
        :return: None
        """
        if self.image is None:
            return

        # From array to QPixmap
        image = QImage(self.marked_image,
                       self.marked_image.shape[1],
                       self.marked_image.shape[0],
                       self.marked_image.strides[0],
                       QImage.Format_RGB888)

        # Convert frame into image and apply the resize
        self.resized_image = (QPixmap.fromImage(image).scaled(self.size().width(),
                                                              self.size().height(),
                                                              Qt.KeepAspectRatio,
                                                              Qt.TransformationMode.SmoothTransformation))

        # If successful apply the image to the QLabel
        self.setPixmap(self.resized_image)

    def set_image(self, image):
        """
        Set the current image of the DropImage
        and update its display.
        :param image: Image to set in the DropImage
        :type image: Mat | ndarray[Any, dtype[generic]] | ndarray
        :return: None
        """
        if self.area:
            self.image = image[self.area[1]:self.area[3], self.area[0]:self.area[2]]
        else:
            self.image = image

        self.marked_image = self.image.copy()
        self.update_image()
        self.setEnabled(True)

    def clear(self):
        self.markers = []
        self.draw_markers()
