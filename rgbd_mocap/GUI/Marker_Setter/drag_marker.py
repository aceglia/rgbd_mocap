from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class DragMarker(QLabel):
    """
    This class inherits the behavior of a QLabel.
    It contains a text to display and react to being
    under the mouse or clicked.
    When clicked the DragMarker is considered focus
    and its appearance changes.
    The display color can be changed via the variables:
    @over_color and @focused_color.
    This class contains an x and y positions and therefore
    can be dragged and placed upon a DropImage
    """
    def __init__(self, text, parent=None):
        super(DragMarker, self).__init__(text, parent=parent)
        self.setFixedHeight(15)
        self.is_focus = False
        self.over_color = 'lightgrey'
        self.focused_color = 'lightblue'
        self.x = -1
        self.y = -1
        self.area = (0, 0, 0, 0)

    def print(self) -> None:
        print(self.text())

    def set_pos(self, x: int, y: int, area: tuple) -> None:
        """
        Set the x and y positions of the DragMarker
        :param x: Position x
        :type x: int
        :param y: Position y
        :type y: int
        :return: None
        """
        self.x = x
        self.y = y
        self.area = area

    def __getitem__(self, item):
        """
        Override the [] method to return the x and y positions
        - [0] correspond to the x position
        - [1] correspond to the y position
        :param item: index
        :type item: int
        :return: x or y position or None if index > 1
        :rtype: int | None
        """
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.area[1]
        elif item == 3:
            return self.area[0]
        return None

    ### Get focus in list
    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """
        On click set focus to the DragMarker
        :param ev: Mouse event
        :type ev: QMouseEvent
        :return: None
        """
        self.parent().set_focused_marker(self)

    ### Drag and drop
    def mouseMoveEvent(self, e) -> None:
        """
        Drag the DragMarker on left click
        :param e: Mouse event
        :type e: QMouseEvent
        :return: None
        """
        self.print()
        if e.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)

            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)

    ### Change name of marker on double click
    def mouseDoubleClickEvent(self, a0):
        """
        Rename the DragMarker on a double click
        :param a0: Mouse event
        :type a0: QMouseEvent
        :return: None
        """
        self.unfocused()
        name, accepted = QInputDialog.getText(self, 'Enter the new marker name', 'Name:')
        if accepted and name:
            self.setText(name)
        self.focused()

    ### Mouse over selection
    def enterEvent(self, a0):
        """
        Change DragMarker appearance when mouse is over
        :param a0: Mouse event
        :type a0: QMouseEvent
        :return: None
        """
        super(DragMarker, self).enterEvent(a0)
        if not self.is_focus:
            self.setStyleSheet(f"border: 1px solid {self.over_color}")

    ### Mouse of selection
    def leaveEvent(self, a0):
        """
        Reset default appearance when mouse leave de DragMarker
        :param a0: Mouse event
        :type a0: QMouseEvent
        :return: None
        """
        super(DragMarker, self).leaveEvent(a0)
        self.setStyleSheet(f"border: 0px solid {self.over_color}")

        if self.is_focus:
            self.setStyleSheet(f"background-color: {self.focused_color}")

    ### If is selected on the list
    def focused(self):
        """
        Set the focus and change the appearance of the DragMarker in consequence
        :return: None
        """
        self.is_focus = True
        self.setStyleSheet(f"background-color: {self.focused_color}")

    ### If another marker is selected from the list
    def unfocused(self):
        """
            Unset the focus and change the appearance of the DragMarker in consequence
            :return: None
        """
        self.is_focus = False
        self.setStyleSheet(f"border: 0px solid {self.over_color}")

    ### If placed on image
    def place_on_image(self, x: int, y: int, area: tuple):
        """
        Set the position of the marker to x, y.
        Remove and hide it from the MarkerList that contained it
        :param x: X position of the marker
        :type x: int
        :param y: Y position of the marker
        :type y: int
        :return: None
        """
        self.set_pos(x, y, area)
        self.parent().remove_marker()
        self.hide()

    def remove_from_image(self):
        """
            Remove the marker form the DropImage it has been placed upon
            and put it back to the MarkerList (also reset its position to -1,-1)
        :return: None
        """
        self.set_pos(-1, -1, (0, 0, 0, 0))
        self.parent().re_add_marker(self)
        self.show()

    def to_dict(self):
        return {'name': self.text(), 'pos': (self.x, self.y)}
