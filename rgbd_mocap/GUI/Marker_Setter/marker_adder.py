from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from rgbd_mocap.GUI.Marker_Setter.marker_list import MarkerList


class AddMarkerEntry(QLineEdit):
    """
    Text entry to write the name of the DragMarker to be added.
    Return Key is linked to the AddMarkerButton.
    """
    def __init__(self, add_marker_button=None, default_text='New maker'):
        """
        Create a AddMarkerEntry to link with an AddMarkerButton
        :param add_marker_button: The AddMarkerButton to be linked with
        :type add_marker_button: AddMarkerButton
        :param default_text: Default text to be displayed in the entry
        :type default_text: str
        """
        super(AddMarkerEntry, self).__init__(default_text)
        self.default_text = default_text

        if add_marker_button is not None:
            self.returnPressed.connect(lambda: add_marker_button.add_marker() or self.reset())

    def reset(self):
        """
        Reset the entry text.
        :return: None
        """
        self.clear()

    def focusInEvent(self, a0, QFocusEvent=None):
        """
        When focused clear the default text.
        :param a0:
        :type a0:
        :param QFocusEvent:
        :type QFocusEvent:
        :return: None
        """
        self.clear()

    def focusOutEvent(self, a0, QFocusEvent=None):
        """
        When unfocused set back the default text.
        :param a0:
        :type a0:
        :param QFocusEvent:
        :type QFocusEvent:
        :return: None
        """
        self.setText(self.default_text)


class AddMarkerButton(QWidget):
    """
    QPushButton linked to an AddMarkerEntry and contained in
    a MarkerAdder. Pressing the button will add a DragMarker
    in the MarkerList with the name written in the entry.
    """
    def __init__(self, marker_list: MarkerList, parent=None):
        super(AddMarkerButton, self).__init__(parent)
        self.marker_list = marker_list
        self.entry = AddMarkerEntry(self)

        self.add_button = QPushButton('Add')
        self.add_button.pressed.connect(self.add_marker)

        ### Set layout
        layout = QHBoxLayout()
        layout.addWidget(self.add_button)
        layout.addWidget(self.entry)

        self.setLayout(layout)

    def add_marker(self):
        """
        Add a DragMarker in the MarkerList with
        the name written in the entry.
        :return: None
        """
        ### Get the text in the entry
        name = self.entry.text()

        ### If the name isn't valid return
        if name is None or name == "":
            return

        # self.entry.reset()
        self.marker_list.add_marker(name)


class LoadAndRemoveMarkerButtons(QWidget):
    """
    A set of two buttons linked with a MarkerList.
    The left button remove the currently selected DragMarker
    of the MarkerList.
    The right button remove all the DragMarker contained in
    the MarkerList.
    """
    def __init__(self, marker_list: MarkerList, parent=None):
        """
        Initialize the RemoveMarkerButtons and link them to
        the given MarkerList
        :param marker_list: MarkerList to link
        :type marker_list: MarkerList
        :param parent: MarkerSetter
        :type parent: MarkerSetter
        """
        super(LoadAndRemoveMarkerButtons, self).__init__(parent)
        self.load_button = QPushButton('Load Marker Set')
        self.load_button.pressed.connect(marker_list.load_marker_set)

        self.save_button = QPushButton('Save Marker Set')
        self.save_button.pressed.connect(marker_list.save_marker_set)

        self.remove_button = QPushButton('Remove')
        self.remove_button.pressed.connect(marker_list.remove_marker)

        self.remove_all_button = QPushButton('Remove All')
        self.remove_all_button.pressed.connect(marker_list.remove_all_marker)

        ### Set Layout
        layout = QGridLayout()
        layout.addWidget(self.load_button, 0, 0, 1, 2)
        layout.addWidget(self.remove_button, 1, 0, 1, 1)
        layout.addWidget(self.remove_all_button, 1, 1, 1, 1)
        self.setLayout(layout)


class HorizontalSeparator(QFrame):
    def __init__(self, width=1):
        """
        Create a horizontal separator.
        :param width: thickness of the separator (by default 1)
        :type width: int
        """
        super(HorizontalSeparator, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setLineWidth(width)


class MarkerAdder(QWidget):
    """
    A QWidget object containing a MarkerList, a
    AddMarkerButton at the top and a RemoveMarkerButtons
    at the bottom.
    """
    def __init__(self, l=[], parent=None):
        """
        Initialize a MarkerAdder widget.
        :param l: List of string containing the names of the DragMarker to be inited in the MarkerList.
        :type l: list[str]
        :param parent: Container parent
        :type parent: QObject
        """
        super(MarkerAdder, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self.list_marker = MarkerList(l, self)
        ### Create the layout with the AddMarkerButton on the top
        layout = QVBoxLayout(self)
        self.add_marker_widget = AddMarkerButton(self.list_marker, self)
        layout.addWidget(self.add_marker_widget)

        ### Separator between AddMarkerButton and the MarkerList
        layout.addWidget(HorizontalSeparator())

        ### Also creating a scrolling area for the MarkerList

        # scroll.setWidget(tree)
        # scroll.setWidgetResizable(True)
        # layout.addWidget(scroll)
        scroll = QScrollArea()
        scroll.setWidget(self.list_marker)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        # making a scrolling area with tree widget
        # tree = QTreeWidget()
        # tree.setHeaderLabels(['Model'])
        # tree.setColumnCount(1)
        # tree.setDragEnabled(False)
        # tree.setAcceptDrops(False)
        # tree.setDefaultDropAction(Qt.MoveAction)
        #
        # # add items to the tree widget
        #
        # parent = QTreeWidgetItem(tree, ['Segments'])
        # tree.removeItemWidget(parent, 0)
        # # parent.addChildren([QTreeWidgetItem(parent, ['Child 1'])])
        # # parent.addChildren([QTreeWidgetItem(parent, ['Child 2'])])
        # parent.removeChild(parent.child(0))
        # parent_2 = QTreeWidgetItem(tree, ['Segments 2'])
        #
        # scroll = QScrollArea()
        # scroll.setWidget(tree)
        # scroll.setWidgetResizable(True)
        # layout.addWidget(scroll)
        # # parent.setFlags(parent.flags() | Qt.ItemIsDropEnabled)
        # # child1 = QTreeWidgetItem(parent, ['Child 1'])
        # # child1.setFlags(child1.flags() | Qt.ItemIsDragEnabled)
        # # child2 = QTreeWidgetItem(parent, ['Child 2'])
        # child3 = QTreeWidgetItem(parent_2, ['Child 3'])
        # #
        # # # add the tree widget to the layout
        # # layout.addWidget(tree)
        #

        ### At the bottom of the widget the RemoveMarkerButtons
        self.remove_marker_widget = LoadAndRemoveMarkerButtons(self.list_marker)
        layout.addWidget(self.remove_marker_widget)

        self.setLayout(layout)

    def dragEnterEvent(self, e):
        """
        Accept drag enter event
        :param e: Drag event
        :type e: QDragEnterEvent
        :return: None
        """
        e.accept()

    def dropEvent(self, e):
        """
        Pass the drop event to the MarkerList
        :param e: Drop event
        :type e: QDropEvent
        :return: None
        """
        self.list_marker.dropEvent(e)

    def markers_to_dict(self):
        markers = []
        for marker in self.list_marker:
            markers.append(marker.text())

        return markers

    def load_markers(self, markers):
        for marker in markers:
            self.list_marker.add_marker(marker)

    def clear(self):
        self.list_marker.remove_all_marker()


if __name__ == '__main__':
    app = QApplication([])
    list = ['Acrom', 'Scapula', 'Arm_l', ]
    w = MarkerAdder(list)
    w.show()

    app.exec_()