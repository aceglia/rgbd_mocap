import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from rgbd_mocap.GUI.Video_cropping.crop_video import VideoCropper


class CropVideoTab(QTabWidget):
    """
    A QTabWidget like Widget with default
    layout in tab when calling the addTab function.
    This Tab Widget also has its tab position set
    to North and all the tabs are closable except
    for the first one opened
    """

    def __init__(self, parent=None):
        super(CropVideoTab, self).__init__(parent)
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(lambda index: self.removeTab(index))
        self.tabBarDoubleClicked.connect(lambda index: self.change_tab_name(index))
        self.currentChanged.connect(lambda index: self.hide_dock(index))

        ### List containing all added tabs
        self.tabs = []

    def addTab(self, widget, qwidget=None, *args, **kwargs):
        """
        This method override the default
        addTab method of the QWidgetTab.
        The first tab to be added will not
        be closable and the default layout
        is centered for the added tabs
        :param widget: Widget to be added in the tab
        :type widget: QWidget
        :param qwidget: Name of the new tab
        :type qwidget: str
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: None
        """
        super(CropVideoTab, self).addTab(widget, qwidget)

        # If it's the first tab then it is not closable
        if self.tabBar().count() == 1:
            self.tabBar().setTabButton(0, QTabBar.RightSide, None)

        self.tabs.append(widget)

    def add_crop(self, widget, name="Crop"):
        super(CropVideoTab, self).addTab(widget, name)
        widget.name = name
        self.tabs.append(widget)

    def change_tab_name(self, index):
        """
        Open an input dialog box to put a new name to the tab
        """
        name, accepted = QInputDialog.getText(self, "Enter a new tab name", "Name:")
        if accepted and name:
            self.tabBar().setTabText(index, name)
            if index != 0:
                self.tabs[index].name = name

    def resizeEvent(self, a0: QResizeEvent) -> None:
        """
        Update all its children to resize correctly
        """

        super(CropVideoTab, self).resizeEvent(a0)
        for tab in self.tabs:
            tab.resizeEvent(a0)  # update_image allow resizing for VideoCropper & Cropped Video

    def set_image(self, image_color, image_depth):
        ### Set the new image color/depth to all tabs
        for tab in self.tabs:
            tab.set_image(image_color, image_depth)

        ### Update only the currently selected tab
        ### with filters to avoid unecessary lagging
        i = self.currentIndex()
        self.tabs[i].update_image()

    def removeTab(self, index):
        ### When a tab is remove delete the VideoEditLinker contained in
        ### and also remvoe it from the list self.tabs to avoid lags
        ### and unecessary RAM usage
        item = self.widget(index)
        self.tabs.remove(item)
        item.deleteLater()
        super().removeTab(index)

    def hide_dock(self, index):
        ### Hide all the dock
        for tab in self.tabs[1:]:
            tab.dock.hide()

        ### But the one of the currently selected tab
        if index != 0:
            self.tabs[index].dock.show()

    def get_crops(self):
        crops = []
        for tab in self.tabs[1:]:
            crops.append((tab.name, tab.ve.area))

        return crops


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setGeometry(300, 300, 700, 500)
    vt = CropVideoTab(main_window)
    vc = VideoCropper(vt)
    vt.addTab(vc, "main image")
    main_window.resizeEvent = lambda a: vc.resizeEvent(a)
    main_window.setCentralWidget(vt)

    # vc.set_image(QPixmap("../color_4179.png"))
    main_window.show()
    sys.exit(app.exec_())
