import sys
import time

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from superqt import QRangeSlider, QLabeledRangeSlider


class Icon(QPushButton):
    """
    Simple class to set icon from default options
    """

    def __init__(self, icon: str = "SP_DialogApplyButton"):
        super(Icon, self).__init__()
        self.update_icon(icon)
        self.setStyleSheet("QPushButton { border: none; }")
        self.setIconSize(QSize(40, 40))

    def update_icon(self, icon: str):
        att = getattr(QStyle, icon)
        self.setIcon(self.style().standardIcon(att))


class VideoControl(QWidget):
    def __init__(self, start_index: int = 100, end_index: int = 1000, hz: int = 60, parent=None):
        super(VideoControl, self).__init__(parent)
        layout = QGridLayout()
        self.value = start_index
        self.setFixedHeight(70)
        self.setMinimumWidth(250)

        ### Init Sliders
        self.slider_interval = QSlider(Qt.Vertical)
        self.slider_interval.setValue(hz)
        self.slider_interval.setRange(hz // 2, hz * 2)
        layout.addWidget(self.slider_interval, 0, 0, 3, 1)

        # Control step
        self.step = 1
        self.button_backward = Icon("SP_MediaSkipBackward")
        layout.addWidget(self.button_backward, 0, 1, 1, 1, Qt.AlignmentFlag.AlignRight)

        self.button_forward = Icon("SP_MediaSkipForward")
        layout.addWidget(self.button_forward, 0, 3, 1, 1, Qt.AlignmentFlag.AlignLeft)

        ### Init Animation
        self.button_start_stop = Icon("SP_MediaPlay")
        layout.addWidget(self.button_start_stop, 0, 2, 1, 1)
        ## Could be better with a QLabeledRangeSlider but is error prone..
        self.slider_anim = QRangeSlider(Qt.Horizontal)
        self.slider_anim.setRange(start_index, end_index)
        self.slider_anim.setValue((start_index, start_index, end_index))
        layout.addWidget(self.slider_anim, 2, 1, 1, 3)

        ### Connect timer value to interval slider
        self.timer = QTimer()
        self.timer.setInterval(self.slider_interval.value())
        self.slider_interval.valueChanged.connect(self.update_interval)

        ### Play button
        self.button_start_stop.clicked.connect(self.start_or_pause)
        self.button_backward.clicked.connect(self.go_backward)
        self.button_forward.clicked.connect(self.go_forward)

        ### Interval
        self.timer.timeout.connect(self.update_anim_value)

        self.setLayout(layout)

    def update_interval(self):
        self.timer.setInterval(1000 // self.slider_interval.value())  # From Hz to millisecond

    def update_anim_value(self):
        value = self.slider_anim.value()[1] + self.step
        if value > self.slider_anim.value()[2]:
            value -= self.slider_anim.value()[2] - self.slider_anim.value()[0]

        if value < self.slider_anim.value()[0]:
            value += self.slider_anim.value()[2] - self.slider_anim.value()[0]

        self.slider_anim.setValue((self.slider_anim.value()[0], value, self.slider_anim.value()[2]))
        self.value = value

    def start_or_pause(self):
        (self.timer.stop() or self.button_start_stop.update_icon("SP_MediaPlay")
         if self.timer.isActive() else
         self.timer.start() or self.button_start_stop.update_icon("SP_MediaPause"))

    def go_backward(self):
        self.step -= 1
        self.button_forward.setDisabled(False)

        if self.step < -4:
            self.button_backward.setDisabled(True)

    def go_forward(self):
        self.step += 1
        self.button_backward.setDisabled(False)

        if self.step > 4:
            self.button_forward.setDisabled(True)

    def adjust(self, min_index, max_index):
        self.setEnabled(True)
        self.slider_anim.setRange(min_index, max_index)
        self.slider_anim.setValue((min_index, min_index, max_index))
        self.value = min_index


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setCentralWidget(VideoControl())
    main_window.show()
    sys.exit(app.exec_())
