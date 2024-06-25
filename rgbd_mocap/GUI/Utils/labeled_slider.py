import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from superqt import QRangeSlider
import cv2


class Position:  # Enum
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class SliderLimits(QWidget):
    """
    A Qwidget containing a slider set with some default
    value and appearance. With its min value to the left
    and its maximum value to the right.
    """

    def __init__(self, minimum=0, maximum=100, default_value: int = None, tick_number: int = 10, parent=None):
        """
        Initialize a SliderLimits with some changeable values.
        :param minimum: The minimum value of the slider
        :type minimum: int
        :param maximum: The maximum value of the slider
        :type maximum: int
        :param default_value: The default value of the slider
        :type default_value: int
        :param tick_number: Number of ticks below the slider
        :type tick_number: int
        :param parent: QObject container parent
        :type parent: QObject
        """
        super(SliderLimits, self).__init__(parent)

        ### Init Slider
        self.slider = QSlider(self)
        self.slider.setMaximum(maximum)
        self.slider.setMinimum(minimum)
        self.slider.setMinimumSize(50, 15)
        self.slider.setMaximumHeight(15)
        self.slider.setRange(minimum, maximum)
        self.slider.setTickInterval((maximum - minimum) // tick_number)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep((maximum - minimum) // 20)
        self.slider.setPageStep((maximum - minimum) // 20)
        self.slider.setOrientation(Qt.Horizontal)
        if default_value is not None:
            self.slider.setValue(default_value)
        else:
            self.slider.setValue((maximum - minimum) // 2)

        ### Init Layout
        self.layout_slider = QHBoxLayout()
        self.layout_slider.setContentsMargins(0, 2, 0, 2)

        ### Minimum label
        self.minimum_label = QLabel(str(minimum))
        self.minimum_label.setFont(QFont(self.minimum_label.fontInfo().styleName(), 7))
        self.layout_slider.addWidget(self.minimum_label)

        ### The slider
        self.layout_slider.addWidget(self.slider)

        ### Maximum label
        self.maximum_label = QLabel(str(maximum))
        self.maximum_label.setFont(QFont(self.maximum_label.fontInfo().styleName(), 7))
        self.layout_slider.addWidget(self.maximum_label)

        ### Set Layout
        self.setLayout(self.layout_slider)


class LabeledSlider(QWidget):
    def __init__(
        self,
        name: str,
        position: Position = Position.UP,
        minimum=0,
        maximum=100,
        default_value: int = None,
        tick_number: int = 10,
        parent=None,
    ):
        super(LabeledSlider, self).__init__(parent)
        self.param = name

        ### Init Slider with bounds
        self.slider = SliderLimits(
            minimum=minimum, maximum=maximum, default_value=default_value, tick_number=tick_number, parent=self
        )

        ### Init label and connect it to update_value
        self.name = name.replace("_", " ").capitalize()
        self.label = QLabel()
        self.label.setMaximumHeight(12)
        self.label.setFont(QFont(self.label.fontInfo().styleName(), 8))
        self.slider.slider.valueChanged.connect(self.update_value)

        ### Define Layout base on the position
        if position == Position.RIGHT or position == Position.LEFT:
            layout = QHBoxLayout()
        else:
            layout = QVBoxLayout()

        ### Define order of the layout
        if position == Position.LEFT or position == Position.UP:
            layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.slider, 0)

        else:
            layout.addWidget(self.slider)
            layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)

        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.update_value()

    def update_value(self):
        self.label.setText(self.name + ": " + str(self.slider.slider.value()))


########################################################################################################################
class RangeSliderLimits(QWidget):
    def __init__(
        self, minimum=0, maximum=100, default_value: int = None, tick_number: int = 10, step: int = 20, parent=None
    ):
        super(RangeSliderLimits, self).__init__(parent)

        ### Init Slider
        self.slider = QRangeSlider(Qt.Horizontal)
        self.slider.setMaximum(maximum)
        self.slider.setMinimum(minimum)
        self.slider.setMinimumSize(50, 15)
        self.slider.setMaximumHeight(15)
        self.slider.setRange(minimum, maximum)
        self.slider.setTickInterval((maximum - minimum) // tick_number)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep((maximum - minimum) // step)
        self.slider.setPageStep((maximum - minimum) // step)
        self.slider.setOrientation(Qt.Horizontal)
        if default_value is not None:
            self.slider.setValue(default_value)
        else:
            self.slider.setValue((minimum, maximum))

        ### Init Layout
        self.layout_slider = QHBoxLayout()
        self.minimum_label = QLabel(str(minimum))
        self.minimum_label.setFont(QFont(self.minimum_label.fontInfo().styleName(), 7))
        self.layout_slider.addWidget(self.minimum_label)
        self.layout_slider.addWidget(self.slider)
        self.maximum_label = QLabel(str(maximum))
        self.maximum_label.setFont(QFont(self.maximum_label.fontInfo().styleName(), 7))
        self.layout_slider.addWidget(self.maximum_label)
        self.layout_slider.setContentsMargins(0, 2, 0, 2)

        ### Set Layout
        self.setLayout(self.layout_slider)
        self.slider.rangeChanged.connect(self.update_range_label)

    def update_range_min(self, min):
        self.slider.setMinimum(min)

    def update_range_max(self, max):
        self.slider.setMinimum(max)

    def update_range_label(self):
        self.minimum_label.setText(str(self.slider.minimum()))
        self.maximum_label.setText(str(self.slider.maximum()))


class LabeledRangeSlider(QWidget):
    def __init__(
        self,
        name: str,
        position: Position = Position.UP,
        minimum=0,
        maximum=100,
        default_value: int = None,
        tick_number: int = 10,
        step: int = 20,
        parent=None,
    ):
        super(LabeledRangeSlider, self).__init__(parent)
        self.param = name

        ### Init Slider with bounds
        self.slider = RangeSliderLimits(
            minimum=minimum,
            maximum=maximum,
            default_value=default_value,
            tick_number=tick_number,
            step=step,
            parent=self,
        )

        ### Init label and connect it to update_value
        self.name = name.replace("_", " ").capitalize()
        self.label = QLabel()
        self.label.setMaximumHeight(12)
        self.label.setFont(QFont(self.label.fontInfo().styleName(), 8))
        self.slider.slider.valueChanged.connect(self.update_value)

        ### Define Layout base on the position
        if position == Position.RIGHT or position == Position.LEFT:
            layout = QHBoxLayout()
        else:
            layout = QVBoxLayout()

        ### Define order of the layout
        if position == Position.LEFT or position == Position.UP:
            layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.slider, 0)

        else:
            layout.addWidget(self.slider)
            layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)

        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.update_value()

    def update_value(self):
        self.label.setText(
            self.name + ": " + str(self.slider.slider.value()[0]) + "-" + str(self.slider.slider.value()[1])
        )
