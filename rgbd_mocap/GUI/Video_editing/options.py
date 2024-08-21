from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from rgbd_mocap.GUI.Utils.labeled_slider import LabeledSlider, LabeledRangeSlider
from rgbd_mocap.GUI.Utils.collapsible_box import CollapsibleBox


class OptionBoxLayout(QVBoxLayout):
    """
    A Box layout that stack the various
    widget of the options list in itself.
    """

    def __init__(self, options=None, parent=None):
        """
        Initialize an OptionBoxLayout and add
        all the options in order in it.
        :param options: Various widget to be contained in the OptionBoxLayout
        :type options: list[QWidget]
        :param parent: QObject container parent
        :type parent: QObject
        """
        super(OptionBoxLayout, self).__init__(parent)
        self.activate = ActivateOptionButton(self)
        self.addWidget(self.activate)

        if options is None:
            options = []

        if isinstance(options, list):
            for option in options:
                self.addWidget(option)
        else:
            self.addWidget(options)

    def __iter__(self):
        """
        Overload the iteration operator.
        :return: The various widgets contained in options
        :rtype: QWidget
        """
        ### Jump the first item which is the Activate button
        for i in range(1, self.count()):
            yield self.itemAt(i).widget()

    def __getitem__(self, item):
        return self.itemAt(item).widget()


class ActivateOptionButton(QToolButton):
    """
    A Checkable button linked to an OptionBoxLayout
    enabling or disabling the options contained in it
    """

    def __init__(self, parent: OptionBoxLayout):
        """
        Initialize an ActivateOptionButton with its
        parent being the OptionBoxLayout to linked with.
        :param parent: OptionBoxLayout parent and container
        :type parent: OptionBoxLayout
        """
        super(ActivateOptionButton, self).__init__()
        self.setText("Apply")
        self.setCheckable(True)
        self.parent = parent
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.pressed.connect(lambda: self.on_pressed(self.isChecked()))

    def on_pressed(self, value=None):
        """
        If check enable all the options in the OptionBoxLayout.
        Else disable all the options.
        :return: None
        """
        for child in self.parent:
            child.setEnabled(value)


class WhiteOption(CollapsibleBox):
    """
    White options class. A CollapsibleBox containing
    the image options for white parameters.
    """

    def __init__(self, parent=None, white_range=(0, 255), **kwargs):
        """
        Initialize a WhiteOption widget.
        :param parent: QObject container parent
        :type parent: QObject
        """
        super(WhiteOption, self).__init__("Adjust white", parent)
        self.name = "white_option"

        self.white_range = LabeledRangeSlider(
            "white_range", minimum=0, maximum=255, default_value=white_range, step=41, parent=self
        )

        ### Add the Labeled Sliders to the Collapsible Boxes
        self.option_layout = OptionBoxLayout([self.white_range])
        self.activate = self.option_layout.activate
        self.setContentLayout(self.option_layout)

    def set_params(self, white_option, white_range, **kwargs):
        ### Set activate
        self.option_layout.activate.setChecked(not white_option)
        self.option_layout.activate.on_pressed(white_option)

        ### Set options
        self.white_range.slider.slider.setValue(white_range)


class BlobOption(CollapsibleBox):
    """
    Blob options class. A CollapsibleBox containing
    the image options for white parameters.
    """

    def __init__(
        self, parent=None, blob_area=(10, 100), convexity=1, circularity=1, distance_between_blobs=1, **kwargs
    ):
        """
        Initialize a BlobOption widget.
        :param parent: QObject container parent
        :type parent: QObject
        """
        super(BlobOption, self).__init__("Adjust blob detection", parent)
        self.name = "blob_option"

        self.blob_area = LabeledRangeSlider("blob_area", minimum=1, maximum=500, default_value=blob_area, parent=self)
        self.circularity = LabeledSlider("circularity", minimum=1, maximum=100, default_value=circularity, parent=self)
        self.convexity = LabeledSlider("convexity", minimum=1, maximum=100, default_value=convexity, parent=self)
        self.distance = LabeledSlider(
            "distance_between_blobs", minimum=1, maximum=100, default_value=distance_between_blobs, parent=self
        )

        ### Add the Labeled Sliders to the Collapsible Boxes
        self.option_layout = OptionBoxLayout([self.blob_area, self.circularity, self.convexity, self.distance])
        self.activate = self.option_layout.activate
        self.setContentLayout(self.option_layout)

    def set_params(self, blob_option, blob_area, convexity, circularity, distance_between_blobs=1, **kwargs):
        ### Set activate
        self.option_layout.activate.setChecked(not blob_option)
        self.option_layout.activate.on_pressed(blob_option)

        ### Set options
        self.blob_area.slider.slider.setValue(blob_area)
        self.convexity.slider.slider.setValue(convexity)
        self.circularity.slider.slider.setValue(circularity)
        self.distance.slider.slider.setValue(distance_between_blobs)


class ClaheOption(CollapsibleBox):
    """
    Clahe filters options class. A CollapsibleBox containing
    the image options for white parameters.
    """

    def __init__(self, parent=None, clahe_clip_limit=3, clahe_grid_size=1, gaussian_blur=0, **kwargs):
        """
        Initialize a ClaheOption widget.
        :param parent: QObject container parent
        :type parent: QObject
        """
        super(ClaheOption, self).__init__("Adjust AHE", parent)
        self.name = "clahe_option"

        self.clahe_clip_limit = LabeledSlider(
            "clahe_clip_limit", minimum=1, maximum=40, default_value=clahe_clip_limit, parent=self
        )
        self.clahe_clip_grid_size = LabeledSlider(
            "clahe_grid_size", minimum=1, maximum=40, default_value=clahe_grid_size, parent=self
        )
        self.clahe_median_blur = LabeledSlider(
            "gaussian_blur", minimum=0, maximum=20, default_value=gaussian_blur, parent=self
        )

        ### Add the Labeled Sliders to the Collapsible Boxes
        self.option_layout = OptionBoxLayout([self.clahe_clip_limit, self.clahe_clip_grid_size, self.clahe_median_blur])
        self.activate = self.option_layout.activate
        self.setContentLayout(self.option_layout)

    def set_params(self, clahe_option, clahe_clip_limit, clahe_grid_size, gaussian_blur, **kwargs):
        ### Set activate
        self.option_layout.activate.setChecked(not clahe_option)
        self.option_layout.activate.on_pressed(clahe_option)

        ### Set options
        self.clahe_clip_limit.slider.slider.setValue(clahe_clip_limit)
        self.clahe_clip_grid_size.slider.slider.setValue(clahe_grid_size)
        self.clahe_median_blur.slider.slider.setValue(gaussian_blur)


class DistanceOption(CollapsibleBox):
    """
    Distance options class. A CollapsibleBox containing
    the image options for white parameters.
    """

    def __init__(self, parent=None, distance_in_centimeters=(0, 7000), **kawrgs):
        """
        Initialize a DistanceOption widget.
        :param parent: QObject container parent
        :type parent: QObject
        """
        super(DistanceOption, self).__init__("Adjust distance", parent)
        self.name = "distance_option"

        self.distance = LabeledRangeSlider(
            "distance_in_centimeters",
            minimum=0,
            maximum=700,
            default_value=distance_in_centimeters,
            step=70,
            parent=self,
        )

        self.contour = OptionButton("use_contour", default_value=True)

        ### Add the Labeled Sliders to the Collapsible Boxes
        self.option_layout = OptionBoxLayout([self.distance, self.contour])
        self.activate = self.option_layout.activate
        self.setContentLayout(self.option_layout)

    def set_params(self, distance_option, distance_in_centimeters, use_contour, **kwargs):
        ### Set activate
        self.option_layout.activate.setChecked(not distance_option)
        self.option_layout.activate.on_pressed(distance_option)

        ### Set options
        self.distance.slider.slider.setValue(distance_in_centimeters)
        self.contour.setChecked(use_contour)


class MaskOption(CollapsibleBox):
    """
    Distance options class. A CollapsibleBox containing
    the image options for white parameters.
    """

    def __init__(self, select_area_button=None, parent=None, **kawrgs):
        """
        Initialize a DistanceOption widget.
        :param parent: QObject container parent
        :type parent: QObject
        """
        super(MaskOption, self).__init__("Masks", parent)
        self.name = "masks_option"

        self.select = select_area_button

        ### Add the Labeled Sliders to the Collapsible Boxes
        self.option_layout = OptionBoxLayout([self.select])
        self.activate = self.option_layout.activate
        self.setContentLayout(self.option_layout)

    def set_params(self, masks_option, **kwargs):
        ### Set activate
        self.option_layout.activate.setChecked(not masks_option)
        self.option_layout.activate.on_pressed(masks_option)

        ### Activate option
        self.select.setChecked(masks_option)


class OptionButton(QCheckBox):
    def __init__(self, name, default_value, parent=None):
        super(QCheckBox, self).__init__(parent=parent)
        self.param = name
        self.setText(name.replace("_", " ").capitalize())
        self.setCheckable(True)
        self.setChecked(default_value)
