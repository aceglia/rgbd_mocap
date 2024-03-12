import sys
import os
import json
from pathlib import Path

import numpy as np

from rgbd_mocap.GUI.Video_editing.options import *
from rgbd_mocap.GUI.Utils.popup import ErrorPopUp
from rgbd_mocap.GUI.Utils.file_dialog import SaveDialog, LoadDialog


class ImageOptionsButtons(QWidget):
    def __init__(self, image_options):
        super(ImageOptionsButtons, self).__init__(image_options)
        self.setFixedHeight(80)

        self.image_option = image_options

        ### Copy and Paste Buttons
        self.copy_button = QPushButton('Copy Parameters')
        self.paste_button = QPushButton('Paste Parameters')

        self.copy_button.pressed.connect(self.copy_)
        self.paste_button.pressed.connect(self.paste)

        self.default_dir = os.getcwd() + os.sep + 'save' + os.sep
        self.parameters = {}

        ### Save and Load buttons
        self.save_button = QPushButton('Save Parameters')
        self.load_button = QPushButton('Load Parameters')

        self.save_button.pressed.connect(self.save)
        self.load_button.pressed.connect(self.load)

        self.dir = self.default_dir

        ### Set Layout
        layout = QGridLayout()
        layout.addWidget(self.copy_button, 0, 0)
        layout.addWidget(self.paste_button, 0, 1)
        layout.addWidget(self.save_button, 1, 0)
        layout.addWidget(self.load_button, 1, 1)
        self.setLayout(layout)

    ### Generic method
    def save_parameters(self, file):
        if not os.path.isdir(Path(file).parent):
            os.makedirs(Path(file).parent)

        with open(file, 'w') as f:
            json.dump(self.image_option.to_dict(), f)

    def load_parameters(self, file):
        with open(file, 'r') as f:
            parameters = json.load(f)
            try:
                self.image_option.set_params(parameters, parameters['mask'])

            except TypeError:
                ErrorPopUp('File could not be loaded, wrong format')

    ### Copy and Paste
    def copy_(self):
        self.save_parameters(self.default_dir + 'tmp')

    def paste(self):
        if os.path.isfile(self.default_dir + 'tmp'):
            self.load_parameters(self.default_dir + 'tmp')
        else:
            ErrorPopUp('Nothing to paste')

    ### Save and Load
    def save(self):
        SaveDialog(parent=self,
                   caption='Saving parameters file',
                   filter='Save File (*.save)',
                   suffix='save',
                   save_method=self.save_parameters,
                   )

    def load(self):
        LoadDialog(parent=self,
                   caption='Load parameters file',
                   filter='Save File (*.save);; Any(*)',
                   load_method=self.load_parameters)


class ImageOptions(QWidget):
    """
    This class init a QWidget with some
    preset information.
    It contains all the modification image options :
        - Blob min area
        - Blob circularity
        - Blob convexity
        - Min White Threshold
        - Max White Threshold
        - Clahe clip limit
        - Clahe clip grid size
        - Min&Max distance
        - Contour remover
        - Masks
    """

    def __init__(self, video_filter, parent=None):
        """
        Initialize an ImageOptions.
        :param parent: QWindow container
        :type parent: QWindow | VideoFilters
        """
        super(ImageOptions, self).__init__(parent)
        self.video_filter = video_filter

        self.params = {
            "blend": 100,
            "white_range": (100, 255),
            "blob_area": (1, 200),
            "convexity": 5,
            "circularity": 5,
            "distance_between_blobs": 1,
            "distance_in_centimeters": (5, 500),
            "clahe_clip_limit": 1,
            "clahe_grid_size": 3,
            "gaussian_blur": 0,
            "use_contour": True,
            "mask": None,
        }

        self.show_params = {
            "white_option": True,
            "blob_option": True,
            "clahe_option": True,
            "distance_option": True,
            "masks_option": False,
        }

        ### Create layout and options list ###
        layout = QVBoxLayout()
        self.options = []

        ### Blend option over all ###
        self.blend_option = LabeledSlider('Show applied filters (%): ',
                                          minimum=0, maximum=100,
                                          default_value=self.params["blend"],
                                          parent=self)
        layout.addWidget(self.blend_option)
        self.blend_option.slider.slider.valueChanged.connect(lambda value:
                                                             self.set_param_value('blend', value) or
                                                             self.video_filter.update())

        ### Create the boxes
        self.white_option = WhiteOption(**self.params)
        self.options.append(self.white_option)
        self.blob_option = BlobOption(**self.params)
        self.options.append(self.blob_option)
        self.clahe_option = ClaheOption(**self.params)
        self.options.append(self.clahe_option)
        self.distance_option = DistanceOption(**self.params)
        self.options.append(self.distance_option)

        # for all options in self (use override __iter__ method)
        for option in self:
            layout.addWidget(option)
        # layout.addStretch()

        self.set_masks_options(self.parent(),layout)
        ### Create the buttons ###
        self.buttons = ImageOptionsButtons(self)
        layout.addWidget(self.buttons, 0, Qt.AlignBottom)

        self.setLayout(layout)
        layout.addStretch()

        ### Link signals
        for options in self:
            ### Need to pass via another function to avoid bad linking
            self.link_options(options)

    def link_options(self, options):
        options.activate.pressed.connect(lambda: self.set_show_params_value(options.name,
                                                                            options.activate.isChecked()) or
                                                 self.video_filter.update())

        ## Connect options sliders
        for option in options.option_layout:
            ### Need to pass via an other function to avoid bad linking
            self.link_option(option)

    def link_option(self, option):
        if isinstance(option, OptionButton):
            option.pressed.connect(lambda: (self.set_param_value(option.param, not option.isChecked()) or
                                            self.video_filter.update()))
        elif isinstance(option, LabeledSlider) or isinstance(option, LabeledRangeSlider):
            option.slider.slider.valueChanged.connect(
                lambda value: (self.set_param_value(option.param, value) or
                               self.video_filter.update()))

    def __getitem__(self, item):
        found = self.params.get(item)
        if found:
            return found

        return self.show_params.get(item)

    def __iter__(self):
        for option in self.options:
            yield option

    def set_param_value(self, key, value):
        # print(key, value)
        self.params[key] = value

    def set_show_params_value(self, key, value):
        # print(key, value)
        self.show_params[key] = value

    def print(self):
        print(self.params)
        print(self.show_params)

    def set_params(self, parameters, mask):
        ### The blend opotion is not contained in a option layout so update by hand
        self.blend_option.slider.slider.setValue(parameters['blend'])

        ### For the other option just update via the option layout
        for option in self.options:
            option.set_params(**parameters)

        ### Need to force the update of show_params
        ### Because on_pressed isn't (cannot ?) be
        ### activated via .set_params()
        for key in self.show_params.keys():
            self.show_params[key] = parameters[key]
        if mask is not None:
            self.set_mask(mask)
        self.video_filter.update()

    def set_masks_options(self, vel, layout):
        self.masks_option = MaskOption(vel.select_area_button, **self.params)
        self.options.append(self.masks_option)
        layout.addWidget(self.masks_option)

        self.link_options(self.masks_option)

    def set_mask(self, mask):
        self.video_filter.parent.set_mask(mask)

    def to_dict(self):
        parameters = {}
        parameters.update(self.params)
        parameters.update(self.show_params)

        vel = self.video_filter.parent
        if parameters['masks_option'] and not np.all(vel.mask) == 1:
            mask = vel.mask.astype(dtype=np.uint8)
            zeros_idx = np.where(mask == 0)
            idx_to_save = [zeros_idx[0].tolist(), zeros_idx[1].tolist()]
            parameters['mask'] = idx_to_save
        else:
            parameters['mask'] = None

        return parameters


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = QMainWindow()
    image_opt = ImageOptions(demo)

    ### ScrollArea Bonus ###
    scroll = QScrollArea(demo)
    scroll.setWidget(image_opt)
    scroll.setWidgetResizable(True)

    demo.setCentralWidget(scroll)
    demo.show()
    sys.exit(app.exec_())
