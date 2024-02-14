import time

import cv2
import os

from ..enums import Rotation
from ..processing.multiprocess_handler import MultiProcessHandler, MarkerSet, SharedFrames
from ..frames.frames import Frames
from ..crop.crop import DepthCheck, Crop
from ..processing.process_handler import ProcessHandler
from ..tracking.utils import print_marker


class ProcessImage:
    ROTATION = None
    SHOW_IMAGE = False

    def __init__(self, config, tracking_options, static_markers=None, multi_processing=False):
        # Options
        self.config = config

        # Printing information
        self.computation_time = 0

        # Image
        self.path = config['directory']
        self.index = config['start_index']
        self.color, depth = self._load_img()

        # Frame
        if not multi_processing:
            self.frames = Frames(self.color, depth)
        else:
            self.frames = SharedFrames(self.color, depth)

        self.static_markers = static_markers

        # Init Markers
        self.marker_sets = self._init_marker_set(self.static_markers, multi_processing)
        self.loading_time = 0

        # Set offsets for the marker_sets
        # Already done in the init_marker_set
        # for i in range(len(self.marker_sets)):
        #     self.marker_sets[i].set_offset_pos(config['crops'][i]['area'][:2])
        self.tracking_options = tracking_options
        self._init_crops()
        # Process
        if not multi_processing:
            self.process_handler = ProcessHandler(self.crops)
        else:
            self.process_handler = MultiProcessHandler(self.marker_sets, self.frames, config, tracking_options)
            self.process_handler.start_process()

    def _init_crops(self):
        self.crops = []
        for i in range(len(self.marker_sets)):
            self.crops.append(Crop(self.config['crops'][i]["area"], self.frames, self.marker_sets[i],
                                   self.config['crops'][i]["filters"],
                        self.tracking_options))

    # Init
    def _init_marker_set(self, static_markers, multi_processing):
        set_names = []
        off_sets = []
        marker_names = []
        base_positions = []
        for i in range(len(self.config['crops'])):
            set_names.append(self.config['crops'][i]['name'])
            off_sets.append(self.config['crops'][i]['area'])
            marker_name = []
            base_position = []
            for j in range(len(self.config['crops'][i]['markers'])):
                marker_name.append(self.config['crops'][i]['markers'][j]['name'])
                base_position.append((self.config['crops'][i]['markers'][j]['pos'][1],
                                     self.config['crops'][i]['markers'][j]['pos'][0]))
            marker_names.append(marker_name)
            base_positions.append(base_position)

        marker_sets: list[MarkerSet] = []
        for i in range(len(set_names)):
            marker_set = MarkerSet(set_names[i], marker_names[i], multi_processing)
            marker_set.set_markers_pos(base_positions[i])
            marker_set.set_offset_pos(off_sets[i][:2])
            depth_cropped = self.frames.get_crop(off_sets[i])[1]
            for marker in marker_set:
                marker.set_depth(DepthCheck.check(marker.get_pos(), depth_cropped, 0, 10000)[0])
                if static_markers and marker.name in static_markers:
                    marker.is_static = True
            marker_sets.append(marker_set)

        return marker_sets

    # Loading
    def _load_img(self):
        color, depth = None, None
        while color is None or depth is None:
            color, depth = load_img(self.path, self.index, self.ROTATION)
            if color is None or depth is None:
                self.index += 1
                if self.index == self.config['end_index']:
                    return None, None
        return color, depth

    def _update_img(self, color, depth):
        if color is None or depth is None:
            return False

        self.frames.set_images(color, depth)

        return True

    # Processing
    def _process_after_loading(self):
        # Update frame
        color, depth = self._load_img()

        # Process image
        self.process_handler.send_and_receive_process()

        if not self._update_img(color, depth):  # If image could not be loaded then skip to the next one
            return False
        return True

    def _process_while_loading(self):
        # Start the processing of the current image
        self.process_handler.send_process()
        # self.blobs = self.process_handler.blobs

        # Load next frame
        color, depth = self._load_img()  # If image could not be loaded then skip to the next one

        # Wait for the end of the processing of the image
        self.process_handler.receive_process()

        # # If image could not be loaded then skip to the next one
        return self._update_img(color, depth)

    def process_next_image(self, process_while_loading=True):
        tik = time.time()

        # Get next image
        self.index += 1

        if self.index == self.config['end_index']:
            return False

        # Process
        if process_while_loading:
            if not self._process_while_loading():
                return True
        else:
            if not self._process_after_loading():
                return True

        if ProcessImage.SHOW_IMAGE:
            cv2.namedWindow('Main image :', cv2.WINDOW_NORMAL)
            im = self.get_processed_image().copy()
            cv2.putText(
                im,
                f"Frame : {self.index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow('Main image :', im)
            if cv2.waitKey(1) == ord('q'):
                return False

        tok = time.time() - tik
        self.computation_time = tok

        return True

    def process_all_image(self):
        total_time = 0

        if ProcessImage.SHOW_IMAGE:
            cv2.namedWindow('Main image :', cv2.WINDOW_NORMAL)

        while self.index != self.config['end_index']:
            tik = time.time()

            # Get next image
            self.index += 1

            # Process
            if not self._process_while_loading():
                continue

            if ProcessImage.SHOW_IMAGE:
                cv2.imshow('Main image :', self.get_processed_image())
                if cv2.waitKey(1) == ord('q'):
                    break

            tok = time.time() - tik
            total_time += tok
            self.computation_time = tok

        self.process_handler.end_process()
        nb_img = self.index - self.config['start_index']
        return total_time, total_time / nb_img

    def get_processed_image(self):
        img = print_marker_sets(self.color, self.marker_sets)
        self.color = self.frames.color.copy()

        return img

    def get_processing_time(self):
        return self.computation_time


def print_marker_sets(frame, marker_sets):
    for i, marker_set in enumerate(marker_sets):
        frame = print_marker(frame, marker_set)

    return frame


def load_img(path, index, rotation=None):  # Possibly change it to also allow the usage of the camera
    color_file = path + os.sep + f"color_{index}.png"
    depth_file = path + os.sep + f"depth_{index}.png"

    if not os.path.isfile(color_file) or not os.path.isfile(depth_file):
        return None, None
    try:
        color_image = cv2.imread(color_file)
        depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    except:
        return None, None

    if rotation is not None and rotation != Rotation.ROTATE_0:
        if rotation != Rotation.ROTATE_180:
            raise NotImplementedError("Only 180 degrees rotation is implemented")
        color_image = cv2.rotate(color_image, rotation.value)
        depth_image = cv2.rotate(depth_image, rotation.value)

    return color_image, depth_image
