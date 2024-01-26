import time

import cv2

from processing.multiprocess_handler import MultiProcessHandler, MarkerSet, SharedFrames
from frames.frames import Frames
from crop.crop import DepthCheck
from processing.process_handler import ProcessHandler
from tracking.test_tracking import print_marker


class ProcessImage:
    ROTATION = None
    SHOW_IMAGE = False

    def __init__(self, config, tracking_options, shared=False):
        # Options
        self.config = config

        # Printing information
        self.computation_time = 0

        # Image
        self.path = config['directory']
        self.index = config['start_index']
        self.color, depth = self._load_img()

        # Frame
        if not shared:
            self.frames = Frames(self.color, depth)
        else:
            self.frames = SharedFrames(self.color, depth)

        # Init Markers
        self.marker_sets = self._init_marker_set(shared)

        # Set offsets for the marker_sets
        for i in range(len(self.marker_sets)):
            self.marker_sets[i].set_offset_pos(config['crops'][i]['area'][:2])

        # Process
        if not shared:
            self.process_handler = ProcessHandler(self.marker_sets, self.frames, config, tracking_options)
        else:
            self.process_handler = MultiProcessHandler(self.marker_sets, self.frames, config, tracking_options)
            self.process_handler.start_process()

    # Init
    def _init_marker_set(self, shared):
        set_names = []
        off_sets = []
        marker_names = []
        base_positions = []

        for i in range(len(self.config['crops'])):
            set_names.append(self.config['crops'][i]['name'])
            off_sets.append(self.config['crops'][i]['area'][:2])

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
            marker_set = MarkerSet(set_names[i], marker_names[i], shared)
            marker_set.set_markers_pos(base_positions[i])
            marker_set.set_offset_pos(off_sets[i])
            for marker in marker_set:
                marker.set_depth(DepthCheck.check(marker.get_pos(), self.frames.depth, 0, 10000)[0])
            marker_sets.append(marker_set)

        return marker_sets

    # Loading
    def _load_img(self):
        color, depth = load_img(self.path, self.index, self.ROTATION)
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
        if not self._update_img(color, depth):  # If image could not be loaded then skip to the next one
            return False

        # Process image
        self.process_handler.send_and_receive_process()

        return True

    def _process_while_loading(self):
        # Start the processing of the current image
        self.process_handler.send_process()

        # Load next frame
        color, depth = self._load_img()  # If image could not be loaded then skip to the next one

        # Wait for the end of the processing of the image
        self.process_handler.receive_process()

        # # If image could not be loaded then skip to the next one
        return self._update_img(color, depth)

    def process_next_image(self):
        tik = time.time()

        # Get next image
        self.index += 1

        if self.index == self.config['end_index']:
            return False

        # Process
        if not self._process_while_loading():
            return True

        if ProcessImage.SHOW_IMAGE:
            cv2.imshow('Main image :', self._get_processed_image())
            if cv2.waitKey(1) == ord('q'):
                return False

        tok = time.time() - tik
        self.computation_time = tok

        return True

    def process_all_image(self):
        total_time = 0

        while self.index != self.config['end_index']:
            tik = time.time()

            # Get next image
            self.index += 1

            # Process
            if not self._process_while_loading():
                continue

            if ProcessImage.SHOW_IMAGE:
                cv2.imshow('Main image :', self._get_processed_image())
                if cv2.waitKey(1) == ord('q'):
                    break

            tok = time.time() - tik
            total_time += tok
            self.computation_time = tok

        self.process_handler.end_process()
        nb_img = self.index - self.config['start_index']
        return total_time, total_time / nb_img

    def _get_processed_image(self):
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
    color_file = path + f"color_{index}.png"
    depth_file = path + f"depth_{index}.png"

    color_image = cv2.imread(color_file, cv2.COLOR_BGR2RGB)
    depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

    if rotation is not None:
        color_image = cv2.flip(color_image, rotation)
        depth_image = cv2.flip(depth_image, rotation)

    return color_image, depth_image
