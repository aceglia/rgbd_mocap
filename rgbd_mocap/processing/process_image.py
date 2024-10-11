import time

import cv2
import os

from ..enums import Rotation
from ..processing.multiprocess_handler import MultiProcessHandler, MarkerSet, SharedFrames
from ..frames.frames import Frames
from ..crops.crop import DepthCheck, Crop
from ..processing.process_handler import ProcessHandler
from ..tracking.utils import print_marker, check_tracking_config_file


class ProcessImage:
    ROTATION = None
    SHOW_IMAGE = False

    def __init__(self, config, tracking_options, static_markers=None, bounded_markers=None,
                 multi_processing=False, from_dlc=False, dlc_model_path=None, processor=None, dlc_marker_names=(),
                 ignore_all_checks=False, dlc_enhance_markers=(),
                 downsample_ratio=None):
        # Options
        self.config = config
        self.from_dlc = from_dlc
        dlc_marker_names = dlc_marker_names if from_dlc else ()
        dlc_enhance_markers = dlc_enhance_markers if from_dlc else ()

        # Printing information
        self.computation_time = 0
        config, is_markers = check_tracking_config_file(config, dlc_marker_names, dlc_enhance_markers)
        if not is_markers and tracking_options["optical_flow"] and from_dlc:
            raise RuntimeError("You cannot use optical flow while using solely markers from Deeplabcut.")
        # Image
        self.count = 0
        self.downsample_ratio = downsample_ratio
        self.path = config['directory']
        self.index = config['start_index']
        self.masks = config['masks']
        self._dispatch_mask()
        self.first_image_loaded = False
        return_color = False if (self.from_dlc and not tracking_options["optical_flow"] or (from_dlc and len(dlc_enhance_markers) == 0)) else True
        self.color, self.depth, _ = self._load_img(return_color=return_color)

        # Frame
        if not multi_processing:
            self.frames = Frames(self.color, self.depth, self.index, self.downsample_ratio)
        else:
            self.crops = None
            self.frames = SharedFrames(self.color, self.depth, self.index, self.downsample_ratio)

        self.static_markers = static_markers
        self.bounded_markers, self.bounded_markers_bounds = bounded_markers

        # Init Markers
        self.marker_sets = self._init_marker_set(self.static_markers,
                                                 self.bounded_markers,
                                                 self.bounded_markers_bounds,
                                                 multi_processing,
                                                 dlc_enhance_markers, 
                                                 self.downsample_ratio)
        self.loading_time = 0
        self.last_index = 0

        # Set offsets for the marker_sets
        # Already done in the init_marker_set
        # for i in range(len(self.marker_sets)):
        #     self.marker_sets[i].set_offset_pos(config['crops'][i]['area'][:2])
        self.tracking_options = tracking_options
        self._init_crops(from_dlc=from_dlc, dlc_model_path=dlc_model_path, processor=processor,
                         dlc_enhance_markers=dlc_enhance_markers, ignore_all_checks=ignore_all_checks)
        # Process
        if not multi_processing:
            self.process_handler = ProcessHandler(self.crops)
        else:
            self.process_handler = MultiProcessHandler(self.marker_sets, self.frames, config, tracking_options)
            self.process_handler.start_process()

    def _dispatch_mask(self):
        for crop in self.config["crops"]:
            for n, mask in enumerate(self.masks):
                if mask is None:
                    continue
                if mask["name"] == crop['name']:
                    crop["filters"]["mask"] = self.masks[n]["value"]
                    break

    def _init_crops(self, from_dlc=False, dlc_model_path=None, processor=None, dlc_enhance_markers=(), ignore_all_checks=False):
        self.crops = []
        for i in range(len(self.marker_sets)):
            self.crops.append(Crop(self.config['crops'][i]["area"], self.frames, self.marker_sets[i],
                                   self.config['crops'][i]["filters"],
                        self.tracking_options, from_dlc, dlc_model_path, processor,
                                   ignore_all_checks=ignore_all_checks))

    # Init
    def _init_marker_set(self, static_markers, bounded_markers, bounds, multi_processing, dlc_enhance_markers=(), downsample_ratio=1):
        set_names = []
        off_sets = []
        marker_names = []
        base_positions = []
        dlc_marker = []
        dlc_markers = []
        ignore_from_dlc_all = []
        for i in range(len(self.config['crops'])):
            set_names.append(self.config['crops'][i]['name'])
            area = self.config['crops'][i]['area']
            self.config['crops'][i]['area'] = [int(a * downsample_ratio) for a in area]
            off_sets.append(self.config['crops'][i]['area'])
            marker_name = []
            base_position = []
            ignore_from_dlc = []
            for j in range(len(self.config['crops'][i]['markers'])):
                marker_name.append(self.config['crops'][i]['markers'][j]['name'])
                base_position.append((self.config['crops'][i]['markers'][j]['pos'][1] * downsample_ratio,
                                     self.config['crops'][i]['markers'][j]['pos'][0] * downsample_ratio))
            if "dlc_markers" in self.config['crops'][i].keys():
                for j in range(len(self.config['crops'][i]['dlc_markers'])):
                    if self.config['crops'][i]['dlc_markers'][j]['name'] not in marker_name:
                        marker_name.append(self.config['crops'][i]['dlc_markers'][j]['name'])
                        base_position.append((self.config['crops'][i]['dlc_markers'][j]['pos'][1] * downsample_ratio,
                                             self.config['crops'][i]['dlc_markers'][j]['pos'][0] * downsample_ratio))
                    else:
                        ignore_from_dlc.append(self.config['crops'][i]['dlc_markers'][j]['name'])
                        print(f"!!WARNING: Marker {ignore_from_dlc[-1]} is in the tracking config file and is provided through DLC."
                              f"The DLC value will be ignored please choose carefully the tracking algorithm for this "
                              f"marker.")
                    dlc_marker.append(self.config['crops'][i]['dlc_markers'][j]['name'])
            marker_names.append(marker_name)
            base_positions.append(base_position)
            dlc_markers.append(dlc_marker)
            ignore_from_dlc_all.append(ignore_from_dlc)

        marker_sets: list[MarkerSet] = []
        for i in range(len(set_names)):
            marker_set = MarkerSet(set_names[i], marker_names[i], multi_processing, downsample_ratio=downsample_ratio)
            marker_set.markers_from_dlc = dlc_markers[i]
            marker_set.ignore_from_dlc = ignore_from_dlc_all[i]
            marker_set.dlc_enhance_markers = dlc_enhance_markers
            marker_set.set_markers_pos(base_positions[i])
            marker_set.set_offset_pos(off_sets[i][:2])
            depth_cropped = self.frames.get_crop(off_sets[i])[1]
            for marker in marker_set:
                if marker.name in marker_set.markers_from_dlc and marker.name not in marker_set.dlc_enhance_markers and marker.name not in marker_set.ignore_from_dlc:
                    marker.from_dlc = True
                    marker.set_depth(-1)
                else:
                    marker.set_depth(DepthCheck.check(marker.get_pos(), depth_cropped, 0, 10000)[0])
                if marker.name in marker_set.dlc_enhance_markers:
                    marker.set_depth(-1)
                if static_markers and marker.name in static_markers:
                    marker.is_static = True
                if bounded_markers and marker.name in bounded_markers:
                    marker.set_bounds(bounds[bounded_markers.index(marker.name)], downsample_ratio)
            marker_sets.append(marker_set)

        return marker_sets

    # Loading
    def _load_img(self, return_color=True):
        color, depth = None, None
        count = 1 if self.first_image_loaded else 0
        while (color is None and return_color is True) or depth is None:
            color, depth = load_img(self.path, self.index + count, self.ROTATION, return_color)
            if (color is None and return_color is True) or depth is None:
                count += 1
                if self.index == self.config['end_index']:
                    return None, None
        return color, depth, count

    def _update_img(self, color, depth, count):
        if not self.first_image_loaded:
            print("load_first")
            self.first_image_loaded = True
        else:
            self.last_index = self.index
            self.index += count
            self.frames.set_images(color, depth, self.index)

    # Processing
    def _process_after_loading(self):
        # Update frame
        return_color = False if (self.from_dlc and not self.tracking_options["optical_flow"]) else True
        color, depth, count = self._load_img(return_color=return_color)

        tic = time.time()
        # Process image
        self.process_handler.send_and_receive_process()

        self.computation_time = time.time() - tic

        return color, depth, count

    def _process_while_loading(self):
        # Start the processing of the current image
        self.process_handler.send_process()
        # self.blobs = self.process_handler.blobs
        return_color = False if (self.from_dlc and not self.tracking_options["optical_flow"]) else True
        # Load next frame
        color, depth, count = self._load_img(return_color=return_color)  # If image could not be loaded then skip to the next one

        # Wait for the end of the processing of the image
        self.process_handler.receive_process()

        # # If image could not be loaded then skip to the next one
        return color, depth, count

    def process_next_image(self, process_while_loading=True):
        tik = time.time()
        self._update_img(self.color, self.depth, self.count)

        if self.index == self.config['end_index']:
            return False
        # Process
        if process_while_loading:
            self.color, self.depth, self.count = self._process_while_loading()
        else:
            self.color, self.depth, self.count = self._process_after_loading()

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
        # Get next image
        # self.index += 1
        tok = time.time() - tik
        # self.computation_time = tok

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
        if self.frames.color is None:
            raise RuntimeError("Show image not available when using from_dlc.")
        img = print_marker_sets(self.frames.color.copy(), self.marker_sets)
        return img

    def get_processing_time(self):
        return self.computation_time


def print_marker_sets(frame, marker_sets):
    for i, marker_set in enumerate(marker_sets):
        frame = print_marker(frame, marker_set)

    return frame


def load_img(path, index, rotation=None, return_color=True):  # Possibly change it to also allow the usage of the camera
    color_file = path + os.sep + f"color_{index}.png"
    depth_file = path + os.sep + f"depth_{index}.png"

    if not os.path.isfile(color_file) or not os.path.isfile(depth_file):
        return None, None
    try:
        color_image = None if not return_color else cv2.imread(color_file, cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    except:
        return None, None

    if rotation is not None and rotation != Rotation.ROTATE_0:
        if rotation != Rotation.ROTATE_180:
            raise NotImplementedError("Only 180 degrees rotation is implemented")
        color_image = None if not return_color else cv2.rotate(color_image, rotation.value)
        depth_image = cv2.rotate(depth_image, rotation.value)

    return color_image, depth_image
