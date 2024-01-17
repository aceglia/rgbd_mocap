import os
import sys

import cv2

import numpy as np
from multiprocessing import Queue, Process
from frames.shared_frames import SharedFrames
from markers.marker_set import MarkerSet
from crop.crop import Crop
from tracking.test_tracking import print_blobs, print_marker, print_estimated_positions, set_marker_pos


class ProcessHandler:
    STOP = 42
    CONTINUE = 1
    RESET = 2

    def __init__(self, markers_sets: list[MarkerSet], shared_frame: SharedFrames, options, tracking_option):
        self.queue_arg_list = []
        self.queue_res = Queue()
        self.queue_end_proc = Queue()
        self.process_list = []

        arguments = {'queue_res': self.queue_res,
                     'queue_end': self.queue_end_proc,
                     }

        for i in range(len(markers_sets)):
            queue_arg = Queue()
            queue_arg = queue_arg
            marker_set = markers_sets[i]
            option = options['crops'][i]

            process = Process(target=ProcessHandler._process_function,
                              args=(i,
                                    queue_arg,
                                    marker_set,
                                    shared_frame,
                                    option,
                                    tracking_option,
                                    arguments),
                              daemon=True)

            self.queue_arg_list.append(queue_arg)
            self.process_list.append(process)

    def start_process(self):
        for process in self.process_list:
            process.start()

    def send_process(self, order=1):
        for queue in self.queue_arg_list:
            queue.put(order)

        for _ in self.queue_arg_list:
            res = self.queue_res.get()
            # print(f"[Process {res}: Returned]")

    def end_process(self):
        for queue in self.queue_arg_list:
            queue.put(ProcessHandler.STOP)

        ### Wait for the process to stop
        for _ in self.process_list:
            res = self.queue_end_proc.get()
            # print(f"[Process {res}: Stopped]")

        ### When all the process are stopped join them
        for process in self.process_list:
            process.join()

        print('All process stopped')

    @staticmethod
    def _process_function(index,
                         queue_arg,
                         marker_set: MarkerSet,
                         shared_frame: SharedFrames,
                         crop_option,
                         tracking_option,
                         arguments):
        print(f"[Process {index}: Started]")

        # Init Crop
        crop = Crop(crop_option['area'], shared_frame, marker_set, crop_option['filters'], tracking_option)
        print(marker_set)

        while True:
            arg = queue_arg.get()

            if arg == ProcessHandler.STOP:
                break

            elif arg == ProcessHandler.CONTINUE:
                blobs, positions, estimate_positions = crop.track_markers()
                set_marker_pos(marker_set, positions)

                img = crop.filter.filtered_frame
                img = print_blobs(img, blobs)
                img = print_estimated_positions(img, estimate_positions)
                img = print_marker(img, marker_set)

                cv2.imshow(f"{crop_option['name']} {index}", img)
                cv2.waitKey(1)

            elif arg == ProcessHandler.RESET:
                print(f"[Process {index}: Resetting]")

            else:
                print(f"[Process {index}: Order {arg} not implemented]")

            ### When executed its order send back to res queue its index
            arguments['queue_res'].put(index)

        arguments["queue_end"].put(index)
