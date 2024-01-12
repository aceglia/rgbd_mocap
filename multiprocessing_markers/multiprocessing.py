import os
import sys

import cv2

import numpy as np
from multiprocessing import Queue, Process
from frames.frames import SharedFrames
from rgbd_mocap.marker_class import MarkerSet


class ProcessHandler:
    STOP = 42
    CONTINUE = 1
    RESET = 2

    def __init__(self, markers_sets, shared_frame: SharedFrames, crop_areas, masks_params, method):
        self.queue_arg_list = []
        self.queue_res = Queue()
        self.queue_end_proc = Queue()
        self.process_list = []

        arguments = {'color_frame_shared': shared_frame.color,
                     'depth_frame_shared': shared_frame.depth,
                     'method': method,
                     'queue_res': self.queue_res,
                     'queue_end': self.queue_end_proc,
                     }
        for i in range(len(markers_sets)):
            queue_arg = Queue()
            queue_arg = queue_arg
            marker_set_shared_memories = markers_sets[i].get_shared_memories()
            crop_area = crop_areas[i]
            mask_params = masks_params[i]

            # (i,
            #  marker_set,
            #  X_np_color,
            #  X_np_depth,
            #  crop_area,
            #  mask_param,
            #  flags,
            #  depth_scale,
            #  clipping_color,
            #  optical_flow_params,) = arguments[i]

            process = Process(target=process_function,
                              args=(i,
                                    queue_arg,
                                    marker_set_shared_memories,
                                    crop_area,
                                    mask_params,
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


def process_function(index,
                     queue_arg,
                     marker_set_shared_memories,
                     crop_area,
                     mask_params,
                     arguments):
    print(f"[Process {index}: Started]")

    # Init Crops
    color_cropped = arguments['color_frame_shared'][crop_area[1]: crop_area[3], crop_area[0]:crop_area[2]]
    depth_cropped = arguments['depth_frame_shared'][crop_area[1]: crop_area[3], crop_area[0]:crop_area[2]]

    # Init MarkerSet
    marker_set = MarkerSet.set_shared_memories(marker_set_shared_memories)

    print(marker_set)

    while True:
        arg = queue_arg.get()

        if arg == ProcessHandler.STOP:
            break

        elif arg == ProcessHandler.CONTINUE:
            blobed_img = arguments['method'](True, color_cropped, depth_cropped, marker_set, mask_params)
            cv2.imshow(f'Crop {index}', blobed_img)
            cv2.waitKey(1)

        elif arg == ProcessHandler.RESET:
            print(f"[Process {index}: Resetting]")

        else:
            print(f"[Process {index}: Order {arg} not implemented]")

        ### When executed its order send back to res queue its index
        arguments['queue_res'].put(index)

    arguments["queue_end"].put(index)

