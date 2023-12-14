import os
import sys

import cv2

import rgbd_mocap.marker_class
import numpy as np
from multiprocessing import Queue, Process, Lock, RawArray


class SharedFrames:
    def __init__(self, color_frame, depth_frame):
        if color_frame is None or depth_frame is None:
            raise ValueError(f'{self}: color_frame and depth frame should be init.')

        self.width = color_frame.shape[0]
        self.height = color_frame.shape[1]

        color_array = RawArray('c', self.width * self.height * 3)  # 'c' -> value between 0-255
        depth_array = RawArray('i', self.width * self.height)  # 'i' -> int32

        self.color = np.frombuffer(color_array, dtype=np.uint8).reshape((self.width, self.height, 3))
        self.depth = np.frombuffer(depth_array, dtype=np.int32).reshape((self.width, self.height))

        np.copyto(self.color, color_frame)
        np.copyto(self.depth, depth_frame)

    def shape_error(self, got, expected):
        raise ValueError(
            f'{self}: Given array has a wrong shape, got "{got.shape}" expected "{expected.shape}".')

    def set_images(self, color_frame, depth_frame):
        if color_frame.shape != self.color.shape:
            self.shape_error(color_frame, self.color)

        if depth_frame.shape != self.depth.shape:
            self.shape_error(depth_frame, self.depth)

        np.copyto(self.color, color_frame)
        np.copyto(self.depth, depth_frame)

    def get_images(self):
        return self.color, self.depth


class ProcessHandler:
    STOP = 42
    CONTINUE = 1
    RESET = 2

    def __init__(self, markers_sets, shared_frame: SharedFrames, crop_areas, method):
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

            # (i,
            #  kinematics_marker_set,
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
            print(f"[Process {self.queue_res.get()}: Returned]")

    def end_process(self):
        for queue in self.queue_arg_list:
            queue.put(ProcessHandler.STOP)

        ### Wait for the process to stop
        for _ in self.process_list:
            print(f"[Process {self.queue_end_proc.get()}: Stopped]")

        ### When all the process are stopped join them
        for process in self.process_list:
            process.join()

        print('All process stopped')


def process_function(index,
                     queue_arg,
                     marker_set_shared_memories,
                     crop_area,
                     arguments):
    print(f"[Process {index}: Started]")

    while True:
        arg = queue_arg.get()

        if arg == ProcessHandler.STOP:
            break

        elif arg == ProcessHandler.CONTINUE:
            crop = arguments['color_frame_shared'][crop_area[1]: crop_area[3], crop_area[0]:crop_area[2]]
            cv2.imshow(f'Crop {index}', crop)
            cv2.waitKey(1)

        elif arg == ProcessHandler.RESET:
            print(f"[Process {index}: Resetting]")

        else:
            print(f"[Process {index}: Order {arg} not implemented]")

        ### When executed its order send back to res queue its index
        arguments['queue_res'].put(index)

    arguments["queue_end"].put(index)

