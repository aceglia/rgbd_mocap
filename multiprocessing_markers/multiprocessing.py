import rgbd_mocap.marker_class
from multiprocessing import Queue, Process, Lock


class ProcHandler:
    def __init__(self, markers_sets, shared_frame, crop_areas):
        self.queue_arg_list = []
        self.queue_res = Queue()
        self.queue_end_proc = Queue()
        self.process_list = []

        for i in range(len(markers_sets)):
            arguments = {
                'marker_set_shared_memories': markers_sets[i].get_shared_memories(),  # TODO
                'color_frame_shared': None,  # TODO
                'depth_frame_shared': None,  # TODO
                'crop_areas': crop_areas[i],  # TODO
            }

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

            queue_arg = Queue()
            process = Process(target=proc_function, args=arguments, daemon=True)

            self.queue_arg_list.append(queue_arg)
            self.process_list.append(process)
            # process.start()

    def start_process(self):
        for process in self.process_list:
            process.start()

    def end_process(self):
        ### Tell to the proc to stop
        for _ in self.process_list:
            self.queue_end_proc.get()

        ### When all the process are stopped join them
        for process in self.process_list:
            process.join()


def init_multiprocessing(markers):
    pass

    queue_arg_list = []
    queue_res = Queue()
    queue_end_proc = Queue()
    proc_list = []
    # lock = Lock()

    for i, marker_set in enumerate(markers):
        marker_set_shared_memories = marker_set.get_shared_memories()  # TODO
        color_frame_shared = None  # TODO
        depth_frame_shared = None  # TODO
        crop_area = None  # TODO

        (i,
         kinematics_marker_set,
         X_np_color,
         X_np_depth,
         crop_area,
         mask_param,
         flags,
         depth_scale,
         clipping_color,
         optical_flow_params,) = arguments[i]

        queue_arg = Queue()
        process = Process(target=partial_get_frame_process, args=(i,
                                                                  kinematics_marker_set,
                                                                  X_np_color,
                                                                  X_np_depth,
                                                                  crop_area,
                                                                  mask_param,
                                                                  flags,
                                                                  depth_scale,
                                                                  clipping_color,
                                                                  optical_flow_params,
                                                                  queue_arg,
                                                                  queue_res,
                                                                  queue_end_proc,
                                                                  lock), daemon=True)

        queue_arg_list.append(queue_arg)
        proc_list.append(process)
        process.start()
