from multiprocessing import Queue, Process
from rgbd_mocap.frames.shared_frames import SharedFrames
from rgbd_mocap.markers.marker_set import MarkerSet
from rgbd_mocap.crop.crop import Crop
from rgbd_mocap.tracking.test_tracking import set_marker_pos
from rgbd_mocap.processing.handler import Handler


class MultiProcessHandler(Handler):
    def __init__(self, markers_sets: list[MarkerSet], shared_frame: SharedFrames, options, tracking_option):
        super().__init__()
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

            process = Process(target=MultiProcessHandler._process_function,
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

    def receive_process(self):
        for _ in self.queue_arg_list:
            res = self.queue_res.get()
            # print(f"[Process {res}: Returned]")

    def send_and_receive_process(self, order=1):
        self.send_process(order)
        self.receive_process()

    def end_process(self):
        for queue in self.queue_arg_list:
            queue.put(MultiProcessHandler.STOP)

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

            if arg == Handler.STOP:
                break

            elif arg == Handler.CONTINUE:
                blobs, positions, estimate_positions = crop.track_markers()
                set_marker_pos(marker_set, positions)

                Handler.show_image(f"{crop_option['name']} {index}",
                                   crop.filter.filtered_frame,
                                   blobs=blobs,
                                   markers=crop.marker_set,
                                   estimated_positions=estimate_positions)

            elif arg == Handler.RESET:
                print(f"[Process {index}: Resetting]")
                crop.re_init(marker_set, tracking_option)

            else:
                print(f"[Process {index}: Order {arg} not implemented]")

            ### When executed its order send back to res queue its index
            arguments['queue_res'].put(index)

        arguments["queue_end"].put(index)