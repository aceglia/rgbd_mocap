import asyncio
from datetime import datetime
# import qtm_rt
import os
import cv2
import pyrealsense2 as rs
import numpy as np
import json
import datetime
from biosiglive import ViconClient, DeviceType
import multiprocessing as mp
import time
# from pyScienceMode2.rehastim_interface import Stimulator


class Synchronizer:
    def __init__(self, from_rgbd = True, from_qualysis = True, use_trigger = True, with_motomed = True):
        self.from_rgbd = from_rgbd
        self.from_qualysis = from_qualysis
        self.pipeline = None
        self.participant = None
        self.align = None
        self.dic_config_cam = {}
        self.queue_trigger_rgbd = None
        self.queue_trigger_qualysis = None
        self.trigger_event = None
        self.queue_motomed = None
        self.interface = None
        self.use_trigger = use_trigger
        self.file_name = "data"
        self.config_file_name = None
        self.with_motomed = with_motomed
        now = datetime.datetime.now()
        self.date_time = now.strftime("%d-%m-%Y_%H_%M_%S")

        self.motomed_gear = 5
        if from_rgbd:
            self.queue_trigger_rgbd = mp.Queue()
            self.trigger_event = mp.Event()
            self.queue_buffer_idx = mp.Queue()
            self.queue_color = mp.Manager().Queue()
            self.queue_depth = mp.Manager().Queue()
            self.reccording_event = mp.Event()
            self.queue_frame = mp.Manager().Queue()
            self.last_frame_queue = mp.Manager().Queue()
            self.saving_event = mp.Event()
            self.saving_finished_event = mp.Event()

        if from_qualysis:
            self.queue_trigger_qualysis = mp.Manager().Queue()
        if with_motomed :
            self.queue_motomed = mp.Manager().Queue()
        if use_trigger:
            self.init_trigger()

    async def qualysis_recording(self):
        connection = await qtm_rt.connect("192.168.1.212", version="1.18")
        async with qtm_rt.TakeControl(connection, "s2mlab"):
            await connection.new()
            while True:
                data_triger = 0
                try:
                    data_triger = self.queue_trigger_qualysis.get_nowait()
                except:
                    pass
                if data_triger > 1.5:
                    break
            curr_time = datetime.datetime.now()
            await connection.start()
            await asyncio.sleep(2)
            print(curr_time.strftime('%H:%M:%S.%f'))

    def init_camera_pipeline(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

        self.pipeline.start(config)
        d_profile = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
        d_intr = d_profile.get_intrinsics()
        scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        c_profile = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
        c_intr = c_profile.get_intrinsics()
        deth_to_color = d_profile.get_extrinsics_to(c_profile)
        r = np.array(deth_to_color.rotation).reshape(3, 3)
        t = np.array(deth_to_color.translation)
        self.dic_config_cam = {"camera_name": device_product_line,
               'depth_scale': scale,
               'depth_fx_fy': [d_intr.fx, d_intr.fy],
               'depth_ppx_ppy': [d_intr.ppx, d_intr.ppy],
               'color_fx_fy': [c_intr.fx, c_intr.fy],
               'color_ppx_ppy': [c_intr.ppx, c_intr.ppy],
               'depth_to_color_trans': t.tolist(),
               'depth_to_color_rot': r.tolist(),
               "model_color": c_intr.model.name,
               "model_depth": d_intr.model.name,
               "dist_coeffs_color": c_intr.coeffs,
               "dist_coeffs_depth": d_intr.coeffs,
               "size_color": [c_intr.width, c_intr.height],
               "size_depth": [d_intr.width, d_intr.height],
               "color_rate": c_profile.fps(),
               "depth_rate": d_profile.fps()
               }
        # self.config_file_name = f"config_camera_files\config_camera_{self.date_time}.json"
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def init_trigger(self):
        self.interface = ViconClient(ip="192.168.1.211", system_rate=120, init_now=False)

    def start_motomed(self):
        port = "COM6"
        motomed = Stimulator(port, show_log=False, with_motomed=True).motomed
        motomed.init_phase_training(arm_training=True)
        while True:
            try:
                trigger = self.queue_motomed.get_nowait()
            except:
                trigger = 0
                time.sleep(0.001)
            if trigger > 1.5:
                motomed.start_phase(speed=60, gear=self.motomed_gear, active=True, go_forward=True, spasm_detection=False)
                break

    def get_trigger(self):
        self.interface.init_client()
        self.interface.get_frame()
        self.interface.add_device(
            nb_channels=1,
            device_type=DeviceType.Generic,
            name="trigger",
            rate=2160,
        )
        # break when everything is started
        while True:
            trigger_data = self.interface.get_device_data(device_name="trigger")
            if trigger_data is not None:
                if len(trigger_data) > 0:
                    trigger_data = np.mean(trigger_data)
                else:
                    trigger_data = 0
                if trigger_data > 1.5:
                    self.trigger_event.set()
                # if self.from_rgbd:
                #     try:
                #         self.queue_trigger_rgbd.get_nowait()
                #     except:
                #         pass
                #     self.queue_trigger_rgbd.put_nowait(trigger_data)
                #
                # if self.with_motomed:
                #     try:
                #         self.queue_motomed.get_nowait()
                #     except:
                #         pass
                #     self.queue_motomed.put_nowait(trigger_data)

    def start_qualysis(self):
        asyncio.get_event_loop().run_until_complete(self.qualysis_recording())

    def get_rgbd(self, shared_image_to_save, shared_depth_to_save, shared_frame_number, shared_buffer_idx):
        if self.from_rgbd:
            self.init_camera_pipeline()
            with open(f"D:\Documents\Programmation\pose_estimation\{self.config_file_name}", 'w') as outfile:
                json.dump(self.dic_config_cam, outfile, indent=4)
        loop_time_list = []
        save_data = False
        self.config_file_name = f"config_camera_files\config_camera_{self.participant}.json"
        i = 0
        buffer_idx = 0
        count = 0
        total_buffer_idx = 0
        shared_image_to_save = np.frombuffer(shared_image_to_save, dtype=np.uint8).reshape(
            (480, 848, 3, self.buffer_size))
        shared_depth_to_save = np.frombuffer(shared_depth_to_save, dtype=np.uint16).reshape(
            (480, 848, self.buffer_size))
        shared_frame_number = np.frombuffer(shared_frame_number).reshape((1, self.buffer_size))
        shared_buffer_idx = np.frombuffer(shared_buffer_idx).reshape((1))

        while True:
            aligned_frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(aligned_frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.uint16)
            color_image = np.asanyarray(color_frame.get_data())
            if save_data:
                # put into shared memory:
                tic = time.time()
                # np.copyto(shared_image_to_save[:, :, :, buffer_idx], color_image)
                # np.copyto(shared_depth_to_save[:, :, buffer_idx], depth_image)
                shared_image_to_save[:, :, :, buffer_idx] = color_image
                shared_depth_to_save[:, :, buffer_idx] = depth_image
                shared_frame_number[:, buffer_idx] = color_frame.frame_number
                # np.copyto(shared_frame_number[:, buffer_idx], np.array(frame_number, dtype=int))
                if not self.saving_event.is_set():
                    self.saving_event.set()
                # try:
                #     self.queue_buffer_idx.get_nowait()
                # except:
                #     pass
                # self.queue_buffer_idx.put_nowait(total_buffer_idx)
                shared_buffer_idx[0] = total_buffer_idx
                buffer_idx += 1
                total_buffer_idx += 1
                if buffer_idx == self.buffer_size:
                    buffer_idx = 0
                    count += 1
                if self.use_trigger and self.trigger_event.is_set():
                    if i >= 80:
                        break
                    else:
                        self.trigger_event.clear()
                else:
                    if total_buffer_idx >= 1500:
                        self.saving_event.clear()
                        # wait until process finished
                        print("stop recording..."
                              "wait until all data are saved")
                        self.saving_finished_event.wait()
                        break

            if not save_data:
                if self.use_trigger and self.trigger_event.is_set():
                    save_data = True
                elif cv2.waitKey(1) & 0xFF == ord('q'):
                    save_data = True
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.addWeighted(depth_colormap, 0.8, color_image, 0.8, 0, color_image)
                if len(loop_time_list) > 20:
                    cv2.putText(color_image,
                                f"FPS = {1 / np.mean(loop_time_list[-20:])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 2,
                                cv2.LINE_AA)
                elif len(loop_time_list) > 0:
                    cv2.putText(color_image,
                                f"FPS = {1 / np.mean(loop_time_list)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 2,
                                cv2.LINE_AA)
                cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
                cv2.imshow('RealSense', color_image)
                if save_data:
                    cv2.destroyAllWindows()

        self.pipeline.stop()

    def save_rgbd_from_buffer(self, shared_image_to_save, shared_depth_to_save, shared_frame_number, shared_buffer_idx):
        # while True:
        buffer_local_idx = 0
        idx_buffer = -1
        buffer_idx = 0
        count = 0
        shared_image_to_save = np.frombuffer(shared_image_to_save, dtype=np.uint8).reshape(
            (480, 848, 3, self.buffer_size))
        shared_depth_to_save = np.frombuffer(shared_depth_to_save, dtype=np.uint16).reshape(
            (480, 848, self.buffer_size))
        shared_frame_number = np.frombuffer(shared_frame_number).reshape((1, self.buffer_size))
        shared_buffer_idx = np.frombuffer(shared_buffer_idx).reshape((1))
        nb_file_saved = 0
        if not os.path.exists(
                f'D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}'):
            os.makedirs(
                f'D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}')
        # self.saving_event.wait()
        tic_init = time.time()
        init_count = 0
        while buffer_local_idx != idx_buffer or init_count == 0:
            self.saving_event.wait()
            tic = time.time()
            if idx_buffer - buffer_local_idx >= self.buffer_size:
                print(f"Increase the buffer size ..., you have {idx_buffer - buffer_local_idx} difference")
            if buffer_local_idx == 0:
                print("saving ....")
            idx_buffer = shared_buffer_idx[0]
            if idx_buffer == -1:
                continue
            color_image = shared_image_to_save[:, :, :, buffer_idx]
            depth_image = shared_depth_to_save[:, :, buffer_idx]
            frame_number = int(shared_frame_number[0, buffer_idx])
            # print("time to get data from buffer :", time.time() - tic)
            # print("local_buffer_idx :", buffer_local_idx, "buffer_idx :", idx_buffer, "frame_number :", frame_number,)
            buffer_local_idx += 1
            buffer_idx += 1
            if buffer_idx == self.buffer_size:
                buffer_idx = 0
                count += 1
            tic = time.time()
            cv2.imwrite(
                f'D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}\depth_{frame_number}.png',
                depth_image)
            cv2.imwrite(
                f'D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}\color_{frame_number}.png',
                color_image)
            print("time to save :", time.time() - tic)
            nb_file_saved += 1
            print("remaining number of frame to save :", idx_buffer - buffer_local_idx)
            if init_count == 0:
                init_count += 1
            # print("nb_file_saved: ", nb_file_saved, "frame_number :", frame_number)

        # number of file in the dir
        print("total time to save :", time.time() - tic_init)
        os.remove(f'D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}\color_{-1}.png')
        os.remove(f'D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}\depth_{-1}.png')
        nb_file = len(os.listdir(
            f'D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}'))

        print("all data are saved"
              f"nb_file : {nb_file/2}"
              "DO NOT FORGET TO SAVE DELAY")
        self.saving_finished_event.set()

    def start(self):
        width, height = 480, 848
        # raw_array = mp.RawArray('c', width * height * 3)
        # raw_array_depth = mp.RawArray('c', width * height * 2)
        self.buffer_size = 300
        raw_array_buffer = mp.RawArray('c', width * height * 3 * self.buffer_size)
        raw_array_frame = mp.RawArray('i', 2 * self.buffer_size)
        raw_array_depth_buffer = mp.RawArray('c', width * height * 2 * self.buffer_size)
        raw_buffer_idx = mp.RawArray('i', 2)
        processes = []
        print("initializing_process...")
        if self.from_rgbd:
            p = mp.Process(target=Synchronizer.get_rgbd, args=(
                self,raw_array_buffer, raw_array_depth_buffer, raw_array_frame, raw_buffer_idx))
            processes.append(p)
            # p = mp.Process(target=Synchronizer.bufferize, args=(self,))
            # processes.append(p)
            p = mp.Process(target=Synchronizer.save_rgbd_from_buffer,
                           args=(self, raw_array_buffer, raw_array_depth_buffer, raw_array_frame, raw_buffer_idx),
                           )
            processes.append(p)
        if self.use_trigger:
            p = mp.Process(target=Synchronizer.get_trigger, args=(self,), daemon=True)
            processes.append(p)
        if self.with_motomed:
            p = mp.Process(target=Synchronizer.start_motomed, args=(self,))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    sync = Synchronizer(from_rgbd=True, from_qualysis=False, use_trigger=False, with_motomed=False)
    sync.file_name = "test"
    sync.participant = "test"
    sync.start()
    #
    # import time
    # import numpy as np
    # from multiprocessing import Process, Queue, RawArray
    #
    #
    # class ProcessFunc:  # Enum
    #     STOP = 0
    #     SAVE = 1
    #
    #
    # def process_function(process_index, array, arg_queue, finished_queue, stop_queue):
    #     while True:
    #         arg = arg_queue.get()
    #
    #         if arg == ProcessFunc.STOP:
    #             print(f"Stopping process number {process_index}")
    #             break
    #
    #         # Check if the array has been updated with the right number
    #         arg, number = arg
    #         # print(np.all(array == number))
    #
    #         # Here you can call the saving function
    #
    #         # Send the result of the verification
    #         finished_queue.put(np.all(array == number))
    #
    #     stop_queue.put(process_index)
    #
    #
    # def main(process_number, number_of_operations):
    #     tik = time.time()
    #
    #     arg_queue_list = []  # List of the queues sending the arguments for the process
    #     finished_queue = Queue()  # Queue receiving from the process when its job is finished
    #     stop_queue = Queue()  # Queue receiving from the process when it stopped
    #     proc_list = []  # List of the different process
    #
    #     ### Init shared memory array
    #     width = 300
    #     height = 300
    #     raw_array = RawArray('c', width * height * 3)
    #     shared_image = np.frombuffer(raw_array, dtype=np.uint8).reshape((width, height, 3))
    #
    #     ### Init the process
    #     for process_index in range(process_number):
    #         arg_queue = Queue()
    #         p = Process(target=process_function,
    #                     args=(process_index,
    #                           shared_image,
    #                           arg_queue,
    #                           finished_queue,
    #                           stop_queue),
    #                     daemon=True)
    #
    #         arg_queue_list.append(arg_queue)
    #         proc_list.append(p)
    #         p.start()
    #
    #     tok = time.time()
    #     print('Process initialisation finished: ' + str(tok - tik))
    #     tik = tok
    #
    #     ### Start the core program
    #     for i in range(number_of_operations):
    #         current_image = np.ones_like(shared_image, dtype=np.uint8) * i  # image to save
    #         np.copyto(shared_image, current_image)  # copy the image to the shared memory array
    #
    #         # For all process wait until they all finished saving
    #         for _ in arg_queue_list:
    #             result = finished_queue.get()
    #             if not result:
    #                 print('The array has not been updated')
    #
    #     tok = time.time()
    #     print('Time to run:', str(tok - tik))
    #     tik = tok
    #
    #     # Stop the process
    #     for q in arg_queue_list:
    #         q.put(ProcessFunc.STOP)
    #
    #     # Wait for the process to finish then kill them
    #     for _ in arg_queue_list:
    #         stop_queue.get()
    #
    #     for proc in proc_list:
    #         proc.kill()
    #
    #     return time.time() - tik
    #
    #
    # process_number = 3
    # number_of_operations = 100
    # print('Time to stop:', main(process_number, number_of_operations))