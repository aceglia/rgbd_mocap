from datetime import datetime
import tkinter as tk

import os
import cv2
from matplotlib.pylab import False_, rayleigh
import pyrealsense2 as rs
import numpy as np
import json
import datetime
from biosiglive import ViconClient, DeviceType
import multiprocessing as mp
from multiprocessing import RawArray
import time
from rgbd_mocap.frames.shared_frames import SharedFrames


class Synchronizer:
    def __init__(self, use_trigger=True, fps=60, show_images=True, buffer_size=30, 
                 start_delay=0, stop_delay=200, from_bag_file=False, bag_path="", n_save_process=3):
        
        self.from_bag_file = from_bag_file
        self.bag_file_path = bag_path
        self.show_images = show_images
        self.buffer_size = buffer_size
        self.start_delay = start_delay
        self.stop_delay = stop_delay
        self.pipeline = None
        self.participant = None
        self.align = None
        self.dic_config_cam = {}
        self.interface = None
        self.use_trigger = use_trigger
        self.file_name = "data"
        self.config_file_name = None
        self.event_started = [mp.Event()] * (n_save_process + 1)

        now = datetime.datetime.now()
        self.date_time = now.strftime("%d-%m-%Y_%H_%M_%S")
        self.nb_save_process = n_save_process
        self.buffer_size = max(self.buffer_size, self.nb_save_process)
        self.fps = fps
        self.frame_queue = mp.Queue()


        self.trigger_start_event = mp.Event()
        self.trigger_stop_event = mp.Event()
        self.init_trigger()

        # self.size = (480, 848)
        self.size = (480, 640)

        self.color_shape = self.size + (3, self.buffer_size)
        self.depth_shape = self.size + (self.buffer_size,)


    def init_camera_pipeline(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, self.size[1], self.size[0], rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.size[1], self.size[0], rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        d_profile = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
        d_intr = d_profile.get_intrinsics()
        scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        c_profile = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
        c_intr = c_profile.get_intrinsics()
        deth_to_color = d_profile.get_extrinsics_to(c_profile)
        r = np.array(deth_to_color.rotation).reshape(3, 3)
        t = np.array(deth_to_color.translation)

        self.dic_config_cam = {
            "camera_name": device_product_line,
            "depth_scale": scale,
            "depth_fx_fy": [d_intr.fx, d_intr.fy],
            "depth_ppx_ppy": [d_intr.ppx, d_intr.ppy],
            "color_fx_fy": [c_intr.fx, c_intr.fy],
            "color_ppx_ppy": [c_intr.ppx, c_intr.ppy],
            "depth_to_color_trans": t.tolist(),
            "depth_to_color_rot": r.tolist(),
            "model_color": c_intr.model.name,
            "model_depth": d_intr.model.name,
            "dist_coeffs_color": c_intr.coeffs,
            "dist_coeffs_depth": d_intr.coeffs,
            "size_color": [c_intr.width, c_intr.height],
            "size_depth": [d_intr.width, d_intr.height],
            "color_rate": c_profile.fps(),
            "depth_rate": d_profile.fps(),
        }

        self.config_file_name = f"config_camera_files\config_camera_{self.date_time}.json"
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        with open(f"D:\Documents\Programmation\pose_estimation\{self.config_file_name}", "w") as outfile:
            json.dump(self.dic_config_cam, outfile, indent=4)


    def dialog_box(self):
        self.master = tk.Tk()
        self.button_stop = tk.Button(master=self.master, text="Stop\nrecording", command=self.stop_and_destroy)
        self.button_start = tk.Button(master=self.master, text="Start\nrecording", command=self.start)
        self.master.title("Annotation tool")
        self.button_start.pack(side=tk.TOP)
        self.button_stop.pack(side=tk.BOTTOM)

    def stop_and_destroy(self):
        self.master.destroy
        self.trigger_stop_event.set()
    
    def start_reccording(self):
        self.trigger_start_event.set()

    def init_trigger(self):
        if not self.use_trigger:
            return
        self.interface = ViconClient(ip="192.168.1.211", system_rate=120, init_now=False)

    def get_trigger(self):
        if self.use_trigger:
            self.interface.init_client()
            self.interface.get_frame()
            self.interface.add_device(
                nb_channels=1,
                device_type=DeviceType.Generic,
                name="trigger",
                rate=2000,
            )
        init_time = time.time()
        self.event_started[0].set()
        is_started=False
        while True:
            if self.use_trigger:
                trigger_data = self.interface.get_device_data(device_name="trigger")
                if trigger_data is None:
                    continue
                if len(np.where(np.array(trigger_data) > 0.1)[0]) > 0 and not self.trigger_start_event.is_set():
                    self.trigger_start_event.set()
                elif len(np.where(np.array(trigger_data) > 0.1)[0]) > 0 and not self.trigger_stop_event.is_set():
                    self.trigger_stop_event.set()
                    break
            else:
                time.sleep(0.005)
                delay = time.time() - init_time
                if not is_started and delay > self.start_delay:
                    self.trigger_start_event.set()
                    print("start recording...")
                    init_time = time.time()
                    is_started = True
                else:
                    if delay > self.stop_delay:
                        self.trigger_stop_event.set()
                        break

    def get_images(self):
        try:
            aligned_frames = self.pipeline.wait_for_frames()
        except:
            return None, None, None

        aligned_frames = self.align.process(aligned_frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            return None, None, None
        frame_number = color_frame.frame_number
        depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.uint16)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image, frame_number

    def set_shared_memory_images(self, shared_color, shared_depth, color_image, depth_image, idx):
        np.copyto(shared_color[..., idx], color_image)
        np.copyto(shared_depth[..., idx], depth_image)

    def show_cv2_images(self, color, depth, frame_number, fps):
        fps = 0 if not np.isfinite(fps) else fps
        color_image = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.addWeighted(depth_colormap, 0.8, color_image, 0.8, 0, color_image)
        cv2.putText(
            color_image,
            f"FPS: {int(fps)} | frame: {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.waitKey(1)
        cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)
        cv2.imshow("RealSense", color_image)


    def _init_bag_file(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.bag_file_path, repeat_playback=False)
        config.enable_all_streams()
        self.pipeline.start(config)
        device = self.pipeline.get_active_profile().get_device()
        playback = device.as_playback()
        playback.set_real_time(False)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def wait_all(self):
        for i in range(len(self.event_started)):
            self.event_started[i].wait()
        return

    def get_rgbd(self, shared_color, shared_depth):
        shared_color = np.frombuffer(shared_color, dtype=np.uint8).reshape((self.color_shape))
        shared_depth = np.frombuffer(shared_depth, dtype=np.uint16).reshape((self.depth_shape))
        if not self.from_bag_file:
            self.init_camera_pipeline()
        else:
            self._init_bag_file()
        loop_time_list = []
        buffer_idx = 0
        count = 0
        self.wait_all()
        while True:
            if self.trigger_stop_event.is_set():
                break
            tic = time.time()
            color_image, depth_image, frame_number = self.get_images()
            if color_image is None:
                continue
            if not self.trigger_start_event.is_set() and self.show_images:
                fps = 1 / np.mean(loop_time_list[-20:])
                self.show_cv2_images(color_image, depth_image, frame_number, fps)
            elif self.trigger_start_event.is_set():
                if count == 0:
                    cv2.destroyAllWindows()
                buffer_idx = frame_number % self.buffer_size
                self.set_shared_memory_images(shared_color, shared_depth, color_image, depth_image, buffer_idx)
                self.frame_queue.put_nowait((frame_number, buffer_idx))
                count += 1
            loop_time_list.append(time.time() - tic)
        print(f"stop recording...nb frame: {len(loop_time_list)}, in {np.array(loop_time_list).sum()}" 
              "wait until all data are saved")
        self.pipeline.stop()

    def save_rgbd_from_buffer(self, shared_color, shared_depth, i):
        shared_color = np.frombuffer(shared_color, dtype=np.uint8).reshape((self.color_shape))
        shared_depth = np.frombuffer(shared_depth, dtype=np.uint16).reshape((self.depth_shape))
        path = f"D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}"
        if not os.path.exists(path):
            os.makedirs(path)
        self.event_started[i].set()
        while True:
            try:
                queue = self.frame_queue.get(0.01)
            except:
                if self.trigger_stop_event.is_set():
                    break
                continue
            shared_idx = queue[1]
            frame_number = queue[0]
            depth_image = shared_depth[..., shared_idx]
            color_image = shared_color[..., shared_idx]
            cv2.imwrite(
                    f"D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}\depth_{frame_number}.png",
                    depth_image,
                )
            cv2.imwrite(
                    f"D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}\color_{frame_number}.png",
                    cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB),
                )

    def start(self):
        color_array = RawArray("c", int(np.prod(self.color_shape)))  # 'c' -> value between 0-255
        depth_array = RawArray("H", int(np.prod(self.depth_shape)))  # 'H' -> uint16
        processes = []
        p = mp.Process(target=Synchronizer.get_rgbd, args=(self, color_array, depth_array,), daemon=True)
        processes.append(p)
        for i in range(self.nb_save_process):
            p = mp.Process(target=Synchronizer.save_rgbd_from_buffer, args=(self, color_array, depth_array, i,), daemon=True)
            processes.append(p)
        p = mp.Process(target=Synchronizer.get_trigger, args=(self,), daemon=True)
        processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    sync = Synchronizer(use_trigger=False, start_delay=1, stop_delay=30, from_bag_file=True, 
                        bag_path=r"test.bag", n_save_process=2, show_images=True)
    sync.fps = 60
    sync.file_name = "tets"
    sync.participant = "P00"
    sync.start()
