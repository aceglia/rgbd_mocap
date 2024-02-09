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
    def __init__(self, from_rgbd=True, from_qualysis=True, use_trigger=True, with_motomed=True, fps=60):
        self.from_rgbd = from_rgbd
        self.from_qualysis = from_qualysis
        self.pipeline = None
        self.participant = None
        self.align = None
        self.dic_config_cam = {}
        self.queue_trigger_rgbd = None
        self.queue_trigger_qualysis = None
        self.queue_motomed = None
        self.interface = None
        self.use_trigger = use_trigger
        self.file_name = "data"
        self.config_file_name = None
        self.with_motomed = with_motomed
        now = datetime.datetime.now()
        self.date_time = now.strftime("%d-%m-%Y_%H_%M_%S")
        self.nb_save_process = 3
        self.fps = fps
        self.motomed_gear = 5
        if from_rgbd:
            self.queue_trigger_rgbd = mp.Manager().Queue()
            self.queue_color = [mp.Manager().Queue()] * self.nb_save_process
            self.queue_depth = [mp.Manager().Queue()] * self.nb_save_process
            self.reccording_event = mp.Event()
            self.queue_frame = [mp.Manager().Queue()] * self.nb_save_process
            self.last_frame_queue = [mp.Manager().Queue()] * self.nb_save_process

        if from_qualysis:
            self.queue_trigger_qualysis = mp.Manager().Queue()
        if with_motomed:
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
            print(curr_time.strftime("%H:%M:%S.%f"))

    def init_camera_pipeline(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, self.fps)

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
                motomed.start_phase(
                    speed=60, gear=self.motomed_gear, active=True, go_forward=True, spasm_detection=False
                )
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
            # print(trigger_data)
            if trigger_data is not None:
                if len(trigger_data) > 0:
                    # trigger_data = np.mean(trigger_data)
                    trigger_data = 5 if len(np.where(trigger_data > 0.1)[0]) > 0 else 0
                else:
                    trigger_data = 0
                if self.from_qualysis:
                    try:
                        self.queue_trigger_qualysis.get_nowait()
                    except:
                        pass

                    self.queue_trigger_qualysis.put_nowait(trigger_data)
                if self.from_rgbd:
                    try:
                        self.queue_trigger_rgbd.get_nowait()
                    except:
                        pass
                    self.queue_trigger_rgbd.put_nowait(trigger_data)
                if self.with_motomed:
                    try:
                        self.queue_motomed.get_nowait()
                    except:
                        pass
                    self.queue_motomed.put_nowait(trigger_data)

    def start_qualysis(self):
        asyncio.get_event_loop().run_until_complete(self.qualysis_recording())

    def get_rgbd(self):
        self.init_camera_pipeline()
        loop_time_list = []
        save_data = False
        self.config_file_name = f"config_camera_files\config_camera_{self.participant}.json"
        i = 0
        last_frame_number = 0
        nb_process = 0
        all_frame_saved = [0] * self.nb_save_process
        tic_init = -1
        while True:
            tic = time.time()
            aligned_frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(aligned_frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.uint16)
            color_image = np.asanyarray(color_frame.get_data())

            depth_image_to_save = depth_image.copy()
            color_image_to_save = color_image.copy()
            frame_number = color_frame.frame_number
            if not save_data:
                if cv2.waitKey(1) & self.use_trigger:
                    try:
                        data_trigger = self.queue_trigger_rgbd.get_nowait()
                    except:
                        data_trigger = 0
                    if data_trigger > 1.5:
                        save_data = True
                elif cv2.waitKey(1) & 0xFF == ord("q"):
                    save_data = True
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.addWeighted(depth_colormap, 0.8, color_image, 0.8, 0, color_image)
                if len(loop_time_list) > 20:
                    cv2.putText(
                        color_image,
                        f"FPS = {1 / np.mean(loop_time_list[-20:])}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                elif len(loop_time_list) > 0:
                    cv2.putText(
                        color_image,
                        f"FPS = {1 / np.mean(loop_time_list)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("RealSense", color_image)
                if save_data:
                    tic_init = time.time()
                    print("start recording...")
                    cv2.destroyAllWindows()
            if save_data:
                if all_frame_saved[0] % 500 == 0:
                    print("time: ", time.time() - tic_init)
                self.queue_depth[nb_process].put_nowait(depth_image_to_save)
                self.queue_color[nb_process].put_nowait(color_image_to_save)
                self.queue_frame[nb_process].put_nowait(frame_number)
                all_frame_saved[nb_process] += 1
                if nb_process == self.nb_save_process - 1:
                    nb_process = 0
                else:
                    nb_process += 1

                if self.use_trigger:
                    if i >= 60:
                        try:
                            data_trigger = self.queue_trigger_rgbd.get_nowait()
                        except:
                            data_trigger = 0
                        if data_trigger > 1.5 or i >= 7200:
                            break
                else:
                    if i >= 7200:
                        break

                if i == 0:
                    with open(f"D:\Documents\Programmation\pose_estimation\{self.config_file_name}", "w") as outfile:
                        json.dump(self.dic_config_cam, outfile, indent=4)
                i += 1
            loop_time_list.append(time.time() - tic)
        print("stop recording..." "wait until all data are saved")
        for i in range(self.nb_save_process):
            print(all_frame_saved[i])
            self.last_frame_queue[i].put_nowait(all_frame_saved[i])
        self.pipeline.stop()

    def save_rgbd_from_buffer(self, nb_process):
        # while True:
        last_frame_number = 0
        final_frame = -1
        frame_number = 0
        tic_init = time.time()
        if not os.path.exists(
            f"D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}"
        ):
            os.makedirs(
                f"D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}"
            )

        nb_frame = 0
        init_count = 0
        while nb_frame != final_frame or init_count == 0:
            try:
                color_image_to_save = self.queue_color[nb_process].get()
                depth_image_to_save = self.queue_depth[nb_process].get()
                frame_number = self.queue_frame[nb_process].get()
                cv2.imwrite(
                    f"D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}\depth_{frame_number}.png",
                    depth_image_to_save,
                )
                cv2.imwrite(
                    f"D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}\color_{frame_number}.png",
                    color_image_to_save,
                )
                try:
                    final_frame = self.last_frame_queue[nb_process].get_nowait()
                except:
                    pass
                nb_frame += 1
                if init_count == 0:
                    init_count += 1
                if final_frame != -1:
                    if nb_frame % 60 == 0:
                        print(f"remaining frame to save for process {nb_process}: {final_frame - nb_frame}")
            except:
                pass
        print(
            len(
                os.listdir(
                    f"D:\Documents\Programmation\pose_estimation\data_files\{self.participant}\{self.file_name}_{self.date_time}"
                )
            )
        )
        print("saving time: {}".format(time.time() - tic_init))
        print("all data are saved" "DO NOT FORGET TO SAVE DELAY")

    def start(self):
        processes = []
        if self.from_qualysis:
            p = mp.Process(target=Synchronizer.start_qualysis, args=(self,))
            processes.append(p)
        if self.from_rgbd:
            p = mp.Process(target=Synchronizer.get_rgbd, args=(self,))
            processes.append(p)
            for i in range(self.nb_save_process):
                p = mp.Process(target=Synchronizer.save_rgbd_from_buffer, args=(self, i))
                processes.append(p)
        if self.use_trigger:
            p = mp.Process(target=Synchronizer.get_trigger, args=(self,))
            processes.append(p)
        if self.with_motomed:
            p = mp.Process(target=Synchronizer.start_motomed, args=(self,))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    sync = Synchronizer(from_rgbd=True, from_qualysis=False, use_trigger=True, with_motomed=False)
    sync.fps = 60
    sync.file_name = ("wheelchair")
    sync.participant = ("P16")
    sync.start()
