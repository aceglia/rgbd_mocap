from datetime import datetime

import os
import cv2

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    pass
try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    pass
import numpy as np
import json
import datetime
import multiprocessing as mp
import time
from rgbd_mocap.enums import ColorResolution, DepthResolution


class RGBDRecorder:
    def __init__(
        self,
        camera_type="realsense",
        from_rgbd=True,
        fps=60,
        threads_number=3,
        depth=DepthResolution.R_848x480,
        color=ColorResolution.R_848x480,
        align_color=True,
    ):
        self.camera_type = camera_type
        self.dropped_images = 0
        self.zed_image_count = 0
        self.from_rgbd = from_rgbd
        self.pipeline = None
        self.participant = None
        self.align = None
        self.dic_config_cam = {}
        self.queue_trigger_rgbd = None
        self.file_name = "data"
        self.config_file_name = None
        now = datetime.datetime.now()
        self.date_time = now.strftime("%d-%m-%Y_%H_%M_%S")
        self.nb_save_process = threads_number
        self.fps = fps
        self.queue_trigger_rgbd = mp.Manager().Queue()
        self.queue_color = [mp.Manager().Queue()] * self.nb_save_process
        self.queue_depth = [mp.Manager().Queue()] * self.nb_save_process
        self.reccording_event = mp.Event()
        self.queue_frame = [mp.Manager().Queue()] * self.nb_save_process
        self.last_frame_queue = [mp.Manager().Queue()] * self.nb_save_process
        self.depth = depth
        self.align_color = align_color
        self.color = color

    def _inti_real_sense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        if self.depth is not None:
            config.enable_stream(rs.stream.depth, self.depth[0], self.depth[1], rs.format.z16, self.fps)
        if self.color is not None:
            config.enable_stream(rs.stream.color, self.color[0], self.color[1], rs.format.bgr8, self.fps)

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
        if self.align_color:
            align_to = rs.stream.color
            self.align = rs.align(align_to)

    def _init_zed(self, zed_params):
        self.zed = sl.Camera()
        # Set configuration parameters
        input_type = sl.InputType()
        init = sl.InitParameters(input_t=input_type)
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.camera_fps = self.fps

        self.pipeline = sl.RuntimeParameters()

        # Prepare new image size to retrieve half-resolution images
        image_size = self.zed.get_camera_information().camera_configuration.resolution
        image_size.width = image_size.width / 2
        image_size.height = image_size.height / 2
        self.image_size_zed = image_size
        self.image_size = [image_size.width, image_size.height]

        # Declare your sl.Mat matrices
        self.zed_image = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        self.zed_depth = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U16_C1)

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters

        self.dic_config_cam = {
            "camera_name": "Zed2i",
            "depth_scale": 1,
            "left_color_fx_fy": [calibration_params.left_cam.fx, calibration_params.left_cam.fy],
            "left_color_ppx_ppy": [calibration_params.left_cam.cx, calibration_params.left_cam.cy],
            "right_color_fx_fy": [calibration_params.right_cam.fx, calibration_params.right_cam.fy],
            "right_color_ppx_ppy": [calibration_params.right_cam.cx, calibration_params.right_cam.cy],
            "depth_to_color_trans": [calibration_params.stereo_transform.get_translation().get()[0], 0, 0],
            "depth_to_color_rot": np.eye(3).tolist(),
            "dist_coeffs_color": calibration_params.left_cam.disto,
            "size_color": self.image_size,
            "size_depth": self.image_size,
            "color_rate": self.zed.get_camera_information().camera_configuration.fps,
            "depth_rate": self.zed.get_camera_information().camera_configuration.fps,
        }

    def init_camera_pipeline(self, camera_type="realsense"):
        if camera_type == "realsense":
            self._inti_real_sense()
        elif camera_type == "zed":
            self._init_zed(self)
        else:
            raise RuntimeError("Camera type not recognize.")

    @staticmethod
    def _show_images(color_to_show, loop_time):
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(color_to_show, alpha=1), cv2.COLORMAP_JET)
        cv2.addWeighted(depth_colormap, 0.8, color_to_show, 0.8, 0, color_to_show)
        if len(loop_time) > 20:
            cv2.putText(
                color_to_show,
                f"FPS = {1 / np.mean(loop_time[-20:])}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        elif len(loop_time) > 0:
            cv2.putText(
                color_to_show,
                f"FPS = {1 / np.mean(loop_time)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)
        cv2.imshow("RealSense", color_to_show)

    def _get_frame_from_realsense(self):
        aligned_frames = self.pipeline.wait_for_frames()
        if self.align_color:
            aligned_frames = self.align.process(aligned_frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.uint16)
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image

    def _get_frame_from_zed(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT, sl.MEM.CPU, self.image_size_zed)
            self.zed.retrieve_measure(self.zed_depth, sl.MEASURE.DEPTH)
            image_ocv = self.zed_image.get_data()[..., :3]

            depth_image = self.zed_depth.get_data()
            depth_image[np.isnan(depth_image)] = 0
            depth_image[np.isinf(depth_image)] = 0
            return depth_image, image_ocv
        else:
            return None, None

    def get_frame(self):
        if self.camera_type == "realsense":
            return self._get_frame_from_realsense()
        elif self.camera_type == "zed":
            return self._get_frame_from_zed()
        else:
            raise RuntimeError("Camera type not recognize.")

    def get_frame_number(self, **kwargs):
        if self.camera_type == "realsense":
            return kwargs["color_frame"].frame_number
        elif self.camera_type == "zed":
            gap = self.zed.get_frame_dropped_count() - self.dropped_images
            self.dropped_images = self.zed.get_frame_dropped_count()
            self.zed_image_count += gap + 1
            return self.zed_image_count

    def get_rgbd(self, keep_image=False):
        self.init_camera_pipeline(self.camera_type)
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
            depth_image, color_image = self.get_frame()
            if depth_image is None or color_image is None:
                continue
            depth_image_to_save = depth_image.copy()
            color_image_to_save = color_image.copy()
            frame_number = self.get_frame_number
            if not save_data:
                if i == 0:
                    print(
                        "Ready to record. Please press 's' to start recording."
                        " If you want to keep alive the image turn keep_image to True."
                    )
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    save_data = True
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                self._show_images(color_image, loop_time_list)
                if save_data:
                    tic_init = time.time()
                    print("start recording...")
                    if not keep_image:
                        cv2.destroyAllWindows()
            if save_data:
                if all_frame_saved[0] % 500 == 0:
                    print("time: ", time.time() - tic_init)
                if keep_image:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    self._show_images(color_image, loop_time_list)
                self.queue_depth[nb_process].put_nowait(depth_image_to_save)
                self.queue_color[nb_process].put_nowait(color_image_to_save)
                self.queue_frame[nb_process].put_nowait(frame_number)
                all_frame_saved[nb_process] += 1
                if nb_process == self.nb_save_process - 1:
                    nb_process = 0
                else:
                    nb_process += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if i == 0:
                    with open(rf".\{self.config_file_name}", "w") as outfile:
                        json.dump(self.dic_config_cam, outfile, indent=4)
                i += 1
            loop_time_list.append(time.time() - tic)
        if keep_image:
            cv2.destroyAllWindows()
        print("stop recording..." "wait until all data are saved")
        for i in range(self.nb_save_process):
            print(all_frame_saved[i])
            self.last_frame_queue[i].put_nowait(all_frame_saved[i])
        self.pipeline.stop()

    def save_rgbd_from_buffer(self, nb_process, save_dir=".\images"):
        # while True:
        last_frame_number = 0
        final_frame = -1
        frame_number = 0
        tic_init = time.time()
        if not os.path.exists(rf"{save_dir}\{self.participant}\{self.file_name}_{self.date_time}"):
            os.makedirs(rf"{save_dir}\{self.participant}\{self.file_name}_{self.date_time}")

        nb_frame = 0
        init_count = 0
        while nb_frame != final_frame or init_count == 0:
            try:
                color_image_to_save = self.queue_color[nb_process].get()
                depth_image_to_save = self.queue_depth[nb_process].get()
                frame_number = self.queue_frame[nb_process].get()
                cv2.imwrite(
                    rf"{save_dir}\{self.participant}\{self.file_name}_{self.date_time}\depth_{frame_number}.png",
                    depth_image_to_save,
                )
                cv2.imwrite(
                    rf"{save_dir}\{self.participant}\{self.file_name}_{self.date_time}\{self.participant}\{self.file_name}_{self.date_time}\color_{frame_number}.png",
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
                        print(f"Remaining frame to save for process {nb_process}: {final_frame - nb_frame}")
            except:
                pass
        print(
            len(os.listdir(rf"{save_dir}\{self.participant}\{self.file_name}_{self.date_time}")), "images were saved."
        )
        print("saving time: {}".format(time.time() - tic_init))

    def start(self, keep_image=False):
        processes = []
        p = mp.Process(
            target=RGBDRecorder.get_rgbd,
            args=(
                self,
                keep_image,
            ),
        )
        processes.append(p)
        for i in range(self.nb_save_process):
            p = mp.Process(target=RGBDRecorder.save_rgbd_from_buffer, args=(self, i))
            processes.append(p)
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()


if __name__ == "__main__":
    rec = RGBDRecorder(from_rgbd=True, camera_type="zed")
    rec.fps = 30
    rec.file_name = "demo"
    rec.participant = "P0"
    rec.start(keep_image=False)
