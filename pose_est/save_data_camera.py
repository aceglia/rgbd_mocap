import os
import cv2
import pyrealsense2 as rs
import numpy as np
import json
import datetime
from biosiglive import ViconClient, DeviceType

pipeline = rs.pipeline()
config = rs.config()
# bag_file_path = r"D:\Documents\20230421_150805.bag"
# config.enable_device_from_file(bag_file_path)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

pipeline.start(config)
d_profile = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
d_intr = d_profile.get_intrinsics()
scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
c_profile = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
c_intr = c_profile.get_intrinsics()
deth_to_color = d_profile.get_extrinsics_to(c_profile)
r = np.array(deth_to_color.rotation).reshape(3, 3)
t = np.array(deth_to_color.translation)
dic = {"camera_name": device_product_line,
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
#
sensor_dep = pipeline.get_active_profile().get_device().first_depth_sensor()
sensor_dep.set_option(rs.option.enable_auto_exposure, True)
save_data = False

file_name = "data"
now = datetime.datetime.now()
date_time = now.strftime("%d-%m-%Y_%H_%M_%S")
file_name_config = f"D:\Documents\Programmation\pose_estimation\config_camera_files\config_camera_{date_time}.json"
# if os.path.exists(file_name):
#     os.remove(file_name)
i = 0
align_to = rs.stream.color
align = rs.align(align_to)
import time
loop_time_list = []
use_trigger = False
plot_trigger = False

if use_trigger:
    interface = ViconClient(ip="127.0.0.1", system_rate=120)
    interface.add_device(
        nb_channels=1,
        device_type=DeviceType.Generic,
        name="Trigger",
        rate=2160,
    )
last_frame_number = 0
last_saving_time = 0
from biosiglive import save
b_size = 60
buffer = np.ndarray((b_size, 480, 848, 3))
buffer_d = np.ndarray((b_size, 480, 848))

buff_count = 0
while True:
    tic = time.time()
    aligned_frames = pipeline.wait_for_frames()
    aligned_frames = align.process(aligned_frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # aligned_depth_frame = rs.decimation_filter(2).process(aligned_depth_frame)
    color_frame = aligned_frames.get_color_frame()
    frame_number = color_frame.frame_number
    if frame_number - last_frame_number > 1:
        print("frame lost jump {} frame, last_saving_time: {}".format((frame_number - last_frame_number), last_saving_time))
    last_frame_number = frame_number
    if not aligned_depth_frame or not color_frame:
        continue
    depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.uint16)
    color_image = np.asanyarray(color_frame.get_data())
    depth_image_to_save = depth_image.copy()
    color_image_to_save = color_image.copy()
    tic = time.time()
    buffer[buff_count, :, :, :] = color_image_to_save
    buffer_d[buff_count, :, :] = depth_image_to_save
    buff_count += 1
    if buff_count == 2:
        buff_count = 0
    print("buffer saving time: {}".format(time.time() - tic))

    if not save_data:
        if use_trigger:
            trigger_data = interface.get_device_data(device_name="Trigger")
            if np.mean(trigger_data) > 0.5:
                save_data = True
                with open(r"config_camera_files\c" + f"onfig_camera_{date_time}.json", 'w') as outfile:
                    json.dump(dic, outfile, indent=4)
                print("saving ...")
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            if 1 / np.mean(loop_time_list) < 52:
                print(f"FPS = {1 / np.mean(loop_time_list)} not enought FPS pleas wait ...")
            else:
                save_data = True
                with open(file_name_config, 'w') as outfile:
                    json.dump(dic, outfile, indent=4)
                print("saving ...")

        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # color_image = np.where(
        #     (depth_image_3d > 11 / scale) | (
        #             depth_image_3d <= 0),
        #     20,
        #     color_image,
        # )
        cv2.addWeighted(depth_colormap, 0.8, color_image, 0.8, 0, color_image)
        cv2.putText(color_image,
                    f"FPS = {1 / np.mean(loop_time_list[-20:])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        if save_data:
            cv2.destroyAllWindows()
    else:
        tic = time.time()
        if not os.path.exists(f'D:\Documents\Programmation\pose_estimation\data_files\{file_name}_{date_time}'):
            os.makedirs(f'D:\Documents\Programmation\pose_estimation\data_files\{file_name}_{date_time}')
        cv2.imwrite(f'D:\Documents\Programmation\pose_estimation\data_files\{file_name}_{date_time}\depth_{i}.jpeg', buffer_d[i, :, :])
        cv2.imwrite(f'D:\Documents\Programmation\pose_estimation\data_files\{file_name}_{date_time}\color_{i}.jpeg', buffer[i, :, :, :])
        i += 1
        last_saving_time = time.time() - tic
        # cv2.waitKey(1)
    loop_time_list.append(time.time() - tic)
    # print(f"loop time: {loop_time_list[-1]}")
    # print(f"average loop time: {np.mean(loop_time_list)}")
    # print(f"average fps: {1 / np.mean(loop_time_list)}")
