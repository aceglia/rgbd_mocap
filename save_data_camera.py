import os
import cv2
import pyrealsense2 as rs
import numpy as np
import json
import datetime

pipeline = rs.pipeline()
config = rs.config()
# bag_file_path = r"D:\Documents\20230421_150805.bag"
# config.enable_device_from_file(bag_file_path)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
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
now = datetime.datetime.now()
date_time = now.strftime("%d-%m-%Y_%H_%M_%S")
file_name = f'config_camera_{date_time}.json'
if os.path.exists(file_name):
    os.remove(file_name)
with open(f'config_camera_{date_time}.json', 'w') as outfile:
    json.dump(dic, outfile, indent=4)
sensor_dep = pipeline.get_active_profile().get_device().first_depth_sensor()
# sensor_dep.set_option(rs.option.enable_auto_exposure, 1)

# sensor_dep.set_option(rs.option.max_distance, 0.5)
# sensor_dep.set_option(rs.option.exposure, 50000)
# exp = sensor_dep.get_option(rs.option.exposure)
save_data = False
i = 0
align_to = rs.stream.color
align = rs.align(align_to)
# sensor_dep = pipeline.get_active_profile().get_device().first_depth_sensor()
while True:
    aligned_frames = pipeline.wait_for_frames()
    aligned_frames = align.process(aligned_frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.uint16)
    color_image = np.asanyarray(color_frame.get_data())
    depth_image_to_save = depth_image.copy()
    color_image_to_save = color_image.copy()
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.addWeighted(depth_colormap, 0.8, color_image, 0.8, 0, color_image)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    if cv2.waitKey(10):
        if 0xFF == ord('q'):
            break
        elif 0xFF == ord('s'):
            save_data = True
    save_data = True
    # create folder
    if save_data:
        # print(f"saving ...")
        if not os.path.exists(f'D:\Documents\Programmation\pose_estimation\data_{date_time}'):
            os.makedirs(f'D:\Documents\Programmation\pose_estimation\data_{date_time}')
        cv2.imwrite(f'D:\Documents\Programmation\pose_estimation\data_{date_time}\depth_{i}.png', depth_image_to_save)
        cv2.imwrite(f'D:\Documents\Programmation\pose_estimation\data_{date_time}\color_{i}.png', color_image_to_save)
        i += 1