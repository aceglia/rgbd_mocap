import glob

import pyrealsense2 as rs
import numpy as np
import cv2
import os


def read_bag_file(bag_file_path):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file_path, repeat_playback=False)
    config.enable_all_streams()
    pipeline.start(config)
    device = pipeline.get_active_profile().get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    # pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    # pipeline_profile = config.resolve(pipeline_wrapper)
    # device = pipeline_profile.get_device()
    # d_profile = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
    # d_intr = d_profile.get_intrinsics()
    # scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    # c_profile = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
    # c_intr = c_profile.get_intrinsics()
    # deth_to_color = d_profile.get_extrinsics_to(c_profile)
    # r = np.array(deth_to_color.rotation).reshape(3, 3)
    # t = np.array(deth_to_color.translation)
    # device_product_line = str(device.get_info(rs.camera_info.product_line))
    # dic_config_cam = {
    #     "camera_name": device_product_line,
    #     "depth_scale": scale,
    #     "depth_fx_fy": [d_intr.fx, d_intr.fy],
    #     "depth_ppx_ppy": [d_intr.ppx, d_intr.ppy],
    #     "color_fx_fy": [c_intr.fx, c_intr.fy],
    #     "color_ppx_ppy": [c_intr.ppx, c_intr.ppy],
    #     "depth_to_color_trans": t.tolist(),
    #     "depth_to_color_rot": r.tolist(),
    #     "model_color": c_intr.model.name,
    #     "model_depth": d_intr.model.name,
    #     "dist_coeffs_color": c_intr.coeffs,
    #     "dist_coeffs_depth": d_intr.coeffs,
    #     "size_color": [c_intr.width, c_intr.height],
    #     "size_depth": [d_intr.width, d_intr.height],
    #     "color_rate": c_profile.fps(),
    #     "depth_rate": d_profile.fps(),
    # }
    # import json
    # with open("camera_config.json", "w") as f:
    #     json.dump(dic_config_cam, f, indent=4)

    # Start streaming

    # Get stream profile and camera intrinsics
    align_to = rs.stream.color
    align = rs.align(align_to)
    intrinsics = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # Read frames
    frames = []
    while True:
        # Wait for a coherent pair of frames: depth and color
        # poll frame
        frames = pipeline.wait_for_frames()
        if frames is None:
            break
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        color_image = np.asanyarray(color_frame.get_data())
        frame_number = color_frame.frame_number
        color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # file_dir = bag_file_path.replace(".bag", "")
        # if not os.path.exists(file_dir):
        #     os.makedirs(file_dir)
        # cv2.imwrite(f'{file_dir}/color_{frame_number}.png', color_rgb)
        # depth_dir = bag_file_path.replace(".bag", "depth")
        # if not os.path.exists(depth_dir):
        #     os.makedirs(depth_dir)
        # cv2.imwrite(f'{depth_dir}/depth_{frame_number}.png', depth_image)
    pipeline.stop()


if __name__ == '__main__':
    bag_file_dir = r"F:\CIME_LOC\tmp_videos"

    # bagfile = glob.glob(os.path.join(bag_file_dir, "20250422_161857.bag"))[0]
    bagfile = "test.bag"
    read_bag_file(bagfile)

