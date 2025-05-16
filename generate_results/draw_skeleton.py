import glob

import os

import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from rgbd_mocap.camera.camera_converter import CameraConverter
from C3DtoTRC import WriteTrcFromMarkersData
from biosiglive import save, OfflineProcessing
import pandas as pd


connections = [(17, 15),
               (15, 0),
               (0, 16),
               (16, 18),
               (0, 1),
               (1, 2),
               (2, 3),
               (3, 4),
               (1, 5),
               (5, 6),
               (6, 7),
               (1, 8),
               (8, 9),
               (9, 10),
               (8, 12),
               (10, 11),
               (11, 22),
               (22, 23),
               (11, 24),
               (12, 13),
               (13, 14),
               (14, 21),
               (14, 19),
               (19, 20),
]
body25_keypoints = [
    "Nose",                # 0
    "Neck",                # 1
    "Right Shoulder",      # 2
    "Right Elbow",         # 3
    "Right Wrist",         # 4
    "Left Shoulder",       # 5
    "Left Elbow",          # 6
    "Left Wrist",          # 7
    "Mid Hip",             # 8
    "Right Hip",           # 9
    "Right Knee",          # 10
    "Right Ankle",         # 11
    "Left Hip",            # 12
    "Left Knee",           # 13
    "Left Ankle",          # 14
    "Right Eye",           # 15
    "Left Eye",            # 16
    "Right Ear",           # 17
    "Left Ear",            # 18
    "Left Big Toe",        # 19
    "Left Small Toe",      # 20
    "Left Heel",           # 21
    "Right Big Toe",       # 22
    "Right Small Toe",     # 23
    "Right Heel"           # 24
]
key_points_reduced = [
    "Nose",                # 0
    "Neck",                # 1
    "RShoulder",           # 2
    "RElbow",              # 3
    "RWrist",              # 4
    "LShoulder",           # 5
    "LElbow",              # 6
    "LWrist",              # 7
    "MidHip",              # 8
    "RHip",                # 9
    "RKnee",               # 10
    "RAnkle",              # 11
    "LHip",                # 12
    "LKnee",               # 13
    "LAnkle",              # 14
    "REye",                # 15
    "LEye",                # 16
    "REar",                # 17
    "LEar",                # 18
    "LBigToe",             # 19
    "LSmallToe",           # 20
    "LHeel",               # 21
    "RBigToe",             # 22
    "RSmallToe",           # 23
    "RHeel",               # 24
]

def get_3d_coordinates(keypoints, depth_img, camera):
    keypoints_3d = []
    for i in range(keypoints.shape[0]):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        z = depth_img[int(y), int(x)] * camera.depth_scale
        keypoints_3d.append([x, y, z])
    return camera.get_markers_pos_in_meter(np.array(keypoints_3d)).T


def write_bio(key_points, file_name):
    key_points_final = key_points
    key_points_final[key_points == 0] = np.nan
    # extrapolate missing markers

    data_filled_extr = np.zeros((3, key_points_final.shape[1], key_points_final.shape[2]))
    for i in range(3):
        data_df = pd.DataFrame(key_points_final[i, :, :], body25_keypoints)
        data_filled_extr[i, :, :] = data_df.interpolate(method="linear", axis=1)

    # filter markers data using low pass
    markers_filtered = np.zeros_like(data_filled_extr)
    for i in range(3):
        low_pass = OfflineProcessing(data_rate=60)
        low_pass.lpf_lcut = 3
        markers_filtered[i, ...] = low_pass.process_generic_signal(data_filled_extr[i, ...],
                                                                   None,
                                                                   False, True, False, False, False,
                                                                   )
    data = {"key_points_ini": key_points,
            "key_points_filtered": markers_filtered,
            "key_point_names": body25_keypoints,
            "key_point_reduced_names": key_points_reduced,
            "key_points_connections": connections}
    save(data, file_name, safe=False)


def write_trc(key_points, file_name):
    name_mapping = [["Nose", "Neck", "LShoulder", "RShoulder","LElbow","RElbow",
"LWrist", "RWrist", "LHip", "RHip",
"LKnee", "RKnee", "LAnkle", "RAnkle", "LBigToe", "LSmallToe",
"LHeel", "RBigToe","RSmallToe","RHeel", "CHip"],
[0, 1, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24, 8]]
    key_points_names = None
    key_points_finals = np.zeros((3, len(name_mapping[0]), key_points.shape[-1]))
    key_points_names = []
    count = 0
    for i in range(key_points.shape[1]):
        for k in range(len(name_mapping[0])):
            if i == name_mapping[1][k]:
                key_points_names.append(name_mapping[0][k])
                key_points_finals[:, count, :] = key_points[:, i, :]
                count += 1
                break
    key_points_finals[key_points_finals == 0] = np.nan
    # extrapolate missing markers
    import pandas as pd
    data_filled_extr = np.zeros((3, key_points_finals.shape[1], key_points_finals.shape[2]))
    for i in range(3):
        data_df = pd.DataFrame(key_points_finals[i, :, :], key_points_names)
        data_filled_extr[i, :, :] = data_df.interpolate(method="linear", axis=1)

    # filter markers data using low pass
    markers_filtered = np.zeros_like(data_filled_extr)
    from biosiglive import OfflineProcessing
    for i in range(3):
        low_pass = OfflineProcessing(data_rate=60)
        low_pass.lpf_lcut = 3
        markers_filtered[i, ...] = low_pass.process_generic_signal(data_filled_extr[i, ...],
                                                       None,
                                                       False, True, False, False, False,
                                                       )
    # plt.figure("markers")
    # for i in range(key_points_finals.shape[1]):
    #     plt.subplot(5, 5, i+1)
    #     plt.plot(key_points_finals[0, i, :])
    #     plt.plot(markers_filtered[0, i, :])
    # plt.show()

    WriteTrcFromMarkersData(file_name, markers=markers_filtered, marker_names=key_points_names,
                            data_rate=60, cam_rate=60, n_frames=key_points_finals.shape[-1]).write_trc()


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

camera = CameraConverter()
camera.set_intrinsics("camera_config.json")
camera.set_extrinsics("camera_config.json")
# read keypoints from json file

file_name = "20250422_162024"
color_path_tmp = fr"F:\CIME_LOC\tmp_videos\{file_name}"
depth_path_tmp = fr"F:\CIME_LOC\tmp_videos\{file_name}depth"
key_points_file = fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons"
all_key_points_files = glob.glob(key_points_file + "/*.json")
idxs = [int(os.path.basename(file).split("_")[1].split(".")[0]) for file in all_key_points_files]
idxs = sorted(idxs)
frame_nr = 0
n_image = len(all_key_points_files)
count = 0
key_points_3d = None
while count != n_image-1:
    frame_nr += 1
    depth_image = cv2.imread(depth_path_tmp + f"\depth_{idxs[frame_nr]}.png", cv2.IMREAD_ANYDEPTH)
    with open(os.path.join(key_points_file, f"color_{idxs[frame_nr]}_keypoints.json"), 'r') as f:
        keypoints = json.load(f)
    keypoints = keypoints["people"][0]["pose_keypoints_2d"]
    keypoints = np.array(keypoints).reshape((-1, 3))
    keypoints_3d = get_3d_coordinates(keypoints, depth_image, camera).T[..., None]
    key_points_3d = keypoints_3d if key_points_3d is None else np.concatenate((key_points_3d, keypoints_3d), axis=2)
    count += 1

write_bio(key_points_3d, fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons\{file_name}.bio")

# write_trc(keypoints_3d, fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons\{file_name}.trc")

plt.figure("3d plot")
ax = plt.axes(projection='3d')
keypoints_3d = key_points_3d[..., 0].T
ax.scatter3D(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2])
# draw line for each connection
for connection in connections:
    ax.plot3D([keypoints_3d[connection[0], 0], keypoints_3d[connection[1], 0]],
              [keypoints_3d[connection[0], 1], keypoints_3d[connection[1], 1]],
              [keypoints_3d[connection[0], 2], keypoints_3d[connection[1], 2]],
              color='r')
set_axes_equal(ax)
plt.show()
# plt.figure("2d plot")
# # plt.scatter(keypoints[:, 0], keypoints[:, 1])
# # plot text for each keypoint
# [plt.text(keypoints[i, 0], keypoints[i, 1], str(i)) for i in range(keypoints.shape[0])]
# # draw line for each connection
# for connection in connections:
#     plt.plot([keypoints[connection[0], 0], keypoints[connection[1], 0]],
#              [keypoints[connection[0], 1], keypoints[connection[1], 1]],
#              color='r')
# plt.imshow(cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB))
# plt.show()
# print(keypoints)


