import cv2
import numpy as np
import os
import glob

# Path to the folder with the images
path = r"Q:\Projet_hand_bike_markerless\RGBD\P9\only_rgbd_11-01-2024_17_27_39"
all_color_files = glob.glob(path + "/color*.png")
all_depth_files = glob.glob(path + "/depth*.png")
# check where there is a gap in the numbering
idx = []
for f in all_depth_files:
    idx.append(int(f.split("\\")[-1].split("_")[-1].removesuffix(".png")))
idx.sort()
frame_width = 848
frame_height = 480
windows_destroyed = False

import time

out = cv2.VideoWriter(path + '\P9_test.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
60, (frame_width, frame_height))
for i in range(1000, 1800):
    try:
        depth = cv2.imread(path + f"\depth_{idx[i]}.png", cv2.IMREAD_ANYDEPTH)
    except:
        continue
    if depth is None:
        continue
    depth = np.where(
        (depth > 1.4 / (0.0010000000474974513)) | (depth <= 0),
        0,
        depth,
    )
    min_depth = np.min(depth[depth > 0])
    # min_depth = 200
    max_depth = np.max(depth)
    normalize_depth = (depth - min_depth) / (max_depth - min_depth)
    normalize_depth[depth == 0] = 0
    normalize_depth = normalize_depth * 255
    depth = normalize_depth.astype(np.uint8)
    depth_3d = np.dstack((depth, depth, depth))
    out.write(depth_3d)
    # cv2.waitKey(16)
