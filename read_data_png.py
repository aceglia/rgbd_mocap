import cv2
import numpy as np
import os
import glob

# Path to the folder with the images
path = "Q:\Projet_hand_bike_markerless\RGBD\P1\gear_5_28-07-2023_16_11_45"
all_color_files = glob.glob(path + "/color*.png")
all_depth_files = glob.glob(path + "/depth*.png")
frame_width = 848
frame_height = 480
windows_destroyed = False

import time

# out = cv2.VideoWriter('P1_test.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
# 50, (frame_width, frame_height))
for i in range(100, int(len(all_color_files))):
    tic = time.time()
    color, depth = (cv2.imread(path + f"/color_{i}.png"),
                    cv2.imread(path + f"/depth_{i}.png", cv2.IMREAD_ANYDEPTH))
    color = cv2.rotate(color, cv2.ROTATE_180)
    # out.write(color)

    depth = cv2.rotate(depth, cv2.ROTATE_180)
    depth_colormap = cv2.convertScaleAbs(depth, alpha=0.03)
    # draw all contours
    depth_3d = np.dstack((depth_colormap, depth_colormap, depth_colormap))
    # cv2.addWeighted(depth_colormap, 0.8, color, 0.8, 0, color)
    # if cv2.waitKey(int((1/80)*1000)) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     windows_destroyed = True
    # out.write(depth_3d)
    if not windows_destroyed:
        cv2.putText(
            color, f"FPS = {1 / (time.time() - tic)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.imshow("RealSense", color)
    cv2.waitKey(16)
