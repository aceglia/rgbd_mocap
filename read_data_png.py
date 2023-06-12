import cv2
import numpy as np
import os
import glob

# Path to the folder with the images
path = "data_25-04-2023_16_55_00"
all_color_files = glob.glob(path + "/color*.png")
all_depth_files = glob.glob(path + "/depth*.png")
frame_width = 848
frame_height = 480
out = cv2.VideoWriter('pedalage_25-04-2023_16_55_00.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
for i in range(0, int(len(all_color_files)/2)):
    color, depth = cv2.imread(path + f"/color_{i}.png"), cv2.imread(path + f"/depth_{i}.png", cv2.IMREAD_ANYDEPTH)
    out.write(color)
    # depth = np.asarray(depth, dtype=np.uint16)
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    # bld = 0.5 * color + (1 - 0.5) * depth_colormap
    # new_color = (depth_colormap -  color * 0.8) / (1-0.8)
    # cv2.addWeighted(depth_colormap, 0.5, color, 0.5, 0)
    # cv2.addWeighted(depth_colormap, 1/0.5, color, 1/0.5, 0, color)
    # cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL) # or cv2.INPAINT_NS for Navier-Stokes algorithm
    # cv2.imshow('RealSense', new_color)
    # cv2.waitKey(160)
#all color file