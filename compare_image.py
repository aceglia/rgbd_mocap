import cv2
import numpy as np

if __name__ == '__main__':

    data_dir = r"D:\Documents\Programmation\pose_estimation\data_files\P16\gear_20_25-01-2024_14_46_57\test_images"
    new_filter = data_dir + r"\new_filter.png"
    new_depth = data_dir + r"\new_depth.png"
    new_with_blobs = data_dir  + r"\new_with_blobs.png"
    new_color = data_dir+ r"\before_new_filter.png"
    new_kalman = data_dir + r"\new_kalman.png"
    new_optical_flow = data_dir + r"\new_optical_flow.png"
    new_optical_flow_image = data_dir + r"\new_image_optical_flow.png"
    optical_flow_image = data_dir + r"\image_optical_flow.png"
    new_optical_flow_depth = data_dir + r"\new_depth_optical_flow.png"
    optical_flow_depth = data_dir + r"\depth_optical_flow.png"

    optical_flow = data_dir + r"\optical_flow.png"
    kalman = data_dir + r"\kalman.png"
    filter = data_dir + r"\filter.png"
    with_blobs = data_dir + r"\with_blobs.png"
    color = data_dir + r"\before_filter.png"
    news = [new_color,  new_filter, new_with_blobs, new_kalman, new_optical_flow, new_optical_flow_image, new_optical_flow_depth, new_depth]
    olds = [color, filter, with_blobs, kalman, optical_flow, optical_flow_image, optical_flow_depth, optical_flow_depth]
    for i in range(len(news)):
        new = cv2.imread(news[i])
        old = cv2.imread(olds[i])
        print(f"new and old {str(olds[i])} are equal:", np.array_equal(new, old))