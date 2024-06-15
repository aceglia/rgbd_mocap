import cv2
import os
import glob
import numpy as np


if __name__ == '__main__':
    participants = ["P10"]#, "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    main_path = "Q:\Projet_hand_bike_markerless\RGBD"
    # main_path = "data_files"
    # empty_depth = np.zeros((nb_frame * (len(participants) - 1), 480, 848, 3), dtype=np.uint8)
    all_kps = []
    count = 0
    for p, participant in enumerate(participants):
        files = os.listdir(f"{main_path}{os.sep}{participant}")
        files = [file for file in files if "only" in file and "less" not in file and "more" not in file]
        for file in files:
            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            file_full = "image_processed.avi"
            non_augmented = "video_labeled_non_augmented_alone.avi"
            video_full = cv2.VideoCapture(path + os.sep + file_full)
            video_non_augmented = cv2.VideoCapture(path + os.sep + non_augmented)
            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            all_color_files = glob.glob(path + "/color*.png")
            all_depth_files = glob.glob(path + "/depth*.png")
            # check where there is a gap in the numbering
            idx = []
            for f in all_depth_files:
                idx.append(int(f.split("\\")[-1].split("_")[-1].removesuffix(".png")))
            idx.sort()
            count = 1000
            while True:
                ret, frame_full = video_full.read()
                ret, frame_non_augmented = video_non_augmented.read()
                if frame_full is None or frame_non_augmented is None:
                    break
                color = cv2.imread(path + f"\color_{idx[count]}.png")
                final_image = cv2.addWeighted(frame_full, 0.5, frame_non_augmented, 0.5, 0)
                # empty_depth[count] = np.concatenate((frame_full, frame_non_augmented), axis=1)
                # empty_depth = np.concatenate((frame_full, frame_non_augmented), axis=1)
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                cv2.imshow("frame", final_image)
                cv2.waitKey(16)
                count += 1
