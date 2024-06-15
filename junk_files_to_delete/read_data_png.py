import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Path to the folder with the images
main_path = r"../data_files"
participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
for participant in participants:
    files = os.listdir(f"{main_path}{os.sep}{participant}")
    files = [file for file in files if "only" in file and "less" not in file and "more" not in file]
    for file in files:
        path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
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

        # out = cv2.VideoWriter(path + '\P9_test.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
        # 60, (frame_width, frame_height))
        first_max = -1
        first_min = -1
        for i in range(300):
            try:
                depth = cv2.imread(path + f"\depth_{idx[i]}.png", cv2.IMREAD_ANYDEPTH)
            except:
                continue
            if depth is None:
                continue
            depth = np.where(
                (depth > 1.2 / (0.0010000000474974513)) | (depth <= 0 / (0.0010000000474974513)),
                0,
                depth,
            )
            # min_depth = np.min(depth[depth > 0])
            # first_min = np.median(np.sort(depth[depth > 0].flatten())[:10]) if first_min == -1 else first_min
            # print(first_min)
            first_min = 0.4 / (0.0010000000474974513)
            first_max = np.median(np.sort(depth.flatten())[-30:]) if first_max == -1 else first_max

            # max_depth = 1.2 / (0.0010000000474974513)
            normalize_depth = (depth - first_min) / (first_max - first_min)
            normalize_depth[depth == 0] = 0
            normalize_depth = normalize_depth * 255
            depth = normalize_depth.astype(np.uint8)
            # plot histogramm of the depth image
            hist_eq = cv2.equalizeHist(depth)
            # hist_eq[hist_eq == 255] = 0
            hist_3d = np.dstack((hist_eq, hist_eq, hist_eq))
            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # # kernel = np.array([[-1, -1, -1],
            # #                    [-1, 9, -1],
            # #                    [-1, -1, -1]])
            # sharpened = cv2.filter2D(hist_3d, -1,
            #                          kernel)
            depth_3d = np.dstack((depth, depth, depth))
            depth_colormap = cv2.applyColorMap(hist_3d, cv2.COLORMAP_JET)
            depth_colormap[depth_3d == 0] = 0
            # kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
            kernel = np.array([[-1, 0, -1],
                               [-1, 7, -1],
                               [-1, 0, -1]])
            sharpened = cv2.filter2D(hist_3d, -1,
                                     kernel)
            # sharpened[sharpened == 255] = 0
            cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
            cv2.imshow("depth", sharpened)

            # cv2.waitKey(0)
            alpha = 0.5
            # control brightness by 50
            beta = 0
            image2 = cv2.convertScaleAbs(depth_3d, alpha=alpha, beta=beta)
            # applying the sharpening kernel to the input image & displaying it.
            cv2.waitKey(15)
            # cv2.namedWindow("init", cv2.WINDOW_NORMAL)
            # cv2.imshow("init", depth_3d)
            hist_eq = cv2.equalizeHist(depth)
            # hist_eq[hist_eq == 255] = 0
            hist_3d = np.dstack((hist_eq, hist_eq, hist_eq))
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # kernel = np.array([[-1, -1, -1],
            #                    [-1, 9, -1],
            #                    [-1, -1, -1]])
            sharpened = cv2.filter2D(hist_3d, -1,
                                     kernel)

            # hist, bins = np.histogram(hist_eq.flatten(), 256, [1, 256])
            # # emboss image
            #
            # cdf = hist.cumsum()
            # cdf_normalized = cdf * float(hist.max()) / cdf.max()
            #
            # plt.plot(cdf_normalized, color='b')
            # plt.hist(hist_eq.flatten(), 256, [1, 256], color='r')
            # plt.xlim([0, 256])
            # plt.legend(('cdf', 'histogram'), loc='upper left')
            # # plt.show()
            # cv2.namedWindow("hist_eq", cv2.WINDOW_NORMAL)
            # cv2.imshow("hist_eq", hist_eq)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            depth_clahe = clahe.apply(depth)
            # plt.figure("clahe")
            # hist, bins = np.histogram(depth_clahe.flatten(), 256, [4, 256])
            # # emboss image
            #
            # cdf = hist.cumsum()
            # cdf_normalized = cdf * float(hist.max()) / cdf.max()
            #
            # plt.plot(cdf_normalized, color='b')
            # plt.hist(depth_clahe.flatten(), 256, [4, 256], color='r')
            # plt.xlim([0, 256])
            # plt.legend(('cdf', 'histogram'), loc='upper left')
            # plt.show()

            kernel_structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            # Top Hat Transform
            topHat = cv2.morphologyEx(depth, cv2.MORPH_TOPHAT, kernel_structuring)  # Black Hat Transform
            blackHat = cv2.morphologyEx(depth, cv2.MORPH_BLACKHAT, kernel_structuring)
            res = depth + topHat - blackHat
            # cv2.namedWindow("res", cv2.WINDOW_NORMAL)
            # cv2.imshow("res", res)
            # cv2.namedWindow("hist_eq", cv2.WINDOW_NORMAL)
            # cv2.imshow("hist_eq", hist_eq)
            #
            # cv2.namedWindow("depth_clahe", cv2.WINDOW_NORMAL)
            # cv2.imshow("depth_clahe", sharpened)
            # cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
            # cv2.imshow("depth", image2
            #            )
            # cv2.waitKey(0)

            # out.write(depth_3d)
            # cv2.waitKey(16)
