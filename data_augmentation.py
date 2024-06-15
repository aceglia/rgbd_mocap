import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import cv2
import os
from numba import jit
import time
import glob
from biosiglive import load
from imgaug.augmentables import Keypoint, KeypointsOnImage
import csv
from pathlib import Path
import shutil

def add_to_csv(image_path, markers_pos, markers_names, csv_path):
    file = csv_path
    # write headers if file does not exist
    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scorer", "", ""] + ["Automatic labeling"] * 2 * len(markers_names))
            bodyparts = []
            x_y = []
            for i in range(len(markers_names)):
                bodyparts.append((markers_names[i]))
                bodyparts.append((markers_names[i]))
                x_y.append("x")
                x_y.append("y")
            writer.writerow(["bodyparts", "", ""] + bodyparts)
            writer.writerow(["coords", "", ""] + x_y)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        markers_pos_list = []
        for i in range(len(markers_names)):
            if markers_pos.shape[1] == 12 and i == 0:
                markers_pos_list.append(None)
                markers_pos_list.append(None)
            markers_pos_list.append(markers_pos[0, i])
            markers_pos_list.append(markers_pos[1, i])
        writer.writerow(["labeled-data", "data_test"] + [Path(image_path).stem + ".png"] + markers_pos_list)


def get_augmented_images(images, keypoints=None):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.15))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.6, 1.3), "y": (0.6, 1.3)},
                translate_percent={"x": (-0.15, 0.15), "y": (-0.2, 0.2)},
                rotate=(-30, 30),
                # shear=(-16, 16),
                order=[0, 1],
                cval=(0, 1),
                mode="constant"
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           # sometimes(iaa.Dropout((0.01, 0.05), per_channel=0.5)),

                           # In some images distort local areas with varying strength.
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )
    if keypoints is not None:
        return seq(images=images, keypoints=keypoints)
    return seq(images=images)


def save_augmented_images(images, kps_aug, path, marker_names):
    out = cv2.VideoWriter(
        path + "/annotated_images_aug.avi",
        cv2.VideoWriter_fourcc("M", "J", "P3", "G"), 20, (848, 480))
    csv_path = training_path + "/CollectedData_Ame.csv"
    for i in range(images.shape[0]):
        # if kps_aug is None:
        #     image_path = training_path + rf"/depth_{i}.png"
        #     cv2.imwrite(image_path, images[i, ...])
        #     continue
        markers = kps_aug[i].to_xy_array().T
        image_path = training_path + rf"/depth_{i}.png"
        cv2.imwrite(image_path, images[i, ...])
        marker_names_tmp = marker_names if markers.shape[1] == len(marker_names) else marker_names[1:]
        add_to_csv(image_path, markers, marker_names_tmp, csv_path)
        image_to_save = kps_aug[i].draw_on_image(images[i, ...].copy(), size=10)
        # cv2.imshow("image", image_to_save)

        # cv2.waitKey(0)
        out.write(image_to_save)


def get_label_image(participant_to_exclude=None):
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    # main_path = "Q:\Projet_hand_bike_markerless\RGBD"
    main_path = "data_files"
    nb_frame = 30
    empty_depth = np.zeros((nb_frame * (len(participants) -1 ), 480, 848, 3), dtype=np.uint8)
    all_kps = []
    count = 0
    for p, participant in enumerate(participants):
        if participant == participant_to_exclude:
            continue
        files = os.listdir(f"{main_path}{os.sep}{participant}")
        files = [file for file in files if "only" in file and "less" not in file and "more" not in file]
        for file in files:
            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            if not os.path.isfile(path + "/marker_pos_multi_proc_3_crops_pp.bio"):
                continue
            print("getting data from ", file, "for participant ", participant, "...")
            markers_data = load(path + "/marker_pos_multi_proc_3_crops_pp.bio")
            markers = markers_data["markers_in_pixel"]
            frame_idx = markers_data["frame_idx"]
            occlusions = markers_data["occlusions"]
            marker_names = markers_data["markers_names"][:, 0]
            oc_list = [None] * (occlusions.shape[0] - 1)
            oc_list_ones = np.ones((occlusions.shape[1]))
            for i in range(1, occlusions.shape[0]):
                oc_list_ones_tmp = np.ones((occlusions.shape[1]))
                oc_list[i-1] = list(np.argwhere(occlusions[i, :] == False))
                oc_list_ones_tmp[oc_list[i-1]] = 0
                oc_list_ones = oc_list_ones * oc_list_ones_tmp
            final_visible_idx = np.argwhere(oc_list_ones == 1).flatten()
            final_visible_idx_rand = np.sort(np.random.choice(final_visible_idx, nb_frame))
            markers = markers[:, :, final_visible_idx_rand]
            frame_idx = np.array(frame_idx)[final_visible_idx_rand]
            xiph_threeshold = np.mean(markers[2, 0, :]) - 0.1
            first_min = -1
            first_max = -1
            for i in range(nb_frame):
                depth = cv2.imread(path + f"/depth_{frame_idx[i]}.png", cv2.IMREAD_ANYDEPTH)
                # cv2.imshow("depth", depth)
                key_point_list = []
                depht_value_xiph = depth[markers[1, 0, i].astype(int), markers[0, 0, i].astype(int)] * 0.0010000000474974513
                if depht_value_xiph > xiph_threeshold:
                    key_point_list.append(Keypoint(x=int(markers[0, 0, i]), y=int(markers[1, 0, i])))
                # else:
                #     print("xiphoid process is not visible")
                #     print("depth value at xiphoid process: ", depht_value_xiph)
                #     print("threeshold: ", xiph_threeshold)
                for j in range(1, markers.shape[1]):
                    key_point_list.append(Keypoint(x=int(markers[0, j, i]), y=int(markers[1, j, i])))
                all_kps.append(KeypointsOnImage(key_point_list, shape=(480, 848, 3)))
                first_min = np.median(np.sort(depth.flatten()[depth.flatten().nonzero()])[:10]) if first_min == -1 else first_min
                # depth = cv2.resize(depth, (848 // 2, 480 // 2), interpolation=cv2.INTER_NEAREST)

                tic = time.time()
                depth = np.where(
                    (depth > 1.2 / (0.0010000000474974513)) | (depth <= 0.2 / (0.0010000000474974513)),
                    0,
                    depth,
                )
                # compute_surface_normals(depth)

                empty_depth[count, ...] = compute_surface_normals(depth)
                print("time to compute surface normals: ", time.time() - tic)

                # min_depth = np.min(depth[depth > 0])
                # print(first_min)
                # min_depth = np.min(depth[depth > 0])
                # first_min = np.median(np.sort(depth[depth > 0].flatten())[:10]) if first_min == -1 else first_min
                # print(first_min)
                # first_min = 0.4 / (0.0010000000474974513)
                # first_max = np.median(np.sort(depth.flatten())[-30:]) if first_max == -1 else first_max
                # normalize_depth = (depth - first_min) / (first_max - first_min)
                # normalize_depth[depth == 0] = 0
                # normalize_depth = normalize_depth * 255
                # depth = normalize_depth.astype(np.uint8)
                # cv2.imshow("depth", depth)
                # cv2.waitKey(0)

                # first_min = 0.4 / (0.0010000000474974513)
                # first_min = np.median(np.sort(depth[depth > 0].flatten())[:10]) if first_min == -1 else first_min
                # first_max = np.median(np.sort(depth.flatten())[-30:]) if first_max == -1 else first_max
                # depth_list = []
                # ranges = [0.4, 0.8, 0.95, 1.2]
                # for d in range(3):
                #     depth_tmp = np.where(
                #         (depth > ranges[d+1] / (0.0010000000474974513)) | (depth <= ranges[d]  / (0.0010000000474974513)),
                #         0,
                #         depth,
                #     )
                #     min = np.median(np.sort(depth.flatten()[depth_tmp.flatten().nonzero()])[:10])
                #     max = np.median(np.sort(depth_tmp.flatten())[-30:])
                #     normalize_depth = (depth_tmp - min) / (max - min)
                #     normalize_depth[depth_tmp == 0] = 0
                #     normalize_depth = normalize_depth * 255
                #     depth_tmp = normalize_depth.astype(np.uint8)
                #     depth_list.append(depth_tmp)
                # #     cv2.imshow(f"depth_{d}", depth_tmp)
                # # cv2.waitKey(0)
                #
                # depth_3d = np.dstack((depth_list[0], depth_list[1], depth_list[2]))

                # depth_3d = np.dstack((depth, depth, depth))
                # aply depth color map
                # depth_colormap = cv2.applyColorMap(empty_depth[count, ...], cv2.COLORMAP_JET)
                # depth_colormap[empty_depth[count, ...] == 0] = 0
                # empty_depth[count, ...] = depth_colormap
                # hist_eq = cv2.equalizeHist(depth)
                # hist_3d = np.dstack((hist_eq, hist_eq, hist_eq))
                # # depth_3d = np.dstack((depth, depth, depth))
                # depth_colormap = cv2.applyColorMap(hist_3d, cv2.COLORMAP_JET)
                # depth_colormap[depth_3d == 0] = 0
                # kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
                # kernel = np.array([[-1, 0, -1],
                #                    [-1, 7, -1],
                #                    [-1, 0, -1]])
                # empty_depth[count, ...] = cv2.filter2D(depth_3d, -1,
                #                          kernel)
                # augment constrast in the depth image using opencv
                # clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(13, 13))
                # depth = clahe.apply(depth)
                cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
                cv2.imshow("depth", empty_depth[count, ...])
                cv2.waitKey(0)
                count += 1
    return empty_depth, all_kps, marker_names

@jit
def _compute_normal(x, y, depth_map, rows, cols):

    dx = np.gradient(depth_map, axis=1)
    dy = np.gradient(depth_map, axis=0)
    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))

    # norm = np.linalg.norm(normal, axis=2).reshape((rows, cols, 1))
    norm = np.sqrt(np.sum(normal ** 2, axis=2)) #, keepdims=True))
    norm_non_zero = np.ones_like(norm)
    norm_non_zero[norm == 0] = 1
    normal /= norm_non_zero
    # normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)
    return normal


def compute_surface_normals(depth_map):
    rows, cols = depth_map.shape

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1)
    # dx = np.gradient(depth_map, axis=1)
    # dy = np.gradient(depth_map, axis=0)
    # x = x.astype(np.float32)
    # y = y.astype(np.float32)
    # # Calculate the partial derivatives of depth with respect to x and y
    # dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0)
    # dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1)
    # # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    #
    # tic= time.time()
    #
    norm = np.linalg.norm(normal, axis=2).reshape((rows, cols, 1))
    # norm = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    # grad /= np.linalg.norm(grad, axis=0)
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)
    # print("time to compute normal vectors: ", time.time() - tic)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # Save the normal map to a file
    # normal = _compute_normal(x, y, depth_map, rows, cols)
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    return normal_bgr


if __name__ == '__main__':
    np.random.seed(40)
    participants = ["P9"]#, "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    for p, part in enumerate(participants):
        print(f"Processing data augmentation excluding {part}...")
        # training_path = f"Q:\Projet_hand_bike_markerless\RGBD\Training_data\{part}_excluded_normal"
        # if os.path.exists(training_path):
        #     shutil.rmtree(training_path, ignore_errors=True)
        # os.makedirs(training_path)
        images, markers, marker_names = get_label_image(participant_to_exclude=part)
        # images_aug, kps_aug = get_augmented_images(images, markers)
        # save_augmented_images(images_aug, kps_aug, training_path, marker_names)
        #save_augmented_images(images, markers, training_path, marker_names)