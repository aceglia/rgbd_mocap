import numpy as np
import os
import csv
from pathlib import Path
import shutil
import imgaug.augmenters as iaa
import cv2
# from numba import jit
import time
from biosiglive import load
from imgaug.augmentables import Keypoint, KeypointsOnImage

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
        for i in range(markers_pos.shape[1]):
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
    # out = cv2.VideoWriter(
    #     path + "/annotated_images_aug.avi",
    #     cv2.VideoWriter_fourcc("M", "J", "P3", "G"), 20, (848, 480))
    csv_path = training_path + "/CollectedData_Ame.csv"
    for i in range(len(images)):
        # if kps_aug is None:
        #     image_path = training_path + rf"/depth_{i}.png"
        #     cv2.imwrite(image_path, images[i, ...])
        #     continue
        markers = kps_aug[i].to_xy_array().T
        image_path = training_path + rf"/depth_{i}.png"
        cv2.imwrite(image_path, images[i])
        marker_names_tmp = marker_names
        add_to_csv(image_path, markers, marker_names_tmp, csv_path)
        # image_to_save = kps_aug[i].draw_on_image(images[i].copy(), size=10)
        # cv2.imshow("image", image_to_save)
        # cv2.waitKey(0)
        #out.write(image_to_save)


def apply_crop_and_ratio(markers, ratio, area):
    transformed_mark = markers.copy()
    for m in range(markers.shape[1]):
        transformed_mark[:2, m] = markers[:2, m] - area[:2]
        transformed_mark[:2, m] = transformed_mark[:2, m] * ratio
    return transformed_mark.astype(int)

def get_label_image(participant_to_exclude=None):
    import json
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    main_path = "Q:\Projet_hand_bike_markerless\RGBD"

    #main_path = "data_files"
    nb_frame = 500
    nb_cycle = 20
    ratio_down = 3
    empty_depth = []
    all_kps = []
    count = 0
    for p, participant in enumerate(participants):
        if participant == participant_to_exclude:
            continue
        files = os.listdir(f"{main_path}{os.sep}{participant}")
        file_gear_5 = [file for file in files if "gear_5" in file and "less" not in file and "more" not in file]
        files = [file for file in files if "only" in file and "less" not in file and "more" not in file]
        for file in files:
            tracking_config_path = f"{main_path}{os.sep}{participant}{os.sep}" + file_gear_5[0] + f"{os.sep}tracking_config_dlc.json"
            with open(tracking_config_path) as json_file:
                tracking_config = json.load(json_file)
            area = tracking_config["crops"][0]["area"]
            for a in range(len(area)):
                if a in [0, 1, 2]:
                    if a == 0 or a == 1:
                        value = area[a] - 50
                        area[a] = value if value >=0 else 0
                    else:
                        value = area[a] + 50
                        area[a] = value if value <= 848 else 848
                if a == 3:
                    value = area[a] + 50
                    area[a] = value if value <= 460 else 460

            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            if not os.path.isfile(path + "/marker_pos_multi_proc_3_crops_pp.bio"):
                continue
            print("getting data from ", file, "for participant ", participant, "...")
            markers_data = load(path + "/marker_pos_multi_proc_3_crops_pp.bio")
            markers = markers_data["markers_in_pixel"]
            frame_idx = markers_data["frame_idx"]
            occlusions = markers_data["occlusions"]
            marker_names = markers_data["markers_names"][:, 0]
            no_gap_idx = np.linspace(frame_idx[0], frame_idx[-1], frame_idx[-1] - frame_idx[0]).astype(int)
            first_frame_cycles = no_gap_idx[::60]
            cycles = np.sort(np.random.choice(first_frame_cycles, nb_cycle))
            all_cycles = sum([list(range(idx_cycle, idx_cycle + 60)) for idx_cycle in cycles], [])
            all_cycles_reduced = np.sort(np.random.choice(all_cycles, nb_frame))
            oc_list = [None] * (occlusions.shape[0] - 1)
            oc_list_ones = np.ones((occlusions.shape[1]))
            for i in range(1, occlusions.shape[0]):
                oc_list_ones_tmp = np.ones((occlusions.shape[1]))
                oc_list[i-1] = list(np.argwhere(occlusions[i, :] == False))
                oc_list_ones_tmp[oc_list[i-1]] = 0
                oc_list_ones = oc_list_ones * oc_list_ones_tmp
            final_visible_idx = np.argwhere(oc_list_ones == 1).flatten()
            final_visible_idx = [frame_idx[vis_idx] for vis_idx in final_visible_idx]
            # final_visible_idx_rand = np.sort(np.random.choice(final_visible_idx, nb_frame))
            final_idx = [idx_cycle for idx_cycle in all_cycles_reduced if idx_cycle in final_visible_idx and idx_cycle in frame_idx]
            final_idx = [frame_idx.index(idx_cycle) for idx_cycle in final_idx]
            markers = markers[:, :, final_idx]

            frame_idx = np.array(frame_idx)[final_idx]
            xiph_threeshold = np.mean(markers[2, 0, :]) - 0.1
            for i in range(len(frame_idx)):
                depth = cv2.imread(path + f"/depth_{frame_idx[i]}.png", cv2.IMREAD_ANYDEPTH)
                depth_init = depth.copy()
                depth = depth[area[1]: area[3], area[0]:area[2]]
                # cv2.imshow("depth", depth)
                key_point_list = []
                if count % 3 == 0:
                    ratio = 0.8
                elif count % 2 == 0:
                    ratio = 0.9
                else:
                    ratio = 1
                markers_tmp = apply_crop_and_ratio(markers[:, :, i], area=area, ratio=ratio)
                depht_value_xiph = depth[markers_tmp[1, 0].astype(int), markers_tmp[0, 0].astype(int)] * 0.0010000000474974513
                if depht_value_xiph > xiph_threeshold:
                    key_point_list.append(Keypoint(x=markers_tmp[0, 0], y=markers_tmp[1, 0]))
                # else:
                #     print("xiphoid process is not visible")
                #     print("depth value at xiphoid process: ", depht_value_xiph)
                #     print("threeshold: ", xiph_threeshold)
                for j in range(1, markers_tmp.shape[1]):
                    key_point_list.append(Keypoint(x=markers_tmp[0, j], y=markers_tmp[1, j]))
                all_kps.append(KeypointsOnImage(key_point_list, shape=(int(depth.shape[1] * ratio), int(depth.shape[0] * ratio), 3)))
                depth = np.where(
                    (depth > 1.2 / (0.0010000000474974513)) | (depth <= 0.2 / (0.0010000000474974513)),
                    0,
                    depth,
                )

                # compute_surface_normals(depth)
                if ratio != 1:
                    depth = cv2.resize(depth, (int(depth.shape[1] * ratio), int(depth.shape[0] * ratio)))#, interpolation=cv2.INTER_NEAREST)

                empty_depth.append(compute_surface_normals(depth))
                # empty_depth.append(compute_surface_normals_k_nearest(depth))

                # cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
                # cv2.imshow("depth", empty_depth[count])

                # first_min = 0.4 / (0.0010000000474974513)
                # first_max = np.median(np.sort(depth.flatten())[-30:])
                # normalize_depth = (depth - first_min) / (first_max - first_min)
                # normalize_depth[depth == 0] = 0
                # normalize_depth = normalize_depth * 255
                # depth = normalize_depth.astype(np.uint8)

                # cv2.imwrite(r"Q:\Projet_hand_bike_markerless\RGBD\fig_surface_normal.png", empty_depth[count])
                # depth_colormap = cv2.applyColorMap(np.dstack([depth, depth, depth]), cv2.COLORMAP_JET)
                # cv2.imwrite(r"Q:\Projet_hand_bike_markerless\RGBD\fig_surface_normal.png", empty_depth[count])
                # cv2.imwrite(r"Q:\Projet_hand_bike_markerless\RGBD\fig_3D.png",depth_colormap )
                # cv2.imshow("c", depth_colormap)

                # if count % ratio_down == 0:
                #     cv2.waitKey(0)
                # cv2.waitKey(0)
                count += 1
    return empty_depth, all_kps, marker_names

def compute_surface_normals(depth_map, empty_mat = None):
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1)

    normal = empty_mat if empty_mat is not None else np.empty(
        (depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
    normal[..., 0] = -dx
    normal[..., 1] = -dy
    normal[..., 2] = 1.0

    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1.0) * 127.5
    #normal *= 255
    normal = np.clip(normal, 0, 255).astype(np.uint8)
    normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    return normal


from scipy.spatial import cKDTree


def compute_surface_normals_k_nearest(depth_map, k=9):
    # Get the coordinates of all pixels
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack((x, y, depth_map), axis=-1).reshape(-1, 3)

    # Filter out invalid points (where depth is zero or invalid)
    valid_mask = depth_map > 0
    valid_coords = coords[valid_mask.reshape(-1)]

    # Build the k-d tree for fast neighbor search
    tree = cKDTree(valid_coords)

    # Initialize the normal map
    normals = np.zeros_like(coords, dtype=np.float32)

    # Calculate normals for valid points
    for i, point in enumerate(valid_coords):
        _, idx = tree.query(point, k=k)
        neighbors = valid_coords[idx]
        # Center the neighborhood around the origin
        centered_neighbors = neighbors - point
        # Perform SVD
        _, _, vt = np.linalg.svd(centered_neighbors)
        normal = vt[2]  # Normal is the last row of vt
        normals[valid_mask.reshape(-1)][i] = normal

    # Reshape normals to the original image shape
    normals = normals.reshape(height, width, 3)

    # Normalize the normals
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm != 0)

    normals = (normals + 1) * 127.5
    normals = np.clip(normals, 0, 255).astype(np.uint8)
    # Convert normal to BGR format for visualization (assuming RGB input)
    normals_bgr = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)

    return normals_bgr

if __name__ == '__main__':
    prefix = r"Q:\Projet_hand_bike_markerless" if os.name == "nt" else r"/mnt/Projet_hand_bike_markerless"
    np.random.seed(40)
    participants = ["P16"]#, "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    for p, part in enumerate(participants):
        print(f"Processing data augmentation excluding {part}...")
        training_path = f"Q:\Projet_hand_bike_markerless\RGBD\Training_data\{part}_excluded_normal_500_down"
        if os.path.exists(training_path):
            shutil.rmtree(training_path, ignore_errors=True)
        os.makedirs(training_path)
        images, markers, marker_names = get_label_image(participant_to_exclude=part)
        # images_aug, kps_aug = get_augmented_images(images, markers)
        # save_augmented_images(images_aug, kps_aug, training_path, marker_names)
        save_augmented_images(images, markers, training_path, marker_names)