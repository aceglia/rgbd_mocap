import numpy as np
import math
import matplotlib.pyplot    as     plt
from biosiglive import load, save
import cv2
import pyrealsense2 as rs
from mpl_toolkits.mplot3d import Axes3D
from rgbd_mocap.camera.camera_converter import CameraConverter
import json
import time
import os


def get_normal_vector(data, name, normal_vector):
    marker = data[1][:, data[0].index(name), :]
    M1 = marker[:, 0]
    M2 = marker[:, 20]
    M3 = marker[:, 40]
    first_axis_vector = M3 - M1
    second_axis_vector = M2 - M1
    third_axis_vector = np.cross(first_axis_vector, second_axis_vector, axis=0)[:, None]
    third_axis_vector = np.repeat(third_axis_vector, data[1].shape[2], axis=1)
    rt = np.zeros((4, 4, data[1].shape[2]))
    vertical_vector = data[1][:, data[0].index("ster"), :] - data[1][:, data[0].index("xiph"), :]
    antero_vector = np.cross(vertical_vector, third_axis_vector, axis=0)
    medial_vector = -np.cross(vertical_vector, antero_vector, axis=0)
    rt[:3, 0, :] = medial_vector / np.linalg.norm(medial_vector, axis=0)
    rt[:3, 1, :] = antero_vector / np.linalg.norm(antero_vector, axis=0)
    rt[:3, 2, :] = vertical_vector / np.linalg.norm(vertical_vector, axis=0)
    rt[:3, 3, :] = data[1][:, data[0].index("xiph"), :]
    rt[3, 3, :] = 1
    return rt

def get_measure(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["ster_C7"] * 0.001


def add_virtual_markers(data, rt, remove_marker=None, measurement_file=None):
    dist = np.linalg.norm(data[1][:, data[0].index("xiph"), :] - data[1][:, data[0].index("ster"), :], axis=0)
    marker_tec_2 = np.repeat(np.array([0, -0.2, 0, 1])[:, None],data[1].shape[2],  axis = 1)
    l = get_measure(measurement_file) if measurement_file is not None else 0.145
    marker_tec_1 = np.zeros((4, data[1].shape[2]))
    marker_tec_1[2, :] = dist
    marker_tec_1[1, :] = -l
    marker_tec_1[3, :] = 1

    new_markers = np.ndarray((3, 2, data[1].shape[2]))
    for i in range(data[1].shape[2]):
        new_markers[:, 0, i] = np.dot(rt[:, :, i], marker_tec_1[:, i])[:3]
        new_markers[:, 1, i] = np.dot(rt[:, :, i], marker_tec_2[:, i])[:3]

    names = data[0]
    if remove_marker is not None:
        data[1] = np.delete(data[1], data[0].index(remove_marker), axis=1)
        names.pop(data[0].index(remove_marker))
    data_out = np.concatenate((data[1][:, :data[0].index("xiph") + 1, :],
                               new_markers, data[1][:, data[0].index("xiph") + 1:, :]), axis=1)
    names = names[:names.index("xiph") + 1] + ["marker_tec_1", "marker_tec_2"] + names[names.index("xiph") + 1:]


    data_out = (names, data_out)
    # plot(data_out)
    return data_out


def get_label_image(participant, camera):
    main_path = f"{prefix}/RGBD"
    files = os.listdir(f"{main_path}{os.sep}{participant}")
    # file_gear_5 = [file for file in files if "gear_5" in file and "less" not in file and "more" not in file]
    files = [file for file in files if "gear_5" in file and "less" not in file and "more" not in file]
    for file in files:
        tracking_config_path = (
            f"{main_path}{os.sep}{participant}{os.sep}" + file + f"{os.sep}tracking_config_gui_3_crops.json"
        )
        if not os.path.isfile(tracking_config_path):
            continue
        with open(tracking_config_path) as json_file:
            tracking_config = json.load(json_file)

        area = [0, 0, 0, 0]
        area[0] = min([tracking_config["crops"][i]["area"][0] for i in range(len(tracking_config["crops"]))]) - 50
        area[2] = max([tracking_config["crops"][i]["area"][2] for i in range(len(tracking_config["crops"]))]) + 50
        area = np.array(area)
        area = np.clip(area, 0, 848)
        area[1] = min([tracking_config["crops"][i]["area"][1] for i in range(len(tracking_config["crops"]))]) - 50
        area[3] = max([tracking_config["crops"][i]["area"][3] for i in range(len(tracking_config["crops"]))]) + 50
        area[[1, 3]] = np.clip(area[[1, 3]], 0, 460)
        path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
        if not os.path.isfile(path + "/marker_pos_multi_proc_3_crops_pp.bio"):
            continue
        print("getting data from ", file, "for participant ", participant, "...")
        markers_data = load(path + "/marker_pos_multi_proc_3_crops_normal_500_down_b1_ribs_and_cluster_1_with_model_pp_full.bio")
        markers = markers_data["markers_in_pixel"]
        frame_idx = markers_data["frame_idx"]
        occlusions = markers_data["occlusions"]
        marker_names = markers_data["markers_names"][:, 0].tolist()
        markers_in_meter = markers_data["markers_in_meters"]
        ax = None
        random_idx = None
        markers_in_meter_augmented = np.zeros((3, markers_in_meter.shape[1] + 1, markers_in_meter.shape[2]))
        markers_in_meter_augmented[:, :-1, :] = markers_in_meter
        markers_names_augmented = marker_names + ["technical_marker"]
        markers_names_augmented = np.repeat(np.array([markers_names_augmented])[:, None],
                  markers_in_meter_augmented.shape[2], axis=1)
        idx_ster = marker_names.index("ster")
        time_list = []
        for i in range(len(frame_idx)):
            depth = cv2.imread(path + f"/depth_{frame_idx[i]}.png", cv2.IMREAD_ANYDEPTH)
            tic = time.time()
            bb = np.array(
                [
                    [
                        int(markers[0, idx_ster, i]),
                        int(markers[1, idx_ster, i]),
                    ],
                    [
                        int(markers[0, idx_ster, i]),
                        int(markers[1, idx_ster, i]),
                    ],
                ]
            )

            bb[0, 0] = max(bb[0, 0] - 30, 0)
            bb[0, 1] = max(bb[0, 1] - 5, 0)
            bb[1, 0] = min(bb[1, 0] + 30, depth.shape[1])
            bb[1, 1] = min(bb[1, 1] + 35, depth.shape[0])
            # get all pixels within the bb
            pixels = np.argwhere(
                (depth[bb[0, 1] : bb[1, 1], bb[0, 0] : bb[1, 0]] > 0)
                & (depth[bb[0, 1] : bb[1, 1], bb[0, 0] : bb[1, 0]] < 10000)
            )

            # get 20 pixels randomly
            random_idx = np.random.choice(len(pixels), 15, replace=False)
            pixels = pixels[random_idx]
            # get pixel 2d coordinates
            pixels_2d = pixels + [bb[0, 1], bb[0, 0]]
            # get_depth values for the pixels
            depth_values = depth[pixels_2d[:, 0], pixels_2d[:, 1]][:, None] * camera.depth_scale
            pixels_3d = np.concatenate((pixels_2d, depth_values), axis=-1).tolist()
            # project to 3d
            points_in_meters = camera.get_markers_pos_in_meter(pixels_3d).T
            # plot 3d points
            rt = compute_normal(points_in_meters)
            normal = rt[2]
            ster = markers_in_meter[:, idx_ster, i]

            # new 3d point at a distance d from the ster markers along the normal direction
            d = 0.230
            new_point = ster + normal * d
            markers_in_meter_augmented[:, -1, i] = new_point
            time_list.append(time.time() - tic)

            # plot the new point
            # fig = plt.figure("3d")
            # ax = fig.add_subplot(111, projection='3d')
            # for j in range(markers_in_meter.shape[1]):
            #     ax.scatter(markers_in_meter[0, j, i], markers_in_meter[1, j, i], markers_in_meter[2, j, i], c="b")
            # ax.scatter(ster[0], ster[1], ster[2], c="r")
            # ax.scatter(new_point[0], new_point[1], new_point[2], c="g")
            # ax.quiver(ster[0], ster[1], ster[2], normal[0], normal[1], normal[2], color="r")
            # set_axes_equal(ax)
            # plt.show()

            # ax.plot_surface(xx, yy, z, alpha=0.2)
            # ax.scatter(pointsT[0], pointsT[1], pointsT[2])

            # normal_positive = np.median(normal) > 0
            # if not normal_positive:
            #     normal = -normal

            # ax = plot_points(points_in_meters, rt, ax, origin=markers_in_meter[:, idx_ster, i])
            # if i == 50:
            #     plt.show()
            if i!= 0 and i % 500 == 0:
                print(f"{i} iterations done for participant {participant}")
        new_path = path + "/marker_pos_multi_proc_3_crops_normal_500_down_b1_ribs_and_cluster_1_with_model_pp_full.bio"
        new_path = new_path.replace("pp_full", "pp_full_technical_marker")
        markers_data["markers_in_meters"] = markers_in_meter_augmented
        markers_data["markers_names"] = markers_names_augmented
        markers_data["time_to_add_technical_marker"] = time_list
        save(markers_data, new_path, safe=False)
        print(f"Data augmentation done for {file}")
    return


def compute_normal(points):
    centroid = points.mean(axis=0)
    points_mean = points - centroid
    u, sigma, v = np.linalg.svd(points_mean)
    eigen_vectors = v
    if np.mean(eigen_vectors[2]) < 0:
        eigen_vectors[2] = -eigen_vectors[2]
    return eigen_vectors


def plot_points(points, eigen_vectors, ax=None, origin=None):
    centroid = points.mean(axis=0)
    centroid = origin if origin is not None else centroid
    forGraphs = list()
    normal = eigen_vectors[2]
    axis_1 = eigen_vectors[0]
    axis_2 = eigen_vectors[1]
    forGraphs.append(np.array([centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2]]))
    forGraphs_1 = list()
    forGraphs_2 = list()
    for i in range(len(points)):
        forGraphs_1.append(np.array([centroid[0], centroid[1], centroid[2], axis_1[0], axis_1[1], axis_1[2]]))
        forGraphs_2.append(np.array([centroid[0], centroid[1], centroid[2], axis_2[0], axis_2[1], axis_2[2]]))

    # get d coefficient to plane for display
    d = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2]
    pointsT = points.T
    # create x,y for display
    minPlane = int(math.floor(min(min(pointsT[0]), min(pointsT[1]), min(pointsT[2]))))
    maxPlane = int(math.ceil(max(max(pointsT[0]), max(pointsT[1]), max(pointsT[2]))))
    xx, yy = np.meshgrid(range(minPlane, maxPlane), range(minPlane, maxPlane))

    # calculate corresponding z for display
    z = (-normal[0] * xx - normal[1] * yy + d) * 1. / normal[2]

    # matplotlib display code
    forGraphs = np.asarray(forGraphs)
    X, Y, Z, U, V, W = zip(*forGraphs)
    forGraphs_1 = np.asarray(forGraphs_1)
    X1, Y1, Z1, U1, V1, W1 = zip(*forGraphs_1)
    forGraphs_2 = np.asarray(forGraphs_2)
    X2, Y2, Z2, U2, V2, W2 = zip(*forGraphs_2)
    if not ax:
        fig = plt.figure("normals")
        ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, yy, z, alpha=0.2)
    # ax.scatter(pointsT[0], pointsT[1], pointsT[2])
    ax.quiver(X, Y, Z, U, V, W)
    ax.quiver(X1, Y1, Z1, U1, V1, W1, color="r")
    ax.quiver(X2, Y2, Z2, U2, V2, W2, color="g")
    ax.set_xlim([min(pointsT[0]) - 0.1, max(pointsT[0]) + 0.1])
    ax.set_ylim([min(pointsT[1]) - 0.1, max(pointsT[1]) + 0.1])
    ax.set_zlim([min(pointsT[2]) - 0.1, max(pointsT[2]) + 0.1])
    set_axes_equal(ax)
    # plt.show()
    return ax

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

if __name__ == '__main__':
    prefix = r"Q:\Projet_hand_bike_markerless" if os.name == "nt" else r"/mnt/shared/Projet_hand_bike_markerless"
    np.random.seed(40)
    participants = [f"P{i}" for i in range(9, 17)]
    for p, part in enumerate(participants):
        camera_config_path = f"{prefix}/RGBD/config_camera_files/config_camera_{part}.json"
        camera = CameraConverter()
        camera.set_intrinsics(camera_config_path)
        camera.set_extrinsics(camera_config_path)
        get_label_image(part, camera)
