import numpy as np
from scipy.optimize import minimize, minimize_scalar
import cv2

try:
    from skimage.feature import blob_log
except ModuleNotFoundError:
    pass
from .enums import *
import json
import math

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    pass


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)


def find_closest_node(point, node_list):
    closest_node = None
    smallest_distance = float("inf")
    for node in node_list:
        distance = math.sqrt((point[0] - node[0]) ** 2 + (point[1] - node[1]) ** 2)
        if distance < smallest_distance:
            closest_node = node
            smallest_distance = distance
    if closest_node:
        return closest_node[0], closest_node[1]
    else:
        return None, None


def find_closest_blob(center, blobs):
    """
    Find the closest blob to the center
    """
    delta = 10
    center = center.astype(int)
    cx, cy = find_closest_node(center, blobs)
    if cx and cy:
        if cx in range(center[0] - delta, center[0] + delta) and cy in range(center[1] - delta, center[1] + delta):
            final_centers = [cx, cy]
            is_visible = True
        else:
            final_centers = center
            is_visible = False
        return final_centers, is_visible
    else:
        return center, False


def RT(angleX):
    Rototrans = np.array([[np.cos(angleX), -np.sin(angleX)], [np.sin(angleX), np.cos(angleX)]])
    return Rototrans


def objective(x, from_pts, to_pts):
    center_from = np.mean(from_pts, axis=0)
    center_to = np.mean(to_pts, axis=0)
    from_pts = from_pts - center_from
    to_pts = to_pts - center_to
    rototrans = RT(x)
    J = 0
    from_pts_transf = np.zeros(from_pts.shape)
    for i in range(to_pts.shape[0]):
        from_pts_transf[i, :] = np.dot(rototrans, from_pts[i, :])

    for i in range(to_pts.shape[0]):
        closest_idx = closest_node(to_pts[i, :], from_pts_transf)
        J = J + sum((to_pts[i, :] - from_pts_transf[closest_idx, :]) ** 2)
    return J


def label_markers_from_model(centers_from_model, centers_from_blobs, marker_names):
    ordered_labels = label_point_set(centers_from_model, centers_from_blobs, labels=marker_names)
    return ordered_labels


def minimize_points_location(ref_points, target_points, print_stats=True):
    if not isinstance(target_points, np.ndarray):
        target_points = np.array(target_points)
    if not isinstance(ref_points, np.ndarray):
        ref_points = np.array(ref_points)
    sol = minimize_scalar(
        objective,
        method="Brent",
        args=(
            ref_points,
            target_points,
        ),
    )
    if print_stats:
        print(f"objective: {sol.fun}", f"success: {sol.success}")
    theta = sol.x
    rototrans = RT(theta).reshape(2, 2)

    result_points = np.zeros(target_points.shape)
    for i in range(ref_points.shape[0]):
        result_points[i, :] = np.dot(rototrans, ref_points[i, :] - np.mean(ref_points, axis=0)) + np.mean(
            target_points, axis=0
        )
    return result_points


def auto_label(labelized_points, points_to_label, true_labels):
    # labelized_points: points with label
    # points_to_label: points to label
    # returns: a list of labels for points_to_label
    if not isinstance(labelized_points, np.ndarray):
        labelized_points = np.array(labelized_points)
    if not isinstance(points_to_label, np.ndarray):
        points_to_label = np.array(points_to_label)
    labels = []
    for i in range(points_to_label.shape[0]):
        labels.append(true_labels[closest_node(labelized_points[i, :], points_to_label)])
    return labels


def label_point_set(ref_points, target_points, labels, print_stats=True):
    if not isinstance(target_points, np.ndarray):
        target_points = np.array(target_points)
    if not isinstance(ref_points, np.ndarray):
        ref_points = np.array(ref_points)
    result_set = minimize_points_location(ref_points, target_points, print_stats)
    return auto_label(result_set, target_points, labels)


def get_conf_data(conf_file):
    with open(conf_file, "r") as infile:
        data = json.load(infile)
    return data


def check_and_attribute_depth(pos_2d, depth_image, depth_scale=1):
    """
    Check if the depth is valid
    :param pos_2d: 2d position of the marker
    :param depth_image: depth image
    :param depth_scale: depth scale
    :return: depth value
    """
    delta = 10
    if depth_image[pos_2d[1], pos_2d[0]] < 0:
        pos = np.mean(depth_image[pos_2d[1] - delta : pos_2d[1] + delta,
                           pos_2d[0]-delta:pos_2d[0]+delta]) * depth_scale
        is_visible = False
    else:
        pos = depth_image[pos_2d[1], pos_2d[0]] * depth_scale
        is_visible = True
    return pos, is_visible


def distribute_pos_markers(pos_markers_dic: dict, depth_image: list):
    """
    Distribute the markers in a dictionary of markers
    :param pos_markers_dic: dictionary of markers
    :return: list of markers
    """
    markers = np.zeros((3, len(pos_markers_dic.keys())))
    occlusion = []
    c = 0
    for key in pos_markers_dic.keys():
        markers[:2, c] = np.array(pos_markers_dic[key][0], dtype=int)
        markers[2, c] = depth_image[c][pos_markers_dic[key][0][1], pos_markers_dic[key][0][0]]
        occlusion.append(pos_markers_dic[key][1])
        c += 1
    return markers, occlusion


def find_bounds_color(color_frame, depth, method, filter, depth_scale=1):
    """
    Find the bounds of the image
    """

    def nothing(x):
        pass

    if method == DetectionMethod.CV2Contours:
        cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("min threshold", "Trackbars", 30, 255, nothing)
        cv2.createTrackbar("max threshold", "Trackbars", 111, 255, nothing)
        cv2.createTrackbar("clipping distance in meters", "Trackbars", 14, 80, nothing)

    elif method == DetectionMethod.CV2Blobs:
        cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("min area", "Trackbars", 1, 255, nothing)
        cv2.createTrackbar("max area", "Trackbars", 500, 5000, nothing)
        cv2.createTrackbar("color", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("min threshold", "Trackbars", 30, 255, nothing)
        cv2.createTrackbar("max threshold", "Trackbars", 111, 255, nothing)

    elif method == DetectionMethod.SCIKITBlobs:
        cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("min area", "Trackbars", 1, 255, nothing)
        cv2.createTrackbar("max area", "Trackbars", 500, 5000, nothing)
        cv2.createTrackbar("color", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("min threshold", "Trackbars", 30, 255, nothing)
        cv2.createTrackbar("max threshold", "Trackbars", 111, 255, nothing)
    else:
        raise ValueError("Method not supported")
    import matplotlib.pyplot as plt

    plt.figure(num=None, figsize=(8, 6), dpi=80)
    color_frame_init = color_frame.copy()
    while True:
        if method == DetectionMethod.CV2Contours:
            min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
            max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
            clipping_distance = cv2.getTrackbarPos("clipping distance in meters", "Trackbars") / 10
            params = {
                "min_threshold": min_threshold,
                "max_threshold": max_threshold,
                "clipping_distance_in_meters": clipping_distance,
            }
            depth_image_3d = np.dstack((depth, depth, depth))
            color_frame = np.where(
                (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= 0), 155, color_frame_init
            )
            im_from, contours = get_blobs(
                color_frame, method=method, params=params, return_image=True, return_centers=True
            )
            draw_blobs(color_frame, contours)

        elif method == DetectionMethod.CV2Blobs:
            min_area = cv2.getTrackbarPos("min area", "Trackbars")
            max_area = cv2.getTrackbarPos("max area", "Trackbars")
            color = cv2.getTrackbarPos("color", "Trackbars")
            min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
            max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
            if min_area == 0:
                min_area = 1
            if max_area == 0:
                max_area = 1
            if max_area < min_area:
                max_area = min_area + 1
            hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
            # Set up the detector parameters
            params = cv2.SimpleBlobDetector_Params()
            # Filter by color
            params.filterByColor = True
            params.blobColor = color
            params.filterByArea = True
            params.minArea = min_area
            params.maxArea = max_area
            # Create the detector object

            detector = cv2.SimpleBlobDetector_create(params)
            im_from = hsv[:, :, 2]
            im_from = cv2.GaussianBlur(im_from, (5, 5), 0)
            im_from = cv2.inRange(im_from, min_threshold, max_threshold, im_from)
            result_mask = cv2.bitwise_and(color_frame, color_frame, mask=im_from)

            keypoints = detector.detect(im_from)

            result = cv2.drawKeypoints(
                color_frame, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            params = {
                "min_area": min_area,
                "max_area": max_area,
                "color": color,
                "min_threshold": min_threshold,
            }

        elif method == DetectionMethod.SCIKITBlobs:
            from skimage.feature import blob_log
            from math import sqrt

            min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
            max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
            params = {
                "max_sigma": 30,
                "min_sigma": 2,
                "threshold": 0.3,
                "min_threshold": min_threshold,
                "max_threshold": max_threshold,
            }
            im_from, blobs = get_blobs(color_frame, method, params, return_image=True)
            result = color_frame.copy()
            for blob in blobs:
                y, x, area = blob
                result = cv2.circle(result, (int(x), int(y)), int(area), (0, 0, 255), 1)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", color_frame)
        cv2.namedWindow("im_from", cv2.WINDOW_NORMAL)
        cv2.imshow("im_from", im_from)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    return params


def calculate_contour_distance(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1 / 2
    c_y1 = y1 + h1 / 2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2 / 2
    c_y2 = y2 + h2 / 2

    return max(abs(c_x1 - c_x2) - (w1 + w2) / 2, abs(c_y1 - c_y2) - (h1 + h2) / 2)


def merge_cluster(blobs, threshold_distance=5.0):
    current_contours = blobs
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours) - 1):
            for y in range(x + 1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = np.concatenate((current_contours[index1], current_contours[index2]), axis=0)
            del current_contours[index2]
        else:
            break
    return current_contours


def get_blobs(
    frame,
    params,
    method=DetectionMethod.CV2Contours,
    return_image=False,
    area_bounds: tuple = None,
    return_centers=False,
    image_bounds=None,
):
    if not area_bounds:
        area_bounds = (5, 60)
    if image_bounds:
        bounded_frame = frame.copy()[image_bounds[2] : image_bounds[3], image_bounds[0] : image_bounds[1]]
    else:
        bounded_frame = frame.copy()
    if method == DetectionMethod.SCIKITBlobs:
        hsv = cv2.cvtColor(bounded_frame, cv2.COLOR_BGR2HSV)
        im_from = hsv[:, :, 2]
        im_from = cv2.GaussianBlur(im_from, (5, 5), 0)
        im_from = cv2.inRange(im_from, params["min_threshold"], params["max_threshold"], im_from)
        blobs = blob_log(im_from, max_sigma=area_bounds[1], min_sigma=area_bounds[0], threshold=0.3)
        if return_image:
            return im_from, blobs
        else:
            return blobs

    if method == DetectionMethod.CV2Contours:
        hsv = cv2.cvtColor(bounded_frame, cv2.COLOR_BGR2HSV)
        im_from = hsv[:, :, 2]
        im_from = cv2.GaussianBlur(im_from, (5, 5), 0)
        im_from = cv2.inRange(im_from, params["min_threshold"], params["max_threshold"], im_from)
        contours, _ = cv2.findContours(image=im_from, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = merge_cluster(list(contours), threshold_distance=5)
        centers = []
        if return_centers:
            if len(contours) != 0:
                for c in contours:
                    if area_bounds[0] < cv2.contourArea(c) < area_bounds[1]:
                        M = cv2.moments(c)
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if image_bounds:
                            cx = cx + image_bounds[0]
                            cy = cy + image_bounds[2]
                        centers.append((cx, cy))

        if return_image:
            if return_centers:
                return im_from, centers
            else:
                return im_from, contours
        elif return_centers:
            return centers
        else:
            return contours


def set_conf_file_from_camera(pipeline, device):
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    d_profile = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
    d_intr = d_profile.get_intrinsics()
    scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    c_profile = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
    c_intr = c_profile.get_intrinsics()
    deth_to_color = d_profile.get_extrinsics_to(c_profile)
    r = np.array(deth_to_color.rotation).reshape(3, 3)
    t = np.array(deth_to_color.translation)
    dic = {
        "camera_name": device_product_line,
        "depth_scale": scale,
        "depth_fx_fy": [d_intr.fx, d_intr.fy],
        "depth_ppx_ppy": [d_intr.ppx, d_intr.ppy],
        "color_fx_fy": [c_intr.fx, c_intr.fy],
        "color_ppx_ppy": [c_intr.ppx, c_intr.ppy],
        "depth_to_color_trans": t.tolist(),
        "depth_to_color_rot": r.tolist(),
        "model_color": c_intr.model.name,
        "model_depth": d_intr.model.name,
        "dist_coeffs_color": c_intr.coeffs,
        "dist_coeffs_depth": d_intr.coeffs,
        "size_color": [c_intr.width, c_intr.height],
        "size_depth": [d_intr.width, d_intr.height],
        "color_rate": c_profile.fps(),
        "depth_rate": d_profile.fps(),
    }

    with open("camera_conf.json", "w") as outfile:
        json.dump(dic, outfile, indent=4)


def draw_blobs(frame, blobs, color=(255, 0, 0)):
    if blobs is not None:
        for blob in blobs:
            frame = cv2.circle(frame, (int(blob[0]), int(blob[1])), 5, color, 1)
    return frame


def draw_markers(frame,
                 markers_pos,
                 markers_filtered_pos=None,
                 markers_names=None,
                 is_visible=None,
                 scaling_factor=1.0):
    x, y = None, None
    if markers_pos is not None:
        for i in range(markers_pos.shape[1]):
            if markers_pos[0, i] and markers_pos[1, i]:
                color = (0, 255, 0) if bool(is_visible[i]) else (0, 0, 255)
                x, y = int(markers_pos[0, i]), int(markers_pos[1, i])
                frame = cv2.circle(frame, (int(markers_pos[0, i]), int(markers_pos[1, i])), 5, color, 1)
            else:
                if markers_filtered_pos is not None:
                    if markers_filtered_pos[0, i] and markers_filtered_pos[1, i]:
                        color = (0, 0, 255)
                        x, y = int(markers_filtered_pos[0, i]), int(markers_filtered_pos[1, i])
                        frame = cv2.circle(
                            frame, (int(markers_filtered_pos[0, i]), int(markers_filtered_pos[1, i])), 5, color, 1
                        )
            if markers_names:
                if x and y:
                    if markers_pos[2, i]:
                        frame = cv2.putText(
                            frame,
                            str(np.round(markers_pos[2, i]*100, 2)),
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scaling_factor,
                            (0, 255, 0),
                            1,
                        )
    return frame


def bounding_rect(frame, points, color=(255, 0, 0), delta=10):
    point_x = [item for item in points[0] if item is not None]
    point_y = [item for item in points[1] if item is not None]
    x_min, x_max = np.clip(min(point_x) - delta, 0, None).astype(int), np.clip(max(point_x) + delta, None, frame.shape[0]).astype(int)
    y_min, y_max = np.clip(min(point_y) - delta, 0, None).astype(int), np.clip(max(point_y) + delta, None, frame.shape[1]).astype(int)
    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1)
    return frame, (x_min, x_max, y_min, y_max)


def check_filtered_or_true_pos(pos, filtered, occlusions):
    occlusions_idx = np.where(occlusions)[0]
    merged_pose = pos.copy()
    for idx in range(merged_pose.shape[1]):
        if idx not in occlusions_idx:
            merged_pose[:, idx] = filtered[:, idx]
    return merged_pose


