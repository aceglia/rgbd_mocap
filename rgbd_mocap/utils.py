import numpy as np
from scipy.optimize import minimize_scalar
import cv2
import ezc3d

try:
    from skimage.feature import blob_log
except ModuleNotFoundError:
    pass
from .enums import *
import json
import math

try:
    import biorbd
except ModuleNotFoundError:
    pass
try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    pass

import functools

try:
    import haiku as hk
    import jax
    import jax.numpy as jnp
    import mediapy as media
    from tqdm import tqdm
    import tree
    from .tapnet import tapir_model
    from .tapnet.utils import transforms
    from .tapnet.utils import viz_utils
except:
    pass


def build_online_model_init(frames, query_points):
    """Initialize query features for the query points."""
    model = tapir_model.TAPIR(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)

    feature_grids = model.get_feature_grids(frames, is_training=False)
    query_features = model.get_query_features(
        frames,
        is_training=False,
        query_points=query_points,
        feature_grids=feature_grids,
    )
    return query_features


def build_online_model_predict(frames, query_features, causal_context):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=query_features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True,
    )
    causal_context = trajectories["causal_context"]
    del trajectories["causal_context"]
    return {k: v[-1] for k, v in trajectories.items()}, causal_context


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    pred_occ = jax.nn.sigmoid(occlusions)
    pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
    visibles = pred_occ < 0.5  # threshold
    return visibles


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


def construct_initial_causal_state(num_points, num_resolutions):
    value_shapes = {
        "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
    }
    fake_ret = {k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()}
    return [fake_ret] * num_resolutions * 4


def convert_select_points_to_query_points(frame, points):
    """Convert select points to query points.

    Args:
      points: [num_points, 2], [t, y, x]
    Returns:
      query_points: [num_points, 3], [t, y, x]
    """
    points = np.stack(points)
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
    query_points[:, 0] = frame
    query_points[:, 1] = points[:, 1]
    query_points[:, 2] = points[:, 0]
    return query_points


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)


def create_c3d_file(marker_pos, marker_names, save_path: str, fps=60):
    """
    Write data to a c3d file
    """
    c3d = ezc3d.c3d()
    # Fill it with random data
    c3d["parameters"]["POINT"]["RATE"]["value"] = [fps]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_names
    c3d["data"]["points"] = marker_pos

    # Write the data
    c3d.write(save_path)


def start_idx_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data["start_frame"]


def find_closest_node(point, node_list):
    closest_node = None
    smallest_distance = float("inf")
    for node in node_list:
        distance = math.sqrt((node[0] - point[0]) ** 2 + (node[1] - point[1]) ** 2)
        if distance < smallest_distance:
            closest_node = node
            smallest_distance = distance
    if closest_node is not None:
        return closest_node[0], closest_node[1], smallest_distance
    else:
        return None, None, None


def find_closest_node_3d(point, node_list):
    closest_node = None
    smallest_distance = float("inf")
    idx = 0
    for n, node in enumerate(node_list):
        distance = math.sqrt((node[0] - point[0]) ** 2 + (node[1] - point[1]) ** 2 + (node[2] - point[2]) ** 2)
        if distance < smallest_distance:
            closest_node = node
            smallest_distance = distance
            idx = n
    if closest_node is not None:
        return closest_node[0], closest_node[1], closest_node[2], smallest_distance, idx
    else:
        return None, None, None


def find_closest_blob(center, blobs, delta=10, return_distance=False):
    """
    Find the closest blob to the center
    """
    delta = delta
    # delta = 15
    center = np.array(center)
    if len(center.shape) == 2:
        center = center[:, 0]
    cx, cy, distance = find_closest_node(center, blobs)
    if cx and cy:
        if distance <= delta:
            # if center[0] - delta <= cx <= center[0] + delta and center[1] - delta <= cy <= center[1] + delta:
            final_centers = np.array([cx, cy])
            is_visible = True
        else:
            final_centers = center
            is_visible = False
        if return_distance:
            return final_centers, is_visible, distance
        else:
            return final_centers, is_visible
    else:
        if return_distance:
            return center, False, -1
        else:
            return center, False


def find_closest_blob_3D(center, blobs, delta=10, return_distance=False):
    """
    Find the closest blob to the center
    """
    delta = delta
    # delta = 15
    center = np.array(center)
    if len(center.shape) == 2:
        center = center[:, 0]
    cx, cy, cz, distance, idx = find_closest_node_3d(center, blobs)
    if cx and cy and cz:
        if distance <= delta:
            # if center[0] - delta <= cx <= center[0] + delta and center[1] - delta <= cy <= center[1] + delta:
            final_centers = np.array([cx, cy, cz])
            is_visible = True
        else:
            final_centers = center
            is_visible = False
        if return_distance:
            return final_centers, is_visible, distance, idx
        else:
            return final_centers, is_visible, idx
    else:
        if return_distance:
            return center, False, -1
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
    sub = np.mean(ref_points, axis=0)
    result_points = np.zeros(target_points.shape)
    t = np.mean(target_points, axis=0)
    for i in range(ref_points.shape[0]):
        result_points[i, :] = np.dot(rototrans, ref_points[i, :] - sub) + t
    return result_points, rototrans, t, sub


def auto_label(labelized_points, points_to_label, true_labels):
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
    result_set, rt = minimize_points_location(ref_points, target_points, print_stats)
    return result_set, auto_label(result_set, target_points, labels), rt


def get_conf_data(conf_file):
    with open(conf_file, "r") as infile:
        data = json.load(infile)
    return data


def check_and_attribute_depth(pos_2d, depth_image, depth_scale=0.01):
    """
    Check if the depth is valid
    :param pos_2d: 2d position of the marker
    :param depth_image: depth image
    :param depth_scale: depth scale
    :return: depth value
    """
    delta = 8
    if isinstance(pos_2d, list):
        pos_2d = np.array(pos_2d)
    pos_2d = pos_2d.astype(int)
    if depth_image[pos_2d[1], pos_2d[0]] <= 0:
        pos = (
            np.median(depth_image[pos_2d[1] - delta : pos_2d[1] + delta, pos_2d[0] - delta : pos_2d[0] + delta])
            * depth_scale
        )
        if not np.isfinite(pos):
            pos = -1
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


def find_closest_markers_in_model(marker_to_check, markers_model, same_idx, model):
    center, _, idx = find_closest_blob_3D(marker_to_check, markers_model[:, :, 0].T, delta=5)
    return idx


def get_blobs(
    frame,
    params,
    method=DetectionMethod.CV2Contours,
    return_image=False,
    return_centers=False,
    image_bounds=None,
    threshold_distance=2,
    depth=None,
    clipping_color=None,
    depth_scale=None,
):
    if image_bounds:
        bounded_frame = frame.copy()
        bounded_frame = bounded_frame[image_bounds[2] : image_bounds[3], image_bounds[0] : image_bounds[1]]
        bounded_depth = depth.copy()
        bounded_depth = bounded_depth[image_bounds[2] : image_bounds[3], image_bounds[0] : image_bounds[1]]
    else:
        bounded_frame = frame.copy()
        image_bounds = (0, frame.shape[1], 0, frame.shape[0])
        bounded_depth = depth.copy()
    im_from_init = bounded_frame.copy()
    centers = []
    blobs = []
    im_from = None
    for i in range(1):
        im_from = im_from_init
        if "min_dist" not in params.keys():
            params["min_dist"] = 0
        if params["use_bg_remover"]:
            im_from = background_remover(
                im_from_init,
                bounded_depth,
                params["clipping_distance_in_meters"],
                depth_scale,
                clipping_color,
                params["min_dist"],
                params["use_contour"],
            )
        if im_from is None:
            continue
        try:
            im_from = cv2.cvtColor(im_from, cv2.COLOR_RGB2GRAY)
        except cv2.error:
            continue
        if "blur" not in params.keys():
            params["blur"] = 5
        clahe = cv2.createCLAHE(
            clipLimit=params["clahe_clip_limit"], tileGridSize=(params["clahe_autre"], params["clahe_autre"])
        )
        im_from = clahe.apply(im_from)
        im_from = cv2.GaussianBlur(im_from, (params["blur"], params["blur"]), 0)
        if method == DetectionMethod.SCIKITBlobs:
            raise RuntimeError("Method not implemented")
            # blobs = blob_log(im_from, max_sigma=area_bounds[1], min_sigma=area_bounds[0], threshold=0.3)

        if method == DetectionMethod.CV2Contours:
            # im_from = im_from_init.copy()
            # im_from = cv2.cvtColor(im_from, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(
                clipLimit=params["clahe_clip_limit"], tileGridSize=(params["clahe_autre"], params["clahe_autre"])
            )
            im_from = clahe.apply(im_from)
            contours, _ = cv2.findContours(image=im_from, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
            # contours = merge_cluster(list(contours), threshold_distance=3)
            # circles = cv2.HoughCircles(im_from, cv2.HOUGH_GRADIENT, dp=1.5,
            #                         minDist=10, param1=220, param2=50, maxRadius=30)
            # centers = []
            # if circles is not None:
            #     circles = np.uint16(np.around(circles))
            #     for i in circles[0, :]:
            #         # Draw outer circle
            #         cx, cy = i[0], i[1]
            #         if image_bounds:
            #             cx = cx + image_bounds[0]
            #             cy = cy + image_bounds[2]
            #         centers.append((cx, cy))
            #         # cv2.circle(im_from, (i[0], i[1]), i[2], (255, 0, 0), 2)
            #     blobs.append(circles)
            #
            if return_centers:
                if len(contours) != 0:
                    for c in contours:
                        M = cv2.moments(c)
                        print(c)
                        if M["m00"] == 0:
                            pass
                        else:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            if image_bounds:
                                cx = cx + image_bounds[0]
                                cy = cy + image_bounds[2]
                            centers.append((cx, cy))
                        blobs.append(c)

        if method == DetectionMethod.CV2Blobs:
            params_detector = cv2.SimpleBlobDetector_Params()
            params_detector.minThreshold = params["min_threshold"]
            params_detector.maxThreshold = params["max_threshold"]
            params_detector.filterByColor = True
            params_detector.blobColor = params["blob_color"]
            params_detector.minDistBetweenBlobs = 1
            params_detector.filterByArea = True
            params_detector.minArea = params["min_area"]
            params_detector.maxArea = params["max_area"]
            params_detector.filterByCircularity = True
            params_detector.minCircularity = params["circularity"]
            params_detector.filterByConvexity = True
            params_detector.minConvexity = params["convexity"]
            params_detector.filterByInertia = False
            # params_detector.minInertiaRatio = 0.1
            # Create the detector object
            detector = cv2.SimpleBlobDetector_create(params_detector)
            keypoints = detector.detect(im_from)
            for blob in keypoints:
                blobs.append(blob)
                if return_centers:
                    centers.append((int(blob.pt[0] + image_bounds[0]), int(blob.pt[1] + image_bounds[2])))
    if return_image:
        if return_centers:
            return im_from, centers
        else:
            return im_from, blobs
    elif return_centers:
        return centers
    else:
        return blobs


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


def background_remover(frame, depth, clipping_distance, depth_scale, clipping_color, min_dist=0, use_contour=True):
    depth_image_3d = np.dstack((depth, depth, depth))
    if use_contour:
        white_frame = np.ones_like(frame) * 255
        im_for_mask = np.where(
            (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= min_dist / depth_scale),
            clipping_color,
            white_frame,
        )
        gray = cv2.cvtColor(im_for_mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            c = max(contours, key=cv2.contourArea)
            mask = np.ones_like(frame) * clipping_color
            cv2.drawContours(mask, [c], contourIdx=-1, color=(255, 255, 255), thickness=-1)
            final = np.where(mask == (255, 255, 255), frame, clipping_color)
        except ValueError:
            final = np.where(
                (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= min_dist / depth_scale),
                clipping_color,
                frame,
            )
        # mask = np.ones_like(frame) * clipping_color
        # cv2.drawContours(mask, [c], contourIdx=-1, color=(255, 255, 255), thickness=-1)
        # final = np.where(mask == (255, 255, 255), frame, clipping_color)
    else:
        final = np.where(
            (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= min_dist / depth_scale),
            clipping_color,
            frame,
        )
    return final


def draw_blobs(frame, blobs, color=(255, 0, 0), scale=5):
    if blobs is not None:
        for blob in blobs:
            frame = cv2.circle(frame, (int(blob[0]), int(blob[1])), scale, color, 1)
    return frame


def draw_markers(
    frame,
    markers_pos,
    markers_filtered_pos=None,
    markers_names=None,
    is_visible=None,
    scaling_factor=1.0,
    circle_scaling_factor=5,
    markers_reliability_index=None,
    thickness=1,
    color=None,
):
    frame = frame.copy()
    try:
        markers_pos.shape[1]
    except IndexError:
        pass
    is_visible = np.ones(markers_pos.shape[1], dtype=bool) if is_visible is None else is_visible
    if markers_pos is not None:
        for i in range(markers_pos.shape[1]):
            x, y = None, None
            if np.isfinite(markers_pos[0, i]) and np.isfinite(markers_pos[1, i]):
                if not color:
                    color_tmp = (0, 255, 0) if bool(is_visible[i]) else (0, 0, 255)
                else:
                    color_tmp = color
                x, y = int(markers_pos[0, i]), int(markers_pos[1, i])
                frame = cv2.circle(
                    frame,
                    (int(markers_pos[0, i]), int(markers_pos[1, i])),
                    int(circle_scaling_factor),
                    color_tmp,
                    thickness,
                )

            # else:
            #     if markers_filtered_pos is not None:
            #         if np.isfinite(markers_filtered_pos[0, i]) and np.isfinite(markers_filtered_pos[1, i]):
            #             color = (0, 0, 255)
            #             x, y = int(markers_filtered_pos[0, i]), int(markers_filtered_pos[1, i])
            #             frame = cv2.circle(
            #                 frame, (int(markers_filtered_pos[0, i]), int(markers_filtered_pos[1, i])), 5, color, 1
            #             )
            if markers_names:
                if x and y:
                    frame = cv2.putText(
                        frame,
                        str(markers_names[i]),
                        (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scaling_factor,
                        color_tmp,
                        1,
                    )
            if markers_reliability_index is not None:
                if x and y:
                    frame = cv2.putText(
                        frame,
                        str(markers_reliability_index[i]),
                        (x - 30, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scaling_factor,
                        color_tmp,
                        1,
                    )
    return frame


def ortho_norm_basis(vector, idx):
    # build an orthogonal basis fom a vector
    basis = []
    v = np.random.random(3)
    vector_norm = vector / np.linalg.norm(vector)
    z = np.cross(v, vector_norm)
    z_norm = z / np.linalg.norm(z)
    y = np.cross(vector_norm, z)
    y_norm = y / np.linalg.norm(y)
    if idx == 0:
        basis = np.append(vector_norm, np.append(y_norm, z_norm)).reshape(3, 3).T
        if np.linalg.det(basis) < 0:
            basis = np.append(vector_norm, np.append(y_norm, -z_norm)).reshape(3, 3).T
    elif idx == 1:
        basis = np.append(y_norm, np.append(vector_norm, z_norm)).reshape(3, 3).T
        if np.linalg.det(basis) < 0:
            basis = np.append(y_norm, np.append(vector_norm, -z_norm)).reshape(3, 3).T
    elif idx == 2:
        basis = np.append(z_norm, np.append(y_norm, vector_norm)).reshape(3, 3).T
        if np.linalg.det(basis) < 0:
            basis = np.append(-z_norm, np.append(y_norm, vector_norm)).reshape(3, 3).T
    return basis


def bounding_rect(frame, points, color=(255, 0, 0), delta=10):
    point_x = [item for item in points[0] if item is not None]
    point_y = [item for item in points[1] if item is not None]
    min_x, max_x = int(min(point_x) - delta), int(max(point_x) + delta)
    min_y, max_y = int(min(point_y) - delta), int(max(point_y) + delta)
    x_min, x_max = np.clip(min_x, 0, frame.shape[1]), np.clip(max_x, 0, frame.shape[1])
    y_min, y_max = np.clip(min_y, 0, frame.shape[0]), np.clip(max_y, 0, frame.shape[0])
    # x_min, x_max, y_min, y_max = 0, frame.shape[1], 0, frame.shape[0]
    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1)
    return frame, (int(x_min), int(x_max), int(y_min), int(y_max))


def check_filtered_or_true_pos(pos, filtered, occlusions):
    occlusions_idx = np.where(occlusions)[0]
    merged_pose = pos.copy()
    for idx in range(merged_pose.shape[1]):
        if idx not in occlusions_idx:
            merged_pose[:, idx] = filtered[:, idx]
    return merged_pose


def rotate_frame(color, depth, rotation):
    if rotation == 90 or rotation == Rotation.ROTATE_90:
        color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180 or rotation == Rotation.ROTATE_180:
        color = cv2.rotate(color, cv2.ROTATE_180)
        depth = cv2.rotate(depth, cv2.ROTATE_180)
    elif rotation == 270 or rotation == Rotation.ROTATE_270:
        color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation != 0 and rotation != Rotation.ROTATE_0:
        raise ValueError("Rotation value not supported")
    return color, depth
