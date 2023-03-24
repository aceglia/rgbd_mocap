import numpy as np
from scipy.optimize import minimize, minimize_scalar
import cv2
from .enums import *


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def list_authorized_color_resolution(camera_type: str = "D455"):
    if camera_type == "D455":
        return [(424, 240),
                (480, 270),
                (640, 360),
                (640, 480),
                (848, 480),
                (1280, 720),
                (1280, 800)
                ]
    else:
        raise ValueError("Camera type not supported")


def list_authorized_depth_resolution(camera_type: str = "D455"):
    if camera_type == "D455":
        return [(256, 144),
                (424, 240),
                (480, 270),
                (640, 360),
                (640, 400),
                (640, 480),
                (848, 100),
                (848, 480),
                (1280, 720),
                (1280, 800)
                ]
    else:
        raise ValueError("Camera type not supported")


def list_authorized_fps(camera_type: str = "D455"):
    if camera_type == "D455":
        return [5, 15, 25, 30, 60, 90, 100]
    else:
        raise ValueError("Camera type not supported")


def RT(angleX):
    Rototrans = np.array([[
        np.cos(angleX), -np.sin(angleX)],
        [np.sin(angleX), np.cos(angleX)]]
    )
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
        J = J + sum((to_pts[i, :] - from_pts_transf[closest_idx, :])**2)
    return J


def minimize_points_location(ref_points, target_points, print_stats=True):
    if not isinstance(target_points, np.ndarray):
        target_points = np.array(target_points)
    if not isinstance(ref_points, np.ndarray):
        ref_points = np.array(ref_points)
    sol = minimize_scalar(objective, method="Brent", args=(ref_points, target_points,),)
    if print_stats:
        print(f"objective: {sol.fun}", f"success: {sol.success}")
    theta = sol.x
    rototrans = RT(theta).reshape(2, 2)

    result_points = np.zeros(target_points.shape)
    for i in range(ref_points.shape[0]):
        result_points[i, :] = np.dot(rototrans, ref_points[i, :] - np.mean(ref_points, axis=0)) + np.mean(target_points, axis=0)
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
    # ref_points: points with label
    # target_points: points to label
    # labels: labels of ref_points
    # returns: a list of labels for target_points
    if not isinstance(target_points, np.ndarray):
        target_points = np.array(target_points)
    if not isinstance(ref_points, np.ndarray):
        ref_points = np.array(ref_points)
    result_set = minimize_points_location(ref_points, target_points, print_stats)
    return auto_label(result_set, target_points, labels)


def find_bounds_color(frame, method, filter):
    """
    Find the bounds of the image
    """
    def nothing(x):
        pass

    if method == DetectionMethod.CV2Contours:
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("min threshold", "Trackbars", 30, 255, nothing)
        cv2.createTrackbar("max threshold", "Trackbars", 111, 255, nothing)

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

    while True:
        if method == DetectionMethod.CV2Contours:
            min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
            max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
            params = {"min_threshold": min_threshold, "max_threshold": max_threshold}
            im_from, contours = get_blobs(frame, method, params, return_image=True)
            result = frame.copy()
            for c, contour in enumerate(contours):
                if cv2.contourArea(contour) > 5 and cv2.contourArea(contour) < 50:
                    M = cv2.moments(contour)
                    # cX = int(M["m10"] / M["m00"]) + area_x[0]
                    # cY = int(M["m01"] / M["m00"]) + area_y[0]
                    cv2.drawContours(image=result, contours=contour, contourIdx=-1, color=(255, 0, 0), thickness=2,
                                     lineType=cv2.LINE_AA)

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
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
            result_mask = cv2.bitwise_and(frame, frame, mask=im_from)

            tic = time.time()
            keypoints = detector.detect(im_from)
            print(time.time() - tic)

            result = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 255, 0),
                                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            params = {"min_area": min_area, "max_area": max_area, "color": color, "min_threshold": min_threshold,}
        elif method == DetectionMethod.SCIKITBlobs:
            from skimage.feature import blob_log
            from math import sqrt
            min_threshold = cv2.getTrackbarPos("min threshold", "Trackbars")
            max_threshold = cv2.getTrackbarPos("max threshold", "Trackbars")
            params = {"max_sigma": 30,
                    "min_sigma": 2,
                    "threshold": 0.3,
                      "min_threshold": min_threshold,
                        "max_threshold": max_threshold
                      }
            im_from, blobs = get_blobs(frame, method, params, return_image=True)
            result = frame.copy()
            for blob in blobs:
                y, x, area = blob
                result = cv2.circle(result, (int(x), int(y)), int(area), (0, 0, 255), 1)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", result)
        cv2.namedWindow("im_from", cv2.WINDOW_NORMAL)
        cv2.imshow("im_from", im_from)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    return params

import time
def get_blobs(frame, method, params, return_image=False):
    if method == DetectionMethod.SCIKITBlobs:
        from skimage.feature import blob_log
        from math import sqrt
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        im_from = hsv[:, :, 2]
        im_from = cv2.GaussianBlur(im_from, (5, 5), 0)
        im_from = cv2.inRange(im_from, params["min_threshold"], params["max_threshold"], im_from)
        blobs = blob_log(im_from, max_sigma=30, min_sigma=2, threshold=0.3)
        if return_image:
            return im_from, blobs
        else:
            return blobs

    if method == DetectionMethod.CV2Contours:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        im_from = hsv[:, :, 2]
        im_from = cv2.GaussianBlur(im_from, (5, 5), 0)
        im_from = cv2.inRange(im_from, params["min_threshold"], params["max_threshold"], im_from)
        contours, _ = cv2.findContours(image=im_from, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        if return_image:
            return im_from, contours
        else:
            return contours
