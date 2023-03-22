import numpy as np
from scipy.optimize import minimize, minimize_scalar
import cv2


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


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


def find_bounds_color(frame, depth):
    """
    Find the bounds of the image
    """
    def nothing(x):
        pass

    import pyrealsense2 as rs
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.namedWindow("Trackbars_u")
    cv2.createTrackbar("U - H", "Trackbars_u", 0, 255, nothing)
    cv2.createTrackbar("U - S", "Trackbars_u", 0, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars_u", 0, 255, nothing)
    hsv_low = np.zeros((250, 500, 3), np.uint8)
    hsv_high = np.zeros((250, 500, 3), np.uint8)
    depth_scale = 0.0010000000474974513
    while True:
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars_u")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars_u")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars_u")

        hsv_low[:] = (l_h, l_s, l_v)
        hsv_high[:] = (u_h, u_s, u_v)
        cv2.imshow("Trackbars_u", hsv_high)
        cv2.imshow("Trackbars", hsv_low)
        # lower = np.array([l_h, l_s, l_v])
        # upper = np.array([u_h, u_s, u_v])
        lower = np.array([l_h])
        upper = np.array([u_h])
        clipping_distance_in_meters = 1.4  # 1 meter
        clip_distance = clipping_distance_in_meters / depth_scale
        clipping_distance_in_meters_min = 0.5  # 1 meter
        clip_distance_min = clipping_distance_in_meters_min / depth_scale
        grey_color = 153
        #aligned_depth = depth_rgb_registration(depth, frame)
        d_intr = [429.627, 429.627, 422.179, 243.999]
        c_intr = [418.856, 418.447, 418.145, 245.17]
        d_to_c = [[0.99999577, - 0.00241159, - 0.00164173, -0.05902871],
                  [0.00239989, 0.99997199, - 0.00709178, 0.00020985],
                  [0.00165878, 0.00708781, 0.99997348, 0.00028736],
                  [0, 0, 0, 1]]
        intrinsics_depth = np.array([[d_intr[0], 0, d_intr[2]],
                                     [0, d_intr[1], d_intr[3]],
                                     [0, 0, 1]], dtype=float)
        intrinsics_color = np.array([[c_intr[0], 0, c_intr[2]],
                                     [0, c_intr[1], c_intr[3]],
                                     [0, 0, 1]], dtype=float)
        depth = cv2.rgbd.registerDepth(intrinsics_depth, intrinsics_color, None, np.array(d_to_c, dtype=float), depth, (frame.shape[1], frame.shape[0]), True)
        depth_image_3d = np.dstack((depth,depth,depth)) #depth image is 1 channel, color is 3 channels
        frame = np.where((depth_image_3d > clip_distance) | (depth_image_3d <= clip_distance_min), grey_color, frame)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_RAINBOW)
        images = np.hstack((frame, depth_colormap))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        mask = cv2.inRange(h, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        imgray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        # imgray = clahe.apply(imgray)
        contours, hierarchy = cv2.findContours(image=imgray,
                                               mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        for c, contour in enumerate(contours):
            if cv2.contourArea(contour) > 5 and cv2.contourArea(contour) < 1000:
                M = cv2.moments(contour)
                # cX = int(M["m10"] / M["m00"]) + area_x[0]
                # cY = int(M["m01"] / M["m00"]) + area_y[0]
                cv2.drawContours(image=result, contours=contour, contourIdx=-1, color=(255, 0, 0), thickness=2,
                                     lineType=cv2.LINE_AA)

        # params = cv2.SimpleBlobDetector_Params()
        # detector = cv2.SimpleBlobDetector_create(params)
        # imgray_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # keypoints = detector.detect(imgray_scaled)
        # blobs = cv2.drawKeypoints(imgray_scaled, keypoints, np.array([]), (0, 0, 255),
        #                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        alpha = 0.6
        frame_bis = frame.copy()
        cv2.addWeighted(depth_colormap, alpha, frame_bis, 1 - alpha,
                        0, frame_bis)

        cv2.namedWindow(f'mask', cv2.WINDOW_NORMAL)
        cv2.imshow(f'mask', result)
        cv2.namedWindow(f'hsv', cv2.WINDOW_NORMAL)
        cv2.imshow(f'hsv', h)
        cv2.namedWindow(f'depth', cv2.WINDOW_NORMAL)
        cv2.imshow(f'depth', images)
        cv2.waitKey(0)
        # cv2.namedWindow(f'blob', cv2.WINDOW_NORMAL)
        # cv2.imshow(f'blob', imgray)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return lower, upper


def align(depth_data, rgb_data):
    d_intr = [429.627, 429.627, 422.179, 243.999]
    c_intr = [420.684, 420.273, 418.145, 245.17]
    d_to_c = [[0.99999577, - 0.00241159, - 0.00164173,-0.05902871], [0.00239989, 0.99997199, - 0.00709178,  0.00020985],
          [0.00165878, 0.00708781, 0.99997348, 0.00028736],
          [0, 0, 0, 1]]
    # Assume the intrinsics and extrinsics parameters for depth and RGB are available
    depth_intrinsics = d_intr  # [fx, fy, cx, cy]  # focal length x, focal length y, principal point x, principal point y
    # depth_to_rgb_extrinsics = [[r11, r12, r13, tx],
    #                            [r21, r22, r23, ty],
    #                            [r31, r32, r33, tz],
    #                            [0, 0, 0, 1]]  # 4x4 transformation matrix
    depth_to_rgb_extrinsics = d_to_c
    # Create an empty array for the aligned depth data
    aligned_depth = np.zeros_like(depth_data)
    rgb_intrinsics = c_intr  # [fx, fy, cx, cy]  # focal length x, focal length y, principal point x, principal point y

    # Loop through each pixel in the depth image and calculate its corresponding pixel in the RGB image
    for v in range(depth_data.shape[0]):
        for u in range(depth_data.shape[1]):
            # Get the depth value for the current pixel
            depth_value = depth_data[v, u]

            # Convert the pixel to camera coordinates using the depth intrinsics
            x = (u - depth_intrinsics[2]) * depth_value / depth_intrinsics[0]
            y = (v - depth_intrinsics[3]) * depth_value / depth_intrinsics[1]

            # Apply the depth to RGB extrinsics transformation to get the corresponding pixel in the RGB image
            xyz_depth = np.array([x, y, depth_value, 1]).reshape((4, 1))
            xyz_rgb = np.array(depth_to_rgb_extrinsics) @ xyz_depth
            u_rgb = xyz_rgb[0, 0] * rgb_intrinsics[0] / xyz_rgb[2, 0] + rgb_intrinsics[2]
            v_rgb = xyz_rgb[1, 0] * rgb_intrinsics[1] / xyz_rgb[2, 0] + rgb_intrinsics[3]

            # Check if the corresponding pixel in the RGB image is within its boundaries
            if (u_rgb >= 0 and u_rgb < rgb_data.shape[1] and v_rgb >= 0 and v_rgb < rgb_data.shape[0]):
                # Copy the RGB values to the corresponding pixel in the aligned depth image
                aligned_depth[v, u] = rgb_data[int(v_rgb), int(u_rgb)]

    return aligned_depth
            # depth = [0,
            # 0,
            # 0,
            # 0,
            # 0,]
            # color = [-0.0554703,
            #                 0.0659396,
            #                 0.000437566,
            #                 0.000100082, - 0.020705]


#import numpy as np


def depth_rgb_registration(depthData, rgbData):
    # ,
    #                        fx_d, fy_d, cx_d, cy_d,
    #                        fx_rgb, fy_rgb, cx_rgb, cy_r
    #                        extrinsics):
    depthHeight = depthData.shape[0]
    depthWidth = depthData.shape[1]
    depthScale = 0.0010000000474974513
    d_intr = [429.627, 429.627, 422.179, 243.999]
    c_intr = [420.684, 420.273, 418.145, 245.17]
    d_to_c = [[0.99999577, - 0.00241159, - 0.00164173,-0.05902871], [0.00239989, 0.99997199, - 0.00709178,  0.00020985],
          [0.00165878, 0.00708781, 0.99997348, 0.00028736],
          [0, 0, 0, 1]]
    cv2.rgbd.depthRegistration(d_intr, c_intr, None, np.array(d_to_c), depthData, (depthWidth, depthHeight), depthDilation=False)
    fx_d, fy_d, cx_d, cy_d = d_intr[0], d_intr[1], d_intr[2], d_intr[3]
    fx_rgb, fy_rgb, cx_rgb, cy_rgb = c_intr[0], c_intr[1], c_intr[2], c_intr[3]
    extrinsics = np.array(d_to_c)

    # Aligned will contain X, Y, Z, R, G, B values in its planes
    aligned = np.zeros((depthHeight, depthWidth, 6), dtype=np.float32)

    for v in range(depthHeight):
        for u in range(depthWidth):
            # Apply depth intrinsics
            z = float(depthData[v, u]) / depthScale
            x = float((u - cx_d) * z) / fx_d
            y = float((v - cy_d) * z) / fy_d

            # Apply the extrinsics
            transformed = np.dot(extrinsics[:3,:3], np.array([x, y, z])) + extrinsics[:3, 3]
            aligned[v, u, 0] = transformed[0]
            aligned[v, u, 1] = transformed[1]
            aligned[v, u, 2] = transformed[2]

    for v in range(depthHeight):
        for u in range(depthWidth):
            # Apply RGB intrinsics
            x = (aligned[v, u, 0] * fx_rgb / aligned[v, u, 2]) + cx_rgb
            y = (aligned[v, u, 1] * fy_rgb / aligned[v, u, 2]) + cy_rgb

            # "x" and "y" are indices into the RGB frame, but they may contain
            # invalid values (which correspond to the parts of the scene not visible
            # to the RGB camera.
            # Do we have a valid index?
            rgbWidth = rgbData.shape[1]
            rgbHeight = rgbData.shape[0]
            if (x > rgbWidth or y > rgbHeight or
                    x < 1 or y < 1 or np.isnan(x) or np.isnan(y)):
                continue

            x = np.round(x).astype(int)
            y = np.round(y).astype(int)

            aligned[v, u, 3] = np.float32(rgbData[y, x, 0])/255.0
            aligned[v, u, 4] = np.float32(rgbData[y, x, 1])/255.0
            aligned[v, u, 5] = np.float32(rgbData[y, x, 2])/255.0
    return aligned
def registerDepth(intrinsics_depth, intrinsics_color,extrinsics, depth):

    width, height = depth.shape[1], depth.shape[0]
    out = np.zeros((height, width))
    y, x = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    z = depth.reshape(1, -1)
    intrinsics_depth = np.array([[intrinsics_depth[0], 0, intrinsics_depth[1]],
                                 [0, intrinsics_depth[2], intrinsics_depth[3]],
                                 [0, 0, 1]], dtype=float)
    intrinsics_color = np.array([[intrinsics_color[0], 0, intrinsics_color[1]],
                                 [0, intrinsics_color[2], intrinsics_color[3]],
                                 [0, 0, 1]], dtype=float)
    x = (x - intrinsics_depth[0, 2]) / intrinsics_depth[0, 0]
    y = (y - intrinsics_depth[1, 2]) / intrinsics_depth[1, 1]
    pts = np.vstack((x * z, y * z, z))
    pts = extrinsics[:3, :3] @ pts + extrinsics[:3, 3:]
    pts = intrinsics_color @ pts
    px = np.round(pts[0, :] / pts[2, :])/255.0
    py = np.round(pts[1, :] / pts[2, :])/255.0
    mask = (px >= 0) * (py >= 0) * (px < width) * (py < height)
    out[py[mask].astype(int), px[mask].astype(int)] = pts[2, mask]
    return out

