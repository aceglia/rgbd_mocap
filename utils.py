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


class MarkerSet:
    """
    This class is used to store the marker information
    """
    def __init__(self, nb_markers: int, marker_names: list, image_idx: int):
        """
        init markers class with number of markers, names and image index

        Parameters
        ----------
        nb_markers : int
            number of markers
        marker_names : list
            list of names for the markers
        image_idx : list
            index of the image where the marker set is located
        """
        self.nb_markers = nb_markers
        self.image_idx = image_idx
        self.marker_names = marker_names
        self.pos = np.zeros((2, nb_markers, 1))
        self.speed = np.zeros((2, nb_markers, 1))
        self.marker_set_model = None
        self.markers_idx_in_image = []
        self.estimated_area = []
        self.next_pos = np.zeros((2, nb_markers, 1))
        self.model = None

    @staticmethod
    def compute_speed(pos, pos_old, dt=1):
        """
        Compute the speed of the markers
        """
        return (pos - pos_old) / dt

    @staticmethod
    def compute_next_position(speed, pos, dt=1):
        """
        Compute the next position of the markers
        """
        return pos + speed * dt

    def update_speed(self):
        for i in range(2):
            self.speed[i, :] = self.compute_speed(self.pos[i, :, -1], self.pos[i, :, -2])

    def update_next_position(self):
        """
        Update the next position of the markers
        """
        next_pos=[]
        for i in range(2):
            # next_pos.append(np.concatenate((self.next_pos[i, :, :], self.compute_next_position(self.speed[i, :], self.pos[i, :, -1])[:, np.newaxis]), axis=1))
            self.next_pos[i, :] = self.compute_next_position(self.speed[i, :], self.pos[i, :, -1])[:, np.newaxis]


def find_bounds_color(frame):
    """
    Find the bounds of the image
    """
    def nothing(x):
        pass


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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        mask = cv2.inRange(h, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        imgray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        imgray = clahe.apply(imgray)
        contours, hierarchy = cv2.findContours(image=imgray,
                                               mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        for c, contour in enumerate(contours):
            if cv2.contourArea(contour) > 5 and cv2.contourArea(contour) < 1000:
                M = cv2.moments(contour)
                # cX = int(M["m10"] / M["m00"]) + area_x[0]
                # cY = int(M["m01"] / M["m00"]) + area_y[0]
                cv2.drawContours(image=result, contours=contour, contourIdx=-1, color=(255, 0, 0), thickness=2,
                                     lineType=cv2.LINE_AA)

        params = cv2.SimpleBlobDetector_Params()
        detector = cv2.SimpleBlobDetector_create(params)
        imgray_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(imgray_scaled)
        blobs = cv2.drawKeypoints(imgray_scaled, keypoints, np.array([]), (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.namedWindow(f'mask', cv2.WINDOW_NORMAL)
        cv2.imshow(f'mask', result)
        cv2.namedWindow(f'blob', cv2.WINDOW_NORMAL)
        cv2.imshow(f'blob', imgray)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return lower, upper