import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from casadi import SX, nlpsol, dot, sin, cos, einstein, mtimes, MX,blockcat


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
