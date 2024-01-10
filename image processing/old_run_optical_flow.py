import cv2
import math
import numpy as np
from rgbd_mocap.utils import find_closest_blob, check_and_attribute_depth


def _run_optical_flow(
        self,
        idx,
        color,
        prev_color,
        prev_pos,
        kalman_filter,
        blob_detector,
        markers_visible_names,
        use_optical_flow,
        error_threshold=10,
        use_tapir=False,
):
    if use_optical_flow:
        if not use_tapir:
            prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (9, 9), 0)
            color_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            color_gray = cv2.GaussianBlur(color_gray, (9, 9), 0)
            if isinstance(prev_pos, list):
                prev_pos = np.array(prev_pos, dtype=np.float32)
            if isinstance(prev_pos, np.ndarray):
                prev_pos = prev_pos.astype(np.float32)
            new_markers_pos_optical_flow, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                color_gray,
                prev_pos,
                None,
                **self.optical_flow_params,
            )
        else:
            new_markers_pos_optical_flow = []
    else:
        new_markers_pos_optical_flow = []

    count = 0
    markers_pos_list = []
    last_pos = []
    for m, marker in enumerate(self.marker_sets[idx].markers):
        last_pos.append(marker.pos[:2])  # literally recreating prev_pos
        if marker in self.static_markers:  # check directly via the marker
            count += 1  # ?
            continue
        past_last_pos = np.copy(self.last_pos[idx])  # ?
        pos_optical_flow = []
        pos_kalman = []
        if marker.name in markers_visible_names:
            if use_optical_flow:
                if not use_tapir:
                    # if st[count] == 1 and err[count] < error_threshold:
                    pos_optical_flow.append(
                        new_markers_pos_optical_flow[count])  # recreated each time why put it in a list ?
                    center = np.array((pos_optical_flow[-1])).astype(
                        int)  # casting in np.array may not be necessary, handled in find_closest_blob
                    blob_center, optical_flow_visible = find_closest_blob(center, self.blobs[idx], delta=8)
                    if optical_flow_visible:  # why returning a boolean when just checking if blob_center is None suffice ?
                        pos_optical_flow[-1] = blob_center
                    optical_flow_visible = False  # Why always reset it to False
                    # else:
                    #     marker.pos = [None, None, None]
                else:
                    new_markers_pos_optical_flow.append(marker.pos[:2])
            else:
                optical_flow_visible = False
            if kalman_filter:
                pos_kalman.append(marker.predict_from_kalman())
                if blob_detector:
                    center = np.array((marker.pos[:2])).astype(int)
                    blob_center, kalman_visible = find_closest_blob(center, self.blobs[idx], delta=8)
                else:
                    kalman_visible = False

                if kalman_visible and optical_flow_visible:
                    marker.pos[:2] = [
                        (blob_center[j] + pos_optical_flow[-1][j]) / 2 for j in range(len(blob_center))
                    ]
                elif kalman_visible and not optical_flow_visible:
                    marker.pos[:2] = blob_center
                elif not kalman_visible and optical_flow_visible:
                    marker.pos[:2] = pos_optical_flow[-1]
                elif not kalman_visible and not optical_flow_visible:
                    if use_optical_flow:
                        marker.pos[:2] = new_markers_pos_optical_flow[count]
                    else:
                        marker.pos[:2] = center

                curent_pos = self.marker_sets[idx].markers[count].pos[:2]
                all_pos = self.marker_sets[idx].get_markers_pos()[:, :count]
                # if idx == 0:
                for p in range(all_pos.shape[1]):
                    pos = all_pos[:, p]
                    if int(pos[0]) == int(curent_pos[0]) and int(pos[1]) == int(curent_pos[1]):
                        dist_p = math.sqrt(
                            (pos[0] - past_last_pos[p][0]) ** 2 + (pos[1] - past_last_pos[p][1]) ** 2
                        )
                        dist_current = math.sqrt(
                            (curent_pos[0] - past_last_pos[m][0]) ** 2
                            + (curent_pos[1] - past_last_pos[m][1]) ** 2
                        )
                        if dist_p < dist_current:
                            marker.correct_from_kalman(past_last_pos[m])
                            marker.pos[:2] = past_last_pos[m]
                        else:
                            self.marker_sets[idx].markers[p].correct_from_kalman(past_last_pos[p])
                            self.marker_sets[idx].markers[p].pos[:2] = past_last_pos[p]
                            marker_depth, self.marker_sets[idx].markers[p].is_depth_visible = check_and_attribute_depth(
                                self.marker_sets[idx].markers[p].pos[:2], self.depth_cropped[idx],
                                depth_scale=self.depth_scale
                            )
                            if abs(marker_depth - self.marker_sets[idx].markers[p].pos[2]) > 0.08:
                                marker_depth = self.marker_sets[idx].markers[p].pos[2]
                                self.marker_sets[idx].markers[p].is_depth_visible = False
                            # elif marker.is_depth_visible and \
                            elif (
                                    marker_depth > self.mask_params[idx]["min_dist"]
                                    and marker_depth < self.mask_params[idx]["clipping_distance_in_meters"]
                            ):
                                self.marker_sets[idx].markers[p].pos[2] = marker_depth
                            self.marker_sets[idx].markers[p].set_global_pos(
                                self.marker_sets[idx].markers[p].pos,
                                [self.start_crop[0][idx], self.start_crop[1][idx]],
                            )
            count += 1
        markers_pos_list.append(marker.pos[:2])
        # for j, marker_pos in enumerate(marker.pos[:2]):
        #     j_bis = 0 if j == 1 else 1
        #     if marker_pos is not None and marker_pos > self.color_cropped[idx].shape[j_bis] - 1:
        #         marker.pos[j] = self.color_cropped[idx].shape[j_bis] - 1

        # Just check bounds range and list not needed at all
        if marker.pos[0] and int(marker.pos[0]) not in list(range(0, self.color_cropped[idx].shape[1] - 1)):
            marker.pos[0] = (
                self.color_cropped[idx].shape[1] - 1 if marker.pos[0] > self.color_cropped[idx].shape[1] - 1 else 0
            )
        if marker.pos[1] and int(marker.pos[1]) not in list(range(0, self.color_cropped[idx].shape[0] - 1)):
            marker.pos[1] = (
                self.color_cropped[idx].shape[0] - 1 if marker.pos[1] > self.color_cropped[idx].shape[0] - 1 else 0
            )

        if marker.pos[0] is not None:
            marker_depth, marker.is_depth_visible = check_and_attribute_depth(
                marker.pos[:2], self.depth_cropped[idx], depth_scale=self.depth_scale
            )

            if abs(marker_depth - marker.pos[2]) > 0.08:
                marker_depth = marker.pos[2]
                marker.is_depth_visible = False
            # elif marker.is_depth_visible and \
            elif (
                    marker_depth > self.mask_params[idx]["min_dist"]
                    and marker_depth < self.mask_params[idx]["clipping_distance_in_meters"]
            ):
                marker.pos[2] = marker_depth

        marker.set_global_pos(marker.pos, [self.start_crop[0][idx], self.start_crop[1][idx]])
    self.last_pos[idx] = last_pos


def image_gray_and_blur(image, blur_size):
    return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                            (blur_size, blur_size),
                            0)


def check_bounds(i, max_bound, min_bound=0):
    if i < min_bound:
        return min_bound, False

    elif i > max_bound:
        return max_bound, False

    return i, True


def _run_optical_flow_(
        color,
        prev_color,
        prev_pos: list,  # List of the previous markers positions (x, y)
        marker_set,
        kalman_filter,
        blob_detector,
        blobs,
        markers_visible_names,
        use_optical_flow,
        error_threshold=10,
        use_tapir=False,
        optical_flow_params={},
):
    new_markers_pos_optical_flow = []

    if use_optical_flow:
        prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_RGB2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (9, 9), 0)
        color_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        color_gray = cv2.GaussianBlur(color_gray, (9, 9), 0)
        if isinstance(prev_pos, list):
            prev_pos = np.array(prev_pos, dtype=np.float32)
        if isinstance(prev_pos, np.ndarray):
            prev_pos = prev_pos.astype(np.float32)
        new_markers_pos_optical_flow, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            color_gray,
            prev_pos,
            None,
            **self.optical_flow_params,
        )

    count = 0
    markers_pos_list = []
    last_pos = []
    for i, marker in enumerate(marker_set.markers):
        if marker.is_static:  # check directly via the marker
            continue

        # past_last_pos = np.copy(self.last_pos[idx]) # ?
        pos_optical_flow = []
        pos_kalman = []
        if marker.is_visible:
            if not use_optical_flow:
                optical_flow_visible = False

            else:
                if use_tapir:
                    new_markers_pos_optical_flow.append(marker.pos[:2])

                else:
                    # if st[count] == 1 and err[count] < error_threshold:
                    center = np.array(new_markers_pos_optical_flow[i],
                                      dtype=np.int32)  # transformation to np.array int32 may not be necessary, handled in the finc_closest_blob
                    blob_center, optical_flow_visible = find_closest_blob(center, blobs, delta=8)

                    if blob_center:
                        pos_optical_flow = blob_center

                    optical_flow_visible = False  # usefull ??

            if kalman_filter:
                pos_kalman.append(marker.predict_from_kalman())
                if blob_detector:
                    center = np.array((marker.pos[:2])).astype(int)
                    blob_center, kalman_visible = find_closest_blob(center, blobs, delta=8)
                else:
                    kalman_visible = False

                if kalman_visible and optical_flow_visible:
                    marker.pos[:2] = (blob_center + pos_optical_flow[-1]) / 2
                elif kalman_visible and not optical_flow_visible:
                    marker.pos[:2] = blob_center
                elif not kalman_visible and optical_flow_visible:
                    marker.pos[:2] = pos_optical_flow[-1]
                elif not kalman_visible and not optical_flow_visible:
                    if use_optical_flow:
                        marker.pos[:2] = new_markers_pos_optical_flow[count]
                    else:
                        marker.pos[:2] = center

                curent_pos = marker_set[count].pos[:2]
                all_pos = marker_set.get_markers_pos()[:, :count]
                # if idx == 0:
                for p in range(all_pos.shape[1]):
                    pos = all_pos[:, p]
                    if int(pos[0]) == int(curent_pos[0]) and int(pos[1]) == int(curent_pos[1]):
                        dist_p = math.sqrt(
                            (pos[0] - past_last_pos[p][0]) ** 2 + (pos[1] - past_last_pos[p][1]) ** 2
                        )
                        dist_current = math.sqrt(
                            (curent_pos[0] - past_last_pos[i][0]) ** 2
                            + (curent_pos[1] - past_last_pos[i][1]) ** 2
                        )
                        if dist_p < dist_current:
                            marker.correct_from_kalman(past_last_pos[i])
                            marker.pos[:2] = past_last_pos[i]
                        else:
                            self.marker_sets[idx].markers[p].correct_from_kalman(past_last_pos[p])
                            self.marker_sets[idx].markers[p].pos[:2] = past_last_pos[p]

                            marker_depth, self.marker_sets[idx].markers[p].is_depth_visible = check_and_attribute_depth(
                                self.marker_sets[idx].markers[p].pos[:2], self.depth_cropped[idx],
                                depth_scale=self.depth_scale
                            )

                            if abs(marker_depth - self.marker_sets[idx].markers[p].pos[2]) > 0.08:  # ?
                                marker_depth = self.marker_sets[idx].markers[p].pos[2]
                                self.marker_sets[idx].markers[p].is_depth_visible = False
                            # elif marker.is_depth_visible and \
                            elif (
                                    marker_depth > self.mask_params[idx]["min_dist"]
                                    and marker_depth < self.mask_params[idx]["clipping_distance_in_meters"]
                            ):
                                self.marker_sets[idx].markers[p].pos[2] = marker_depth

                            self.marker_sets[idx].markers[p].set_global_pos(
                                self.marker_sets[idx].markers[p].pos,
                                [self.start_crop[0][idx], self.start_crop[1][idx]],
                            )
            count += 1
        markers_pos_list.append(marker.pos[:2])
        # for j, marker_pos in enumerate(marker.pos[:2]):
        #     j_bis = 0 if j == 1 else 1
        #     if marker_pos is not None and marker_pos > self.color_cropped[idx].shape[j_bis] - 1:
        #         marker.pos[j] = self.color_cropped[idx].shape[j_bis] - 1
        marker.pos[0] = check_bounds(marker.pos[0], color.shape[1] - 1)
        marker.pos[1] = check_bounds(marker.pos[1], color.shape[0] - 1)

        # if marker.pos[0] and int(marker.pos[0]) not in list(range(0, self.color_cropped[idx].shape[1] - 1)):
        #     marker.pos[0] = (
        #         self.color_cropped[idx].shape[1] - 1 if marker.pos[0] > self.color_cropped[idx].shape[1] - 1 else 0
        #     )
        # if marker.pos[1] and int(marker.pos[1]) not in list(range(0, self.color_cropped[idx].shape[0] - 1)):
        #     marker.pos[1] = (
        #         self.color_cropped[idx].shape[0] - 1 if marker.pos[1] > self.color_cropped[idx].shape[0] - 1 else 0
        #     )

        if marker.pos[0] is not None:
            marker_depth, marker.is_depth_visible = check_and_attribute_depth(
                marker.pos[:2], self.depth_cropped[idx], depth_scale=self.depth_scale
            )

            if abs(marker_depth - marker.pos[2]) > 0.08:
                marker_depth = marker.pos[2]
                marker.is_depth_visible = False
            # elif marker.is_depth_visible and \
            elif (
                    marker_depth > self.mask_params[idx]["min_dist"]
                    and marker_depth < self.mask_params[idx]["clipping_distance_in_meters"]
            ):
                marker.pos[2] = marker_depth

        marker.set_global_pos(marker.pos, [self.start_crop[0][idx], self.start_crop[1][idx]])
    self.last_pos[idx] = last_pos
