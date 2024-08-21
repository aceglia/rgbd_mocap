import cv2
import numpy as np


def image_gray_and_blur(image, blur_size):
    return cv2.GaussianBlur(image, (blur_size, blur_size), 0)


def background_remover(frame, depth, clipping_distance, depth_scale, clipping_color, min_dist=0, use_contour=True):
    depth_image_3d = depth #np.dstack((depth, depth, depth))
    final = np.where(
        (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= min_dist / depth_scale),
        clipping_color,
        frame,
    )
    if use_contour:
        white_frame = np.ones_like(frame) * 255
        im_for_mask = np.where(
            (depth_image_3d > clipping_distance / depth_scale) | (depth_image_3d <= min_dist / depth_scale),
            clipping_color,
            white_frame,
        )
        # gray = cv2.cvtColor(im_for_mask, cv2.COLOR_BGR2GRAY)
        gray = im_for_mask
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            c = sorted(contours, key=cv2.contourArea)
        except ValueError:
            return final
        mask = np.ones_like(frame) * clipping_color
        if len(c) > 0:
            cv2.drawContours(mask, [c[-1]], contourIdx=-1, color=(255, 255, 255), thickness=-1)
            if len(c) > 1:
                cv2.drawContours(mask, [c[-2]], contourIdx=-1, color=(255, 255, 255), thickness=-1)
            # final = np.where(mask == (255, 255, 255), frame, clipping_color)
            final = np.where(mask == 255, frame, clipping_color)

    return final


class OpticalFlow:
    BLUR = 5
    optical_flow_parameters = {
        'winSize': (15, 15),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }

    def __init__(self, frame, depth, marker_set):
        positions = []
        for mark in marker_set.markers:
            if mark.name in marker_set.markers_from_dlc and mark.name not in marker_set.dlc_enhance_markers:
                continue
            else:
                positions.append(mark.pos)
        frame = background_remover(frame, depth, 1.4, 0.0010000000474974513, 100)
        self.frame = image_gray_and_blur(frame, OpticalFlow.BLUR)
        self.previous_frame = self.frame.copy()
        self.previous_positions = np.array(positions, dtype=np.float32)
        self.value = None

    def get_optical_flow_pos(self, frame, depth, marker_set):
        self.previous_frame = self.frame.copy()
        frame = background_remover(frame, depth, 1.4, 0.0010000000474974513,
                                   100, use_contour=True)
        self.depth = depth
        self.frame = image_gray_and_blur(frame, OpticalFlow.BLUR)
        self.value = cv2.calcOpticalFlowPyrLK(
            self.previous_frame,
            self.frame,
            self.previous_positions,
            None,
            **OpticalFlow.optical_flow_parameters,
        )

        all_pos = np.ndarray((marker_set.nb_markers, 2))
        all_x = np.ndarray((marker_set.nb_markers, 1))
        all_y = np.ndarray((marker_set.nb_markers, 1))
        count = 0
        for m, mark in enumerate(marker_set.markers):
            if mark.name in marker_set.markers_from_dlc and mark.name not in marker_set.dlc_enhance_markers:
                all_pos[m, :] = [0, 0]
                all_x[m, :] = 0
                all_y[m, :] = 0
            else:
                all_pos[m, :] = self.value[0][count]
                all_x[m, :] = self.value[1][count]
                all_y[m, :] = self.value[2][count]
                count += 1

        self.value = [all_pos, all_x, all_y]

        # if len(self.value_to_add) != 0:
        #     self.value = list(self.value)
        #     self.value[0] = np.concatenate((self.value[0], np.zeros((len(self.value_to_add), 2))), axis=0)
        #     self.value[1] = np.concatenate((self.value[1], np.zeros((len(self.value_to_add), 1))), axis=0)
        #     self.value[2] = np.concatenate((self.value[2], np.zeros((len(self.value_to_add), 1))), axis=0)
        return self.value

    def set_positions(self, marker_set):
        positions = []
        value_to_add = []
        for mark in marker_set.markers:
            if mark.name in marker_set.markers_from_dlc and mark.name not in marker_set.dlc_enhance_markers:
                value_to_add.append([mark.pos])
            else:
                positions.append(mark.pos)
        self.previous_positions = np.array(positions, dtype=np.float32)
        # self.value_to_add = value_to_add

    def __getitem__(self, item):
        if self.value is not None:
            return self.value[0][item], self.value[1][item], self.value[2][item]
