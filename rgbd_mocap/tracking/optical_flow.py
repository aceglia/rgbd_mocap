import cv2
import numpy as np


def image_gray_and_blur(image, blur_size):
    return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (blur_size, blur_size), 0)


def background_remover(frame, depth, clipping_distance, depth_scale, clipping_color, min_dist=0, use_contour=True):
    depth_image_3d = np.dstack((depth, depth, depth))
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
        gray = cv2.cvtColor(im_for_mask, cv2.COLOR_BGR2GRAY)
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
            final = np.where(mask == (255, 255, 255), frame, clipping_color)
    return final


class OpticalFlow:
    BLUR = 5
    optical_flow_parameters = {
        'winSize': (15, 15),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }

    def __init__(self, frame, depth, positions):
        frame = background_remover(frame, depth, 1.4, 0.0010000000474974513, 100)
        self.frame = image_gray_and_blur(frame, OpticalFlow.BLUR)
        self.previous_frame = self.frame.copy()
        self.previous_positions = np.array(positions, dtype=np.float32)
        self.value = None

    def get_optical_flow_pos(self, frame, depth):
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

        return self.value

    def set_positions(self, positions):
        self.previous_positions = np.array(positions, dtype=np.float32)

    def __getitem__(self, item):
        if self.value is not None:
            return self.value[0][item], self.value[1][item], self.value[2][item]
