import cv2
import numpy as np


def image_gray_and_blur(image, blur_size):
    return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                            (blur_size, blur_size),
                            0)


class OpticalFlow:
    BLUR = 9
    optical_flow_parameters = {
        'winSize': (15, 15),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }

    def __init__(self, frame, positions):
        self.frame = image_gray_and_blur(frame, OpticalFlow.BLUR)
        self.previous_frame = self.frame.copy()
        self.previous_positions = np.array(positions, dtype=np.float32)

        self.value = None

    def get_optical_flow_pos(self, frame):
        self.previous_frame = self.frame
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
