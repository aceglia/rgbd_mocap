import time

import numpy as np
import cv2

from frames.frames import Frames


class Filter:
    def __init__(self, options):
        self.frame = None
        self.filtered_frame = None

        self.options = options

        ### Clahe options
        self.clahe = cv2.createCLAHE(
            clipLimit=self.options["clahe_clip_limit"],
            tileGridSize=(self.options["clahe_grid_size"],
                          self.options["clahe_grid_size"])
        )

        ### Blobs options
        self.blobs_param = cv2.SimpleBlobDetector_Params()
        self.blobs_param.minThreshold = self.options["white_range"][0]
        self.blobs_param.maxThreshold = self.options["white_range"][1]
        self.blobs_param.filterByColor = True
        self.blobs_param.blobColor = 255
        self.blobs_param.minDistBetweenBlobs = self.options['distance_between_blobs']
        self.blobs_param.filterByArea = True
        self.blobs_param.minArea = self.options["blob_area"][0]
        self.blobs_param.maxArea = self.options["blob_area"][1]
        self.blobs_param.filterByCircularity = True
        self.blobs_param.minCircularity = self.options["circularity"] / 100
        self.blobs_param.filterByConvexity = True
        self.blobs_param.minConvexity = self.options["convexity"] / 100
        self.blobs_param.filterByInertia = False

    # Getter
    def get_filtered_frame(self):
        if self.filtered_frame is not None:
            return self.filtered_frame
        else:
            raise Warning('No image has been filtered.')

    ##### Masks functions #############################
    def _white_range_mask(self):
        gray = cv2.cvtColor(self.frame.color, cv2.COLOR_BGR2GRAY)

        mask = np.ones(gray.shape, dtype=np.uint8)
        # mask[gray < self.image_options["white_range"][0]] = 0
        # mask[gray > self.image_options["white_range"][1]] = 0

        return mask

    def _distance_range_mask(self):
        mask = np.ones(self.frame.depth.shape, dtype=np.uint8)

        mask[self.frame.depth < self.options["distance_in_centimeters"][0] * 10] = 0
        mask[self.frame.depth > self.options["distance_in_centimeters"][1] * 10] = 0

        if not self.options["use_contour"]:
            return mask

        thresh = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            c = sorted(contours, key=cv2.contourArea)
        except ValueError:
            return mask
        if len(c) > 0:
            final = np.zeros(self.frame.depth.shape, dtype=np.uint8)
            cv2.drawContours(final, [c[-1]], contourIdx=-1, color=1, thickness=-1)
            if len(c) > 1:
                cv2.drawContours(final, [c[-2]], contourIdx=-1, color=1, thickness=-1)

            return final

        return mask

    ##### Filters functions #############################
    def _clahe_filter(self):
        img = cv2.cvtColor(self.frame.color, cv2.COLOR_RGB2GRAY)
        img = self.clahe.apply(img)

        if self.options["gaussian_blur"]:
            img = cv2.GaussianBlur(img, (self.options["gaussian_blur"] * 2 + 1,
                                         self.options["gaussian_blur"] * 2 + 1), 0)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        self.filtered_frame = img

    ##### Blob detection ###############################
    def _blob_detector(self):
        detector = cv2.SimpleBlobDetector_create(self.blobs_param)

        keypoints = detector.detect(self.filtered_frame)
        centers = []
        for blob in keypoints:
            centers.append((int(blob.pt[1]), int(blob.pt[0])))

        return centers

    ##### Main functions ####################################
    def _apply_masks(self, masks):
        mask = np.ones(self.frame.depth.shape, dtype=np.uint8)

        for m in masks:
            mask[:] *= m[:]

        self.filtered_frame[mask == 0] = 0

    def apply_filters(self):
        # Clahe filter
        if self.options["clahe_option"]:
            self._clahe_filter()

        # Masks
        masks = []
        if self.options["white_option"]:
            masks.append(self._white_range_mask())
        if self.options["distance_option"]:
            masks.append(self._distance_range_mask())
        if self.options['masks_option'] and self.options['mask']:
            masks.append(self.options['mask'])

        self._apply_masks(masks)

    def get_blobs(self, frame: Frames):
        self.frame = frame
        self.filtered_frame = self.frame.color.copy()

        self.apply_filters()

        return self._blob_detector()
