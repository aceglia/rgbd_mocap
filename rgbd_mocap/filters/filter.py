import numpy as np
import cv2

from ..frames.frames import Frames


class Filter:
    def __init__(self, options):
        self.frame = None
        self.filtered_frame = None
        self.filtered_depth = None

        self.options = options
        ### Clahe options
        self.clahe = cv2.createCLAHE(
            clipLimit=self.options["clahe_clip_limit"],
            tileGridSize=(self.options["clahe_grid_size"],
                          self.options["clahe_grid_size"])
        )
        self.white_range = self.options["white_range"]
        ### Blobs options
        self.blobs_param = cv2.SimpleBlobDetector_Params()
        # self.blobs_param.minThreshold = self.options["white_range"][0]
        # self.blobs_param.maxThreshold = self.options["white_range"][1]
        self.blobs_param.filterByColor = True
        self.blobs_param.blobColor = 255
        # self.blobs_param.blobColor = 0
        self.blobs_param.minRepeatability = 1

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

    def _distance_range_mask(self):
        mask_color = 20
        from ..crops.crop import DepthCheck
        mask = np.ones(self.frame.depth.shape, dtype=np.uint8)
        min_dist = self.options["distance_in_centimeters"][0] / 100 / DepthCheck.DEPTH_SCALE
        max_dist = self.options["distance_in_centimeters"][1] / 100 / DepthCheck.DEPTH_SCALE
        mask[self.frame.depth < min_dist] = 0
        mask[self.frame.depth > max_dist] = 0
        self.filtered_depth = np.where((self.frame.depth > max_dist), -1, self.frame.depth)

        if not self.options["use_contour"]:
            self.filtered_frame[mask == 0] = mask_color
            return

        thresh = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            c = sorted(contours, key=cv2.contourArea)
        except ValueError:
            self.filtered_frame[mask == 0] = mask_color
            return

        if len(c) > 0:
            mask = np.zeros(self.frame.depth.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c[-1]], contourIdx=-1, color=1, thickness=-1)
            if len(c) > 1:
                cv2.drawContours(mask, [c[-2]], contourIdx=-1, color=1, thickness=-1)
            self.filtered_frame[mask == 0] = mask_color
            return

        self.filtered_frame[mask == 0] = mask_color
        return

        # return mask
    ##### Filters functions #############################
    def _clahe_filter(self):
        if len(self.filtered_frame.shape) == 3:
            self.filtered_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_RGB2GRAY)

        # self.filtered_frame = cv2.GaussianBlur(self.filtered_frame, (3, 3), 0)

        self.filtered_frame = self.clahe.apply(self.filtered_frame)
        ret, self.filtered_frame = cv2.threshold(self.filtered_frame,
                                                 int(self.options["white_range"][0]),
                                                 int(self.options["white_range"][1]), 0)
        # self.filtered_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_RGB2HSV)[:, :, 1]

        if self.options["gaussian_blur"] and self.options["gaussian_blur"] > 0:
            self.filtered_frame = cv2.GaussianBlur(self.filtered_frame, (self.options["gaussian_blur"] * 2 + 1,
                                         self.options["gaussian_blur"] * 2 + 1), 0)

    ##### Blob detection ###############################
    def _blob_detector(self):
        if not self.options['blob_option']:
            return []
        self.blobs_detector = cv2.SimpleBlobDetector_create(self.blobs_param)
        keypoints = self.blobs_detector.detect(self.filtered_frame)
        centers = [(int(blob.pt[0]), int(blob.pt[1])) for blob in keypoints]
        return centers

    ##### Main functions ####################################
    def _apply_masks(self, masks):
        mask = np.ones(self.frame.depth.shape, dtype=np.uint8)

        for m in masks:
            mask[:] *= m[:]

        self.filtered_frame[mask == 0] = 20

    def apply_filters(self):
        # Clahe filters
        # Masks
        if self.options["clahe_option"]:
            self._clahe_filter()
        if self.options["distance_option"]:
            self._distance_range_mask()
        if self.options['masks_option'] and self.options['mask'] is not None:
            self.filtered_frame[self.options['mask'][0], self.options['mask'][1]] = 20


    def get_blobs(self, frame: Frames):
        self.frame = frame
        self.filtered_frame = self.frame.color.copy()
        self.filtered_depth = self.frame.depth.copy()
        self.apply_filters()

        return self._blob_detector()
