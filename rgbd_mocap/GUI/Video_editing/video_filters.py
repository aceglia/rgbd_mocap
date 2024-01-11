import time

import numpy as np
import cv2
from Video_editing.video_edit import VideoEdit
from Video_editing.image_options import ImageOptions
# from video_edit_linker import VideoEditLinker


class VideoFilters:
    def __init__(self, parent):
        self.parent = parent
        self.image_options = ImageOptions(video_filter=self, parent=parent)

        ### Clahe options
        self.clahe = cv2.createCLAHE(
            clipLimit=self.image_options["clahe_clip_limit"],
            tileGridSize=(self.image_options["clahe_grid_size"],
                          self.image_options["clahe_grid_size"])
        )

        ### Blobs options
        self.blobs_param = cv2.SimpleBlobDetector_Params()
        self.blobs_param.minThreshold = self.image_options["white_range"][0]
        self.blobs_param.maxThreshold = self.image_options["white_range"][1]
        self.blobs_param.filterByColor = True
        self.blobs_param.blobColor = 255
        self.blobs_param.minDistBetweenBlobs = self.image_options['distance_between_blobs']
        self.blobs_param.filterByArea = True
        self.blobs_param.minArea = self.image_options["blob_area"][0]
        self.blobs_param.maxArea = self.image_options["blob_area"][1]
        self.blobs_param.filterByCircularity = True
        self.blobs_param.minCircularity = self.image_options["circularity"] / 100
        self.blobs_param.filterByConvexity = True
        self.blobs_param.minConvexity = self.image_options["convexity"] / 100
        self.blobs_param.filterByInertia = False

    ###### Update functions #############################
    def update_params(self):
        ### Update the blob detector params
        if self.image_options.show_params["blob_option"]:
            self.blobs_param.minThreshold = self.image_options["white_range"][0]
            self.blobs_param.maxThreshold = self.image_options["white_range"][1]

            self.blobs_param.minArea = self.image_options["blob_area"][0]
            self.blobs_param.maxArea = self.image_options["blob_area"][1]
            self.blobs_param.minCircularity = self.image_options["circularity"] / 100
            self.blobs_param.minConvexity = self.image_options["convexity"] / 100
            self.blobs_param.minDistBetweenBlobs = self.image_options['distance_between_blobs']

        ### Update the clahe filters
        if self.image_options.show_params["clahe_option"]:
            self.clahe.setClipLimit(self.image_options["clahe_clip_limit"])
            self.clahe.setTilesGridSize((self.image_options["clahe_grid_size"],
                                         self.image_options["clahe_grid_size"]))

    ##### Filters/Masks functions #############################
    def white_range_mask(self, ve: VideoEdit):
        gray = cv2.cvtColor(ve.filtered_frame, cv2.COLOR_BGR2GRAY)

        mask = np.ones(gray.shape, dtype=np.uint8)
        # mask[gray < self.image_options["white_range"][0]] = 0
        # mask[gray > self.image_options["white_range"][1]] = 0

        return mask

    def distance_range_mask(self, ve: VideoEdit):
        mask = np.ones(ve.depth_frame.shape, dtype=np.uint8)

        mask[ve.depth_frame < self.image_options["distance_in_centimeters"][0] * 10] = 0
        mask[ve.depth_frame > self.image_options["distance_in_centimeters"][1] * 10] = 0

        if not self.image_options["use_contour"]:
            return mask

        thresh = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            c = sorted(contours, key=cv2.contourArea)
        except ValueError:
            return mask
        if len(c) > 0:
            final = np.zeros(ve.depth_frame.shape, dtype=np.uint8)
            cv2.drawContours(final, [c[-1]], contourIdx=-1, color=1, thickness=-1)
            if len(c) > 1:
                cv2.drawContours(final, [c[-2]], contourIdx=-1, color=1, thickness=-1)

            return final

        return mask

    def clahe_filter(self, ve: VideoEdit):
        img = cv2.cvtColor(ve.color_frame, cv2.COLOR_RGB2GRAY)
        img = self.clahe.apply(img)

        if self.image_options["gaussian_blur"]:
            img = cv2.GaussianBlur(img, (self.image_options["gaussian_blur"] * 2 + 1,
                                         self.image_options["gaussian_blur"] * 2 + 1), 0)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def blob_detector(self, ve: VideoEdit):
        detector = cv2.SimpleBlobDetector_create(self.blobs_param)

        keypoints = detector.detect(ve.filtered_frame)
        centers = []
        for blob in keypoints:
            centers.append((int(blob.pt[1]), int(blob.pt[0])))

        ve.blobs = centers

    ##### Main functions ####################################
    def update(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """
        self.update_params()
        if self.parent.ve.color_frame is not None:
            self.apply_filters(self.parent.ve)

    def apply_filters(self, ve: VideoEdit):
        tik = time.time()
        ve.filtered_frame = ve.color_frame.copy()
        ve.blobs = []
        if self.image_options["clahe_option"]:
            ve.apply_filter(self.clahe_filter)

        masks = []
        # if self.image_options["white_option"]:
        #     masks.append(self.white_range_mask(ve))
        if self.image_options["distance_option"]:
            masks.append(self.distance_range_mask(ve))
        ve.apply_mask(masks)

        if self.image_options["blob_option"]:
            self.blob_detector(ve)

        ve.apply_blend(self.image_options["blend"])
        ve.apply_blob_detect()

        ve.update()
        print(time.time() - tik)
