from enum import Enum


class DetectionMethod(Enum):
    """
    The different types of detection methods that can be used.
    """

    CV2Blobs = "blobs"
    CV2Contours = "contours"
    SCIKITBlobs = "scikit_blobs"


class FilterType(Enum):
    """
    The different types of detection methods that can be used.
    """

    HSV = "hsv"
    Gray = "gray"
