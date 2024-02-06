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


class ColorResolution(Enum):
    """
    The different types of color resolutions that can be used.
    """

    R_424x240 = (424, 240)
    R_480x270 = (480, 270)
    R_640x360 = (640, 360)
    R_640x480 = (640, 480)
    R_848x480 = (848, 480)
    R_1280x720 = (1280, 720)
    R_1280x800 = (1280, 800)


class DepthResolution(Enum):
    """
    The different types of color resolutions that can be used.
    """

    R_256x144 = (256, 144)
    R_424x240 = (424, 240)
    R_480x270 = (480, 270)
    R_640x360 = (640, 360)
    R_640x400 = (640, 400)
    R_640x480 = (640, 480)
    R_848x100 = (848, 100)
    R_848x480 = (848, 480)
    R_1280x720 = (1280, 720)
    R_1280x800 = (1280, 800)


class FrameRate(Enum):
    """
    The different types of frame rates that can be used.
    """

    FPS_5 = 5
    FPS_15 = 15
    FPS_25 = 25
    FPS_30 = 30
    FPS_60 = 60
    FPS_90 = 90


class Rotation(Enum):
    """
    The different types of frame rotation using opencv that can be used.
    """

    ROTATE_0 = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270
