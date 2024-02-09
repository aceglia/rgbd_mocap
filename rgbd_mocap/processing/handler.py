import cv2

from ..markers.marker_set import MarkerSet
from ..tracking.position import Position


class Handler:
    SHOW_BLOBS = True
    SHOW_ESTIMATION = True
    SHOW_MARKERS = True
    SHOW_CROPS = False
    STOP = 42
    CONTINUE = 1
    RESET = 2

    def __init__(self):
        pass

    # Interact with process
    def send_process(self, order=1):
        pass

    def receive_process(self):
        pass

    def send_and_receive_process(self, order=1):
        pass

    def end_process(self):
        pass

    # Show Crops
    @staticmethod
    def show_image(crop_name, image, blobs=None, markers=None, estimated_positions=None):
        if not Handler.SHOW_CROPS:
            return

        if Handler.SHOW_BLOBS and blobs is not None:
            image = print_blobs(image, blobs)

        if Handler.SHOW_ESTIMATION and estimated_positions is not None:
            image = print_estimated_positions(image, estimated_positions)

        if Handler.SHOW_MARKERS and markers is not None:
            image = print_marker(image, markers, use_off_set=False)

        cv2.imshow(crop_name, image)
        cv2.waitKey(1)


def print_blobs(frame, blobs, size=4, color=(0, 255, 0)):
    img = frame.copy()
    for blob in blobs:
        img[blob[1] - size:blob[1] + size, blob[0] - size:blob[0] + size] = color

    return img


def print_marker(frame, marker_set: MarkerSet, use_off_set=True):
    visible = []
    not_visible = []
    color_ok = (0, 255, 0)
    color_not_ok = (0, 0, 255)

    off_set = marker_set[0].crop_offset
    if not use_off_set:
        off_set = (0, 0)

    for marker in marker_set:
        if marker.is_visible:
            frame = cv2.putText(frame, marker.name,
                                (marker.pos[0] + 10 + off_set[0], marker.pos[1] + 10 + off_set[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color_ok, 1)
            visible.append(marker.pos[:2] + off_set)

            if marker.is_depth_visible:
                frame = cv2.putText(frame, str(marker.depth // 10),
                                    (marker.pos[0] + off_set[0], marker.pos[1] + 20 + off_set[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ok, 1)

        else:
            frame = cv2.putText(frame, marker.name,
                                (marker.pos[0] + 10 + off_set[0], marker.pos[1] + 10 + off_set[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color_not_ok, 1)
            not_visible.append(marker.pos[:2] + off_set)

    frame = print_blobs(frame, visible, size=2, color=color_ok)
    return print_blobs(frame, not_visible, size=2, color=color_not_ok)


def print_position(frame, positions: list[Position]):
    blob_static = []
    blob_visible = []
    blob_not_visible = []
    for pos in positions:
        if isinstance(pos, tuple):
            blob_static.append(pos)
            continue

        if pos.visibility:
            blob_visible.append(pos.position)
        else:
            blob_not_visible.append(pos.position)

    frame = print_blobs(frame, blob_static, size=3, color=(120, 120, 120))
    frame = print_blobs(frame, blob_visible, size=3, color=(255, 155, 0))
    return print_blobs(frame, blob_not_visible, size=3, color=(255, 0, 0))


def print_estimated_positions(frame, estimated_pos: list[list[Position]]):
    for pos in estimated_pos:
        frame = print_position(frame, pos)

    return frame
