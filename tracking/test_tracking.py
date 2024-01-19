import cv2
import numpy as np
from tracking.tracking_markers import Tracker, Position, MarkerSet, List, CropFrames
from frames.frames import Frames

params_detector = cv2.SimpleBlobDetector_Params()
params_detector.minThreshold = 150
params_detector.maxThreshold = 220
params_detector.filterByColor = True
params_detector.blobColor = 255
detector = cv2.SimpleBlobDetector_create(params_detector)


def get_blobs(frame):
    keypoints = detector.detect(frame)

    blobs = []
    for blob in keypoints:
        blobs.append((int(blob.pt[0]), int(blob.pt[1])))

    return blobs


def print_blobs(frame, blobs, size=4, color=(0, 255, 0)):
    img = frame.copy()
    for blob in blobs:
        img[blob[1] - size:blob[1] + size, blob[0] - size:blob[0] + size] = color

    return img


def print_marker(frame, marker_set: MarkerSet):
    visible = []
    not_visible = []
    color_ok = (0, 255, 0)
    color_not_ok = (0, 0, 255)

    off_set = marker_set[0].crop_offset
    for marker in marker_set:
        if marker.is_visible:
            frame = cv2.putText(frame, marker.name,
                                (marker.pos[0] + 10 + off_set[0], marker.pos[1] + 10 + off_set[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color_ok, 1)
            visible.append(marker.pos[:2] + off_set)

            if marker.is_depth_visible:
                frame = cv2.putText(frame, str(marker.pos[2] // 10),
                                    (marker.pos[0] + off_set[0], marker.pos[1] + 20 + off_set[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ok, 1)

        else:
            frame = cv2.putText(frame, marker.name,
                                (marker.pos[0] + 10 + off_set[0], marker.pos[1] + 10 + off_set[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color_not_ok, 1)
            not_visible.append(marker.pos[:2] + off_set)

    frame = print_blobs(frame, visible, size=2, color=color_ok)
    return print_blobs(frame, not_visible, size=2, color=color_not_ok)


def print_position(frame, positions: List[Position]):
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


def print_estimated_positions(frame, estimated_pos: List[List[Position]]):
    for pos in estimated_pos:
        frame = print_position(frame, pos)

    return frame


def set_marker_pos(marker_set: MarkerSet, positions: List[Position]):
    assert len(marker_set.markers) == len(positions)

    for i in range(len(positions)):
        if positions[i] != ():
            marker_set[i].set_pos(positions[i].position)
            marker_set[i].set_visibility(positions[i].visibility)
        else:
            marker_set[i].set_visibility(False)


def main():
    # Image
    path = "test_image/"
    name = 'marker'
    angle = 5
    all_color_files = [path + f"{name}_{i}.png" for i in range(0, 360, angle)]
    color_images = [cv2.imread(file) for file in all_color_files[:]]

    frame = Frames(color_images[0], color_images[0])
    image = CropFrames((0, 0, frame.width, frame.height), frame)

    # Marker Set
    marker_set = MarkerSet('Test', ['a', 'b', 'c', 'd'])

    # Base positions
    base_positions = [(184, 99), (186, 242), (391, 341), (249, 482)]
    marker_set.set_markers_pos(base_positions)

    Tracker.DELTA = 20
    tracker = Tracker(image, marker_set, naive=False, optical_flow=True, kalman=False)

    # cv2.imshow('test', color_images[0])
    # cv2.waitKey(0)

    quit_press = False
    while not quit_press:
        for i in range(len(color_images)):
            frame.set_images(color_images[i], color_images[i])
            image.get_images()

            img = image.color
            blobs = get_blobs(image.color)
            # img = print_blobs(image.color, blobs)

            positions, estimate_positions = tracker.track(image, blobs)

            set_marker_pos(marker_set, positions)
            # tracker.correct()

            # img = print_blobs(img, blobs, size=5)
            img = print_estimated_positions(img, estimate_positions)
            img = print_marker(img, marker_set)

            cv2.imshow('test', img)

            if cv2.waitKey(10 * angle) == ord('q'):
                quit_press = True
                break


if __name__ == '__main__':
    main()
