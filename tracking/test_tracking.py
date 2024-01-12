import cv2
import numpy as np
from tracking_markers import Tracker, Position, MarkerSet, List, Frames


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
    blobs = []
    color = (0, 0, 255)

    for marker in marker_set:
        frame = cv2.putText(frame, marker.name, (marker.pos[0] + 10, marker.pos[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        blobs.append(marker.pos[:2])

    return print_blobs(frame, blobs, size=2, color=color)


def print_position(frame, positions: List[Position]):
    blob_visible = []
    blob_not_visible = []
    for pos in positions:
        if pos.visibility:
            blob_visible.append(pos.position)
        else:
            blob_not_visible.append(pos.position)

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
            marker_set[i].pos[:2] = positions[i].position


def main():
    # Image
    path = "test_image/"
    name = 'marker'
    angle = 2
    all_color_files = [path + f"{name}_{i}.png" for i in range(0, 360, angle)]
    color_images = [cv2.imread(file) for file in all_color_files[:]]

    image = Frames(color_images[0], color_images[0])

    # Marker Set
    marker_set = MarkerSet('Test', ['a', 'b', 'c', 'd'])

    # Base positions
    base_positions = [(184, 99), (186, 242), (391, 341), (249, 482)]
    marker_set.init_kalman_from_pos(base_positions)

    for i in range(len(marker_set.markers)):
        marker_set[i].pos[:2] = base_positions[i]

    Tracker.DELTA = 10
    tracker = Tracker(image, marker_set, naive=True, optical_flow=False, kalman=True)

    # cv2.imshow('test', color_images[0])
    # cv2.waitKey(0)

    quit_press = False
    while not quit_press:
        for i in range(len(color_images)):
            image.set_images(color_images[i], color_images[i])
            img = image.color
            blobs = get_blobs(image.color)
            # img = print_blobs(image.color, blobs)

            positions, estimate_positions = tracker.track(image, blobs)

            set_marker_pos(marker_set, positions)

            img = print_estimated_positions(img, estimate_positions)
            img = print_marker(img, marker_set)

            cv2.imshow('test', img)

            if cv2.waitKey(10 * angle) == ord('q'):
                quit_press = True
                break


if __name__ == '__main__':
    main()
