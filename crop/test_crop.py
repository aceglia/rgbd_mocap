import cv2

from filter.filter import Frames
from crop import Crop
from rgbd_mocap.marker_class import MarkerSet
from tracking.test_tracking import print_blobs, print_marker, print_position, print_estimated_positions, set_marker_pos


filter_options = {
        "blend": 100,
        "white_range": [
          147,
          242
        ],
        "blob_area": [
          10,
          100
        ],
        "convexity": 10,
        "circularity": 10,
        "distance_between_blobs": 10,
        "distance_in_centimeters": [
          20,
          102
        ],
        "clahe_clip_limit": 4,
        "clahe_grid_size": 1,
        "gaussian_blur": 5,
        "use_contour": True,
        "mask": None,
        "white_option": True,
        "blob_option": True,
        "clahe_option": True,
        "distance_option": True,
        "masks_option": True
      }

tracking_options = {
    "naive": True,
    "kalman": True,
    "optical_flow": True,
}

_options = {
    'filter_options': filter_options,
    'tracking_options': tracking_options,
}


def main():
    # Marker Set
    dms = MarkerSet('test', ['d'])
    marker_set = MarkerSet('Test', ['a', 'b', 'c', 'd', 'e'])

    # Base positions  67, 73     60, 89
    d = [(176, 104)]
    base_positions = [(80, 85), (71, 101), (97, 117), (165, 117), (176, 104)]
    # base_positions = [(204, 264), (197, 280), (223, 296), (404, 356)] #, (302, 308)]
    marker_set.init_kalman_from_pos(base_positions)
    dms.init_kalman_from_pos(d)
    dms.markers[0].pos[:2] = d[0]

    for i in range(len(marker_set.markers)):
        marker_set[i].pos[:2] = base_positions[i]

    # Image
    path = '../data_files/P4_session2/gear_20_15-08-2023_10_52_14/'

    all_color_files = [path + f"color_{i}.png" for i in range(600, 700)]
    all_depth_files = [path + f"depth_{i}.png" for i in range(600, 700)]
    color_images = [cv2.flip(cv2.imread(file, cv2.COLOR_BGR2RGB), -1) for file in all_color_files[:]]
    depth_images = [cv2.flip(cv2.imread(file, cv2.IMREAD_ANYDEPTH), -1) for file in all_depth_files[:]]

    # Frame
    frames = Frames(color_images[0], depth_images[0])

    # Area (Full img for test)
    area = (137, 191, 461, 355)
    # area = (0, 0, frames.width, frames.height)

    crop = Crop(area, frames, marker_set, _options)
    img = print_marker(crop.frame.color, marker_set)

    cv2.imshow('blobs', img)
    cv2.waitKey(0)

    for i in range(len(color_images)):
        frames.set_images(color_images[i], depth_images[i])
        crop.frame.get_images()

        blobs, positions, estimated = crop.track_markers()
        set_marker_pos(marker_set, positions)

        img = crop.filter.get_filtered_frame()
        # img = cv2.cvtColor(crop.tracker.optical_flow.frame, cv2.COLOR_GRAY2RGB)

        img = print_blobs(img, blobs)
        img = print_estimated_positions(img, estimated)
        img = print_marker(img, marker_set)

        cv2.imshow('blobs', img)
        cv2.waitKey(20)


if __name__ == '__main__':
    main()
