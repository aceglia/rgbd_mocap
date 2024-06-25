import cv2

from ..filters.filter import Frames
from ..frames.shared_frames import SharedFrames
from crop import Crop
from ..markers.marker_set import MarkerSet
from ..tracking.utils import print_blobs, print_marker, print_estimated_positions, set_marker_pos


filter_option = {
    "blend": 100,
    "white_range": [147, 242],
    "blob_area": [10, 100],
    "convexity": 10,
    "circularity": 10,
    "distance_between_blobs": 10,
    "distance_in_centimeters": [20, 102],
    "clahe_clip_limit": 4,
    "clahe_grid_size": 1,
    "gaussian_blur": 5,
    "use_contour": True,
    "mask": None,
    "white_option": True,
    "blob_option": True,
    "clahe_option": True,
    "distance_option": True,
    "masks_option": True,
}

tracking_options = {
    "naive": True,
    "kalman": True,
    "optical_flow": True,
}

_options = {
    "filter_options": filter_option,
    "tracking_options": tracking_options,
}


def main():
    shared_memory = False
    # Marker Set
    marker_set = MarkerSet("Test", ["a", "b", "c", "d"], shared=shared_memory)  # , 'e'])

    # Base positions  67, 73     60, 89
    base_positions = [(80, 85), (71, 101), (97, 117), (165, 117)]  # , (176, 104)]
    # base_positions = [(204, 264), (197, 280), (223, 296), (404, 356)] #, (302, 308)]
    marker_set.set_markers_pos(base_positions)

    # Image
    path = "../data_files/P4_session2/gear_20_15-08-2023_10_52_14/"

    all_color_files = [path + f"color_{i}.png" for i in range(600, 900)]
    all_depth_files = [path + f"depth_{i}.png" for i in range(600, 900)]
    color_images = [cv2.flip(cv2.imread(file, cv2.COLOR_BGR2RGB), -1) for file in all_color_files[:]]
    depth_images = [cv2.flip(cv2.imread(file, cv2.IMREAD_ANYDEPTH), -1) for file in all_depth_files[:]]

    # Frame
    frames = Frames(color_images[0], depth_images[0])
    if shared_memory:
        frames = SharedFrames(color_images[0], depth_images[0])

    # Area (Full img for test)
    area = (137, 191, 461, 355)
    # area = (0, 0, frames.width, frames.height)

    crop = Crop(area, frames, marker_set, filter_option, tracking_options)
    img = print_marker(crop.frame.color, marker_set)

    cv2.imshow("blobs", img)
    cv2.waitKey(0)

    for i in range(len(color_images)):
        frames.set_images(color_images[i], depth_images[i])

        blobs, positions, estimated = crop.track_markers()
        set_marker_pos(marker_set, positions)

        img = crop.filter.get_filtered_frame()
        # img = cv2.cvtColor(crops.tracker.optical_flow.frame, cv2.COLOR_GRAY2RGB)

        img = print_blobs(img, blobs)
        img = print_estimated_positions(img, estimated)
        img = print_marker(img, marker_set)

        cv2.imshow("blobs", img)
        cv2.waitKey(20)


if __name__ == "__main__":
    main()
