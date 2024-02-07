import cv2

from rgbd_mocap.filter.filter import Filter, Frames


_options = {
        "blend": 100,
        "white_range": [
          147,
          242
        ],
        "blob_area": [
          10,
          500
        ],
        "convexity": 10,
        "circularity": 10,
        "distance_between_blobs": 2,
        "distance_in_centimeters": [
          20,
          102
        ],
        "clahe_clip_limit": 4,
        "clahe_grid_size": 1,
        "gaussian_blur": 1,
        "use_contour": True,
        "mask": None,
        "white_option": True,
        "blob_option": True,
        "clahe_option": True,
        "distance_option": True,
        "masks_option": True
      }


def print_blobs(frame, blobs, size=4, color=(0, 255, 0)):
    img = frame.copy()
    for blob in blobs:
        img[blob[1] - size:blob[1] + size, blob[0] - size:blob[0] + size] = color

    return img


def main():
    path = '../../data_files/P15/gear_5_25-01-2024_11_33_17/'

    color = cv2.imread(path + 'color_1372.png', cv2.COLOR_BGR2RGB)
    depth = cv2.imread(path + 'depth_1372.png', cv2.IMREAD_ANYDEPTH)

    frames = Frames(color, depth)
    filter = Filter(_options)

    blobs = filter.get_blobs(frames)

    img = filter.get_filtered_frame()
    img = print_blobs(img, blobs)

    cv2.imshow('blobs', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
