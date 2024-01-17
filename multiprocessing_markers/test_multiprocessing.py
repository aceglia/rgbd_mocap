import time

import cv2

from multiprocessing_markers.multiprocessing_ import ProcessHandler, MarkerSet, SharedFrames
from multiprocessing_markers.config import config


def load_img(path, index):
    color_file = path + f"color_{index}.png"
    depth_file = path + f"depth_{index}.png"

    color_image = cv2.flip(cv2.imread(color_file, cv2.COLOR_BGR2RGB), -1)
    depth_image = cv2.flip(cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH), -1)

    return color_image, depth_image


def main():
    # Init Marker Sets
    set_names = []
    marker_names = []
    base_positions = []

    for i in range(len(config['crops'])):
        set_names.append(config['crops'][i]['name'])

        mn = []
        bs = []
        for j in range(len(config['crops'][i]['markers'])):
            mn.append(config['crops'][i]['markers'][j]['name'])
            bs.append(config['crops'][i]['markers'][j]['pos'])
            # bs.append((config['crops'][i]['markers'][j]['pos'][1], config['crops'][i]['markers'][j]['pos'][0]))

        marker_names.append(mn)
        base_positions.append(bs)

    print(set_names)
    print(marker_names)
    print(base_positions)

    marker_sets = []
    for i in range(len(set_names)):
        marker_sets.append(MarkerSet(set_names[i], marker_names[i], shared=True))

    for i, marker_set in enumerate(marker_sets):
        marker_set.set_markers_pos(base_positions[i])

    # Image
    path = config['directory']
    index = config['start_index']
    color, depth = load_img(path, index)

    # Frame
    print(color, depth)
    frames = SharedFrames(color, depth)

    # Method
    tracking_options = {
      "naive": True,
      "kalman": True,
      "optical_flow": True,
    }

    process_handler = ProcessHandler(marker_sets, frames, config, tracking_options)
    process_handler.start_process()

    while index != config['end_index']:
        tik = time.time()

        # Get next image
        index += 1
        color, depth = load_img(path, index)
        if color is None or depth is None:
            continue

        tok = time.time()
        print('Loading frame :', tok - tik)

        tik = tok
        frames.set_images(color, depth)

        # Process image
        process_handler.send_process()

        tok = time.time()
        print('Time to compute frame:', tok - tik)

        if cv2.waitKey(1) == ord('q'):
            process_handler.end_process()
            break

    return 0


if __name__ == '__main__':
    if main() == 0:
        print("Everything's fine !")
