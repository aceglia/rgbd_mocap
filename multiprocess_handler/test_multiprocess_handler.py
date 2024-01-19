import time

import cv2
import numpy as np

from multiprocess_handler import ProcessHandler, MarkerSet, SharedFrames
from config import config
from tracking.test_tracking import print_marker
from frames.frames import Frames


def print_marker_sets(frame, marker_sets):
    for i, marker_set in enumerate(marker_sets):
        frame = print_marker(frame, marker_set)

    return frame


def load_img(path, index):
    color_file = path + f"color_{index}.png"
    depth_file = path + f"depth_{index}.png"

    color_image = cv2.flip(cv2.imread(color_file, cv2.COLOR_BGR2RGB), -1)
    depth_image = cv2.flip(cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH), -1)

    return color_image, depth_image


def init_():
    # Init Marker Sets
    set_names = []
    off_sets = []
    marker_names = []
    base_positions = []

    for i in range(len(config['crops'])):
        set_names.append(config['crops'][i]['name'])
        off_sets.append(config['crops'][i]['area'][:2])

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

    marker_sets: list[MarkerSet] = []
    for i in range(len(set_names)):
        marker_sets.append(MarkerSet(set_names[i], marker_names[i], shared=True))

    for i, marker_set in enumerate(marker_sets):
        marker_set.set_markers_pos(base_positions[i])
        marker_set.set_offset_pos(off_sets[i])

    # Image
    path = config['directory']
    index = config['start_index']
    color, depth = load_img(path, index)

    # Frame
    frames = SharedFrames(color, depth)

    # Method
    tracking_options = {
        "naive": False,
        "kalman": True,
        "optical_flow": True,
    }

    process_handler = ProcessHandler(marker_sets, frames, config, tracking_options)
    process_handler.start_process()

    return index, path, frames, process_handler, marker_sets


def main(index, path, frames, process_handler, marker_sets):
    avg_load_time = 0
    avg_frame_time = 0
    avg_total_time = 0
    while index != config['end_index']:
        tik = time.time()

        # Get next image
        index += 1
        color, depth = load_img(path, index)
        if color is None or depth is None:
            continue

        tok = time.time()
        avg_load_time += (tok - tik)

        frames.set_images(color, depth)

        # Process image
        process_handler.send_and_receive_process()

        tak = time.time()
        avg_frame_time += (tak - tok)
        avg_total_time += (tak - tik)

        img = frames.color.copy()
        img = print_marker_sets(img, marker_sets, config['crops'])

        cv2.imshow('Main image :', img)
        if cv2.waitKey(1) == ord('q'):
            process_handler.end_process()
            break

    nb_img = index - config['start_index']
    return avg_load_time / nb_img, avg_frame_time / nb_img, avg_total_time / nb_img


def main_load_while_processing(index, path, frames, process_handler: ProcessHandler, marker_sets):
    avg_load_time = 0
    avg_frame_time = 0
    avg_total_time = 0
    while index != config['end_index']:
        tik = time.time()

        # Process image
        process_handler.send_process()

        tok = time.time()
        # Get next image
        index += 1
        color, depth = load_img(path, index)
        if color is None or depth is None:
            continue

        tuk = time.time()
        avg_load_time += (tuk - tok)

        # Receive from process
        process_handler.receive_process()

        tak = time.time()
        avg_frame_time += (tak - tik) - (tuk - tok)
        avg_total_time += (tak - tik)

        img = frames.color.copy()
        img = print_marker_sets(img, marker_sets)

        # Set next frame
        frames.set_images(color, depth)

        cv2.imshow('Main image :', img)
        if cv2.waitKey(1) == ord('q'):
            process_handler.end_process()
            break

    nb_img = index - config['start_index']
    return avg_load_time / nb_img, avg_frame_time / nb_img, avg_total_time / nb_img


if __name__ == '__main__':
    # Init
    index, path, frames, process_handler, marker_sets = init_()

    # Run
    load_time, frame_time, tot_time = main_load_while_processing(index, path, frames, process_handler, marker_sets)

    print("Everything's fine !")
    print('Average load time :', load_time)
    print('Average computation time :', frame_time)
    print('Average total time :', tot_time)
