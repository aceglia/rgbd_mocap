import cv2

from rgbd_mocap.processing.process_image import ProcessImage
from rgbd_mocap.processing.handler import Handler
from rgbd_mocap.processing.config import config

tracking_options = {
        "naive": False,
        "kalman": True,
        "optical_flow": True,
    }


if __name__ == '__main__':
    # Init
    ProcessImage.ROTATION = -1
    ProcessImage.SHOW_IMAGE = True
    Handler.SHOW_CROPS = True

    PI = ProcessImage(config, tracking_options, multi_processing=True)

    check_first_frame = False
    if check_first_frame:
        PI.process_next_image()
        cv2.waitKey(0)

    # Run
    run_all = False
    # Run everything and return time of the computation
    if run_all:
        total_time, avg_time_per_frame = PI.process_all_image()
        print("Everything's fine !")
        print('Average computation time per frame:', avg_time_per_frame)
        print('Total time :', total_time)

    # Run frame by frame
    else:
        while PI.process_next_image():
            continue
