import cv2

from rgbd_mocap.processing.process_image import ProcessImage
from rgbd_mocap.processing.handler import Handler
from rgbd_mocap.processing.config import load_json
from rgbd_mocap.crop.crop import DepthCheck

tracking_options = {
        "naive": False,
        "kalman": True,
        "optical_flow": True,
    }


if __name__ == '__main__':
    # Init
    from rgbd_mocap.enums import Rotation
    ProcessImage.ROTATION = Rotation.ROTATE_0
    ProcessImage.SHOW_IMAGE = True
    Handler.SHOW_CROPS = True
    DepthCheck.DEPTH_SCALE = 0.0010000000474974513
    config = load_json(r"D:\Documents\Programmation\pose_estimation\data_files\P14\gear_15_project\test.json")
    config["depth_scale"] = DepthCheck.DEPTH_SCALE
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
        import time
        while True:
            tic = time.time()
            if not PI.process_next_image():
                break
            print("Time for one frame :", (time.time() - tic))

