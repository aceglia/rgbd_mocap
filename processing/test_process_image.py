from process_image import ProcessImage, Handler
from processing.config import config

tracking_options = {
        "naive": False,
        "kalman": True,
        "optical_flow": True,
    }


if __name__ == '__main__':
    # Init
    ProcessImage.ROTATION = -1
    ProcessImage.SHOW_IMAGE = True
    Handler.SHOW_CROPS = False

    PI = ProcessImage(config, tracking_options, shared=True)


    # Run
    total_time, avg_time_per_frame = PI.process_all_image()

    print("Everything's fine !")
    print('Average computation time per frame:', avg_time_per_frame)
    print('Total time :', total_time)
