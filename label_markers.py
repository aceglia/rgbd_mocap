import cv2

from rgbd_mocap.RgbdImages import RgbdImages
from rgbd_mocap.markers.marker_set import MarkerSet
import os

def _init_kin_marker_set():
    kinematics_marker_set_shoulder = MarkerSet(
        marker_set_name="shoulder",
        marker_names=[
             "xiph",
             "ster",
             "clavsc",
            # "M1",
            # "M2",
            # "M3",
            # "clavac",
        ],
    )
    kinematics_marker_set_scap = MarkerSet(
        marker_set_name="scap",
        marker_names=[
            # "xiph",
            # "ster",
            # "clavsc",
             "M1",
             "M2",
             "M3",
             "clavac",
        ],
    )
    kinematics_marker_set_arm = MarkerSet(marker_set_name="arm", marker_names=[ "delt",
                                                                                "arm_l",
                                                                                "epic_l"])
    kinematics_marker_set_hand = MarkerSet(marker_set_name="hand", marker_names=["larm_l",
                                                                                 "styl_r",
                                                                                 "styl_u"
                                                                                 ])
    kin_marker_set = [
        kinematics_marker_set_shoulder,
        kinematics_marker_set_scap,
        kinematics_marker_set_arm,
        kinematics_marker_set_hand,
    ]
    return kin_marker_set


def main():
    kin_marker_set = _init_kin_marker_set()
    path_to_camera_config_file = "D:\Documents\Programmation\pose_estimation\config_camera_files\config_camera_P14.json"
    config = r"D:\Documents\Programmation\pose_estimation\data_files\P14\gear_15_22-01-2024_16_26_05\tracking_config_gui.json"
    rgbd = RgbdImages(path_to_camera_config_file)
    from rgbd_mocap.processing.process_handler import Handler
    Handler.SHOW_CROPS = True
    rgbd.initialize_tracking(config,
                             build_kinematic_model=True,
                             multi_processing=False,
                             kin_marker_set=kin_marker_set,
                             model_name="model_test.bioMod")

    import time
    while True:
        tic = time.time()
        cv2.waitKey(0)
        if not rgbd.get_frames(fit_model=True, show_image=True, save_data=False, save_video=False,
                               file_path=rgbd.tracking_config["directory"] + os.sep + "test.bio"):
            break

        print("Time for one frame :", time.time() - tic)
        continue


if __name__ == '__main__':
    main()