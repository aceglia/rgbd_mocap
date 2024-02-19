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
    participants = ["P16", "P11", "P12", "P13", "P14", "P15", "P16"]
    data_files = "D:\Documents\Programmation\pose_estimation\data_files"
    for part in participants:
        files = os.listdir(f"{data_files}{os.sep}{part}")
        files = [file for file in files if
                 "gear_20" in file and os.path.isdir(f"{data_files}{os.sep}{part}{os.sep}" + file)
                 ]
        path_to_camera_config_file = f"D:\Documents\Programmation\pose_estimation\config_camera_files\config_camera_{part}.json"
        for file in files:
            print(f"working on participant {part} for trial {file[:7]}")
            path = f"{data_files}{os.sep}{part}{os.sep}" + file + f"{os.sep}tracking_config_gui.json"
            if not os.path.isfile(path):
                continue
            rgbd = RgbdImages(path_to_camera_config_file)
            rgbd.set_static_markers(["xiph"])
            rgbd.initialize_tracking(path,
                                     build_kinematic_model=False,
                                     multi_processing=False,
                                     use_optical_flow=False,
                                     use_kalman=True,
                                     kin_marker_set=kin_marker_set,
                                     # images_path=r"D:\Documents\Programmation\pose_estimation\data_files\P14\gear_15_22-01-2024_16_26_05",
                                     model_name="model_test.bioMod")
            last_frame = rgbd.tracking_config["start_index"]
            import time
            while True:
                tic = time.time()
                if not rgbd.get_frames(fit_model=False, show_image=True, save_data=False, save_video=False,
                                       file_path=rgbd.tracking_config["directory"] + os.sep + "marker_pos_multi_proc.bio"):
                    break
                iter = rgbd.iter
                print("frame:", rgbd.process_image.index)
                if rgbd.process_image.index - last_frame > 1:
                    print(f"gap of {rgbd.process_image.index - last_frame} images")
                if iter % 500 == 0:
                    print("nb iterations:", iter)
                last_frame = rgbd.process_image.index
                cv2.waitKey(0)
                if rgbd.process_image.index > 2105:
                    cv2.waitKey(0)
                #print("Time for one frame :", rgbd.process_image.computation_time)


if __name__ == '__main__':
    main()