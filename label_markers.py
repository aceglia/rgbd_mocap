import cv2

from rgbd_mocap.RgbdImages import RgbdImages
from rgbd_mocap.markers.marker_set import MarkerSet
from rgbd_mocap.model_creation.rotations import Rotations
from rgbd_mocap.model_creation.translations import Translations
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
        translations=Translations.XYZ,
        rotations=Rotations.XYZ,
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
        translations=Translations.NONE,
        rotations=Rotations.XYZ,
    )
    kinematics_marker_set_arm = MarkerSet(marker_set_name="arm", marker_names=["delt",
                                                                               "arm_l",
                                                                               "epic_l"],
                                          translations=Translations.NONE,
                                          rotations=Rotations.XYZ)
    kinematics_marker_set_hand = MarkerSet(marker_set_name="hand", marker_names=["larm_l",
                                                                                 "styl_r",
                                                                                 "styl_u"
                                                                                 ],
                                           translations=Translations.NONE,
                                           rotations=Rotations.YZ)
    kin_marker_set = [
        kinematics_marker_set_shoulder,
        kinematics_marker_set_scap,
        kinematics_marker_set_arm,
        kinematics_marker_set_hand,
    ]
    return kin_marker_set


def main():
    kin_marker_set = _init_kin_marker_set()
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15"]
    trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    data_files = "Q:\Projet_hand_bike_markerless\RGBD"
    for p, part in enumerate(participants):
        files = os.listdir(f"{data_files}{os.sep}{part}")
        files = [file for file in files if os.path.isdir(f"{data_files}{os.sep}{part}{os.sep}" + file)
                 ]
        final_files = files if not trials else []
        if trials:
            for trial in trials[p]:
                for file in files:
                    if trial in file:
                        final_files.append(file)
        files = final_files
        path_to_camera_config_file = f"Q:\Projet_hand_bike_markerless\RGBD\config_camera_files\config_camera_{part}.json"
        for f, file in enumerate(files):
            print(f"working on participant {part} for trial {file[:7]}")
            path = f"{data_files}{os.sep}{part}{os.sep}" + file + f"{os.sep}tracking_config_gui_3_crops.json"
            if not os.path.isfile(path):
                continue
            rgbd = RgbdImages(path_to_camera_config_file)
            if part in ["P12", "P13", "P15"]:
                rgbd.set_static_markers(["xiph"])
            else:
                rgbd.set_quasi_static_markers(["xiph"], bounds=[[-10, 10]])
            if part in ["P11", "P12", "P13"]:
                rgbd.set_marker_to_exclude(["styl_r"])
            rgbd.initialize_tracking(path,
                                     build_kinematic_model=True,
                                     multi_processing=True,
                                     kin_marker_set=kin_marker_set,
                                     images_path=f"{data_files}{os.sep}{part}{os.sep}" + file,
                                     model_name="model_test.bioMod")
            last_frame = rgbd.tracking_config["start_index"]
            while True:
                if not rgbd.get_frames(fit_model=True, show_image=True, save_data=False, save_video=False,
                                       file_path=rgbd.tracking_config[
                                                     "directory"] + os.sep + "marker_pos_multi_proc_3_crops.bio"):
                    if rgbd.video_object is not None:
                        rgbd.video_object.release()
                    cv2.destroyAllWindows()
                    break
                iter = rgbd.iter
                if rgbd.process_image.index - last_frame > 2:
                    print(f"gap of {rgbd.process_image.index - last_frame} images")
                if iter % 500 == 0:
                    print("nb iterations:", iter)
                # if iter == 2000:
                #     cv2.destroyAllWindows()
                #     rgbd.video_object.release()
                #     break
                last_frame = rgbd.process_image.index


if __name__ == '__main__':
    main()
