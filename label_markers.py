import cv2

from rgbd_mocap.RgbdImages import RgbdImages
from rgbd_mocap.markers.marker_set import MarkerSet
from rgbd_mocap.model_creation.rotations import Rotations
from rgbd_mocap.model_creation.translations import Translations
import json
import os

prefix = "/mnt/shared/" if os.name == "posix" else r"Q:\\"


def get_crop_from_last_config(path):
    with open(path, "r") as f:
        data = json.load(f)
    min_x = [data["crops"][i]["area"][0] for i in range(len(data["crops"]))]
    min_y = [data["crops"][i]["area"][1] for i in range(len(data["crops"]))]
    max_x = [data["crops"][i]["area"][2] for i in range(len(data["crops"]))]
    max_y = [data["crops"][i]["area"][3] for i in range(len(data["crops"]))]
    return [min(min_x) - 100, min(min_y) - 100, max(max_x) + 200, max(max_y) + 200]


def _init_kin_marker_set():
    kinematics_marker_set_shoulder = MarkerSet(
        marker_set_name="shoulder",
        marker_names=[
            "xiph",
            "ster",
            # "ribs",
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
    kinematics_marker_set_arm = MarkerSet(
        marker_set_name="arm",
        marker_names=["delt", "arm_l", "epic_l"],
        translations=Translations.NONE,
        rotations=Rotations.XYZ,
    )
    kinematics_marker_set_hand = MarkerSet(
        marker_set_name="hand",
        marker_names=["larm_l", "styl_r", "styl_u"],
        translations=Translations.NONE,
        rotations=Rotations.YZ,
    )
    kin_marker_set = [
        kinematics_marker_set_shoulder,
        kinematics_marker_set_scap,
        kinematics_marker_set_arm,
        kinematics_marker_set_hand,
    ]
    return kin_marker_set


def main():
    kin_marker_set = _init_kin_marker_set()
    participants = [f"P{i}" for i in range(9, 17)]
    # participants.pop(participants.index("P14"))
    trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)
    trials = [["gear_5", "gear_10", "gear_15", "gear_20"]] * len(participants)

    # trials = [[ "only", "random"]] * len(participants)
    # data_files = "Q:\Projet_hand_bike_markerless\RGBD"
    data_files = f"{prefix}Projet_hand_bike_markerless/RGBD"

    for p, part in enumerate(participants):
        files = os.listdir(f"{data_files}{os.sep}{part}")
        files = [file for file in files if os.path.isdir(f"{data_files}{os.sep}{part}{os.sep}" + file)]
        final_files = files if not trials else []
        if trials:
            for trial in trials[p]:
                for file in files:
                    if trial in file and not "less" in file and not "more" in file:
                        final_files.append(file)
        files = final_files
        path_to_camera_config_file = (
            f"{prefix}Projet_hand_bike_markerless/RGBD/config_camera_files/config_camera_{part}.json"
        )
        path_to_dlc_model = [
            # f"Q:\Projet_hand_bike_markerless\RGBD\Training_data\DLC_projects\{part}_excluded_non_augmented\exported-models\DLC_test_mobilenet_v2_0.5_iteration-0_shuffle-1",
            # f"Q:\Projet_hand_bike_markerless\RGBD\Training_data\DLC_projects\{part}_excluded_hist_eq\exported-models\DLC_test_mobilenet_v2_0.5_iteration-0_shuffle-1",
            rf"C:\Users\User\Documents\Amedeo\DLC_projects\{part}_excluded_normal_times_three\exported-models\DLC_test_mobilenet_v2_0.75_iteration-0_shuffle-1",
            # fr"C:\Users\User\Documents\Amedeo\DLC_projects\{part}_excluded_normal\exported-models\DLC_test_mobilenet_v2_0.35_iteration-0_shuffle-1"
        ]

        saving_names = ["normal_times_three"]
        alone = ["filtered"]  # , "filtered"]
        for f, file in enumerate(files):
            for m, dlc_model_path in enumerate(path_to_dlc_model):
                for a, al in enumerate(alone):
                    print(f"working on participant {part} for trial {file[:7]}")
                    # path = f"{data_files}{os.sep}{part}{os.sep}" + file + f"{os.sep}tracking_config_dlc.json"
                    path = (
                        f"{data_files}{os.sep}{part}{os.sep}" + file + f"{os.sep}tracking_config_gui_3_crops_new.json"
                    )
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"No tracking config file found for {part} in {file}")
                        last_config = (
                            f"{data_files}{os.sep}{part}{os.sep}" + file + f"{os.sep}tracking_config_gui_3_crops.json"
                        )
                        area = get_crop_from_last_config(last_config)
                        tracking_config = {"crops": [{"name": "crop_0", "area": area}]}
                    else:
                        tracking_config = path
                    rgbd = RgbdImages(path_to_camera_config_file)
                    if part in ["P12", "P13", "P15"] and al == "filtered":
                        # rgbd.set_static_markers(["ribs", "xiph"])
                        rgbd.set_static_markers(["xiph"])

                    # rgbd.set_quasi_static_markers(["ribs"], bounds=[[-20, 20]])
                    else:
                        rgbd.set_quasi_static_markers(["xiph"], x_bounds=[[-20, 20]], y_bounds=[[-20, 20]])
                        # rgbd.set_static_markers(["ribs"])
                    # elif al == "filtered":
                    #     rgbd.set_quasi_static_markers(["xiph"], x_bounds=[[-5, 20]], y_bounds=[[-5,20]])
                    # if part in ["P11", "P12", "P13"]:
                    # rgbd.set_marker_to_exclude(["M1", "M2", "M3"])
                    # rgbd.set_dlc_enhance_markers(["M1", "M2", "M3"])
                    rgbd.initialize_tracking(
                        tracking_config,  # path,
                        build_kinematic_model=True,
                        use_kalman=True,  # al == "filtered",
                        use_optical_flow=True,
                        multi_processing=True,
                        kin_marker_set=kin_marker_set,
                        images_path=f"{data_files}{os.sep}{part}{os.sep}" + file,
                        model_name="model_test.bioMod",
                        from_dlc=False,
                        dlc_model_path=dlc_model_path,
                        dlc_marker_names=[
                            "xiph",
                            "ster",
                            "clavsc",
                            "M1",
                            "M2",
                            "M3",
                            "clavac",
                            "delt",
                            "arm_l",
                            "epic_l",
                            "larm_l",
                            "styl_r",
                            "styl_u",
                        ],
                        ignore_all_checks=al == "alone",
                        downsample_ratio=1,
                        # start_idx=None
                    )
                    last_frame = rgbd.tracking_config["start_index"]
                    while True:
                        if not rgbd.get_frames(
                            fit_model=al == "filtered",
                            show_image=False,
                            save_data=True,
                            save_video=True,
                            file_path=rgbd.tracking_config["directory"]
                            + os.sep
                            + f"marker_pos_multi_proc_3_crops_{saving_names[m]}_new.bio",
                            video_name=f"video_labeled_{saving_names[m]}_new",
                        ):
                            if rgbd.video_object is not None:
                                rgbd.video_object.release()
                            cv2.destroyAllWindows()
                            break
                        iter = rgbd.iter
                        if rgbd.process_image.index - last_frame > 2:
                            print(f"gap of {rgbd.process_image.index - last_frame} images")
                        if iter % 500 == 0:
                            print("nb iterations:", iter)
                            # cv2.destroyAllWindows()
                            # break
                        # if iter % 4000 == 0:
                        #     print("nb iterations:", iter)
                        #     if rgbd.video_object is not None:
                        #         rgbd.video_object.release()
                        #     cv2.destroyAllWindows()
                        #     break
                        # print(rgbd.kinematic_model_checker.marker_sets[0].markers[0].get_depth())
                        last_frame = rgbd.process_image.index


if __name__ == "__main__":
    main()
