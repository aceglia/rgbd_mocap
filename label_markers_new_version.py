from rgbd_mocap.RgbdImages import RgbdImages
from rgbd_mocap.markers.marker_set import MarkerSet


def main():
    kinematics_marker_set_shoulder = MarkerSet(
        marker_set_name="shoulder",
        marker_names=[
            "0",#"xiph",
            "1",#"ster",
            "2",#"clavsc",
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
            "3",#"M1",
            "4",#"M2",
            "5",#"M3",
            "6",#"clavac",
        ],
    )
    kinematics_marker_set_arm = MarkerSet(marker_set_name="arm", marker_names=["7", #"delt",
                                                                               "8", #"arm_l",
                                                                               "9",]) #"epic_l"])
    kinematics_marker_set_hand = MarkerSet(marker_set_name="hand", marker_names=["10",# "larm_l",
                                                                                 "11",# "styl_r",
                                                                                "12", # "styl_u"
                                                                                  ])
    kin_marker_set = [
        kinematics_marker_set_shoulder,
        kinematics_marker_set_scap,
        kinematics_marker_set_arm,
        kinematics_marker_set_hand,
    ]
    path_to_camera_config_file = "D:\Documents\Programmation\pose_estimation\config_camera_files\config_camera_P14.json"
    config = r"D:\Documents\Programmation\pose_estimation\data_files\P14\gear_15_project\test.json"
    rgbd = RgbdImages(path_to_camera_config_file)
    rgbd.initialize_tracking(config,
                             build_kinematic_model=True,
                             multi_processing=False,
                             kin_marker_set=kin_marker_set,
                             model_name="model_test.bioMod")
    # model_name = "kinematic_model_22-01-2024_16_26_05.bioMod"
    #                          )
    # rgbd.set_marker_to_exclude(["2", "3", "4"])
    import time
    while True:
        tic = time.time()
        if not rgbd.get_frames(fit_model=True, show_markers=True):
            break
        print("Time for one frame :", time.time() - tic)
        continue


if __name__ == '__main__':
    main()