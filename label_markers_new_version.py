from rgbd_mocap.RgbdImages import RgbdImages


def main():
    path_to_camera_config_file = "D:\Documents\Programmation\pose_estimation\config_camera_files\config_camera_P14.json"
    config = r"D:\Documents\Programmation\pose_estimation\data_files\P14\gear_15_project\test.json"
    rgbd = RgbdImages(path_to_camera_config_file)
    rgbd.initialize_tracking(config, build_kinematic_model=False)
    import time
    while True:
        tic = time.time()
        if not rgbd.get_frames():
            break
        print("Time for one frame :", time.time() - tic)
        continue


if __name__ == '__main__':
    main()