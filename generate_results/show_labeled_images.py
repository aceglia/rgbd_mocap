from rgbd_mocap.marker_class import MarkerSet
from rgbd_mocap.RgbdImages import RgbdImages
from rgbd_mocap.utils import *
from biosiglive import save
import time
import glob
import shutil
import matplotlib.pyplot as plt
import cv2
from biosiglive import load
from biosiglive.file_io.save_and_load import dic_merger
import os


def get_markers_set(camera):
    markers_thorax = MarkerSet(
        marker_set_name="thorax", marker_names=["xiph", "ster", "clavsc", "M1", "M2", "M3", "clavac"], image_idx=0
    )
    markers_arm = MarkerSet(
        marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l", "larm_l", "styl_r", "styl_u"], image_idx=1
    )
    camera.add_marker_set([markers_thorax, markers_arm])

    kinematics_marker_set_shoulder = MarkerSet(
        marker_set_name="shoulder",
        marker_names=[
            "xiph",
            "ster",
            "clavsc",
        ],
    )
    kinematics_marker_set_scap = MarkerSet(
        marker_set_name="scap",
        marker_names=[
            "M1",
            "M2",
            "M3",
            "clavac",
        ],
    )
    kinematics_marker_set_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"])
    kinematics_marker_set_hand = MarkerSet(marker_set_name="hand", marker_names=["larm_l", "styl_r", "styl_u"])
    kin_marker_set = [
        kinematics_marker_set_shoulder,
        kinematics_marker_set_scap,
        kinematics_marker_set_arm,
        kinematics_marker_set_hand,
    ]
    return camera, kin_marker_set


def reprocess_file(camera, images_dir, mask, suffix, frame, file_name):
    camera, kin_marker_set = get_markers_set(camera)
    shutil.copy2(
        rf"{images_dir}{os.sep}t" + f"racking_config.json", rf"{images_dir}{os.sep}t" + f"racking_config_tmp.json"
    )
    camera.start_index += frame
    camera.camera_frame_numbers = camera.camera_frame_numbers[frame:]
    camera.initialize_tracking(
        tracking_conf_file=rf"{images_dir}{os.sep}t" + f"racking_config_tmp.json",
        crop_frame=False,
        mask_parameters=mask,
        label_first_frame=True,
        build_kinematic_model=True,
        method=DetectionMethod.CV2Blobs,
        model_name=f"{images_dir}{os.sep}kinematic_model_{suffix}_tmp.bioMod",
        # rotation_angle=Rotation.ROTATE_180,
        with_tapir=False,
        marker_sets=kin_marker_set,
    )
    while True:
        _, _ = camera.get_frames(
            aligned=False,
            detect_blobs=True,
            label_markers=True,
            bounds_from_marker_pos=True,
            method=DetectionMethod.CV2Blobs,
            filter_with_kalman=True,
            adjust_with_blobs=True,
            use_optical_flow=True,
            fit_model=True,
            model_name=f"{images_dir}{os.sep}kinematic_model_{suffix}_tmp.bioMod",
            show_images=True,
            save_data=True,
            file_name=file_name,
            save_in_video=True,
        )
        if camera.frame_idx % 500 == 0:
            print(f"frame {camera.frame_idx} processed")
        if camera.frame_idx + camera.start_index == len(camera.all_color_files):
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"interrupted at frame {camera.frame_idx}, all frame were not processed yet. ")
            break
    cv2.destroyAllWindows()
    camera.video_object.release()
    os.remove(rf"{images_dir}{os.sep}t" + f"racking_config_tmp.json")
    os.remove(f"{images_dir}{os.sep}kinematic_model_{suffix}_tmp.bioMod")


def merge_markers_pos(init_results, new_results):
    for key in init_results.keys():
        pass


if __name__ == "__main__":
    participants = ["P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    for participant in participants:
        image_path = rf"D:\Documents\Programmation\pose_estimation\data_files\{participant}"
        files = os.listdir(f"{image_path}")
        # files = [file for file in files if file[:7] == "gear_15"]
        files = [file for file in files if "anato" in file and os.path.isdir(f"{image_path}{os.sep}" + file)]
        config_file = (
            rf"D:\Documents\Programmation\pose_estimation\config_camera_files\config_camera_{participant}.json"
        )
        for file in files:
            print("processing file : ", file)
            if not os.path.isfile(f"{image_path}{os.sep}" + file + "/markers_pos.bio"):
                continue
            file = f"{image_path}{os.sep}" + file
            suffix = file[-19:]
            tracking_file = rf"{file}{os.sep}t" + f"racking_config.json"
            start_idx = start_idx_from_json(tracking_file)
            load_markers = load(file + "/markers_pos.bio")
            markers_depth = load_markers["markers_in_meters"]
            if markers_depth.shape[2] < 100:
                continue
            # plt.figure("markers")
            # for i in range(markers_depth.shape[1]):
            #     plt.subplot(4, 4, i+1)
            #     for j in range(1):
            #         plt.plot(markers_depth[j, i, :], "g")
            # plt.show()
            # show_labeled = input("show markers ? y/n")
            # if show_labeled == "n":
            #     continue
            # frame = input("frame ?")
            # frame = int(frame)
            frame = 0
            start_idx = start_idx + frame
            camera = RgbdImages(
                conf_file=config_file,
                images_dir=file,
                start_index=start_idx,
                # stop_index=10,
                downsampled=1,
                load_all_dir=False,
            )
            camera.all_color_files = [0] * camera.start_index + [
                file + os.sep + "color_" + str(f) + ".png" for f in load_markers["camera_frame_idx"]
            ]
            camera.all_depth_files = [0] * camera.start_index + [
                file + os.sep + "depth_" + str(f) + ".png" for f in load_markers["camera_frame_idx"]
            ]
            camera.camera_frame_numbers = load_markers["camera_frame_idx"]
            camera.show_labeled_images(
                load_markers["markers_in_pixel"][:, :, frame:],
                fps=60,
                occlusions=load_markers["occlusions"],
                markers_names=load_markers["markers_names"],
                show_image=False,
                save_video=True,
            )
            # reprocess = input("reprocess file from the current frame ? y/n")
            # if reprocess == "y":
            #     new_frame_idx = camera.frame_idx
            #     print(camera.frame_idx)
            #     if os.path.isfile(f"{file}{os.sep}markers_pos_post_process.bio"):
            #         delete = input("delete markers_pos_post_process.bio ? y/n")
            #         if delete == "y":
            #             os.remove(f"{file}{os.sep}markers_pos_post_process.bio")
            #     reprocess_file(camera, file, True, suffix, new_frame_idx, "markers_pos_post_process")
            # load_new_markers = load(file + "/markers_pos_post_process.bio")
            # old_markers = load(file + "/markers_pos.bio",
            #                    number_of_line=load_markers["camera_frame_idx"].index(load_new_markers["camera_frame_idx"][0]))
            # all_markers = dic_merger(old_markers, load_new_markers)
            # save(all_markers, file + "/markers_pos_post_process.bio", safe=False)
