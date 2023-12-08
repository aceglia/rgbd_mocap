from rgbd_mocap.marker_class import MarkerSet
from rgbd_mocap.RgbdImages import RgbdImages
from rgbd_mocap.utils import *
from biosiglive import save
import time
import shutil
import matplotlib.pyplot as plt
import cv2
import os

if __name__ == "__main__":
    with_camera = False
    images_dir = None
    participants = ["P8"]  # , "P3_session2", "P4_session2", "P2_session2"]  # "P4_session2",
    start_idx = [85] * len(participants)
    init_batch = False
    save_data = False
    delete_old_data = False
    c = 0
    data_files = "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files"
    for participant in participants:
        tracking_files = []
        files = os.listdir(f"{data_files}{os.sep}{participant}")
        files = [file for file in files if file[:7] == "gear_15"]
        for file in files:
            suffix = file[-19:]
            trial = file[:-20]
            if not with_camera:
                if participant:
                    images_dir = f"{data_files}{os.sep}{participant}{os.sep}{trial}_{suffix}"
                    config_file = f"/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/config_camera_files{os.sep}config_camera_{participant}.json"
                else:
                    images_dir = f"{file}"
                    config_file = f"/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/config_camera_files{os.sep}config_camera_{suffix}.json"
                # image_file = r"D:\Documents\Programmation\vision\image_camera_trial_1_800.bio.gzip"
                # os.remove(fr"{images_dir}{os.sep}t" + f"racking_config.json")
                # break
                if os.path.isfile(rf"{images_dir}{os.sep}t" + f"racking_config.json"):
                    tracking_conf = {"crop": False, "mask": False, "label": False, "build_kinematic_model": True}
                    # start_idx[c] = start_idx_from_json(fr"{images_dir}{os.sep}t" + f"racking_config.json")
                else:
                    if len(tracking_files) == 0:
                        tracking_conf = {"crop": True, "mask": True, "label": True, "build_kinematic_model": True}
                    else:
                        shutil.copy(tracking_files[0], rf"{images_dir}{os.sep}t" + f"racking_config.json")
                        tracking_conf = {"crop": False, "mask": False, "label": True, "build_kinematic_model": True}

                print("working on : ", images_dir)
                camera = RgbdImages(
                    conf_file=config_file,
                    images_dir=images_dir,
                    start_index=start_idx[c],
                    # stop_index=10,
                    downsampled=1,
                    load_all_dir=False,
                )
                c += 1
            else:
                camera = RgbdImages()
                camera.init_camera(
                    ColorResolution.R_848x480,
                    DepthResolution.R_848x480,
                    FrameRate.FPS_60,
                    FrameRate.FPS_60,
                    align=True,
                )
            camera.markers_to_exclude_for_ik = ["M1", "M2", "M3"]
            camera.ik_method = "kalman"  # "kalman" or "least_squares"
            camera.clipping_color = 20
            camera.is_frame_aligned = False

            # ----------- from back ---------------- #
            # markers_shoulder = MarkerSet(marker_set_name="shoulder", marker_names=["T5", "C7", "RIBS_r", "Clavsc", "Scap_AA", "Scap_IA", "Acrom"], image_idx=0)
            # markers_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"], image_idx=1)
            # markers_hand = MarkerSet(marker_set_name="hand", marker_names=['larm_l', "styl_r", "styl_u"], image_idx=2)
            # camera.add_marker_set([markers_shoulder, markers_arm, markers_hand])
            # kinematics_marker_set_shoulder = MarkerSet(marker_set_name="shoulder", marker_names=["T5" ,"C7", "RIBS_r", "Clavsc"])
            # kinematics_marker_set_scapula = MarkerSet(marker_set_name="scapula", marker_names=[ "Scap_AA", "Scap_IA", "Acrom"])
            # kinematics_marker_set_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"])
            # kinematics_marker_set_hand = MarkerSet(marker_set_name="hand", marker_names=['larm_l', "styl_r", "styl_u"])

            # ------------------ from front 3 crops -----------------#
            # markers_shoulder = MarkerSet(marker_set_name="shoulder", marker_names=["xiph", "ster", "clavsc"
            #                                                                        , "M1",
            #                                                                        "M2", "M3", "Clavac"], image_idx=0)
            # markers_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"], image_idx=2)
            # markers_hand = MarkerSet(marker_set_name="hand", marker_names=["larm_l", "styl_r", "styl_u"], image_idx=3)
            # camera.add_marker_set([markers_shoulder, markers_arm, markers_hand])
            #
            # ------------------ from front 4 crops -----------------#
            markers_thorax = MarkerSet(marker_set_name="thorax", marker_names=["xiph", "ster", "clavsc"], image_idx=0)
            markers_shoulder = MarkerSet(
                marker_set_name="cluster", marker_names=["M1", "M2", "M3", "Clavac"], image_idx=1
            )
            markers_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"], image_idx=2)
            markers_hand = MarkerSet(marker_set_name="hand", marker_names=["larm_l", "styl_r", "styl_u"], image_idx=3)
            camera.add_marker_set([markers_thorax, markers_shoulder, markers_arm, markers_hand])
            kinematics_marker_set_shoulder = MarkerSet(
                marker_set_name="shoulder",
                marker_names=[
                    "xiph",
                    "ster",
                    "clavsc",
                    "M1",
                    "M2",
                    "M3",
                    "Clavac",
                ],
            )
            kinematics_marker_set_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"])
            kinematics_marker_set_hand = MarkerSet(marker_set_name="hand", marker_names=["larm_l", "styl_r", "styl_u"])
            kin_marker_set = [
                kinematics_marker_set_shoulder,
                kinematics_marker_set_arm,
                kinematics_marker_set_hand,
            ]
            camera.initialize_tracking(
                tracking_conf_file=rf"{images_dir}{os.sep}t" + f"racking_config.json",
                crop_frame=tracking_conf["crop"],
                mask_parameters=tracking_conf["mask"],
                label_first_frame=tracking_conf["label"],
                build_kinematic_model=tracking_conf["build_kinematic_model"],
                method=DetectionMethod.CV2Blobs,
                model_name=f"{images_dir}{os.sep}kinematic_model_{suffix}.bioMod",
                marker_sets=kin_marker_set,
                rotation_angle=Rotation.ROTATE_180,
                with_tapir=False,
            )
            tracking_files.append(rf"{images_dir}{os.sep}t" + f"racking_config.json")
            camera.stop_index = 5
            count = 0
            if delete_old_data and os.path.isfile(f"{images_dir}{os.sep}markers_kalman.bio"):
                os.remove(f"{images_dir}{os.sep}markers_kalman.bio")
            last_camera_frame = 0
            while True:
                tic = time.time()
                color_cropped, depth_cropped = camera.get_frames(
                    aligned=False,
                    detect_blobs=True,
                    label_markers=True,
                    bounds_from_marker_pos=False,
                    method=DetectionMethod.CV2Blobs,
                    filter_with_kalman=True,
                    adjust_with_blobs=True,
                    use_optical_flow=False,
                    fit_model=True,
                    rotation_angle=Rotation.ROTATE_180,
                    model_name=f"{images_dir}{os.sep}kinematic_model_{suffix}.bioMod",
                    show_images=True,
                    save_data=save_data,
                )
                if camera.camera_frame_numbers[camera.frame_idx] - last_camera_frame > 1:
                    print("nb_jump_frame:", camera.camera_frame_numbers[camera.frame_idx] - last_camera_frame)
                    print("frame_idx:", camera.frame_idx)
                last_camera_frame = camera.camera_frame_numbers[camera.frame_idx]
                if camera.frame_idx % 500 == 0:
                    print(camera.frame_idx)
                    print(time.time() - tic)
                if camera.frame_idx == len(camera.color_images) - 1 or camera.frame_idx == camera.stop_index - 1:
                    cv2.destroyAllWindows()
                    break
                cv2.waitKey(1)

        # # -------- decoment these lines to see poind cloud and 3d markers --------
        # import open3d as o3d
        # pt_cloud = o3d.geometry.PointCloud()
        # pt_cloud.points = o3d.utility.Vector3dVector(markers_in_meters.T)
        # pt_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # sphere_list = []
        # for pt in pt_cloud.points:
        #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        #     sphere.translate(pt)
        #     sphere.paint_uniform_color([0.8, 0.2, 0.2])  # Set color of spheres to red
        #     sphere_list.append(sphere)
        #
        # # Create line sets for x-, y-, and z- axes
        # lineset = o3d.geometry.LineSet()
        #
        # # x-axis (red)
        # lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
        # lineset.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0]]))
        # lineset.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
        #
        # # y-axis (green)
        # lineset2 = o3d.geometry.LineSet()
        # lineset2.lines = o3d.utility.Vector2iVector([[0, 1]])
        # lineset2.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 1, 0]]))
        # lineset2.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
        #
        # # z-axis (blue)
        # lineset3 = o3d.geometry.LineSet()
        # lineset3.lines = o3d.utility.Vector2iVector([[0, 1]])
        # lineset3.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1]]))
        # lineset3.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]]))
        #
        # intrinsics = o3d.camera.PinholeCameraIntrinsic()
        # intrinsics.set_intrinsics(width=depth_image.shape[0], height=depth_image.shape[1], fx=camera.depth_fx_fy[0],
        #                           fy=camera.depth_fx_fy[1],
        #                           cx=camera.depth_ppx_ppy[0], cy=camera.depth_ppx_ppy[1])
        # tic = time.time()
        # depth_3d_image = o3d.geometry.Image(depth_image)
        # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        # color_3d = o3d.geometry.Image(color)
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_3d, depth_3d_image,
        #                                                                       convert_rgb_to_intensity=False)
        # pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        # pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd1, lineset, lineset2, lineset3] + sphere_list)
