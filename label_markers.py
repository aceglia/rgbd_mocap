from pose_est.marker_class import MarkerSet
from pose_est.RgbdImages import RgbdImages
from pose_est.utils import *
import shutil
import matplotlib.pyplot as plt
import cv2
import os

if __name__ == "__main__":
    with_camera = False
    # suffix = "01-06-2023_14_38_55"
    suffix = "07-06-2023_14_47_31"
    # suffix = "07-06-2023_14_49_14"
    suffix = "04-07-2023_14_13_49"
    # suffix = "04-07-2023_15_28_08"
    suffix = "06-07-2023_18_17_59"
    suffix = "19-07-2023_16_42_48"
    # suffix = "27-06-2023_17_22_24"
    suffix = "28-07-2023_16_11_45"
    # suffix = "28-07-2023_16_17_51"
    suffix = "04-08-2023_11_52_30"
    suffix = "08-08-2023_11_12_08"
    suffix = "15-08-2023_09_13_53"
    file_name = f"gear_5"
    images_dir = None
    participants = ["P3_session2"]
    for participant in participants:
        tracking_files = []
        files = os.listdir(f"data_files\{participant}")
        files = [file for file in files if file[:4] == "gear"]
        for file in files:
            suffix = file[-19:]
            trial = file[:-20]
            if not with_camera:
                if participant:
                    images_dir = f"data_files\{participant}\{trial}_{suffix}"
                    config_file = f"config_camera_files\config_camera_{participant}.json"
                else:
                    images_dir = f"data_files\data_{suffix}"
                    config_file = f"config_camera_files\config_camera_{suffix}.json"
                # image_file = r"D:\Documents\Programmation\vision\image_camera_trial_1_800.bio.gzip"
                camera = RgbdImages(conf_file=config_file, images_dir=images_dir,
                                    start_index=0, stop_index=1000, downsampled=1)
                camera.markers_to_exclude_for_ik = ["epic_l"]
                # camera = RgbdImages(conf_file=r"config_camera_mod.json", merged_images=image_file)
            else:
                camera = RgbdImages()
                camera.init_camera(
                    ColorResolution.R_848x480,
                    DepthResolution.R_848x480,
                    FrameRate.FPS_60,
                    FrameRate.FPS_60,
                    align=True,
                )
            camera.ik_method = "kalman"  # "kalman" or "least_squares"
            camera.clipping_color = 20
            camera.is_frame_aligned = False
            markers_shoulder = MarkerSet(marker_set_name="shoulder", marker_names=["T5", "C7", "RIBS_r", "Clavsc", "Scap_AA", "Scap_IA", "Acrom"], image_idx=0)
            markers_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"], image_idx=1)
            markers_hand = MarkerSet(marker_set_name="hand", marker_names=['larm_l', "styl_r", "styl_u"], image_idx=2)
            # markers_shoulder = MarkerSet(marker_names=["C7", "Scap_AA", "Scap_IA", "Acrom", "clav_AC", "clav_SC"], image_idx=0)
            # markers_arm = MarkerSet(marker_names=["delt", "arm_l", "epic_l"], image_idx=1)
            # markers_hand = MarkerSet(marker_names=["styl_u", "styl_r", "h_up", "h_down"], image_idx=2)
            camera.add_marker_set([markers_shoulder, markers_arm, markers_hand])
            # camera.add_marker_set([markers_arm])
            kinematics_marker_set_shoulder = MarkerSet(marker_set_name="shoulder", marker_names=["T5" ,"C7", "RIBS_r", "Clavsc"])
            kinematics_marker_set_scapula = MarkerSet(marker_set_name="scapula", marker_names=[ "Scap_AA", "Scap_IA", "Acrom"])
            kinematics_marker_set_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"])
            kinematics_marker_set_hand = MarkerSet(marker_set_name="hand", marker_names=['larm_l', "styl_r", "styl_u"])

            if os.path.isfile(fr"{images_dir}\t" + f"racking_config.json"):
                tracking_conf = {"crop": False, "mask": False, "label": False, "build_kinematic_model": False}
            else:
                if len(tracking_files) == 0:
                    tracking_conf = {"crop": True, "mask": True, "label": True, "build_kinematic_model": True}
                else:
                    shutil.copy(tracking_files[0], fr"{images_dir}\t" + f"racking_config.json")
                    tracking_conf = {"crop": False, "mask": False, "label": True, "build_kinematic_model": True}

            camera.initialize_tracking(
                tracking_conf_file=fr"{images_dir}\t" + f"racking_config.json",
                crop_frame=tracking_conf["crop"],
                mask_parameters=tracking_conf["mask"],
                label_first_frame=tracking_conf["label"],
                build_kinematic_model=tracking_conf["build_kinematic_model"],
                method=DetectionMethod.CV2Blobs,
                model_name=f"{images_dir}\kinematic_model_{suffix}.bioMod",
                marker_sets=[
                    kinematics_marker_set_shoulder,
                    kinematics_marker_set_scapula,
                    kinematics_marker_set_arm,
                    kinematics_marker_set_hand
                ],
                rotation_angle=Rotation.ROTATE_180,
                with_tapir=False,
            )
            tracking_files.append(fr"{images_dir}\t" + f"racking_config.json")
            # mask_params = camera.mask_params
            # fig = plt.figure()
            import time
            count = 0
            # import open3d as o3d
            # camera.frame_idx = 0

            from biosiglive import save

            if os.path.isfile(f"{images_dir}\markers_{suffix}.bio"):
                os.remove(f"{images_dir}\markers_{suffix}.bio")
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
                    fit_model=True,
                    rotation_angle=Rotation.ROTATE_180,
                    model_name=f"{images_dir}\kinematic_model_{suffix}.bioMod",
                )
                if camera.frame_idx % 500 == 0:
                    print(camera.frame_idx)
                    print(time.time() - tic)

                if not isinstance(color_cropped, list):
                    color_cropped = [color_cropped]
                    depth_cropped = [depth_cropped]

                # for i in range(len(color_cropped)):
                # #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped[i], alpha=0.03), cv2.COLORMAP_JET)
                # #     cv2.addWeighted(depth_colormap, 0.5, color_cropped[i], 0.5, 0, color_cropped[i])
                #     cv2.namedWindow("cropped_final_" + str(i), cv2.WINDOW_NORMAL)
                #     cv2.imshow("cropped_final_" + str(i), color_cropped[i])

                color = camera.color_frame.copy()
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera.depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                # cv2.addWeighted(depth_colormap, 0.5, color, 0.5, 0, color)
                depth_image = camera.depth_frame.copy()
                markers_pos, markers_names, occlusions, reliability_idx = camera.get_global_markers_pos()
                markers_in_meters, _, _, _ = camera.get_global_markers_pos_in_meter(markers_pos)
                # print(markers_pos)

                dic = {"markers_in_meters": markers_in_meters[:, :, np.newaxis],
                       "markers_names": markers_names,
                       "occlusions": occlusions,
                       "reliability_idx": reliability_idx,
                       "time_to_process": time.time() - tic}
                save(dic, f"{images_dir}\markers_kalman_test.bio")

                color = draw_markers(
                    color,
                    markers_pos=markers_pos,
                    markers_names=markers_names,
                    is_visible=occlusions,
                    scaling_factor=0.5,
                    markers_reliability_index=reliability_idx,
                )
                cv2.namedWindow("color", cv2.WINDOW_NORMAL)
                cv2.imshow("color", color)
                # print(time.time() - tic)
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