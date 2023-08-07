import os

import numpy as np

from pose_est.marker_class import MarkerSet
from pose_est.RgbdImages import RgbdImages
from pose_est.utils import *
import matplotlib.pyplot as plt
import cv2

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
    file_name = f"gear_10"
    participant = "P2"
    # participant = None

    if not with_camera:
        if participant:
            images_dir = f"data_files\{participant}\{file_name}_{suffix}"
            config_file = f"config_camera_files\config_camera_{participant}.json"
        else:
            images_dir = f"data_files\data_{suffix}"
            config_file = f"config_camera_files\config_camera_{suffix}.json"
        # image_file = r"D:\Documents\Programmation\vision\image_camera_trial_1_800.bio.gzip"
        camera = RgbdImages(conf_file=config_file, images_dir=images_dir,
                            start_index=310, stop_index=850, downsampled=1)
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

    camera.initialize_tracking(
        tracking_conf_file=r"tracking_config_files\t" + f"racking_conf_{suffix}.json",
        crop_frame=False,
        mask_parameters=False,
        label_first_frame=False,
        build_kinematic_model=False,
        method=DetectionMethod.CV2Blobs,
        model_name=f"kinematic_model\kinematic_model_{suffix}.bioMod",
        marker_sets=[
            kinematics_marker_set_shoulder,
            kinematics_marker_set_scapula,
            kinematics_marker_set_arm,
            kinematics_marker_set_hand
        ],
        rotation_angle=Rotation.ROTATE_180,
        with_tapir=True,
    )
    mask_params = camera.mask_params
    fig = plt.figure()
    import time
    count = 0
    # import open3d as o3d
    camera.frame_idx = 0

    from biosiglive import save

    # if os.path.isfile(f"markers_{suffix}.bio"):
    #     os.remove(f"markers_{suffix}.bio")
    while True:
        tic = time.time()
        color_cropped, depth_cropped = camera.get_frames(
            aligned=False,
            detect_blobs=True,
            label_markers=True,
            bounds_from_marker_pos=False,
            method=DetectionMethod.CV2Blobs,
            filter_with_kalman=False,
            adjust_with_blobs=True,
            fit_model=True,
            rotation_angle=Rotation.ROTATE_180,
            model_name=f"kinematic_model\kinematic_model_{suffix}.bioMod",
        )
        print(camera.frame_idx)
        if not isinstance(color_cropped, list):
            color_cropped = [color_cropped]
            depth_cropped = [depth_cropped]

        for i in range(len(color_cropped)):
        #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped[i], alpha=0.03), cv2.COLORMAP_JET)
        #     cv2.addWeighted(depth_colormap, 0.5, color_cropped[i], 0.5, 0, color_cropped[i])
            cv2.namedWindow("cropped_final_" + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow("cropped_final_" + str(i), color_cropped[i])

        color = camera.color_frame.copy()
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera.depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.addWeighted(depth_colormap, 0.5, color, 0.5, 0, color)
        depth_image = camera.depth_frame.copy()
        markers_pos, markers_names, occlusions, reliability_idx = camera.get_global_markers_pos()
        markers_in_meters, _, _, _ = camera.get_global_markers_pos_in_meter(markers_pos)
        # print(markers_pos)

        dic = {"markers_in_meters": markers_in_meters[:, :, np.newaxis], "markers_names": markers_names}
        # save(dic, f"markers_{suffix}.bio")

        color = draw_markers(
            color,
            markers_pos=markers_pos,
            markers_names=markers_names,
            is_visible=occlusions,
            scaling_factor=0.5,
            markers_reliability_index=reliability_idx,
        )
        # from biosiglive import MskFunctions, InverseKinematicsMethods
        cv2.namedWindow("color", cv2.WINDOW_NORMAL)
        cv2.imshow("color", color)
        # biomod_model = f"kinematic_model_{suffix}.bioMod"
        # markers_pos = markers_in_meters[:, :, np.newaxis]
        # markers_pos = np.repeat(markers_pos, 10, axis=2)
        # funct = MskFunctions(model=biomod_model, data_buffer_size=10)
        # q_recons, _ = funct.compute_inverse_kinematics(markers_pos[:, :, :],
        #                                                method=InverseKinematicsMethods.BiorbdLeastSquare, kalman_freq=100)
        # import bioviz
        # b = bioviz.Viz(model_path=biomod_model, show_floor=False)
        # b.load_movement(q_recons)  # Q from kalman array(nq, nframes)
        # b.load_experimental_markers(markers_pos)  # experimental markers array(3, nmarkers, nframes)
        # b.exec()
        print(time.time() - tic)
        # cv2.waitKey(16)
        if camera.frame_idx > 380:
            while True:
                if cv2.waitKey(16) & 0xFF == ord('q'):
                    break
        else:
            cv2.waitKey(16)
        # gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # equ = cv2.equalizeHist(gray)
        # clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
        # cl1 = clahe.apply(gray)
        # res = np.hstack((gray, equ))  # stacking images side-by-side
        # cv2.namedWindow("CLahe", cv2.WINDOW_NORMAL)
        # cv2.imshow("CLahe", cl1)
        # cv2.namedWindow("equ", cv2.WINDOW_NORMAL)
        # cv2.imshow("equ", equ)
        # cv2.namedWindow("color", cv2.WINDOW_NORMAL)
        # cv2.imshow("color", gray)
        # cv2.waitKey(1000000)
        # if count == 1:
        #     cv2.waitKey(1000000)
        # else:
        #     cv2.waitKey(1)
        # count += 1
        # import pickle
        # with open("markers_pos_test.pkl", "wb") as f:
        #     pickle.dump({"mark_pos": markers_pos, "mark_pos_in_meters": markers_in_meters}, f)

        # ax = plt.axes(projection='3d')
        # ax.scatter3D(markers_in_meters[0, :], markers_in_meters[1, :], markers_in_meters[2, :], c=markers_in_meters[2, :])
        # for i, txt in enumerate(markers_names):
        #     ax.text(markers_in_meters[0, i], markers_in_meters[1, i], markers_in_meters[2, i], txt)
        #
        # # Set axis labels and title
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_title('3D Scatter Plot')
        # plt.show()
        #

        # # decoment these lines to see poind cloud and 3d markers
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




