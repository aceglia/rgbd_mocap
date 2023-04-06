from pose_est.marker_class import MarkerSet
from pose_est.RgbdImages import RgbdImages
from pose_est.utils import *
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    with_camera = False
    if not with_camera:
        file_path = r"D:\Documents\Programmation\vision\image_camera_trial_1_short.bio.gzip"
        camera = RgbdImages(conf_file="config_camera_mod.json", merged_images=file_path)
    else:
        camera = RgbdImages()
        camera.init_camera(
            ColorResolution.R_848x480,
            DepthResolution.R_848x480,
            FrameRate.FPS_30,
            FrameRate.FPS_30,
            align=True,
        )
    camera.is_frame_aligned = True
    markers_shoulder = MarkerSet(marker_names=["C7", "Scap_AA", "Scap_IA", "Acrom", "Clav_AC", "Clav_SC"], image_idx=0)
    markers_arm = MarkerSet(marker_names=["delt", "arm_l", "epic_l"], image_idx=1)
    markers_hand = MarkerSet(marker_names=["styl_u", "h_up", "h_down", "ped_l", "ped_r"], image_idx=2)
    camera.add_marker_set([markers_shoulder, markers_arm, markers_hand])
    camera.initialize_tracking(
        tracking_conf_file="tracking_conf_cycling.json",
        crop_frame=False,
        mask_parameters=False,
        label_first_frame=False,
        method=DetectionMethod.CV2Contours,
    )
    mask_params = camera.mask_params
    fig = plt.figure()
    import time
    count = 0
    import open3d as o3d
    while True:
        color_cropped, depth_cropped = camera.get_frames(
            aligned=True,
            detect_blobs=True,
            label_markers=True,
            area_bounds=(4, 60),
            bounds_from_marker_pos=True,
            method=DetectionMethod.CV2Contours,
        )
        if not isinstance(color_cropped, list):
            color_cropped = [color_cropped]
            depth_cropped = [depth_cropped]

        # for i in range(len(color_cropped)):
        #     # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped[i], alpha=0.03), cv2.COLORMAP_JET)
        #     # cv2.addWeighted(depth_colormap, 0.5, color_cropped[i], 0.5, 0, color_cropped[i])
        #     cv2.namedWindow("cropped_" + str(i), cv2.WINDOW_NORMAL)
        #     cv2.imshow("cropped_" + str(i), color_cropped[i])

        color = camera.color_frame.copy()
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera.depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.addWeighted(depth_colormap, 0.5, color, 0.5, 0, color)

        depth_image = camera.depth_frame.copy()
        markers_pos, markers_names, occlusions = camera.get_merged_global_markers_pos()
        color = draw_markers(
            color,
            markers_pos=markers_pos,
            markers_names=markers_names,
            is_visible=occlusions,
            scaling_factor=0.5,
        )
        count += 1
        # cv2.namedWindow("Global", cv2.WINDOW_NORMAL)
        # cv2.imshow("Global", color)
        # cv2.waitKey(100)
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(markers_pos[0, :], markers_pos[1, :], markers_pos[2, :], c=markers_pos[2, :])
        # for i, txt in enumerate(markers_names):
        #     ax.text(markers_pos[0, i], markers_pos[1, i], markers_pos[2, i], txt)
        #

        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(width=depth_image.shape[0], height=depth_image.shape[1], fx=camera.depth_fx_fy[0],
                                  fy=camera.depth_fx_fy[1],
                                  cx=camera.depth_ppx_ppy[0], cy=camera.depth_ppx_ppy[1])
        tic = time.time()
        depth_3d_image = o3d.geometry.Image(depth_image)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color_3d = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_3d, depth_3d_image,
                                                                              convert_rgb_to_intensity=False)
        pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        print("time to create point cloud: ", time.time() - tic)
        #
        o3d.visualization.draw_geometries([pcd1])
        # Visualize point cloud
        # pointcloud += pcd1
        # vis.add_geometry(pointcloud)
        # # vis.update_geometry()
        # vis.poll_events()
        # vis.update_renderer()
        import time
        time.sleep(0.05)

        # Set axis labels and title
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_title('3D Scatter Plot')
        # plt.show()
        #



