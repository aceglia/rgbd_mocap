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
        tracking_conf_file="tracking_conf.json",
        crop_frame=False,
        mask_parameters=False,
        label_first_frame=False,
        method=DetectionMethod.CV2Contours,
    )
    mask_params = camera.mask_params
    fig = plt.figure()
    count =0
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

        for i in range(len(color_cropped)):
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped[i], alpha=0.03), cv2.COLORMAP_JET)
            # cv2.addWeighted(depth_colormap, 0.5, color_cropped[i], 0.5, 0, color_cropped[i])
            cv2.namedWindow("cropped_" + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow("cropped_" + str(i), color_cropped[i])

        color = camera.color_frame.copy()
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
        cv2.namedWindow("Global", cv2.WINDOW_NORMAL)
        cv2.imshow("Global", color)
        cv2.waitKey(100)
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(markers_pos[0, :], markers_pos[1, :], markers_pos[2, :], c=markers_pos[2, :])
        # for i, txt in enumerate(markers_names):
        #     ax.text(markers_pos[0, i], markers_pos[1, i], markers_pos[2, i], txt)
        # 
        # # Set axis labels and title
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_title('3D Scatter Plot')
        # plt.show()
        #



