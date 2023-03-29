from pose_est.marker_class import MarkerSet, Marker
from pose_est.RgbdImages import RgbdImages
from pose_est.utils import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import cv2

if __name__ == "__main__":
    with_camera = False
    if not with_camera:
        file_path = "videos/image_camera_trial_1_short.bio.gzip"
        camera = RgbdImages(conf_file="config_camera_mod.json", merged_images=file_path)
    else:
        camera = RgbdImages()
        camera.init_camera(
            list_authorized_color_resolution()[4],
            list_authorized_depth_resolution()[5],
            list_authorized_fps()[-3],
            list_authorized_fps()[-3],
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
            cv2.namedWindow("cropped_" + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow("cropped_" + str(i), color_cropped[i])

        color = camera.color_frame.copy()
        markers_pos, markers_names, occlusions = camera.get_merged_global_markers_pos()
        depth = camera.get_markers_depth()
        color = draw_markers(
            color,
            markers_pos=markers_pos,
            markers_names=markers_names,
            is_visible=occlusions,
        )

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(markers_pos[0, :], markers_pos[1, :], depth, c=depth, cmap='Greens')

        # Set axis labels and title
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('3D Scatter Plot')

        cv2.namedWindow("Global", cv2.WINDOW_NORMAL)
        cv2.imshow("Global", color)
        cv2.waitKey(5)

        # plt.show()
