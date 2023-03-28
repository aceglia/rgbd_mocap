from pose_est.marker_class import MarkerSet, Marker
from pose_est.RgbdImages import RgbdImages
from pose_est.utils import *
import cv2

if __name__ == '__main__':
    with_camera = False
    if not with_camera:
        file_path = r'D:\Documents\Programmation\vision\image_camera_trial_1_800.bio.gzip'
        camera = RgbdImages(conf_file="config_camera_mod.json", merged_images=file_path)
    else:
        camera = RgbdImages()
        camera.init_camera(list_authorized_color_resolution()[4],
                           list_authorized_depth_resolution()[5],
                           list_authorized_fps()[-3],
                           list_authorized_fps()[-3],
                           align=True
                           )
    camera.is_frame_aligned = True
    markers_shoulder = MarkerSet(marker_names=["C7", "Scap_AA", "Scap_IA", "Acrom", "Clav_AC", "Clav_SC"],
                                 image_idx=0)
    markers_arm = MarkerSet(marker_names=["delt", "arm_l", "epic_l"],
                                 image_idx=1)
    markers_hand = MarkerSet(marker_names=["styl_u", "h_up", "h_down", "ped_l", "ped_r"],
                                 image_idx=2)
    camera.add_marker_set([markers_shoulder, markers_arm, markers_hand])
    camera.initialize_tracking(
        tracking_conf_file="tracking_conf.json",
                               crop_frame=False,
                               mask_parameters=False,
                               label_first_frame=False,
                               method=DetectionMethod.CV2Contours)
    mask_params = camera.mask_params
    while True:
        color_cropped, depth_cropped = camera.get_frames(cropped=True, aligned=True, detect_blobs=True,
                                                         label_markers=True, area_bounds=(4, 60),
                                                         bounds_from_marker_pos=True,
                                                         method=DetectionMethod.CV2Contours)
        if not isinstance(color_cropped, list):
            color_cropped = [color_cropped]
            depth_cropped = [depth_cropped]

        for i in range(len(color_cropped)):
            cv2.namedWindow("cropped_" + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow("cropped_" + str(i), color_cropped[i])
        cv2.waitKey(10)


