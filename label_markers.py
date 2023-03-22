from pose_est.marker_class import MarkerSet
from pose_est.RgbdImages import RgbdImages
import cv2

if __name__ == '__main__':
    crop = True
    file_path = r'D:\Documents\Programmation\vision\image_camera_trial_1_short.bio.gzip'
    camera = RgbdImages(conf_file="config_camera.json", merged_images=file_path)
    if crop:
        start_crop, end_crop = camera.select_cropping()
        print(start_crop, end_crop)
    else:
        camera.set_cropping_area([[211, 276, 379], [209, 314, 235]], [[326, 415, 533], [315, 413, 372]])

    color_cropped, depth_cropped = camera.get_frames(cropped=False)
    if not isinstance(color_cropped, list):
        color_cropped = [color_cropped]
        depth_cropped = [depth_cropped]
    for i in range(len(color_cropped)):
        cv2.namedWindow("cropped_" + str(i), cv2.WINDOW_NORMAL)
    while True:
        color_cropped, depth_cropped = camera.get_frames(cropped=False)
        if not isinstance(color_cropped, list):
            color_cropped = [color_cropped]
            depth_cropped = [depth_cropped]
        for i in range(len(color_cropped)):
            cv2.imshow("cropped_" + str(i), color_cropped[i])
        cv2.waitKey(10)


