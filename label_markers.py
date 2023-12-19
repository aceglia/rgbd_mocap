from rgbd_mocap.utils import *
from utils.kin_marker_set import KinMarkerSet
from utils.inits import init_RgbdImages, init_tracking_conf
import time
import cv2
import os

# path_to_project = "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/"
path_to_project = "/home/user/KaelFacon/Project/rgbd_mocap/"


def main():
    with_camera = False
    images_dir = None
    participants = ["P4_session2"]  # , "P3_session2", "P4_session2", "P2_session2"]  # "P4_session2", "P8"
    start_idx = [85] * len(participants)
    init_batch = False
    save_data = False
    delete_old_data = False
    c = 0
    data_files = f"{path_to_project}data_files"
    for participant in participants:
        tracking_files = []
        files = [file for file in os.listdir(f"{data_files}{os.sep}{participant}") if file[:7] == "gear_20"]
        for file in files:
            suffix = file[-19:]
            trial = file[:-20]

            ### Dict containing various paths to init the camera
            files_paths = {"path_to_project": path_to_project,
                           "participant": participant,
                           "data_files": data_files,
                           "trial": trial,
                           "suffix": suffix,
                           "file": file,
                           "start_idx": start_idx[c],
                           }
            c += 1

            ### Init camera (Rgbd_image)
            camera = init_RgbdImages(with_camera, files_paths)
            images_dir = files_paths['images_dir']  # Should be updated/set in the 'init_RgbdImages' function

            ### Get the tracking configuration from file
            tracking_config = init_tracking_conf(images_dir, tracking_files)
            tracking_config_file = f"{images_dir}{os.sep}tracking_config.json"

            ### Init KinMarkerSet and add MarkersSets to the camera
            kin_marker_set = KinMarkerSet(camera, KinMarkerSet.BACK_3)

            camera.initialize_tracking(
                tracking_conf_file=tracking_config_file,
                method=DetectionMethod.CV2Blobs,
                model_name=f"{files_paths['images_dir']}{os.sep}kinematic_model_{suffix}.bioMod",
                marker_sets=kin_marker_set,
                rotation_angle=Rotation.ROTATE_180,
                with_tapir=False,
                **tracking_config,
            )
            tracking_files.append(tracking_config_file)
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

                if cv2.waitKey(1) == ord("q"):
                    break

            camera.process_handler.end_process()

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


if __name__ == "__main__":
    main()
