import numpy as np

from pose_est.marker_class import MarkerSet
from scipy import linalg
from pose_est.RgbdImages import RgbdImages
from pose_est.msk_utils import *
from vtk import *
from biosiglive import MskFunctions, PlotType, LivePlot, InverseKinematicsMethods
from pose_est.msk_utils import _init_casadi_function, perform_biomechanical_pipeline
from pose_est.utils import *
import shutil
from scipy.interpolate import CubicSpline
import biorbd
import biorbd_casadi as biorbd_ca
import matplotlib.pyplot as plt
import cv2
import os

try:
    import casadi as ca
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
except ImportError:
    pass


def order_markers_from_names(ordered_name, data_name, data):
    if isinstance(data_name, np.ndarray):
        data_name = data_name.tolist()
    reordered_data = np.zeros((3, len(ordered_name), data.shape[2]))
    for i, name in enumerate(ordered_name):
        if not isinstance(name, str):
            name = name.to_string()
        reordered_data[:, i, :] = data[:, data_name.index(name), :]
    return reordered_data


def compute_joint_reaction_force(model, q, qdot, tau, f_ext, return_all=False):
    pass


def snapshot(windows):
    w2if = vtkWindowToImageFilter()
    w2if.SetInput(windows)
    w2if.Update()
    vtk_image = w2if.GetOutput()
    return vtk_image


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
    save_data = False
    delete_old_data = False
    file_name = f"gear_5"
    images_dir = None
    participants = ["P4_session2", "P3_session2"]  # "P4_session2",
    start_idx = [0] * len(participants)
    # start_idx[0] = 0

    init_batch = False
    c = 0
    data_files = "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files"
    for participant in participants:
        tracking_files = []
        files = os.listdir(f"{data_files}{os.sep}{participant}")
        if participant == "P4_session2":
            files = ["gear_15_15-08-2023_10_45_08"]
        else:
            files = [file for file in files if file[:7] == "gear_20"]
        for file in files:
            # if participant == "P3_session2" and file in ["gear_10_15-08-2023_09_21_39", "gear_15_15-08-2023_09_28_39",
            #                                              "gear_20_15-08-2023_09_35_38"]:
            #     continue
            suffix = file[-19:]
            trial = file[:-20]
            if not with_camera:
                if participant:
                    images_dir = f"{data_files}{os.sep}{participant}{os.sep}{trial}_{suffix}"
                    config_file = f"config_camera_files{os.sep}config_camera_{participant}.json"
                else:
                    images_dir = f"{data_files}{os.sep}data_{suffix}"
                    config_file = f"config_camera_files{os.sep}config_camera_{suffix}.json"
                # image_file = r"D:\Documents\Programmation\vision\image_camera_trial_1_800.bio.gzip"
                if os.path.isfile(rf"{images_dir}{os.sep}t" + f"racking_config.json"):
                    tracking_conf = {"crop": False, "mask": False, "label": False, "build_kinematic_model": False}
                    start_idx[c] = start_idx_from_json(rf"{images_dir}{os.sep}t" + f"racking_config.json")
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
                camera.markers_to_exclude_for_ik = ["epic_l"]
                # camera.markers_to_exclude_for_ik = ["RIBS_r", "styl_r"]

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
            markers_shoulder = MarkerSet(
                marker_set_name="shoulder",
                marker_names=["T5", "C7", "RIBS_r", "Clavsc", "Scap_AA", "Scap_IA", "Acrom"],
                image_idx=0,
            )
            markers_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"], image_idx=1)
            markers_hand = MarkerSet(marker_set_name="hand", marker_names=["larm_l", "styl_r", "styl_u"], image_idx=2)
            # markers_shoulder = MarkerSet(marker_names=["C7", "Scap_AA", "Scap_IA", "Acrom", "clav_AC", "clav_SC"], image_idx=0)
            # markers_arm = MarkerSet(marker_names=["delt", "arm_l", "epic_l"], image_idx=1)
            # markers_hand = MarkerSet(marker_names=["styl_u", "styl_r", "h_up", "h_down"], image_idx=2)
            camera.add_marker_set([markers_shoulder, markers_arm, markers_hand])
            # camera.add_marker_set([markers_arm])
            kinematics_marker_set_shoulder = MarkerSet(
                marker_set_name="shoulder", marker_names=["T5", "C7", "RIBS_r", "Clavsc"]
            )
            kinematics_marker_set_scapula = MarkerSet(
                marker_set_name="scapula", marker_names=["Scap_AA", "Scap_IA", "Acrom"]
            )
            kinematics_marker_set_arm = MarkerSet(marker_set_name="arm", marker_names=["delt", "arm_l", "epic_l"])
            kinematics_marker_set_hand = MarkerSet(marker_set_name="hand", marker_names=["larm_l", "styl_r", "styl_u"])

            camera.initialize_tracking(
                tracking_conf_file=rf"{images_dir}{os.sep}t" + f"racking_config.json",
                crop_frame=tracking_conf["crop"],
                mask_parameters=tracking_conf["mask"],
                label_first_frame=tracking_conf["label"],
                build_kinematic_model=tracking_conf["build_kinematic_model"],
                method=DetectionMethod.CV2Blobs,
                model_name=f"{images_dir}{os.sep}kinematic_model_{suffix}.bioMod",
                marker_sets=[
                    kinematics_marker_set_shoulder,
                    kinematics_marker_set_scapula,
                    kinematics_marker_set_arm,
                    kinematics_marker_set_hand,
                ],
                rotation_angle=Rotation.ROTATE_180,
                with_tapir=False,
            )
            tracking_files.append(rf"{images_dir}{os.sep}t" + f"racking_config.json")
            # mask_params = camera.mask_params
            # fig = plt.figure()
            import time

            camera.stop_index = 5
            count = 0
            # import open3d as o3d
            # camera.frame_idx = 0

            from biosiglive import save

            if delete_old_data and os.path.isfile(f"{images_dir}{os.sep}markers_kalman.bio"):
                os.remove(f"{images_dir}{os.sep}markers_kalman.bio")
            model_path = f"{data_files}{os.sep}P4_session2{os.sep}model_depth_scaled.bioMod"
            model_path = "wu_bras_gauche_seth.bioMod"
            ik_function = MskFunctions(model=model_path, data_buffer_size=6)
            model_ca = biorbd_ca.Model(model_path)
            # marker_plot = LivePlot(name="markers", plot_type=PlotType.Skeleton)
            # marker_plot.init(model_path=f"{data_files}{os.sep}P4_session2{os.sep}model_depth_scaled.bioMod", show_floor=False,
            #                  show_muscles=False)
            # plot_curve = LivePlot(
            #     name="curve",
            #     rate=60,
            #     plot_type=PlotType.Curve,
            #     nb_subplots=2,
            #     # channel_names=["1", "2", "3", "4"],
            # )
            # plot_curve.init(plot_windows=200,
            #                 # y_labels=["Strikes", "Strikes", "Force (N)", "Force (N)"])
            #                 )
            from biosiglive import RealTimeProcessing

            # marker_plot.viz.set_camera_position(0, 0, -5)
            # marker_plot.viz.set_camera_roll(-179)
            # marker_plot.viz.set_camera_zoom(1)
            # marker_plot.viz.set_camera_focus_point(0, 0, 0)
            # marker_plot.viz.vtk_window.avatar_widget.GetRenderWindow().SetSize(848,480)
            rt_proc_method = [RealTimeProcessing(60, 5), RealTimeProcessing(60, 5), RealTimeProcessing(60, 5)]
            t = [0]
            q_df = np.zeros((16, 2))
            q_dot_df = np.zeros((16, 2))
            q_proc = np.zeros((16, 10))
            q_extr = np.zeros((16, 13))
            solver = None
            ca_function = _init_casadi_function(model_ca)
            mus_act, res_tau = None, None
            freq = 100  # Hz
            params = biorbd.KalmanParam(freq)
            kalman = biorbd.KalmanReconsMarkers(ik_function.model, params)
            if os.path.isfile("ik.bio"):
                os.remove("ik.bio")
            scaling_factor = (100, 10)
            time_ik, time_id, time_static_opt, time_process, total_time = [], [], [], [], []
            emg = load_data(
                "P4",
                "gear_15",
                all_paths={"trial_dir": f"{data_files}{os.sep}P4_session2{os.sep}gear_15_15-08-2023_10_45_08{os.sep}"},
                model=ik_function.model,
            )

            from biosiglive import ExternalLoads

            forces = ExternalLoads()
            forces.add_external_load(
                point_of_application=[0, 0, 0],
                applied_on_body="radius_left_pro_sup_left",
                express_in_coordinate="ground",
                name="hand_pedal",
                load=np.zeros((6, 1)),
            )
            while True:
                # if camera.frame_idx == 10:
                #     break
                # if camera.frame_idx > 2 and camera.frame_idx < len(camera.all_color_files) - 1:
                #     last_idx = int(camera.all_color_files[start_idx[c - 1] + camera.frame_idx - 1].split("\\")[-1].split("_")[-1].removesuffix(".png"))
                #     current_idx = int(
                #     camera.all_color_files[start_idx[c - 1] + camera.frame_idx].split("\\")[-1].split("_")[-1].removesuffix(".png"))
                #     if current_idx - last_idx > 8:
                #         print(f"Difference between frames is too big at idx {camera.frame_idx}, skipping trial...")
                #         # if os.path.isfile(f"{images_dir}\markers_kalman.bio"):
                #         #     os.remove(f"{images_dir}\markers_kalman.bio")
                #         cv2.destroyAllWindows()
                #         break

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
                    model_name=f"{images_dir}{os.sep}kinematic_model_{suffix}.bioMod",
                )
                if camera.frame_idx % 500 == 0:
                    print(camera.frame_idx)

                if not isinstance(color_cropped, list):
                    color_cropped = [color_cropped]
                    depth_cropped = [depth_cropped]
                # distance between two markers
                # for i in range(len(color_cropped)):
                #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped[i], alpha=0.03), cv2.COLORMAP_JET)
                #     cv2.addWeighted(depth_colormap, 0.5, color_cropped[i], 0.5, 0, color_cropped[i])
                #     cv2.namedWindow("cropped_final_" + str(i), cv2.WINDOW_NORMAL)
                #     cv2.imshow("cropped_final_" + str(i), color_cropped[i])
                tic = time.time()
                color = camera.color_frame.copy()
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera.depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                # cv2.addWeighted(depth_colormap, 0.5, color, 0.5, 0, color)
                depth_image = camera.depth_frame.copy()
                markers_pos, markers_names, occlusions, reliability_idx = camera.get_global_markers_pos()
                markers_in_meters, _, _, _ = camera.get_global_markers_pos_in_meter(markers_pos)
                # print(markers_pos)
                time_to_get_markers = time.time() - tic
                # print(camera.frame_idx)
                if camera.frame_idx > 0:
                    time_bt_2_frames = (1 / 60) * (
                        camera.camera_frame_numbers[camera.frame_idx]
                        - camera.camera_frame_numbers[camera.frame_idx - 1]
                    )
                    time_bt_2_frames = np.round(time_bt_2_frames, 3)
                    if len(t) > 0:
                        t.append(t[-1] + time_bt_2_frames)
                    else:
                        t.append(time_bt_2_frames)

                # cv2.namedWindow("color", cv2.WINDOW_NORMAL)
                # cv2.imshow("color", color)
                # cv2.waitKey(10000)
                dic = {
                    "markers_in_meters": markers_in_meters[:, :, np.newaxis],
                    "markers_in_pixel": markers_pos[:, :, np.newaxis],
                    "markers_names": markers_names,
                    "occlusions": occlusions,
                    "reliability_idx": reliability_idx,
                    "time_to_process": camera.time_to_get_frame + time_to_get_markers,
                    "time": t[-1],
                    "frame_idx": camera.camera_frame_numbers[camera.frame_idx],
                }
                if save_data:
                    save(dic, f"{images_dir}{os.sep}markers_kalman.bio")
                model_ordered_markers_names = [
                    "C7",
                    "T5",
                    "RIBS_r",
                    "Clavsc",
                    "Acrom",
                    "Scap_AA",
                    "Scap_IA",
                    "delt",
                    "epic_l",
                    "arm_l",
                    "styl_u",
                    "styl_r",
                    "larm_l",
                ]
                # markers_in_meters, _, _, _ = camera.get_global_markers_pos_in_meter(camera.rt_proc_ma)
                ordered_markers = order_markers_from_names(
                    model_ordered_markers_names, markers_names, markers_in_meters[:, :, np.newaxis]
                )
                rt_matrix = load(rf"{data_files}{os.sep}P4_session2{os.sep}RT_optimal.bio")
                r = rt_matrix["rotation_matrix"]
                T = rt_matrix["translation_matrix"]

                new_ordered_markers = np.zeros(ordered_markers.shape)
                for i in range(ordered_markers.shape[2]):
                    new_ordered_markers[:, :, i] = np.dot(np.array(r), np.array(ordered_markers[:, :, i])) + np.array(T)
                ordered_markers = new_ordered_markers
                time_process = camera.time_to_get_frame
                forces.update_external_load_value(np.array([0, 0, 1, 0, 0, 0]), name="hand_pedal")
                result_biomech = perform_biomechanical_pipeline(
                    ordered_markers, ik_function, camera.frame_idx, forces, scaling_factor, emg
                )
                result_biomech["time"]["process_time"] = time_process
                result_biomech["markers_names"] = model_ordered_markers_names
                result_biomech["frame_idx"] = camera.camera_frame_numbers[camera.frame_idx]
                save(result_biomech, "ik.bio", add_data=True)
                if camera.frame_idx == len(camera.all_color_files) - 1 or (init_batch and count == 5):
                    cv2.destroyAllWindows()
                    break
                count += 1
                # cv2.waitKey(1000)

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
