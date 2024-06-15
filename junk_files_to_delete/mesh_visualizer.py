# examples/Python/Basic/mesh.py

import copy
import numpy as np
import open3d as o3d
import glob
import time
from biosiglive import load
from pathlib import Path
import biorbd

# from pyorerun import BiorbdModel, MultiPhaseRerun
# from pyorerun import BiorbdModel, MultiPhaseRerun

def get_mesh_from_model(model):
    with open(model, "r") as f:
        lines = f.read()
    mesh = []
    next_vtp = 1000
    while next_vtp != -1:
        next_vtp = lines.find("meshfile")
        if next_vtp != -1:
            lines = lines[next_vtp + len(r"meshfile\tGeometry"):]
            next_vtp = lines.find("vtp")
            mesh.append(lines[:next_vtp - 1])
    return mesh
import cv2

if __name__ == "__main__":
    mesh_dir = "/Geometry_left"
    path = "Q:\Projet_hand_bike_markerless\RGBD\P9\only_rgbd_11-01-2024_17_27_39"
    depth_file = glob.glob(path + "/depth*.png")
    color_file = glob.glob(path + "/color*.png")
    idx = []
    for f in depth_file:
        idx.append(int(f.split("\\")[-1].split("_")[-1].removesuffix(".png")))
    idx.sort()
    idx = idx[idx.index(3588):]

    results = load(path + "/result_biomech_only_rgbd_11-01-2024_17_27_39__hist_sharp_new_alone.bio")
    q = results["dlc"]["q_raw"]
    model_path = f"Q:\Projet_hand_bike_markerless\RGBD\P9/models/only_rgbd_11-01-2024_17_27_model_scaled_depth_new_seth.bioMod"
    b_model = biorbd.Model(model_path)
    # biorbd_model = BiorbdModel(model_path)
    # t_span = np.linspace(0, 1, q.shape[1])
    depth_image = cv2.imread(depth_file[0], cv2.IMREAD_ANYDEPTH)
    # depth_image = depth_image.repeat(q.shape[1], axis=2).astype(np.uint16)
    #
    multi_rerun = MultiPhaseRerun()

    multi_rerun.add_phase(t_span=t_span, phase=0, window="animation")
    multi_rerun.add_animated_model(biorbd_model, q, phase=0, window="animation")
    multi_rerun.add_depth_image(name="depth", depth_image=depth_image, phase=0, window="animation")

    multi_rerun.rerun("multi_model_test")

    mesh_names = get_mesh_from_model(model_path)
    mesh_file = glob.glob(mesh_dir + "/*.ply")
    if len(mesh_file) == 0:
        raise FileNotFoundError("No mesh file found")
    all_mesh = []
    vis = o3d.visualization.Visualizer()
    vis.create_window("vis", 848, 480)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = [[
                    0.999995768070221,
                    -0.002411586232483387,
                    -0.001641725655645132
                ],
                    [
                        0.0023998860269784927,
                        0.9999719858169556,
                        -0.007091784384101629
                    ],
                    [
                        0.0016587821301072836,
                        0.0070878141559660435,
                        0.999973475933075
                    ]
                ]
    extrinsic[:3, 3] = [-0.059028711169958115,
        0.00020984711591154337,
        0.00028735643718391657]
    rotated_extrinsic = np.dot(extrinsic, [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    extrinsic = rotated_extrinsic
    # extrinsic[:3, 3] = [-0.077565903867894193,
	# 	0.039082049189110957,
	# 	0.066057441784709292]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    # intrinsic.set_intrinsics(width=848, height=480, fx=429.6269226074219,
    #                           fy=429.6269226074219,
    #                           cx=422.17901611328125, cy=243.9992218017578)
    intrinsic.set_intrinsics(width=848, height=480,
                             fx=419.5417175292969,
                              fy=419.13189697265625,
                              cx=418.1453857421875
                             ,cy=245.16981506347656)

    # Set the camera parameters
    camera_params = o3d.camera.PinholeCameraParameters()

    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extrinsic
    # ctr.convert_from_pinhole_camera_parameters()

    # camera_params = ctr.convert_to_pinhole_camera_parameters()
    # print(camera_params)
    # ctr.set_constant_z_far(1000)
    ctr = vis.get_view_control()
    # ctr.set_constant_z_near(0.1)
    # param = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-05-06-16-02-13.json")
    # ctr.convert_from_pinhole_camera_parameters(param, True)

    # ctr.set_scale()

    # ctr.set_lookat([0, 0, 10])
    # ctr.set_up([0, 1, 0])
    # ctr.translate(848/2, 0)
    # ctr.set_zoom(-100)
    # ctr.set_front([0, 0, -1])
    # vis.register_animation_callback(None)
    # ctr.camera_local_translate(0, 0, -1000)
    # ctr.set_zoom(1.2)
    all_mesh_names = []

    for file in mesh_file:
        # if Path(file).stem == "thorax":
        mesh = o3d.io.read_triangle_mesh(file)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1, 0.706, 0])
        all_mesh_names.append(Path(file).stem)
        all_mesh.append(mesh)
        vis.add_geometry(mesh)
        # break
    import cv2
    depth_image = cv2.imread(depth_file[0], cv2.IMREAD_ANYDEPTH)
    color = cv2.imread(color_file[0])
    import open3d as o3d
    # pt_cloud = o3d.geometry.PointCloud()
    # pt_cloud.points = o3d.utility.Vector3dVector(markers_in_meters.T)
    # pt_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # sphere_list = []
    # for pt in pt_cloud.points:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    #     sphere.translate(pt)
    #     sphere.paint_uniform_color([0.8, 0.2, 0.2])  # Set color of spheres to red
    #     sphere_list.append(sphere)

    # Create line sets for x-, y-, and z- axes
    lineset = o3d.geometry.LineSet()

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
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0.0, 0.0, 0.0]))
    # intrinsics = o3d.camera.PinholeCameraIntrinsic()
    # intrinsics.set_intrinsics(width=depth_image.shape[0], height=depth_image.shape[1], fx=camera.depth_fx_fy[0],
    #                           fy=camera.depth_fx_fy[1],
    #                           cx=camera.depth_ppx_ppy[0], cy=camera.depth_ppx_ppy[1])
    # tic = time.time()
    depth_3d_image = o3d.geometry.Image(depth_image)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color_3d = o3d.geometry.Image(color)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_3d, depth_3d_image,
                                                                          convert_rgb_to_intensity=False)
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # vis.draw_geometries([pcd1, lineset, lineset2, lineset3])
    # vis.add_geometry(pcd1)
    # vis.add_geometry(origin)
    # vis.add_geometry(lineset)
    # vis.add_geometry(lineset2)
    # vis.add_geometry(lineset3)
    previous_rt = [None] * len(all_mesh)
    # o3d.visualization.draw_geometries([pcd1, lineset, lineset2, lineset3])
    ctr = vis.get_view_control()
    # param = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-05-06-16-02-13.json")
    ctr.convert_from_pinhole_camera_parameters(camera_params, True)
    for i in range(q.shape[1]):
        # ctr.set_constant_z_near(0.1)
        color = cv2.imread(f"Q:\Projet_hand_bike_markerless\RGBD\P9\only_rgbd_11-01-2024_17_27_39\color_{idx[i]}.png")

        # b_model.UpdateKinematicsCustom(q[:, i])
        # if i == 0:
        mesh_idx = 0
        for m, mesh in enumerate(all_mesh):
            for j in range(len(b_model.segments())):
                p = b_model.segments()[j].characteristics().mesh().path().absolutePath().to_string()
                if Path(p).stem == all_mesh_names[m]:
                    # mesh_names_idx = mesh_names.index(all_mesh_names[m])
                    mesh_idx = j
                    # print(b_model.segments()[j].name().to_string())
                    break
            # mesh_idx = mesh_names.index(all_mesh_names[m])

            rt_matrix = b_model.allGlobalJCS(q[:, i])[mesh_idx].to_array()
            # rt_matrix = b_model.allGlobalJCS(np.zeros_like(q[:, i]))[mesh_idx].to_array()
            t = rt_matrix[:3, 3]
            r = rt_matrix[:3, :3]
            # mesh.get_oriented_bounding_box().R = r
            # mesh.get_oriented_bounding_box().center = t

            # mesh.set_transform(rt_matrix)
            # r = r.dot([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            # rotate of 90 degrees around x
            # r = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            # new_mesh = mesh
            # new_mesh.paint_uniform_color([0.2, 0.706, 0.3])
            # t[-1] = -t[-1]
            # t[1] = -t[1]

            mesh.get_center()
            # mesh.translate(-mesh.get_center())
            # mesh.rotate(r, (0,0,0))
            # mesh.translate(t)
            if previous_rt[m] is not None:
                mesh.transform(np.linalg.inv([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
                # mesh.translate(-previous_rt[m][:3, 3])
                # mesh.rotate(np.linalg.inv(previous_rt[m][:3, :3]))
                mesh.transform(np.linalg.inv(previous_rt[m]))
                # if mesh_idx == 1:
                #     origin.transform(np.linalg.inv([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
                #     origin.transform(np.linalg.inv(previous_rt[m]))
            center = mesh.get_center()
            # mesh.translate(-center)
            mesh.transform(rt_matrix)
            # if mesh_idx == 1:
            #     origin.transform(rt_matrix)
            #     origin.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            #     vis.update_geometry(origin)

            # mesh.translate(center)
            # mesh.translate(t)
            # mesh.rotate(r, center=(0,0,0))
            previous_rt[m] = rt_matrix

            mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            vis.update_geometry(mesh)

            # mesh.transform(np.linalg.inv(rt_matrix))
            # mesh.transform(np.linalg.inv([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("test.png")
        img = cv2.imread("../test.png")
        # depth_3d = np.dstack((depth_image, depth_image, depth_image))
        # depth_3d = cv2.convertScaleAbs(depth_3d, alpha=0.1)
        mask = np.where(img == [255, 255, 255], color, img)
        im_total = cv2.addWeighted(mask, 0.5, color, 0.5, 0)
        cv2.imshow("test", im_total)
        cv2.waitKey(1)
        # time.sleep(1)
        # o3d.io.write_image("output.png", img_o3d, 9)
