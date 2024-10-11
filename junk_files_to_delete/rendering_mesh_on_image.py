import open3d as o3d
import numpy as np
import cv2
from biosiglive import load
import os
import glob

if __name__ == "__main__":
    participants = ["P9"]  # , "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    main_path = "Q:\Projet_hand_bike_markerless\RGBD"
    # main_path = "data_files"
    # empty_depth = np.zeros((nb_frame * (len(participants) - 1), 480, 848, 3), dtype=np.uint8)
    count = 0
    models = ["non_augmented", "hist_eq", "hist_eq_sharp"]
    n_markers = 13
    all_rmse = []
    all_std = []
    rmse = np.ndarray(
        (
            len(models),
            len(participants) * n_markers,
        )
    )
    std = np.ndarray(
        (
            len(models),
            len(participants) * n_markers,
        )
    )
    colors = ["r", "g", "b"]
    lines = ["-", "--", "-."]
    for p, participant in enumerate(participants):
        files = os.listdir(f"{main_path}{os.sep}{participant}")
        files = [file for file in files if "only" in file and "less" not in file and "more" not in file]
        rmse_file = np.ndarray(
            (
                len(models),
                n_markers,
                len(files),
            )
        )
        std_file = np.ndarray(
            (
                len(models),
                n_markers,
                len(files),
            )
        )
        for f, file in enumerate(files):
            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            all_color_files = glob.glob(path + "/color*.png")
            all_depth_files = glob.glob(path + "/depth*.png")
            # check where there is a gap in the numbering
            idx = []
            for f in all_depth_files:
                idx.append(int(f.split("\\")[-1].split("_")[-1].removesuffix(".png")))
            idx.sort()

            results = load(path + "/result_biomech_only_rgbd_11-01-2024_17_27_39_seth_new_model.bio")
            q = results["dlc"]["q_raw"]
            thorax_mesh = r"D:\Documents\Programmation\pose_estimation\Geometry_left\thorax.vtp"
            intrinsics = o3d.camera.PinholeCameraIntrinsic()
            intrinsics.set_intrinsics(
                width=480,
                height=848,
                fx=429.6269226074219,
                fy=429.6269226074219,
                cx=418.1453857421875,
                cy=245.16981506347656,
            )
            for i in range(q.shape[1]):
                import open3d
                import open3d.visualization.rendering as rendering

                # Create a renderer with a set image width and height
                # render = rendering.OffscreenRenderer(480, 848)
                render = open3d.visualization.Visualizer()

                # setup camera intrinsic values
                # pinhole = open3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)

                # Pick a background colour of the rendered image, I set it as black (default is light gray)
                render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

                # now create your mesh
                mesh = open3d.geometry.TriangleMesh()
                mesh.paint_uniform_color([1.0, 0.0, 0.0])  # set Red color for mesh
                # define further mesh properties, shape, vertices etc  (omitted here)
                # mesh = open3d.io.read_triangle_mesh(thorax_mesh)

                # Define a simple unlit Material.
                # (The base color does not replace the mesh's own colors.)
                mtl = o3d.visualization.rendering.Material()
                mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
                mtl.shader = "defaultUnlit"

                # add mesh to the scene
                # render.scene.add_geometry("MyMeshModel", mesh, mtl)
                camMat = [
                    [0.999995768070221, -0.002411586232483387, -0.001641725655645132],
                    [0.0023998860269784927, 0.9999719858169556, -0.007091784384101629],
                    [0.0016587821301072836, 0.0070878141559660435, 0.999973475933075],
                ]
                # render the scene with respect to the camera
                render.scene.camera.set_projection(camMat, 0.1, 1.0, 640, 480)
                img_o3d = render.render_to_image()

                # we can now save the rendered image right at this point
                # open3d.io.write_image("output.png", img_o3d, 9)

                # Optionally, we can convert the image to OpenCV format and play around.
                # For my use case I mapped it onto the original image to check quality of
                # segmentations and to create masks.
                # (Note: OpenCV expects the color in BGR format, so swap red and blue.)
                img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGR)
                cv2.imshow("cv_output.png", img_cv2)
                cv2.waitKey(0)
