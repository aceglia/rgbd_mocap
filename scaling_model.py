from pose_est.utils import *
import cv2


def RT(angleX):
    Rototrans = np.array([[np.cos(angleX), -np.sin(angleX)], [np.sin(angleX), np.cos(angleX)]])
    return Rototrans

def RT_3D(angleX):
    Rototrans = np.array([[np.cos(angleX), -np.sin(angleX)], [np.sin(angleX), np.cos(angleX)]])
    return Rototrans


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)

def objective(x, from_pts, to_pts):
    center_from = np.mean(from_pts, axis=0)
    center_to = np.mean(to_pts, axis=0)
    from_pts = from_pts - center_from
    to_pts = to_pts - center_to
    rototrans = RT(x)
    J = 0
    from_pts_transf = np.zeros(from_pts.shape)
    for i in range(to_pts.shape[0]):
        from_pts_transf[i, :] = np.dot(rototrans, from_pts[i, :])

    for i in range(to_pts.shape[0]):
        closest_idx = closest_node(to_pts[i, :], from_pts_transf)
        J = J + sum((to_pts[i, :] - from_pts_transf[closest_idx, :]) ** 2)
    return J

def find_2d_pos_from_image(color_frame):
    dic = []
    x_tmp, y_tmp = [], []
    color_frame = color_frame.copy()
    while True:
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                x_tmp.append(x)
                y_tmp.append(y)

            if event == cv2.EVENT_RBUTTONDOWN:
                x_tmp.pop()
                y_tmp.pop()

        cv2.namedWindow(
            f"select markers by click on the image.",
            cv2.WINDOW_NORMAL,
        )
        cv2.setMouseCallback(
            f"select markers by click on the image.",
            click_event,
        )
        for x, y in zip(x_tmp, y_tmp):
            cv2.circle(color_frame, (x, y), 5, (255, 0, 0), -1)

        cv2.namedWindow(
            f"select markers.",
            cv2.WINDOW_NORMAL,
        )
        cv2.imshow(f"select markers.", color_frame)
        cv2.imshow(f"select markers by click on the image.", color_frame)
        if cv2.waitKey(10) == ord("q"):
            break
    cv2.destroyAllWindows()
    return x_tmp, y_tmp

if __name__ == "__main__":
    im = ["scaling_image/color_60.png"
    ,"scaling_image/color_108.png"
    ,"scaling_image/color_228.png"]
    depth = ["scaling_image/depth_60.png"
    , "scaling_image/depth_108.png"
    , "scaling_image/depth_228.png"]


    images_dir = "data_files/data_25-04-2023_16_53_13"
    # camera = RgbdImages(conf_file="config_camera_25-04-2023_16_53_13.json", images_dir=images_dir)
    conf_data = get_conf_data("config_camera_25-04-2023_16_53_13.json")
    depth_scale = conf_data["depth_scale"]
    # extract pointcloud

    # intrinsics = o3d.camera.PinholeCameraIntrinsic()
    # intrinsics.set_intrinsics(width=conf_data["size_depth"][0], height=conf_data["size_depth"][1],
    #                           fx=conf_data["depth_fx_fy"][0],
    #                           fy=conf_data["depth_fx_fy"][1],
    #                           cx=conf_data["depth_ppx_ppy"][0], cy=conf_data["depth_ppx_ppy"][1])
    # depth_3d_image = o3d.geometry.Image(cv2.imread(depth[0], cv2.IMREAD_ANYDEPTH))
    # color = cv2.imread(im[0])
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # color_3d = o3d.geometry.Image(color)
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_3d, depth_3d_image,
    #                                                           convert_rgb_to_intensity=False)
    # pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    # # pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #
    # # pc 2
    # depth_3d_image = o3d.geometry.Image(cv2.imread(depth[1], cv2.IMREAD_ANYDEPTH))
    # color = cv2.imread(im[1])
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # color_3d = o3d.geometry.Image(color)
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_3d, depth_3d_image,
    #                                                             convert_rgb_to_intensity=False)
    # pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    # # pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd1.voxel_down_sample(voxel_size=0.0001)
    # pcd2.voxel_down_sample(voxel_size=0.0001)

    from functools import partial
    import matplotlib.pyplot as plt
    from pycpd import RigidRegistration
    import numpy as np
    import os

    # treg = o3d.t.pipelines.registration
    # import copy
    # def draw_registration_result_original_color(source, target, transformation):
    #     source_temp = copy.deepcopy(source)
    #     source_temp.transform(transformation)
    #     o3d.visualization.draw_geometries([source_temp, target],
    #                                       zoom=0.5,
    #                                       front=[-0.2458, -0.8088, 0.5342],
    #                                       lookat=[1.7745, 2.2305, 0.9787],
    #                                       up=[0.3109, -0.5878, -0.7468])
    #
    # # Initial alignment or source to target transform.
    # voxel_radius = [0.04, 0.02, 0.01]
    # max_iter = [50, 3000, 14]
    # init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64)
    # source = pcd1
    # target = pcd2
    # voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
    #
    # # List of Convergence-Criteria for Multi-Scale ICP:
    # criteria_list = [
    #     treg.ICPConvergenceCriteria(relative_fitness=0.0001,
    #                                 relative_rmse=0.0001,
    #                                 max_iteration=20),
    #     treg.ICPConvergenceCriteria(0.00001, 0.00001, 15),
    #     treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
    # ]
    #
    # # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    # max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
    #
    # # Initial alignment or source to target transform.
    # init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
    #
    # # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    # estimation = treg.TransformationEstimationPointToPlane()
    #
    # # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    # callback_after_iteration = lambda loss_log_map: print(
    #     "Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    #         loss_log_map["iteration_index"].item(),
    #         loss_log_map["scale_index"].item(),
    #         loss_log_map["scale_iteration_index"].item(),
    #         loss_log_map["fitness"].item(),
    #         loss_log_map["inlier_rmse"].item()))
    #
    # # Setting Verbosity to Debug, helps in fine-tuning the performance.
    # # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    #
    # registration_ms_icp = treg.multi_scale_icp(source, target, voxel_sizes,
    #                                            criteria_list,
    #                                            max_correspondence_distances,
    #                                            init_source_to_target, estimation,
    #                                            callback_after_iteration)
    # draw_registration_result_original_color(source, target,
    #                                         registration_icp.transformation)
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    color = ["r", "g", "b"]
    markers_sets = []
    markers_array = []
    for i in range(2):
        pos_marker = []
        depth_markers = []
        color = cv2.imread(im[i])
        depth_image = cv2.imread(depth[i], cv2.IMREAD_ANYDEPTH)
        find_2d_pose = find_2d_pos_from_image(color)
        markers_array.append(np.zeros((len(find_2d_pose[0]), 3)))
        count = 0
        for x, y in zip(find_2d_pose[0], find_2d_pose[1]):
            pos_marker.append([x, y])
            markers_array[i][count, :2] = [x, y]
            deth, is_visible = check_and_attribute_depth([x, y], depth_image, depth_scale)
            markers_array[i][count, 2] = deth
            depth_markers.append((deth, is_visible))
            count += 1
        markers_sets.append([pos_marker, depth_markers])
        # ax.scatter([i[0] for i in pos_marker], [i[1] for i in pos_marker], [i[0] for i in depth_markers], color[i])
    print(markers_array)

    # ax.set_xlabel("X Label")
    # ax.set_ylabel("Y Label")
    # ax.set_zlabel("Z Label")
    # plt.show()

    # X = np.array([[433.        , 101.        ,   0.98200005],
    #    [469.        ,  98.        ,   1.03200005],
    #    [480.        ,  99.        ,   1.05600005],
    #    [404.        , 118.        ,   0.96700005],
    #    [405.        , 174.        ,   0.98300005],
    #    [395.        , 202.        ,   1.00800005],
    #    [387.        , 313.        ,   0.98500005],
    #    [395.        , 368.        ,   0.95600005]])
    # Y = np.array([[477.        , 100.        ,   1.02000005],
    #    [491.        , 101.        ,   1.03000005],
    #    [513.        , 170.        ,   1.06500005],
    #    [433.        , 100.        ,   1.00600005],
    #    [395.        , 167.        ,   1.05400005],
    #    [389.        , 193.        ,   1.07600005],
    #    [353.        , 335.        ,   1.08800005]])

    def visualize(iteration, error, X, Y, ax, save_fig=False):
        plt.cla()
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Target')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Source')
        ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
            iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                  fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')
        ax.view_init(90, -90)
        plt.pause(0.001)

    X = markers_array[0]
    Y = markers_array[1]
    # X = np.loadtxt("fish_source.txt")
    # Y = np.loadtxt("fish_target.txt")
    reg = RigidRegistration(**{'X': X, 'Y': Y})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    reg.register(callback=partial(visualize, ax=ax))
    # plot point cloud using matplotlib
    plt.show()


    # find the transformation between each set of 3d markers




