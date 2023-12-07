import numpy as np
from biosiglive import load
from pyomeca import Markers
import matplotlib.pyplot as plt
from biosiglive import OfflineProcessing, OfflineProcessingMethod, load
import casadi as ca
from scipy.interpolate import interp1d


def rmse(data, data_ref):
    return np.sqrt(np.mean(((data - data_ref) ** 2), axis=-1))


markers_names = ["T5", "C7", "Ribs", "clavAC", "scapAA", "scapia", "acrom", "delt", "arml", "epicm", "stylrad", "stylu"]
markers_names = [
    "test:T5",
    "test:C7",
    "test:Ribs",
    "test:clav_ac",
    "test:scapaa",
    "test:scapia",
    "test:acrom",
    "test:delt",
    "test:larm",
    "test:epicl",
    "test:larm_l",
    "test:styl_r",
    "test:styl_l",
]
markers_vicon = Markers.from_c3d(filename=f"Pedalage_{suffix}.c3d", usecols=markers_names)
markers_vicon = markers_vicon.values[:, :, 72:-80] * 0.001
markers_depth = load(f"markers_{suffix}.bio")
marker_depth_names = markers_depth["markers_names"][0]
markers_depth = markers_depth["markers_in_meters"]

# point_test_calib_vicon = np.concatenate((markers_vicon_calib.values[:3, :, 0:1] * 0.001, markers_vicon[:3, :3, 0:1]), axis=1)
# point_test_calib = np.concatenate((markers_depth_calib["markers_in_meters"][:, :, 0:1], markers_depth[:3, :3, 0:1]), axis=1)
point_test_calib_vicon = markers_vicon_calib.values[:3, :, 0:1] * 0.001
point_test_calib = markers_depth_calib["markers_in_meters"][:, :, 0:1]
point_cloud_source = point_test_calib_vicon[:, :, 0]
point_cloud_target = point_test_calib[:, :, 0]
optimal_rotation, optimal_translation = homogeneous_transform_optimization(
    point_test_calib[:, :, 0], point_test_calib_vicon[:, :, 0]
)
new_markers_depth_calib = np.dot(np.array(optimal_rotation), np.array(point_test_calib[:, :, 0])) + np.array(
    optimal_translation
)
new_markers_depth = np.zeros(markers_depth.shape)
for i in range(markers_depth.shape[2]):
    new_markers_depth[:, :, i] = np.dot(np.array(optimal_rotation), np.array(markers_depth[:, :, i])) + np.array(
        optimal_translation
    )
# from scipy.interpolate import approximate_taylor_polynomial
# from scipy.interpolate import interp1d

new_markers_depth_int = np.zeros((3, markers_vicon.shape[1], int(markers_vicon.shape[2])))
# new_markers_depth_int_taylor = np.zeros((3, markers_vicon.shape[1], int(markers_vicon.shape[2])))
for i in range(3):
    x = np.linspace(0, 100, new_markers_depth.shape[2])
    f_mark = interp1d(x, new_markers_depth[i, :, :])

    x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
    new_markers_depth_int[i, :, :] = f_mark(x_new)

    # f_mark = interp1d(x_new, new_markers_depth_int[i, 0, :])
    # f_taylor = approximate_taylor_polynomial(f_mark, 0, 5, 1, 7)
    # new_markers_depth_int_taylor[i, :, :] = f_taylor(x_new)

err_markers = np.zeros((len(marker_depth_names), 1))
for i in range(len(marker_depth_names)):
    # ignore NaN values
    nan_index = np.argwhere(np.isnan(markers_vicon[:, i, :]))
    new_markers_depth_tmp = np.delete(new_markers_depth_int[:, i, :], nan_index, axis=1)
    new_markers_vicon_int_tmp = np.delete(markers_vicon[:, i, :], nan_index, axis=1)
    nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
    new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
    new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
    err_markers[i, 0] = np.median(
        np.sqrt(np.mean(((new_markers_depth_tmp[:, :] * 1000 - new_markers_vicon_int_tmp[:3, :] * 1000) ** 2), axis=0))
    )
print(err_markers)

fig = plt.figure("vicon")
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])
for i in range(len(markers_names)):
    ax.scatter(markers_vicon[0, i, :], markers_vicon[1, i, :], markers_vicon[2, i, :], c="r")
    ax.scatter(new_markers_depth_int[0, i, :], new_markers_depth_int[1, i, :], new_markers_depth_int[2, i, :], c="b")
    if i < 4:
        ax.scatter(
            markers_vicon_calib.values[0, i, :] * 0.001,
            markers_vicon_calib.values[1, i, :] * 0.001,
            markers_vicon_calib.values[2, i, :] * 0.001,
            c="g",
        )
        ax.scatter(new_markers_depth_calib[0, i], new_markers_depth_calib[1, i], new_markers_depth_calib[2, i], c="y")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

new_markers_depth_filter = np.zeros((3, markers_vicon.shape[1], int(markers_vicon.shape[2])))
for i in range(3):
    for j in range(len(marker_depth_names)):
        count = 0
        nan_index = np.argwhere(np.isnan(new_markers_depth_int[i, j, :]))
        if len(nan_index) > 0:
            new_markers_depth_tmp = np.delete(new_markers_depth_int[i, j, :], nan_index)
            # new_markers_depth_tmp = OfflineProcessing().butter_lowpass_filter(new_markers_depth_tmp,
            #                                           5, 100, 4)
            for k in range(new_markers_depth_int.shape[2]):
                if k not in nan_index:
                    new_markers_depth_filter[i, j, k] = new_markers_depth_tmp[k - count]
                else:
                    new_markers_depth_filter[i, j, k] = np.nan
                    count += 1
        else:
            new_markers_depth_tmp = new_markers_depth_int[i, j, :]
            new_markers_depth_filter[
                i, j, :
            ] = new_markers_depth_tmp  # OfflineProcessing().butter_lowpass_filter(new_markers_depth_tmp,
            #                                                                           5, 100, 4)

# new_markers_depth_filter = OfflineProcessing().butter_lowpass_filter(new_markers_depth_int, 6, 100, 4)

t_depth = np.linspace(0, 1, new_markers_depth.shape[2])
t_vicon = np.linspace(0, 1, markers_vicon.shape[2])
plt.figure("markers_depth")
for i in range(len(marker_depth_names)):
    plt.subplot(4, 4, i + 1)
    for j in range(3):
        plt.plot(t_vicon, new_markers_depth_filter[j, i, :], c="b")
        # plt.plot(t_vicon, new_markers_depth_int_taylor[j, i, :], c='g')
for i in range(len(markers_names)):
    plt.subplot(4, 4, i + 1)
    for j in range(3):
        plt.plot(t_vicon, markers_vicon[j, i, :], c="r")
plt.show()
