from biosiglive import load
import os
import glob
import matplotlib.pyplot as plt
import numpy as np



def compute_error(data, ref):
    shape_idx = 1 if data.shape[0] == 3 else 0
    n_data = data.shape[shape_idx]
    err = np.zeros((n_data))
    for i in range(n_data):
        # remove nan values
        if len(data.shape) == 3:
            nan_index = np.argwhere(np.isnan(ref[:, i, :]))
            data_tmp = np.delete(data[:, i, :], nan_index, axis=1)
            ref_tmp = np.delete(ref[:, i, :], nan_index, axis=1)
            err[i] = np.mean(np.sqrt(np.median(((data_tmp - ref_tmp) ** 2), axis=0)))
        else:
            nan_index = np.argwhere(np.isnan(ref[i, :]))
            data_tmp = np.delete(data[i, :], nan_index, axis=0)
            ref_tmp = np.delete(ref[i, :], nan_index, axis=0)
            err[i] = np.mean(np.sqrt(np.median(((data_tmp - ref_tmp) ** 2), axis=0)))
    return err


def compute_std(data, ref):
    shape_idx = 1 if data.shape[0] == 3 else 0
    n_data = data.shape[shape_idx]
    err = np.zeros((n_data))
    for i in range(n_data):
        # remove nan values
        if len(data.shape) == 3:
            nan_index = np.argwhere(np.isnan(ref[:, i, :]))
            data_tmp = np.delete(data[:, i, :], nan_index, axis=1)
            ref_tmp = np.delete(ref[:, i, :], nan_index, axis=1)
            err[i] = np.mean(np.std(data_tmp - ref_tmp, axis=1))
        else:
            nan_index = np.argwhere(np.isnan(data[i, :]))
            data_tmp = np.delete(data[i, :], nan_index, axis=0)
            ref_tmp = np.delete(ref[i, :], nan_index, axis=0)
            err[i] = np.mean(np.std(data_tmp - ref_tmp, axis=0))
    return err


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def check_frames(data_labeling, data_dlc):
    data = list(np.copy(data_labeling["frame_idx"]))
    ref = list(np.copy(data_dlc["frame_idx"]))
    type_1 = 0
    type = 0
    idx_init = 0
    idx = 0
    datalist = [data_dlc, data_labeling]
    for key in data_dlc.keys():
        if data[0] > ref[0]:
            idx_init = ref.index(data[0])
            ref = ref[idx_init:]
            type_1 = 0
        elif data[0] < ref[0]:
            idx_init = data.index(ref[0])
            data = data[idx_init:]
            type_1 = 1
        if isinstance(data_dlc[key], np.ndarray):
            datalist[type_1][key] = datalist[type_1][key][..., idx_init:]
        else:
            datalist[type_1][key] = datalist[type_1][key][idx_init:]
        if data[-1] < ref[-1]:
            idx = ref.index(data[-1])
            idx += 1
            ref = ref[:idx]
            type = 0
        elif data[-1] > ref[-1]:
            idx = data.index(ref[-1])
            idx += 1
            data = data[:idx]
            type = 1
        if idx != 0:
            if isinstance(datalist[type][key], np.ndarray):
                datalist[type][key] = datalist[type][key][..., :idx]
            else:
                datalist[type][key] = datalist[type][key][:idx]
    if ref != data:
        raise RuntimeError("error in frame_idx")
    return datalist[0], datalist[1]


if __name__ == '__main__':
    participants = ["P11"]#, "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    main_path = "Q:\Projet_hand_bike_markerless\RGBD"
    # main_path = "data_files"
    # empty_depth = np.zeros((nb_frame * (len(participants) - 1), 480, 848, 3), dtype=np.uint8)
    count = 0
    models = ["normal"]
    n_markers = 13
    all_rmse = []
    all_std = []
    rmse = np.ndarray((len(models), len(participants) * n_markers,))
    std = np.ndarray((len(models), len(participants) * n_markers,))
    colors = ["r", "g", "b"]
    lines = ["-", "--", "-."]
    for p, participant in enumerate(participants):
        files = os.listdir(f"{main_path}{os.sep}{participant}")
        files = [file for file in files if "gear_10" in file and "less" not in file and "more" not in file]
        rmse_file = np.ndarray((len(models), n_markers, len(files),))
        std_file = np.ndarray((len(models), n_markers,  len(files),))
        for f, file in enumerate(files):
            path = f"{main_path}{os.sep}{participant}{os.sep}{file}"
            results = load(path + f"/result_offline_{file}_normal_alone.bio")
            plt.figure(f"markers 3D participant {participant}")
            # for m, model in enumerate(models):
                # data_dlc = load(path + f"/marker_pos_multi_proc_3_crops_{model}_filtered_pp.bio")
                # data_dlc, data_labeling = check_frames(data_labeling, data_dlc)
                # for key in data_dlc.keys():
                #     if key in ["markers_in_meters", "markers_in_pixel"]:
                #         ster_tmp = np.copy(data_dlc[key][:, 1, :])
                #         xiph_tmp = np.copy(data_dlc[key][:, 0, :])
                #         data_dlc[key][:, 0, :] = ster_tmp
                #         data_dlc[key][:, 1, :] = xiph_tmp
                # rmse_file[m, :, f] = compute_error(data_dlc["markers_in_meters"]* 1000, data_labeling["markers_in_meters"]* 1000)
                # std_file[m, :, f] = compute_std(data_dlc["markers_in_meters"]* 1000, data_labeling["markers_in_meters"]* 1000)

            for i in range(n_markers):
                plt.subplot(3, 5, i + 1)
                for j in range(3):
                    plt.plot(results["depth"]["markers"][j, i, :] * 1000, color=colors[0])
                    plt.plot(results["dlc"]["markers"][j, i, :] * 1000, color=colors[1], alpha=0.8)
                    plt.plot(results["vicon"]["markers"][j, i, :] * 1000, color=colors[2], alpha=0.8)
            plt.figure(f"angles participant {participant}")
            for i in range(results["dlc"]["q"].shape[0]):
                plt.subplot(4, 4, i + 1)
                plt.plot(results["depth"]["q"][i, :] * 180/3.14, color=colors[0])
                plt.plot(results["dlc"]["q"][i, :] * 180 / 3.14, color=colors[1], alpha=0.8)
                plt.plot(results["vicon"]["q"][i, :] * 180 / 3.14, color=colors[2], alpha=0.8)
            plt.show()
                # error on pixel
                # error on 3D
                # plot traj
        for j in range(len(models)):
            rmse[j, p * n_markers : n_markers * (p + 1 )] = np.mean(rmse_file[j, :, :], axis=1)
            std[j, p * n_markers  : n_markers * (p + 1 )] = np.mean(std_file[j, :, :], axis=1)
    print(rmse, std)
    all_rmse.append(rmse.mean(axis=1).round(2))
    all_std.append(std.mean(axis=1).round(2))
    print(all_rmse, all_std)



