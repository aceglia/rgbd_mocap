import os
from biosiglive import load, save
import matplotlib.pyplot as plt
import numpy as np


def merge_files(data, data_1_gap=None, data_2_gap=None, final_end_idx=None, participant=None, file=None):
    first_idx = data_1_gap["frame_idx"][0] if data_1_gap else None
    second_idx = data_2_gap["frame_idx"][0] if data_2_gap else None
    idxs = [first_idx, second_idx, final_end_idx]
    # number of non None in list
    n = sum(x is not None for x in idxs)
    if n == 0:
        return data
    data_set = [data, data_1_gap, data_2_gap]
    if participant == "P10" and "gear_15" in file:
        data_1_tmp = {}
        end_idx = data_1_gap["frame_idx"].index(data_2_gap["frame_idx"][0])
        for key in data.keys():
            data_1_tmp[key] = data_1_gap[key][..., :end_idx] \
                if isinstance(data_1_gap[key], np.ndarray) else data_1_gap[key][:end_idx]

        data_3_tmp = {}
        idx = data_1_gap["frame_idx"].index(data_2_gap["frame_idx"][-1])
        for key in data.keys():
            data_3_tmp[key] = data_1_gap[key][..., idx:] \
                if isinstance(data_1_gap[key], np.ndarray) else data_1_gap[key][idx:]

        data_tmp = {}
        end_idx = data["frame_idx"].index(data_1_gap["frame_idx"][0])
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                data_tmp[key] = np.concatenate((data[key][..., :end_idx], data_1_tmp[key],
                                                data_2_gap[key], data_3_tmp[key]), axis=-1)
            else:
                data_tmp[key] = data[key][:end_idx] + data_1_tmp[key] + data_2_gap[key] + data_3_tmp[key]
        return data_tmp

    if data_1_gap is not None:
        data_tmp = {}
        end_idx = data["frame_idx"].index(data_1_gap["frame_idx"][0])
        for key in data.keys():
            if isinstance(data_1_gap[key], np.ndarray):
                data_tmp[key] = np.concatenate((data[key][..., :end_idx], data_1_gap[key]), axis = -1)
            else:
                data_tmp[key] = data[key][:end_idx] + data_1_gap[key]
    else:
        data_tmp = data

    if data_2_gap is not None:
        data_tmp_2 = {}
        end_idx = data_1_gap["frame_idx"].index(data_2_gap["frame_idx"][0])
        for key in data.keys():
            if isinstance(data_2_gap[key], np.ndarray):
                data_tmp_2[key] = np.concatenate((data_tmp[key][..., :end_idx], data_2_gap[key]), axis = -1)
            else:
                data_tmp_2[key] = data_tmp[key][:end_idx] + data_2_gap[key]
        final_data_to_return = data_tmp_2
    else:
        data_tmp_2 = data_tmp
        final_data_to_return = data_tmp_2

    if final_end_idx is not None:
        data_tmp_3 = {}
        end_idx = data_tmp_2["frame_idx"].index(final_end_idx)
        for key in data.keys():
            if isinstance(data_tmp_2[key], np.ndarray):
                data_tmp_3[key] = data_tmp_2[key][..., :end_idx]
            else:
                data_tmp_3[key] = data_tmp_2[key][:end_idx]
        final_data_to_return = data_tmp_3

    return final_data_to_return


if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    trials = [["gear_20", "gear_10", "gear_15", "gear_20"]] * len(participants)
    # trials[0] = ["gear_15"]
    # trials = [["gear_20"]] * len(participants)
    data_files = "Q:\Projet_hand_bike_markerless\RGBD"
    # data_files = "data_files"
    for p, part in enumerate(participants):
        files = os.listdir(f"{data_files}{os.sep}{part}")
        files = [file for file in files if
                 "gear" in file and os.path.isdir(f"{data_files}{os.sep}{part}{os.sep}" + file)
                 ]
        final_files = files if not trials else []
        if trials:
            for trial in trials[p]:
                for file in files:
                    if trial in file:
                        final_files.append(file)
                        break
        files = final_files
        path_to_camera_config_file = f"Q:\Projet_hand_bike_markerless\RGBD\config_camera_files\config_camera_{part}.json"
        data_1_gap = None
        data_2_gap = None
        for f, file in enumerate(files):
            path = f"{data_files}{os.sep}{part}{os.sep}{file}"
            data = load(path + os.sep + "marker_pos_multi_proc_3_crops.bio", merge=True)
            data["occlusions"] = np.array(data["occlusions"]).reshape(-1, 13).transpose()
            data["markers_names"] = np.array(data["markers_names"]).reshape(-1, 13).transpose()
            if os.path.isfile(path + os.sep + "marker_pos_multi_proc_3_crops_1er_gap.bio"):
                data_1_gap = load(path + os.sep + "marker_pos_multi_proc_3_crops_1er_gap.bio", merge=True)
                data_1_gap["occlusions"] = np.array(data_1_gap["occlusions"]).reshape(-1, 13).transpose()
                data_1_gap["markers_names"] = np.array(data_1_gap["markers_names"]).reshape(-1, 13).transpose()
            if os.path.isfile(path + os.sep + "marker_pos_multi_proc_3_crops_2eme_gap.bio"):
                data_2_gap = load(path + os.sep + "marker_pos_multi_proc_3_crops_2eme_gap.bio", merge=True)
                data_2_gap["occlusions"] = np.array(data_2_gap["occlusions"]).reshape(-1, 13).transpose()
                data_2_gap["markers_names"] = np.array(data_2_gap["markers_names"]).reshape(-1, 13).transpose()

            final_end_idx = None
            if part == "P9" and "gear_5" in file:
                final_end_idx = 7670
            elif part == "P10" and "gear_20" in file:
                final_end_idx = 7200
            elif part == "P10" and "gear_5" in file:
                final_end_idx = 8578
            elif part == "P11" and "gear_20" in file:
                final_end_idx = 7256
            elif part == "P12" and "gear_10" in file:
                final_end_idx = 8763
            elif part == "P12" and "gear_20" in file:
                final_end_idx = 8535
            # data = merge_files(data, data_1_gap, data_2_gap, final_end_idx=final_end_idx, participant=part, file=file)

            markers = data["markers_in_meters"]
            x = data["frame_idx"]
            plt.figure()
            for j in range(markers.shape[1]):
                plt.subplot(4, 4, j + 1)
                for i in range(3):
                    plt.plot(x, markers[i, j, :], "r")
            plt.show()
            save(data, path + os.sep + "marker_pos_multi_proc_3_crops_pp.bio", safe=False)
            print(f"file {file} processed")