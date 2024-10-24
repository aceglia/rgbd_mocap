import os
from biosiglive import load, save
import matplotlib.pyplot as plt
import numpy as np
from processing_data.file_io import get_all_file

prefix = "/mnt/shared" if os.name == "posix" else "Q:"


def merge_files(
    data, data_1_gap=None, data_2_gap=None, final_end_idx=None, participant=None, file=None, start_idx=None
):
    first_idx = data_1_gap["frame_idx"][0] if data_1_gap else None
    second_idx = data_2_gap["frame_idx"][0] if data_2_gap else None
    idxs = [start_idx, first_idx, second_idx, final_end_idx]
    # number of non None in list
    n = sum(x is not None for x in idxs)
    if n == 0:
        return data
    data_set = [data, data_1_gap, data_2_gap]
    # if participant == "P10" and "gear_15" in file:
    #     data_1_tmp = {}
    #     end_idx = data_1_gap["frame_idx"].index(data_2_gap["frame_idx"][0])
    #     for key in data.keys():
    #         data_1_tmp[key] = (
    #             data_1_gap[key][..., :end_idx] if isinstance(data_1_gap[key], np.ndarray) else data_1_gap[key][:end_idx]
    #         )
    #
    #     data_3_tmp = {}
    #     idx = data_1_gap["frame_idx"].index(data_2_gap["frame_idx"][-1])
    #     for key in data.keys():
    #         data_3_tmp[key] = (
    #             data_1_gap[key][..., idx:] if isinstance(data_1_gap[key], np.ndarray) else data_1_gap[key][idx:]
    #         )
    #
    #     data_tmp = {}
    #     end_idx = data["frame_idx"].index(data_1_gap["frame_idx"][0])
    #     for key in data.keys():
    #         if isinstance(data[key], np.ndarray):
    #             data_tmp[key] = np.concatenate(
    #                 (data[key][..., :end_idx], data_1_tmp[key], data_2_gap[key], data_3_tmp[key]), axis=-1
    #             )
    #         else:
    #             data_tmp[key] = data[key][:end_idx] + data_1_tmp[key] + data_2_gap[key] + data_3_tmp[key]
    #     return data_tmp
    data_tmp = {}
    if start_idx is not None and data_1_gap is None:
        start_idx_tmp = data["frame_idx"].index(start_idx)
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                data_tmp[key] = data[key][..., start_idx_tmp:]
            else:
                data_tmp[key] = data[key][start_idx_tmp:]
    else:
        data_tmp = data

    if data_1_gap is not None:
        data_tmp_1 = {}
        end_idx = data["frame_idx"].index(data_1_gap["frame_idx"][0])
        for key in data.keys():
            if isinstance(data_1_gap[key], np.ndarray):
                data_tmp_1[key] = np.concatenate((data_tmp[key][..., :end_idx], data_1_gap[key]), axis=-1)
            else:
                data_tmp_1[key] = data_tmp[key][:end_idx] + data_1_gap[key]
    else:
        data_tmp_1 = data_tmp

    if data_2_gap is not None:
        data_tmp_2 = {}
        end_idx = data_1_gap["frame_idx"].index(data_2_gap["frame_idx"][0])
        for key in data.keys():
            if isinstance(data_2_gap[key], np.ndarray):
                data_tmp_2[key] = np.concatenate((data_tmp_1[key][..., :end_idx], data_2_gap[key]), axis=-1)
            else:
                data_tmp_2[key] = data_tmp_1[key][:end_idx] + data_2_gap[key]
        final_data_to_return = data_tmp_2
    else:
        data_tmp_2 = data_tmp_1
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


if __name__ == "__main__":
    file_name = f"marker_pos_multi_proc_3_crops_normal_times_three_new.bio"
    # file_name = "marker_pos_multi_proc_3_crops_normal_filtered.bio"
    participants = [f"P{i}" for i in range(9, 17)]
    trials = ["gear_5", "gear_10", "gear_15", "gear_20"]
    data_files = f"{prefix}\Projet_hand_bike_markerless\RGBD"
    data_files = "/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files"
    files, parts = get_all_file(participants, data_files, trial_names=trials, to_include="gear")
    for part, file in zip(parts, files):
        path = file
        path_to_camera_config_file = f"/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/config_camera_files/config_camera_{part}.json"
        data_1_gap = None
        data_2_gap = None
        if not os.path.isfile(path + os.sep + file_name):
            print(f"file {file} not processed for participant {part}")
            continue
        data = load(path + os.sep + file_name, merge=True)
        data["occlusions"] = np.array(data["occlusions"]).reshape(-1, 13).transpose()
        data["markers_names"] = np.array(data["markers_names"]).reshape(-1, 13).transpose()
        if os.path.isfile(path + os.sep + file_name[:-4] + "_1er_gap.bio"):
            print(f"file {file} has 1er gap")
            data_1_gap = load(path + os.sep + file_name[:-4] + "_1er_gap.bio", merge=True)
            data_1_gap["occlusions"] = np.array(data_1_gap["occlusions"]).reshape(-1, 13).transpose()
            data_1_gap["markers_names"] = np.array(data_1_gap["markers_names"]).reshape(-1, 13).transpose()
        # # if os.path.isfile(path + os.sep + "marker_pos_multi_proc_3_crops_2eme_gap.bio"):
        # #     data_2_gap = load(path + os.sep + "marker_pos_multi_proc_3_crops_2eme_gap.bio", merge=True)
        # #     data_2_gap["occlusions"] = np.array(data_2_gap["occlusions"]).reshape(-1, 13).transpose()
        # #     data_2_gap["markers_names"] = np.array(data_2_gap["markers_names"]).reshape(-1, 13).transpose()
        # New :
        start_idx = None
        final_end_idx = None
        if part == "P9" and "gear_5" in file:
            final_end_idx = 8015
        elif part == "P10" and "gear_20" in file:
            final_end_idx = 7200
        elif part == "P10" and "gear_5" in file:
            final_end_idx = 8500
        elif part == "P12" and "gear_10" in file:
            final_end_idx = 7832
        # elif part == "P16" and "gear_20" in file:
        #     start_idx = 3600
        # P15 premier gap 6664
        # P16 start gear20 2955

        # final_end_idx = None
        # # if part == "P9" and "gear_5" in file:
        # #     final_end_idx = 8015
        # # elif part == "P10" and "gear_20" in file:
        # #     final_end_idx = 7200
        # # elif part == "P10" and "gear_5" in file:
        # #     final_end_idx = 8578
        # # elif part == "P11" and "gear_20" in file:
        # #     final_end_idx = 7256
        # # elif part == "P12" and "gear_10" in file:
        # #     final_end_idx = 8763
        # # elif part == "P12" and "gear_20" in file:
        # #     final_end_idx = 8535
        # # elif part == "P12" and "only" in file:
        # #     final_end_idx = 5382
        data = merge_files(
            data, data_1_gap, data_2_gap, final_end_idx=final_end_idx, participant=part, file=file, start_idx=start_idx
        )
        #
        markers = data["markers_in_meters"]
        x = data["frame_idx"]
        plt.figure()
        for j in range(markers.shape[1]):
            plt.subplot(4, 4, j + 1)
            for i in range(3):
                plt.plot(x, markers[i, j, :], "r")
        plt.show()
        save(data, path + os.sep + file_name[:-4] + "_pp.bio", safe=False)
        print(f"file {file} processed")
