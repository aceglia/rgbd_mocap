from pathlib import Path
import os
import shutil
import time
import biorbd
from biosiglive import load, save, InverseKinematicsMethods, OfflineProcessing, OfflineProcessingMethod, MskFunctions
import numpy as np
from utils import load_data_from_dlc, _reorder_markers_from_names, _convert_string
from msk_utils import _comment_markers, _compute_new_bounds
from post_process_data import ProcessData
import bioviz


def _compute_new_model_path(file_path, model_path):
    if file_path is not None:
        name = Path(file_path).stem.split("_")[:-1]
        name = "_".join(name)
        parent = str(Path(file_path).parent)
        new_model_path = parent + "/models/" + name + "_" + Path(model_path).stem + ".bioMod"
        if not os.path.isdir(parent + "/models"):
            os.mkdir(parent + "/models")
        if not os.path.isdir(parent + "/models/" + "Geometry"):
            shutil.copytree(str(Path(model_path).parent) + "/Geometry", parent + "/models/" + "Geometry")
    else:
        new_model_path = model_path[:-7] + "_tmp.bioMod"
    return new_model_path


def _compute_ik(msk_function, markers, kalman_freq=120, file_path=None):
    model_path = msk_function.model.path().absolutePath().to_string()
    with open(model_path, "r") as file:
        data = file.read()
    init_idx = data.find("SEGMENT DEFINITION")
    end_idx = data.find("translations xyz // thorax") + len("translations xyz // thorax") + 1
    if "P12" in file_path:
        data_to_insert = f"SEGMENT DEFINITION\n\tsegment thorax_parent\n\t\tparent base\n\t \tRTinMatrix\t0\n    \t\tRT 1.57 -1.57 0 xyz 0 0 0\n\tendsegment\n// Information about ground segment\n\tsegment thorax\n\t parent thorax_parent\n\t \tRTinMatrix\t0\n    \t\tRT 0 0 0 xyz 0 0 0 // thorax\n\t\trotations xyz // thorax\n\t\ttranslations xyz // thorax\n\t\tranges \n\t\t-3 3\n\t\t-3 3\n\t\t-3 3\n\t\t-0.3 0.4\n\t\t-0.3 0.4\n\t\t-0.3 0.4\n"
    else:
        data_to_insert = f"SEGMENT DEFINITION\n\tsegment thorax_parent\n\t\tparent base\n\t \tRTinMatrix\t0\n    \t\tRT 1.57 -1.57 0 xyz 0 0 0\n\tendsegment\n// Information about ground segment\n\tsegment thorax\n\t parent thorax_parent\n\t \tRTinMatrix\t0\n    \t\tRT 0 0 0 xyz 0 0 0 // thorax\n\t\trotations xyz // thorax\n\t\ttranslations xyz // thorax\n\t\tranges \n\t\t-3 3\n\t\t-3 3\n\t\t-3 3\n\t\t-0.1 0.1\n\t\t-0.1 0.1\n\t\t-0.1 0.1\n"
    # data = data[:init_idx] + data_to_insert + data[end_idx:]
    if "vicon" in model_path and markers.shape[1] == 13:
        data = _comment_markers(data)
    data = _compute_new_bounds(data)
    new_model_path = _compute_new_model_path(file_path, model_path)
    with open(new_model_path, "w") as file:
        file.write(data)
    msk_function.model = biorbd.Model(new_model_path)
    q, q_dot = msk_function.compute_inverse_kinematics(markers[:, :, 0:1],
                                                       InverseKinematicsMethods.BiorbdLeastSquare)
    # b = bioviz.Viz(loaded_model=msk_function.model)
    # b.load_movement(q)
    # b.load_experimental_markers(markers)
    # b.exec()
    with open(new_model_path, "r") as file:
        data = file.read()
    print(new_model_path)
    data = data.replace(
       "RT 0 0 0 xyz 0 0 0 // thorax",
       f"RT {q[3, 0]} {q[4, 0]} {q[5, 0]} xyz {q[0, 0]} {q[1, 0]} {q[2, 0]} // thorax",
    )
    # data = data.replace(
    #     "rotations xyz // thorax",
    #     f"//rotations xyz // thorax",
    # )
    # data = data.replace(
    #     "translations xyz // thorax",
    #     f"// translations xyz // thorax",
    # )
    new_model_path = _compute_new_model_path(file_path, model_path)
    with open(new_model_path, "w") as file:
        file.write(data)
    msk_function.clean_all_buffers()
    q[:6, :] = np.zeros((6, 1))
    # q[-1, :] = 0.1
    # q = q[6:, :]
    msk_function.model = biorbd.Model(new_model_path)
    # msk_function.model.UpdateKinematicsCustom(q[:, 0])
    initial_guess = [q[:, 0], np.zeros_like(q)[:, 0], np.zeros_like(q)[:, 0]]
    msk_function.compute_inverse_kinematics(markers[:, :, :],
                                            method=InverseKinematicsMethods.BiorbdLeastSquare,
                                            kalman_freq=kalman_freq,
                                            # initial_guess=initial_guess
                                            )
    q = msk_function.kin_buffer[0].copy()
    # b = bioviz.Viz(loaded_model=msk_function.model)
    # b.load_movement(q)
    # b.load_experimental_markers(markers)
    # b.exec()
    return q


def main(model_dir, participants, processed_data_path, save_data=False, plot=True, results_from_file=False, stop_frame=None):
    source = ["dlc", "depth", "vicon"]#, "minimal_vicon"]
    # model_source = ["depth", "vicon", "depth"]
    model_source = ["depth", "depth", "depth"]
    processed_source = []
    models = ["normal"] #"non_augmented", "hist_eq",
    filtered = ["alone"]
    # processed_data_path =
    for part in participants:
        all_files = os.listdir(f"{processed_data_path}/{part}")

        # all_files = [file for file in all_files if "gear" in file and "result_biomech" not in file and "3_crops" in file]
        all_files = [file for file in all_files if "gear" in file and "less" not in file and "more" not in file]
        for file in all_files:
            if not os.path.isfile(f"Q://Projet_hand_bike_markerless/process_data\{part}\{file.split('_')[0]}_{file.split('_')[1]}_processed_3_crops.bio"):
                continue
            data_init = load(f"Q://Projet_hand_bike_markerless/process_data\{part}\{file.split('_')[0]}_{file.split('_')[1]}_processed_3_crops_rt.bio")
            path = f"{processed_data_path}{os.sep}{part}{os.sep}{file}"
            processed_data = ProcessData()
            vicon_to_depth_idx = processed_data._get_vicon_to_depth_idx(names_vicon=data_init["vicon_markers_names"], names_depth=data_init["depth_markers_names"])
            markers_vicon = data_init["markers_vicon_rotated"][:, vicon_to_depth_idx, :]

            labeled_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_pp.bio"
            dic_to_save = {"file": file,
                           "part": part,
                           "data_path": labeled_data_path}
            model_path = f"{model_dir}/{part}/model_scaled_{model_source[0]}_new_seth.bioMod"
            bio_model = biorbd.Model(model_path)
            markers_names = [_convert_string(i.to_string()) for i in bio_model.markerNames()]

            for model in models:
                for filt in filtered:
                    dlc_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_{model}_{filt}_pp.bio"
                    data_dlc, data_labeling = load_data_from_dlc(labeled_data_path, dlc_data_path, part, file)
                    markers_from_source_in_pixel = [data_dlc["markers_in_meters"], data_labeling["markers_in_meters"]]
                    new_markers_dlc = np.zeros((3,
                                                  data_dlc["markers_in_meters"].shape[1],
                                                  data_dlc["markers_in_meters"].shape[2]
                                                  ))
                    frame_idx = data_dlc["frame_idx"]
                    markers_dlc_hom = np.ones((4, data_dlc["markers_in_meters"].shape[1], data_dlc["markers_in_meters"].shape[2]))
                    markers_dlc_hom[:3, ...] = data_dlc["markers_in_meters"][:3, ...]
                    # for k in range(new_markers_dlc.shape[2]):
                    #     new_markers_dlc[:, :, k] = np.dot(np.array(rt_matrix), markers_dlc_hom[:, :, k])[:3, :]
                    new_markers_dlc = ProcessData()._fill_and_interpolate(data=data_dlc["markers_in_meters"],
                                                       idx=frame_idx,
                                                       shape=markers_vicon.shape[2],
                                                       fill=True)

                    markers_from_source = [data_dlc["markers_in_meters"], data_init["markers_depth_initial"],
                                           markers_vicon]

                    markers_dlc_filtered = np.zeros((3, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
                    markers_depth_filtered = np.zeros((3, markers_from_source[1].shape[1], markers_from_source[1].shape[2]))
                    markers_vicon_filtered = np.zeros((3, markers_from_source[2].shape[1], markers_from_source[2].shape[2]))

                    for i in range(3):
                        markers_depth_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(
                            markers_from_source[1][i, :, :],
                            2, 120, 2)

                    # for i in range(3):
                    #     markers_vicon_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(
                    #         markers_from_source[2][i, :, :],
                    #         2, 120, 2)

                    for i in range(3):
                        markers_dlc_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(
                            new_markers_dlc[i, :, :],
                            2, 120, 2)

                    # if part == "P11":
                    #     markers_vicon_filtered[:, 1, :] = markers_depth_filtered[:, 1, :]
                    # if part == "P16":
                    #     markers_vicon_filtered[:, 4, :] = markers_depth_filtered[:, 4, :]
                    # import json
                    # from scapula_cluster.from_cluster_to_anato import ScapulaCluster
                    # measurements_dir_path = "data_collection_mesurement"
                    # calibration_matrix_dir = "../scapula_cluster/calibration_matrix"
                    # measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{part}.json"))
                    # measurements = measurement_data[f"with_depth"]["measure"]
                    # calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"][
                    #     "calibration_matrix_name"]
                    # new_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                    #                              measurements[4], measurements[5], calibration_matrix)
                    # markers_cluster = markers_dlc_filtered.copy()
                    # anato_pos = new_cluster.process(marker_cluster_positions=markers_dlc_filtered[:, -3:, :] * 1000,
                    #                                 cluster_marker_names=["M1", "M2", "M3"],
                    #                                 save_file=False)
                    # anato_pos_ordered = anato_pos.copy()
                    # anato_pos_ordered[:, 0, :] = anato_pos[:, 0, :]
                    # anato_pos_ordered[:, 1, :] = anato_pos[:, 2, :]
                    # anato_pos_ordered[:, 2, :] = anato_pos[:, 1, :]

                    # stylr_tmp = markers_dlc_filtered[:, 8, :].copy()
                    # stylu_tmp = markers_dlc_filtered[:, 9, :].copy()
                    # markers_dlc_filtered[:, 9, :] = stylr_tmp
                    # markers_dlc_filtered[:, 8, :] = stylu_tmp

                    # markers_cluster[:, 4:4+3, :] = anato_pos_ordered[:3, :, :] * 0.001
                    depth_markers_names = data_init["depth_markers_names"]
                    idx_ts = depth_markers_names.index("scapts")
                    idx_ai = depth_markers_names.index("scapia")
                    depth_markers_names[idx_ts] = "scapia"
                    depth_markers_names[idx_ai] = "scapts"

                    markers_vicon_filtered = _reorder_markers_from_names(
                        markers_vicon, depth_markers_names, markers_names)
                    # stylr_tmp = markers_vicon_filtered[:, -2, :].copy()
                    # stylu_tmp = markers_vicon_filtered[:, -3, :].copy()
                    # markers_vicon_filtered[:, -3, :] = stylr_tmp
                    # markers_vicon_filtered[:, -2, :] = stylu_tmp

                    markers_depth_filtered = _reorder_markers_from_names(
                        markers_depth_filtered, depth_markers_names, markers_names)


                    filtered_markers = [markers_dlc_filtered[:, :-3, :], markers_depth_filtered, markers_vicon_filtered]
                    # import matplotlib.pyplot as plt
                    # plt.figure("markers")
                    # for i in range(markers_depth_filtered.shape[1]):
                    #     plt.subplot(4, 4, i + 1)
                    #     for j in range(3):
                    #         plt.plot(markers_dlc_filtered[j, i, :], "r")
                    #         plt.plot(markers_vicon_filtered[j, i, :], "b")
                    #         plt.plot(markers_depth_filtered[j, i, :], "g")
                    # plt.show()
                    for m, markers in enumerate(filtered_markers):
                        model_path = f"{model_dir}/{part}/model_scaled_{model_source[m]}_new_seth.bioMod"
                        bio_model = biorbd.Model(model_path)
                        msk_function = MskFunctions(model=bio_model, data_buffer_size=markers.shape[2], system_rate=120)

                        # for i in range(3):
                        #     filtered_markers[m][i, :, :] = OfflineProcessing().butter_lowpass_filter(
                        #         markers[i, :, :], 4, 60, 4)
                        # filtered_markers[m] = filtered_markers[m][:, :-3, :]
                        #
                        # stylr_tmp = filtered_markers[m][:, -2, :].copy()
                        # stylu_tmp = filtered_markers[m][:, -3, :].copy()
                        # filtered_markers[m][:, -3, :] = stylr_tmp
                        # filtered_markers[m][:, -2, :] = stylu_tmp
                        # if m == 2:
                        q = _compute_ik(msk_function, markers, file_path=path)
                        dic_to_save[source[m]] = {"q": q, "markers": markers}
                    import matplotlib.pyplot as plt
                    # for i in range(filtered_markers[0].shape[1]):
                    #     plt.subplot(4, filtered_markers[0].shape[1] // 4 + 1, i + 1)
                    #     for j in range(3):
                    #         plt.plot(filtered_markers[0][j, i, :], "c", alpha=0.5)
                    #         plt.plot(filtered_markers[1][j, i, :], "r", alpha=0.5)
                    # plt.show()
                    plt.figure("q")
                    for i in range(dic_to_save["depth"]["q"].shape[0]):
                        plt.subplot(4, 4, i + 1)
                        for s in source:
                            plt.plot(dic_to_save[s]["q"][i, :], label=s)
                    plt.legend()
                    plt.show()
                    if save_data:
                        save(dic_to_save,
                             f"{processed_data_path}/{part}/{file}/result_offline_{Path(file).stem}_{model}_{filt}.bio",
                             safe=False)
                    print(f"Processing participant {part}, trial : {file}")
                    # markers_from_source, names_from_source, forces, f_ext, emg, vicon_to_depth, peaks = load_data(
                    #     processed_data_path, part, file, False
                    # )
                    # model_path = f"{model_dir}/{part}/model_scaled_{source[0]}_seth.bioMod"
                    # if save_data:
                    #     save(all_results, f"{processed_data_path}/{part}/{file}/result_biomech_{Path(file).stem}_{model}_{filt}.bio",
                    #          safe=False)


if __name__ == '__main__':
    model_dir = "Q://Projet_hand_bike_markerless/RGBD"
    # model_dir = "F:\markerless_project"
    participants = [f"P{i}" for i in range(13, 14)]
    # participants += ["P11"]
    # participants.pop(participants.index("P12"))
    processed_data_path = "Q://Projet_hand_bike_markerless/RGBD" #"/mnt/shared/Projet_hand_bike_markerless/process_data"
    # processed_data_path = "F://markerless_project"
    main(model_dir, participants, processed_data_path, save_data=False, results_from_file=False, stop_frame=None,
         plot=False)