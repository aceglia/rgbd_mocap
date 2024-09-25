import os
import numpy as np
import biorbd
from biomech_utils import get_all_file
from msk_utils import compute_cor, compute_ik, compute_so, compute_jrf, compute_id, get_map_activation_idx
try:
    from msk_utils import process_all_frames, get_tracking_idx, reorder_markers
except:
    pass
from biosiglive import MskFunctions, load, save
import bioviz

prefix = "/mnt/shared" if os.name == "posix" else "Q:/"

class BiomechPipeline:
    def __init__(self, stop_frame=None):
        self.stop_frame = stop_frame

    @staticmethod
    def _adjust_idx(data, idx_start, idx_end):
        if idx_start is None and idx_end is None:
            return data
        data_tmp = {}
        for key in data.keys():
            if "names" in key:
                data_tmp[key] = data[key]
                continue
            idx_start = 0 if idx_start is None else idx_start
            if isinstance(data[key], np.ndarray):
                idx_end = data[key].shape[-1] + 1 if idx_end is None else -idx_end
                data_tmp[key] = data[key][..., idx_start:idx_end]
            elif isinstance(data[key], list):
                data_tmp[key] = data[key][idx_start:idx_end]
            else:
                data_tmp[key] = data[key]
        return data_tmp

    def process_all_frames(self, markers, msk_function, source, external_loads, scaling_factor, emg, f_ext,
                           img_idx=None,
                           compute_id=True, compute_so=True, compute_jrf=True, stop_frame=None, file=None,
                           print_optimization_status=False, filter_depth=False, emg_names=None, compute_ik=True,
                           marker_names=None, calibration_matrix=None, measurements=None, part=None, rt_matrix=None,
                           in_pixel=False, range_idx=None):
        final_dic = {}
        # stop_frame = markers.shape[2] if stop_frame is None else stop_frame
        # all_kalman = None
        # if img_idx is None:
        #     img_idx = np.linspace(0, stop_frame, stop_frame)
        # fist_idx = img_idx[0]
        # img_idx = [idx - fist_idx for idx in img_idx]
        stop_frame = stop_frame if stop_frame else markers.shape[2]
        all_kalman = None
        if filter_depth:
            # n_window = 14
            # markers_process = [RealTimeProcessing(60, n_window), RealTimeProcessing(60, n_window),
            #                    RealTimeProcessing(60, n_window)]
            before_kalman = True
            if not in_pixel:
                # kalman_params = [1] * 17 + [1] * 17
                # kalman_params = [1e-3] * 17 + [1e-3] * 17
                # kalman_params[0:8] = [0.4] * 8
                # kalman_params[0+17:8+17] = [1e-3] * 8
                # kalman_params[8] = 1e-2
                # kalman_params[8 + 17] = 1e-3
                measurement_noise = [2] * 17
                proc_noise = [1] * 17
                measurement_noise[:8] = [5] * 8
                proc_noise[:8] = [1e-1] * 8
                measurement_noise[11:14] = [1] * 3
                proc_noise[11:14] = [1] * 3
                kalman_params = measurement_noise + proc_noise
            elif not before_kalman:
                kalman_params = [300] * 14 + [30] * 14
                kalman_params[:4] = [500] * 4
                kalman_params[14:4 + 14] = [5] * 4
                kalman_params[5] = 100
                kalman_params[5 + 14] = 30
                kalman_params[11:11 + 3] = [300] * 3
                kalman_params[11 + 14:11 + 3 + 14] = [30] * 3
                # kalman_params = [300] * 14 + [30] * 14
            else:
                measurement_noise = [30] * 17
                proc_noise = [3] * 17
                measurement_noise[8] = 10
                proc_noise[8] = 3
                measurement_noise[3] = 50
                proc_noise[3] = 1
                measurement_noise[5:8] = [150] * 3
                proc_noise[5:8] = [3] * 3
                measurement_noise[4] = 50
                proc_noise[4] = 1

                kalman_params = measurement_noise + proc_noise

            n_window = 0
        else:
            n_window = 0
            kalman_params = None

        electro_delay = 0
        track_idx = get_tracking_idx(msk_function.model, emg_names)
        map_idx = get_map_activation_idx(msk_function.model, emg_names)
        import json
        measurements_dir_path = "../data_collection_mesurement"
        calibration_matrix_dir = "../../scapula_cluster/calibration_matrix"
        measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{part}.json"))
        measurements = measurement_data[f"with_depth"]["measure"]
        calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"][
            "calibration_matrix_name"]
        new_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                                     measurements[4], measurements[5], calibration_matrix)
        count = 0
        count_dlc = 0
        idx_cluster = 0
        dlc_markers = None
        camera_converter = None
        for i in range_idx:
            tic = time.time()
            # if i > electro_delay:
            #     emg_tmp = emg if emg is None else emg[:, i-electro_delay]
            # else:
            emg_tmp = None
            reorder_names = marker_names
            # if filter_depth:
            if "dlc" in source and filter_depth:
                if i in img_idx:
                    markers_tmp = markers[..., count_dlc:count_dlc + 1]
                    count_dlc += 1
                else:
                    markers_tmp = None
                if count == 0:
                    idx_cluster = marker_names.index("clavac")
                    print(marker_names)
                    marker_names = marker_names[:idx_cluster + 1] + ["scapaa", "scapia", "scapts"] + marker_names[
                                                                                                     idx_cluster + 1:]
                    if in_pixel:
                        camera_converter = CameraConverter()
                        camera_converter.set_intrinsics("config_camera_files/config_camera_P10.json")
                markers_kalman, all_kalman = get_next_frame_from_kalman(all_kalman, markers_tmp,
                                                                        new_cluster, params=kalman_params,
                                                                        rt_matrix=rt_matrix,
                                                                        idx_cluster_markers=idx_cluster, forward=0,
                                                                        camera_converter=camera_converter,
                                                                        in_pixel=in_pixel,
                                                                        convert_cluster_before_kalman=before_kalman)

                # if markers_tmp is not None:
                #     anato_from_cluster = _convert_cluster_to_anato(new_cluster, markers_tmp[:, -3:, :] * 1000) * 0.001
                #     dlc_markers = np.concatenate(
                #         (markers_tmp[:, :idx_cluster + 1, :], anato_from_cluster[:3, ...],
                #          markers_tmp[:, idx_cluster + 1:, :]),
                #         axis=1)
                # # else:
                # #      dlc_markers = markers[..., count_dlc:count_dlc + 1]
                # markers_data = dlc_markers
                # markers_dlc_hom = np.ones((4, markers_data.shape[1], 1))
                # markers_dlc_hom[:3, :, 0] = markers_data[..., 0]
                # markers_data = np.dot(np.array(rt_matrix), markers_dlc_hom[:, :, 0])[:3, :, None]
                # for k in range(markers_kalman.shape[1]):
                #     if k not in list(range(idx_cluster + 1 , idx_cluster + 4)):
                #         markers_kalman[:, i, :] = markers_tmp[]
                markers_tmp, reorder_names = reorder_markers(markers_kalman[:, :-3, None], msk_function.model,
                                                             marker_names[:-3])
                markers_tmp = markers_tmp[:, :, 0]

                # markers_tmp[:, reorder_names.index("ribs")] = np.nan
            else:
                if "dlc" in source:
                    if count == 0:
                        idx_cluster = marker_names.index("clavac")
                        marker_names = marker_names[:idx_cluster + 1] + ["scapaa", "scapia", "scapts"] + marker_names[
                                                                                                         idx_cluster + 1:]
                    markers_tmp, reorder_names = reorder_markers(markers[:, :-3, count:count + 1], msk_function.model,
                                                                 marker_names[:-3])
                    markers_tmp = markers_tmp[:, :, 0]
                else:
                    markers_tmp = markers[:, :, count]
            kalman_freq = 60 if filter_depth else 120
            dic_to_save = self.process_next_frame(markers_tmp, msk_function, count, source, external_loads,
                                             scaling_factor, emg_tmp,
                                             f_ext=f_ext[:, i],
                                             kalman_freq=kalman_freq,
                                             emg_names=emg_names, compute_id=compute_id, compute_ik=compute_ik,
                                             compute_so=compute_so, compute_jrf=compute_jrf, file=file,
                                             print_optimization_status=print_optimization_status,
                                             filter_depth=filter_depth,
                                             tracking_idx=track_idx,
                                             map_emg_idx=map_idx,
                                             n_window=n_window
                                             )
            tim_ti_get_frame = time.time() - tic
            dic_to_save["time"]["time_to_get_frame"] = tim_ti_get_frame
            dic_to_save["tracked_markers"] = markers_tmp[:, :, None]
            if dic_to_save is not None:
                final_dic = dic_merger(final_dic, dic_to_save)
            count += 1
            if count == stop_frame:
                break
        if "dlc" in source:
            final_dic["marker_names"] = reorder_names
        else:
            final_dic["marker_names"] = marker_names
        final_dic["center_of_rot"] = compute_cor(final_dic["q_raw"], msk_function.model)
        return final_dic

    def process_next_frame(self, markers, msk_function, frame_idx, source, external_loads=None,
                           scaling_factor=None, emg=None, kalman_freq=120, emg_names=None, f_ext=None,
                           compute_so=False, compute_id=False, compute_jrf=False, compute_ik=True, file=None,
                           print_optimization_status=False, filter_depth=False, markers_process=None, n_window=14,
                           tracking_idx=None, map_emg_idx=None, calibration_matrix=None,
                           measurements=None, new_cluster=None, viz=None, all_kalman=None, kalman_params=None,
                           rt_matrix=None) -> dict:
        times = {}
        dic_to_save = {"q": None, "q_dot": None, "q_ddot": None,
                       "q_raw": None,
                       "tau": None,
                       "mus_act": None,
                       "emg_proc": None,
                       "res_tau": None,
                       "jrf": None,
                       "time": None,
                       "tracked_markers": None}

        # if markers[0, 0].mean() != 0:
        if compute_ik:
            times, dic_to_save = compute_ik(msk_function,
                                             markers,
                                             frame_idx,
                                             kalman_freq=kalman_freq, times=times, dic_to_save=dic_to_save,
                                             file_path=file, n_window=n_window)
            if compute_id:
                if not compute_ik:
                    raise ValueError("Inverse kinematics must be computed to compute inverse dynamics")
                times, dic_to_save = compute_id(msk_function, f_ext, external_loads, times, dic_to_save)

            if compute_so:
                if not compute_id:
                    raise ValueError("Inverse dynamics must be computed to compute static optimization")
                times, dic_to_save = compute_so(msk_function, emg, times, dic_to_save, scaling_factor,
                                                 print_optimization_status=print_optimization_status,
                                                 emg_names=emg_names, track_idx=tracking_idx, map_emg_idx=map_emg_idx)

            if compute_jrf:
                if not compute_so:
                    raise ValueError("Static optimization must be computed to compute joint reaction forces")
                times, dic_to_save = compute_jrf(msk_function, times, dic_to_save, external_loads)

            times["tot"] = sum(times.values())
            dic_to_save["time"] = times
            return dic_to_save
        else:
            return None


def main(model_dir, participants, processed_data_path, save_data=False, plot=True, results_from_file=False, stop_frame=None,
         source=(), model_source=(), source_to_keep=(), live_filter=False, interpolate_dlc=True, in_pixel=False):


    # emg_names = ["PECM",
    #              "bic",
    #              "tri",
    #              "LAT",
    #              'TRP1',
    #              "DELT1",
    #              'DELT2',
    #              'DELT3']
    processed_source = []
    models = ["normal_500_down_b1"]
    filtered = ["filtered"]

    for part in participants:
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file and "less" not in file and "more" not in file and "result" not in file]
        for file in all_files:
            path = f"{processed_data_path}{os.sep}{part}{os.sep}{file}"
            labeled_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_pp.bio"
            print(f"Processing participant {part}, trial : {file}")
            source_init = ["depth", "vicon", "minimal_vicon"]
            markers_from_source_tmp, names_from_source_tmp, forces, f_ext, emg, vicon_to_depth, peaks, rt = load_data(
                prefix + "/Projet_hand_bike_markerless/process_data", part, f"{file.split('_')[0]}_{file.split('_')[1]}",
                True
            )
            markers_from_source = [None for i in range(len(source))]
            names_from_source = [None for i in range(len(source))]
            for m, mark in enumerate(markers_from_source_tmp):
                if source_init[m] in source:
                    assert mark.shape[1] == len(names_from_source_tmp[m])
                    markers_from_source[source.index(source_init[m])] = mark
                    names_from_source[source.index(source_init[m])] = names_from_source_tmp[m]
            frame_idx = None
            ratio = ["1"]
            model = models[0]
            filt = filtered[0]
            suffix = "_offline" if not live_filter else ""
            file_name_to_save = prefix + f"/Projet_hand_bike_markerless/process_data/{part}/result_biomech_{file.split('_')[0]}_{file.split('_')[1]}_{model}{suffix}.bio"
            all_results = {}
            dlc_times = None
            label_in_pixel = None
            dlc_in_pixel = None
            idx_start, idx_end = None, None
            for r_idx, r in enumerate(ratio):
                print("ratio :", r)
                dlc_data_path = f"{path}{os.sep}marker_pos_multi_proc_3_crops_{model}_ribs_and_cluster_{r}_with_model_pp_full.bio"
                if not os.path.exists(dlc_data_path):
                    continue

                if "dlc" in source:
                    shape = markers_from_source_tmp[0].shape[2]
                    marker_dlc_filtered, frame_idx, names, dlc_times, dlc_in_pixel, label_in_pixel, idx_start, idx_end = get_dlc_data(dlc_data_path,
                        model, filt, part, file, path, labeled_data_path, rt, shape, ratio= r + "_alone", filter=not live_filter, in_pixel=in_pixel)
                    #marker_dlc_filtered[:, 2, :] = np.nan
                    markers_from_source[source.index("dlc")] = marker_dlc_filtered
                    names_from_source[source.index("dlc")] = names

                if not results_from_file:
                    existing_keys = []
                    data = None
                    if os.path.exists(file_name_to_save):
                        try:
                            data = load(file_name_to_save)
                            existing_keys = list(data.keys())
                        except:
                            os.remove(file_name_to_save)
                    for s in range(len(markers_from_source)):
                        if r_idx > 0 and "dlc" not in source[s]:
                            continue
                        src_tmp = f"dlc_{r}" if source[s] == "dlc" else source[s]
                        if source[s] in source_to_keep and source[s] in existing_keys:
                            all_results[source[s]] = data[source[s]]
                            continue
                        elif "dlc" not in source[s]:
                            markers_from_source[s] = self.adjust_idx({"mark": markers_from_source[s]}, idx_start, idx_end)["mark"]
                        # if source[s] == "dlc":
                        #     model_path = f"{model_dir}/{part}/model_scaled_dlc_test_wu_fixed.bioMod"
                        # else:
                        model_path = f"{model_dir}/{part}/model_scaled_{model_source[s]}_new_seth.bioMod"

                        if "vicon" in source[s] or "depth" in source[s]:
                            reorder_marker_from_source, reordered_names = reorder_markers(
                                markers_from_source[s][:, :-3, :], biorbd.Model(model_path), names_from_source[s][:-3])
                        else:
                            reorder_marker_from_source = markers_from_source[s]
                            reordered_names = names
                            # if part == "P15" and "gear_20" in file:
                            #     reorder_marker_from_source = markers_from_source[s]


                            #idx = names.index("xiph")
                            #reorder_marker_from_source[:, idx, :] = np.repeat(reorder_marker_from_source[:, idx, 0], reorder_marker_from_source[:, idx, :].shape[1]).reshape(3, reorder_marker_from_source[:, idx, :].shape[1])
                        # else:
                        #     idx = reordered_names.index("DELT")
                        #     reorder_marker_from_source_filtered = np.zeros((3, reorder_marker_from_source.shape[1], reorder_marker_from_source.shape[2]))
                        #     for i in range(3):
                        #         reorder_marker_from_source_filtered[i, :idx, :] = OfflineProcessing().butter_lowpass_filter(
                        #             reorder_marker_from_source[i, :idx, :],
                        #             4, 120, 2)
                        #         reorder_marker_from_source_filtered[i, idx:, :] = OfflineProcessing().butter_lowpass_filter(
                        #             reorder_marker_from_source[i, idx:, :],
                        #             6, 120, 2)

                        bio_model = biorbd.Model(model_path)
                        msk_function = MskFunctions(model=bio_model, data_buffer_size=20, system_rate=120)
                        if stop_frame is None:
                            stop_frame = (frame_idx[-1] - frame_idx[0]) * 2
                        if (frame_idx[-1] - frame_idx[0]) * 2 < stop_frame:
                            stop_frame = (frame_idx[-1] - frame_idx[0]) * 2
                        if stop_frame % 2 != 0:
                            stop_frame -= 1
                        stop_frame_tmp = int(stop_frame // 2) if live_filter and "dlc" in src_tmp else stop_frame
                        if "dlc" not in source[s] or not live_filter:
                            range_frame = range(stop_frame_tmp)
                        else:
                            range_frame = range(frame_idx[0], frame_idx[-1])
                        result_biomech = process_all_frames(reorder_marker_from_source.copy(), msk_function,
                                                            src_tmp,
                                                            forces, (1000, 10), emg,
                                                            f_ext,
                                                            img_idx=frame_idx,
                                                            compute_ik=True,
                                                            compute_id=True, compute_so=False, compute_jrf=False,
                                                            range_idx=range_frame,
                                                            stop_frame=stop_frame_tmp,
                                                            file=f"{processed_data_path}/{part}" + "/" + file,
                                                            print_optimization_status=False, filter_depth=live_filter,
                                                            emg_names=emg_names,
                                                            measurements=None,
                                                            marker_names=reordered_names,
                                                            part=part,
                                                            rt_matrix=rt,
                                                            in_pixel=in_pixel
                                                            )
                        result_biomech["markers"] = markers_from_source[s]
                        # if "depth" in source[s] or "vicon" in source[s]:
                        #     result_biomech["tracked_markers"] = reorder_marker_from_source[..., :stop_frame]
                        #     result_biomech["marker_names"] = reordered_names
                        all_results[src_tmp] = result_biomech
                        if src_tmp == "dlc_11":
                            b = bioviz.Viz(loaded_model=msk_function.model, show_floor=False)
                            b.load_movement(result_biomech["q_raw"])
                            mark = result_biomech["tracked_markers"]
                            b.load_experimental_markers(mark[:, :, :stop_frame])
                            b.exec()
                        print("Done for source ", src_tmp)
                        if save_data:
                            if "dlc" in src_tmp:
                                all_results[src_tmp]["time"]["time_to_get_markers"] = dlc_times
                                from data_processing.post_process_data import ProcessData
                                from utils import refine_synchro
                                tmp1 = all_results[src_tmp]["tracked_markers"][:, 7, :].copy()
                                tmp2 = all_results[src_tmp]["tracked_markers"][:, 6, :].copy()
                                all_results[src_tmp]["tracked_markers"][:, 6, :] = tmp1
                                all_results[src_tmp]["tracked_markers"][:, 7, :] = tmp2
                                all_results[src_tmp]["marker_names"][6], all_results[src_tmp]["marker_names"][7] = all_results[src_tmp]["marker_names"][7], all_results[src_tmp]["marker_names"][6]
                                dlc_mark_tmp = all_results[src_tmp]["tracked_markers"][:, :, :].copy()
                                dlc_mark_tmp = np.delete(dlc_mark_tmp,
                                                         all_results[src_tmp]["marker_names"].index("ribs"), axis=1)
                                dlc_mark, idx = refine_synchro(all_results["minimal_vicon"]["tracked_markers"][:, :, :],
                                                               dlc_mark_tmp,
                                                               plot_fig=plot)
                                if interpolate_dlc and live_filter:
                                    # idx = 0
                                    from data_processing.post_process_data import ProcessData
                                    for key in all_results[src_tmp].keys():
                                        if isinstance(all_results[src_tmp][key], np.ndarray):
                                            if idx != 0:
                                                dlc_data = all_results[src_tmp][key][..., :-idx]
                                            else:
                                                dlc_data = all_results[src_tmp][key][..., :]
                                            n_roll = 0
                                            if n_roll > 0:
                                                dlc_data = dlc_data[..., :-n_roll]
                                                dlc_data = np.concatenate((dlc_data[..., 0:n_roll], dlc_data), axis=-1)
                                            all_results[src_tmp][key] = ProcessData()._fill_and_interpolate(
                                                dlc_data,
                                                fill=False,
                                                shape=all_results["depth"]["q_raw"].shape[1])
                                else:
                                    for key in all_results[src_tmp].keys():
                                        if isinstance(all_results[src_tmp][key], np.ndarray):
                                            if idx != 0:
                                                all_results[src_tmp][key] = all_results[src_tmp][key][..., :-idx]
                                            else:
                                                all_results[src_tmp][key] = all_results[src_tmp][key][..., :]

            if plot:
                # from utils import refine_synchro
                # dlc_mark_tmp = all_results["dlc_1"]["tracked_markers"][:, :, :].copy()
                # dlc_mark_tmp = np.roll(dlc_mark_tmp, 1, axis=2)
                # dlc_mark_tmp[..., 0] = dlc_mark_tmp[..., 1]
                # dlc_mark_tmp = np.delete(dlc_mark_tmp, all_results["dlc_1"]["marker_names"].index("ribs"), axis=1)
                # dlc_mark, idx = refine_synchro(all_results[source[0]]["tracked_markers"][:, :, :],
                #                                 dlc_mark_tmp,
                #                                plot_fig=plot)
                # if interpolate_dlc and live_filter:
                #     from post_process_data import ProcessData
                #     for key in all_results["dlc"].keys():
                #         if isinstance(all_results["dlc"][key], np.ndarray):
                #             all_results["dlc"][key] = ProcessData()._fill_and_interpolate(all_results["dlc"][key][..., :],
                #                                                                           fill=False,
                #                                                                           shape=all_results["minimal_vicon"]["q_raw"].shape[1])
                import matplotlib.pyplot as plt
                c = ["r", "g", "b", "k"]
                # for s in range(len(source)):
                # if source[s] != "vicon":
                # if live_filter:
                #     all_results["dlc_1"]["tracked_markers"] = all_results["dlc_1"]["tracked_markers"][..., :-idx]
                #     all_results["dlc_1"]["q_raw"] = all_results["dlc_1"]["q_raw"][:, :-idx]

                plt.figure("markers")

                # all_results[source[1]]["tracked_markers"][:, 1, :] = np.repeat(all_results[source[1]]["tracked_markers"][:, 1, 0],  all_results[source[1]]["tracked_markers"].shape[2]).reshape(3, all_results[source[1]]["tracked_markers"].shape[2])
                t = np.linspace(0, stop_frame, all_results["dlc_1"]["tracked_markers"].shape[2])
                count = 0
                for i in range(13):
                    plt.subplot(4, 4, i + 1)
                    if all_results["dlc_1"]["marker_names"][i] == "ribs":
                        count += 1
                    for j in range(3):
                        plt.plot(all_results["depth"]["tracked_markers"][j, i, :stop_frame], c=c[0])
                        plt.plot(t, all_results["dlc_1"]["tracked_markers"][j, count, :stop_frame], c=c[1])
                        plt.plot(all_results["minimal_vicon"]["tracked_markers"][j, i, :stop_frame], c=c[3])
                    plt.title(all_results[source[0]]["marker_names"][i] + all_results["dlc_1"]["marker_names"][count] + all_results["minimal_vicon"]["marker_names"][i])
                    count += 1
                plt.legend([source[0], "dlc_1", "minimal_vicon"])
                factor = 1 # 57.3
                plt.figure("q")
                for i in range(all_results[source[0]]["q_raw"].shape[0]):
                    plt.subplot(4, 4, i + 1)
                    plt.plot(all_results[source[0]]["q"][i, :stop_frame] * 57.3, c=c[0])
                    plt.plot( t, all_results["dlc_1"]["q"][i, :stop_frame] * 57.3, c=c[1])
                    plt.plot(all_results["minimal_vicon"]["q"][i, :stop_frame] * 57.3, c=c[2])
                    plt.plot(all_results["vicon"]["q"][i, :stop_frame] * 57.3, c=c[3])
                plt.legend([source[0], "dlc_1", "minimal_vicon"])
                plt.figure("qdot")
                for i in range(all_results[source[0]]["q_raw"].shape[0]):
                    plt.subplot(4, 4, i + 1)
                    plt.plot(all_results[source[0]]["q_dot"][i, :stop_frame] * factor, c=c[0])
                    plt.plot( t, all_results["dlc_1"]["q_dot"][i, :stop_frame] * factor, c=c[1])
                    plt.plot(all_results["minimal_vicon"]["q_dot"][i, :stop_frame] * factor, c=c[2])
                    plt.plot(all_results["vicon"]["q_dot"][i, :stop_frame] * factor, c=c[3])
                plt.legend([source[0], "dlc_1", "minimal_vicon"])
                plt.figure("qddot")
                for i in range(all_results[source[0]]["q_raw"].shape[0]):
                    plt.subplot(4, 4, i + 1)
                    plt.plot(all_results[source[0]]["q_ddot"][i, :stop_frame] * factor, c=c[0])
                    plt.plot( t, all_results["dlc_1"]["q_ddot"][i, :stop_frame] * factor, c=c[1])
                    plt.plot(all_results["minimal_vicon"]["q_ddot"][i, :stop_frame] * factor, c=c[2])
                    plt.plot(all_results["vicon"]["q_ddot"][i, :stop_frame] * factor, c=c[3])
                plt.legend([source[0], "dlc_1", "minimal_vicon"])
                plt.figure("tau")
                for i in range(all_results[source[0]]["q_raw"].shape[0]):
                    plt.subplot(4, 4, i + 1)
                    plt.plot(all_results[source[0]]["tau"][i, :stop_frame], c=c[0])
                    plt.plot( t, all_results["dlc_1"]["tau"][i, :stop_frame], c=c[1])
                    plt.plot(all_results["minimal_vicon"]["tau"][i, :stop_frame], c=c[2])
                    plt.plot(all_results["vicon"]["tau"][i, :stop_frame], c=c[3])
                plt.legend([source[0], "dlc_1", "minimal_vicon"])
                plt.show()
            if save_data:
                all_results = process_cycles(all_results, peaks, n_peaks=None)
                all_results["image_idx"] = frame_idx
                all_results["emg"] = emg
                all_results["f_ext"] = f_ext
                all_results["peaks"] = peaks
                all_results["ref_in_pixel"] = label_in_pixel
                all_results["dlc_in_pixel"] = dlc_in_pixel
                all_results["rt_matrix"] = rt
                all_results["vicon_to_depth"] = vicon_to_depth
                save(all_results, file_name_to_save, safe=False)
                print("Saved")


def main_new(model_dir, participants, processed_data_path, save_data=True, results_from_file=False, stop_frame=None,
     plot=False, source=None, model_source=None, source_to_keep=[], live_filter=True,
     interpolate_dlc=True, in_pixel=False, emg_names=None):
    biomech_pipeline = BiomechPipeline(stop_frame=stop_frame)
    all_files, mapped_part = get_all_file(participants, processed_data_path)
    for part, file in zip(mapped_part, all_files):
        get_data_from_sources(source, )


if __name__ == '__main__':
    emg_names = ["PectoralisMajorThorax_M",
                 "BIC",
                 "TRI_lat",
                 "LatissimusDorsi_S",
                 'TrapeziusScapula_S',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                 'DeltoideusScapula_P']
    model_dir = prefix + "/Projet_hand_bike_markerless/RGBD"
    participants = [f"P{i}" for i in range(10, 15)]
    participants.pop(participants.index("P12"))
    # participants.pop(participants.index("P15"))
    # participants.pop(participants.index("P16"))
    #participants.pop(participants.index("P14"))
    source = ["depth", "minimal_vicon", "vicon",  "dlc"]
    model_source = ["depth", "minimal_vicon", "vicon", "dlc_ribs"]
    source = ["depth", "minimal_vicon", "vicon",  "dlc"]
    model_source = ["depth", "minimal_vicon", "vicon", "dlc_ribs"]
    processed_data_path = prefix + "/Projet_hand_bike_markerless/RGBD"
    main_new(model_dir, participants, processed_data_path, save_data=True, results_from_file=False, stop_frame=None,
         plot=False, source=source, model_source=model_source, source_to_keep=[], live_filter=True,
         interpolate_dlc=True, in_pixel=False)
    main(model_dir, participants, processed_data_path, save_data=True, results_from_file=False, stop_frame=None,
         plot=False, source=source, model_source=model_source, source_to_keep=[], live_filter=True,
         interpolate_dlc=True, in_pixel=False)

