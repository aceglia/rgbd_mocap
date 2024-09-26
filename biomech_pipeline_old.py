import os

from data_processing.post_process_data import ProcessData

import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import biorbd
from utils_old import load_data, load_data_from_dlc
try:
    from processing_data.biomech_analysis.msk_utils import process_all_frames, get_tracking_idx
except:
    pass

from biosiglive import (
    OfflineProcessing, MskFunctions,
    load, save)
import bioviz


def _convert_string(string):
    return string.lower().replace("_", "")


def rmse(data, data_ref):
    return np.sqrt(np.mean(((data - data_ref) ** 2), axis=-1))





def _interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(markers_depth.shape[0]):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def _interpolate_data_2d(data, shape):
    new_data = np.zeros((data.shape[0], shape))
    x = np.linspace(0, 100, data.shape[1])
    f_mark = interp1d(x, data)
    x_new = np.linspace(0, 100, int(new_data.shape[1]))
    new_data = f_mark(x_new)
    return new_data


def reorder_markers(markers, model, names):
    model_marker_names = [_convert_string(model.markerNames()[i].to_string()) for i in range(model.nbMarkers())]
    assert len(model_marker_names) == len(names)
    assert len(model_marker_names) == markers.shape[1]
    count = 0
    reordered_markers = np.zeros((markers.shape[0], len(model_marker_names), markers.shape[2]))
    final_names = []
    for i in range(len(names)):
        if names[i] == "elb":
            names[i] = "elbow"
        if _convert_string(names[i]) in model_marker_names:
            reordered_markers[:, model_marker_names.index(_convert_string(names[i])),
            :] = markers[:, count, :]
            final_names.append(model.markerNames()[i].to_string())
            count += 1
    return reordered_markers, final_names


def process_cycles(all_results, peaks, n_peaks=None):
    for key in all_results.keys():
        data_size = all_results[key]["q_raw"].shape[1]
        dic_tmp = {}
        for key2 in all_results[key].keys():
            if key2 == "cycle" or key2 == "rt_matrix" or key2 == "marker_names":
                continue
            array_tmp = None
            if not isinstance(all_results[key][key2], np.ndarray):
                dic_tmp[key2] = []
                continue
            if n_peaks and n_peaks > len(peaks) - 1:
                raise ValueError("n_peaks should be less than the number of peaks")
            for k in range(len(peaks) - 1):
                if peaks[k + 1] > data_size:
                    break
                interp_function = _interpolate_data_2d if len(all_results[key][key2].shape) == 2 else _interpolate_data
                if array_tmp is None:
                    array_tmp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], 120)
                    array_tmp = array_tmp[None, ...]
                else:
                    data_interp = interp_function(all_results[key][key2][..., peaks[k]:peaks[k + 1]], 120)
                    array_tmp = np.concatenate((array_tmp, data_interp[None, ...]), axis=0)
            dic_tmp[key2] = array_tmp
        all_results[key]["cycles"] = dic_tmp
    return all_results


def _get_dlc_data(data_path, model, filt, part, file, path, labeled_data_path, rt, shape, ratio, filter=True, in_pixel=False):

    data_dlc, data_labelingn, dlc_names, dlc_times, idx_start, idx_end = load_data_from_dlc(labeled_data_path, data_path, part, file)
    frame_idx = data_dlc["frame_idx"]
    #shape = (frame_idx[-1] - frame_idx[0]) * 2
    # data_nan = ProcessData()._fill_with_nan(new_markers_dlc, frame_idx)
    if filter:
        if in_pixel:
            raise RuntimeError("markers in pixel asked to be filtered.")
        new_markers_dlc = np.zeros((3,
                                    data_dlc["markers_in_meters"].shape[1],
                                    data_dlc["markers_in_meters"].shape[2]
                                    ))
        markers_dlc_hom = np.ones((4, data_dlc["markers_in_meters"].shape[1], data_dlc["markers_in_meters"].shape[2]))
        markers_dlc_hom[:3, ...] = data_dlc["markers_in_meters"][:3, ...]
        for k in range(new_markers_dlc.shape[2]):
            new_markers_dlc[:, :, k] = np.dot(np.array(rt), markers_dlc_hom[:, :, k])[:3, :]
        #shape = len(frame_idx) * 2
        new_markers_dlc = ProcessData()._fill_and_interpolate(data=new_markers_dlc,
                                                              idx=frame_idx,
                                                              shape=shape,
                                                              fill=True)
        import json
        from utils_old import _convert_cluster_to_anato_old
        measurements_dir_path = "data_collection_mesurement"
        calibration_matrix_dir = "../scapula_cluster/calibration_matrix"
        measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{part}.json"
                                          ))
        measurements = measurement_data[f"with_depth"]["measure"]
        calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"][
            "calibration_matrix_name"]
        anato_from_cluster, landmarks_dist = _convert_cluster_to_anato_old(measurements,
                                                                       calibration_matrix,
                                                                       new_markers_dlc[:, -3:, :] * 1000)
        first_idx = dlc_names.index("clavac")
        new_markers_dlc = np.concatenate((new_markers_dlc[:, :first_idx + 1, :],
                                                   anato_from_cluster[:3, :, :] * 0.001,
                                                   new_markers_dlc[:, first_idx + 1:, :]), axis = 1
                                                  )
        new_markers_dlc_filtered = np.zeros((3, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
        for i in range(3):
            new_markers_dlc_filtered[i, :8, :] = OfflineProcessing().butter_lowpass_filter(
                new_markers_dlc[i, :8, :],
                2, 120, 2)
            new_markers_dlc_filtered[i, 8:, :] = OfflineProcessing().butter_lowpass_filter(
                new_markers_dlc[i, 8:, :],
                10, 120, 2)

    else:
        key = "markers_in_meters" if not in_pixel else "markers_in_pixel"
        new_markers_dlc_filtered = data_dlc[key]
    dlc_in_pixel = data_dlc["markers_in_pixel"]
    ref_in_pixel = data_labelingn["markers_in_pixel"]
    return new_markers_dlc_filtered, frame_idx, dlc_names, dlc_times, dlc_in_pixel, ref_in_pixel, idx_start, idx_end


def adjust_idx(data, idx_start, idx_end):
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


def main(model_dir, participants, processed_data_path, save_data=False, plot=True, results_from_file=False, stop_frame=None,
         source=(), model_source=(), source_to_keep=(), live_filter=False, interpolate_dlc=True, in_pixel=False):
    prefix = "/mnt/shared" if os.name == "posix" else "Q:/"
    emg_names = ["PectoralisMajorThorax_M",
                 "BIC",
                 "TRI_lat",
                 "LatissimusDorsi_S",
                 'TrapeziusScapula_S',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                 'DeltoideusScapula_P']
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
                    marker_dlc_filtered, frame_idx, names, dlc_times, dlc_in_pixel, label_in_pixel, idx_start, idx_end = _get_dlc_data(dlc_data_path,
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
                            markers_from_source[s] = adjust_idx({"mark": markers_from_source[s]}, idx_start, idx_end)["mark"]
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
                                from utils_old import refine_synchro
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


if __name__ == '__main__':
    prefix = "/mnt/shared" if os.name == "posix" else "Q:/"
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
    main(model_dir, participants, processed_data_path, save_data=True, results_from_file=False, stop_frame=None,
         plot=False, source=source, model_source=model_source, source_to_keep=[], live_filter=True,
         interpolate_dlc=True, in_pixel=False)

