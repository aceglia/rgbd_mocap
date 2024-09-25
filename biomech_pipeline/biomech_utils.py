import os

prefix = "/mnt/shared" if os.name == "posix" else "Q:/"

def _convert_string(string):
    return string.lower().replace("_", "")

def get_all_file(participants, data_dir):
    all_path = []
    parts = []
    for part in participants:
        all_files = os.listdir(f"{data_dir}{os.sep}{part}")
        all_files = [f"{data_dir}{os.sep}{part}{os.sep}{file}" for file in all_files if "gear" in file and "less" not in file and "more" not in file and "result" not in file]
        parts.append([part for _ in all_files])
        all_path.append(all_files)
    return sum(all_path, []), sum(parts, [])

def get_data_from_sources(participant, file_path, source, out_file=None):
    labeled_data_path = f"{participant}{os.sep}marker_pos_multi_proc_3_crops_pp.bio"
    print(f"Processing participant {participant}, trial : {file_path}")
    source_init = ["depth", "vicon", "minimal_vicon"]
    markers_from_source_tmp, names_from_source_tmp, forces, f_ext, emg, vicon_to_depth, peaks, rt = load_data(
        prefix + "/Projet_hand_bike_markerless/process_data", participant, f"{file_path.split('_')[0]}_{file_path.split('_')[1]}",
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
            marker_dlc_filtered, frame_idx, names, dlc_times, dlc_in_pixel, label_in_pixel, idx_start, idx_end = get_dlc_data(
                dlc_data_path,
                model, filt, part, file, path, labeled_data_path, rt, shape, ratio=r + "_alone", filter=not live_filter,
                in_pixel=in_pixel)
            # marker_dlc_filtered[:, 2, :] = np.nan
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
                    markers_from_source[s] = self.adjust_idx({"mark": markers_from_source[s]}, idx_start, idx_end)[
                        "mark"]
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