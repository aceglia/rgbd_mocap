from processing_data.biomech_analysis.biomech_pipeline import BiomechPipeline
from processing_data.file_io import get_all_file, get_data_from_sources
from processing_data.biomech_analysis.enums import FilteringMethod
import os

prefix = "/mnt/shared" if os.name == "posix" else "Q:/"


def init_kalman_filter_parameters(biomech_pipeline, source):
    if "dlc" in source:
        measurement_noise = [2] * 17
        proc_noise = [1] * 17
        measurement_noise[:8] = [5] * 8
        proc_noise[:8] = [1e-1] * 8
        measurement_noise[11:14] = [1] * 3
        proc_noise[11:14] = [1] * 3
    if "minimal_vicon" in source:
        measurement_noise = [5] * 20
        proc_noise = [1] * 20
        measurement_noise[7:] = [1] * len(measurement_noise[7:])
        proc_noise[7:] = [1] * len(measurement_noise[7:])
        # measurement_noise[5:7] = [2] * len(measurement_noise[7:])
        # proc_noise[5:7] = [1e-2] * len(measurement_noise[7:])
    if "depth" in source:
        measurement_noise = [1] * 20
        proc_noise = [10] * 20
        measurement_noise[:4] = [10] * 4
        proc_noise[:4] = [1] * 4
        measurement_noise[-3:] = [10] * 3
        proc_noise[-3:] = [1] * 3
        #measurement_noise[:4] = [20] * 4
        #proc_noise[:4] = [2] * 4
        # compute from cluster :
        #measurement_noise[10:13] = [1] * 3
        #proc_noise[10:13] = [1] * 3
        # measurement_noise[7:] = [5] * len(measurement_noise[7:])
        # proc_noise[7:] = [1] * len(measurement_noise[7:])
        # measurement_noise[5:7] = [10] * 2
        # proc_noise[5:7] = [1e-2] * 2
    if "vicon" in source:
        measurement_noise = [5] * 20
        proc_noise = [1] * 20
        measurement_noise[7:] = [1] * len(measurement_noise[7:])
        proc_noise[7:] = [1] * len(measurement_noise[7:])
    biomech_pipeline.set_variable("measurement_noise", measurement_noise)
    biomech_pipeline.set_variable("proc_noise", proc_noise)
    return measurement_noise, proc_noise


def init_participant(
    biomech_pipeline, part, forces, f_ext, emg, vicon_to_depth, peaks, rt, trial_short, model_directory
):
    biomech_pipeline.results_dict = {}
    biomech_pipeline.set_variable("external_loads", forces)
    biomech_pipeline.set_variable("f_ext", f_ext)
    biomech_pipeline.set_variable("emg", emg)
    biomech_pipeline.set_variable("vicon_to_depth_idx", vicon_to_depth)
    biomech_pipeline.set_variable("peaks", peaks)
    biomech_pipeline.set_variable("rt_matrix", rt)
    biomech_pipeline.set_variable("trial_name", trial_short)
    biomech_pipeline.set_variable("model_dir", f"{model_directory}/{part}")
    biomech_pipeline.set_variable("scaling_factor", (1000, 10))


def main(
    model_dir,
    participants,
    processed_data_path,
    source,
    save_data=True,
    stop_frame=None,
    plot=False,
    model_source=None,
    source_to_keep=None,
    live_filter_method: list | FilteringMethod = None,
    interpolate_dlc=False,
):
    live_filter_method = live_filter_method if live_filter_method is not None else [FilteringMethod.NONE] * len(source)
    live_filter_method = (
        live_filter_method if isinstance(live_filter_method, list) else [live_filter_method] * len(source)
    )
    biomech_pipeline = BiomechPipeline(stop_frame=stop_frame)
    all_files, mapped_part = get_all_file(
        participants, processed_data_path, to_include=["gear"], to_exclude=["result", "less", "more"]
    )
    markers_rate = 120
    for part, file in zip(mapped_part, all_files):
        trial_short = file.split(os.sep)[-1].split("_")[0] + "_" + file.split(os.sep)[-1].split("_")[1]
        output_file = (
            prefix
            + f"/Projet_hand_bike_markerless/process_data/{part}/result_biomech_{trial_short}_kalman_proc_new.bio"
        )
        markers_dic, forces, f_ext, emg, vicon_to_depth, peaks, rt, dlc_frame_idx = get_data_from_sources(
            part, trial_short, source, model_dir, model_source, live_filter_method, source_to_keep, output_file
        )
        init_participant(biomech_pipeline, part, forces, f_ext, emg, vicon_to_depth, peaks, rt, trial_short, model_dir)
        key_counter = 0
        for key in markers_dic.keys():
            model_path = f"{model_dir}/{part}/model_scaled_{model_source[key_counter]}_new_seth_old.bioMod"
            biomech_pipeline.set_stop_frame(
                stop_frame,
                dlc_frame_idx,
                key,
                live_filter_method[key_counter].value != 0,
                data_shape=markers_dic[key][1].shape[2],
            )
            biomech_pipeline.init_scapula_cluster(part)
            if live_filter_method[key_counter] == FilteringMethod.Kalman:
                init_kalman_filter_parameters(biomech_pipeline, key)
                biomech_pipeline.kalman_instance, biomech_pipeline.n_markers, biomech_pipeline.reordered_idx = (
                    None,
                    None,
                    None,
                )
            markers_rate = 60 if (live_filter_method[key_counter].value != 0 and "dlc" in key) else markers_rate
            biomech_pipeline.set_variable("markers_rate", markers_rate)
            biomech_pipeline.process_all_frames(
                markers_dic[key][1],
                compute_so=True,
                compute_id=True,
                live_filter_method=live_filter_method[key_counter],
                model_path=model_path,
                marker_names=markers_dic[key][0],
            )
            key_counter += 1
            print("Done for source:", key)
        if save_data:
            biomech_pipeline.save(output_file, interpolate_dlc=interpolate_dlc)
        if plot:
            biomech_pipeline.plot_results(plot_by_cycle=False)


if __name__ == "__main__":
    participants = [f"P{i}" for i in range(9, 17)]
    source = [
        "depth",
        "vicon",
        "minimal_vicon",
        # , "dlc_0_8", "dlc_0_9", "dlc_1"
    ]
    model_source = [
        "depth",
        "vicon",
        "minimal_vicon",
        # , "dlc_ribs", "dlc_ribs", "dlc_ribs"
    ]
    filter_method = [
        FilteringMethod.Kalman,
        FilteringMethod.Kalman,
        FilteringMethod.Kalman,
        FilteringMethod.Kalman,
        FilteringMethod.Kalman,
        FilteringMethod.Kalman,
    ]
    model_dir = prefix + "/Projet_hand_bike_markerless/RGBD"
    processed_data_path = prefix + "/Projet_hand_bike_markerless/RGBD"
    main(
        model_dir,
        participants,
        processed_data_path,
        save_data=True,
        stop_frame=None,
        plot=False,
        source=source,
        model_source=model_source,
        live_filter_method=filter_method,
        interpolate_dlc=True,
    )
