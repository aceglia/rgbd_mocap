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
    else:
        measurement_noise = [5] * 16
        proc_noise = [1e-1] * 16
        measurement_noise[7:] = [1] * 11
        proc_noise[7:] = [1] * 11
    biomech_pipeline.set_variable("measurement_noise", measurement_noise)
    biomech_pipeline.set_variable("proc_noise", proc_noise)
    return measurement_noise, proc_noise


def init_participant(biomech_pipeline, part, forces, f_ext, emg, vicon_to_depth, peaks, rt, trial_short, model_dir):
    biomech_pipeline.results_dict = {}
    biomech_pipeline.set_variable("external_loads", forces)
    biomech_pipeline.set_variable("f_ext", f_ext)
    biomech_pipeline.set_variable("emg", emg)
    biomech_pipeline.set_variable("vicon_to_depth_idx", vicon_to_depth)
    biomech_pipeline.set_variable("peaks", peaks)
    biomech_pipeline.set_variable("rt_matrix", rt)
    biomech_pipeline.set_variable("trial_name", trial_short)
    biomech_pipeline.set_variable("model_dir", f"{model_dir}/{part}")


def main(model_dir, participants, processed_data_path, source, save_data=True, stop_frame=None,
             plot=False,
             model_source=None, source_to_keep=None, live_filter_method: list | FilteringMethod = None, interpolate_dlc=False):
    live_filter_method = live_filter_method if live_filter_method is not None else [FilteringMethod.NONE] * len(source)
    live_filter_method = live_filter_method if isinstance(live_filter_method, list) else [live_filter_method] * len(source)
    biomech_pipeline = BiomechPipeline(stop_frame=stop_frame)
    all_files, mapped_part = get_all_file(participants, processed_data_path)
    markers_rate = 120
    for part, file in zip(mapped_part, all_files):
        trial_short = file.split(os.sep)[-1].split('_')[0] + "_" + file.split(os.sep)[-1].split('_')[1]
        output_file = prefix + f"/Projet_hand_bike_markerless/process_data/{part}/result_biomech_{trial_short}.bio"
        markers_dic, forces, f_ext, emg, vicon_to_depth, peaks, rt, dlc_frame_idx = get_data_from_sources(
            part, trial_short, source, model_dir, model_source,
            live_filter_method, source_to_keep,
            output_file)
        init_participant(biomech_pipeline, part, forces, f_ext, emg, vicon_to_depth, peaks, rt, trial_short, model_dir)
        key_counter = 0
        for key in markers_dic.keys():
            model_path = f"{model_dir}/{part}/model_scaled_{model_source[key_counter]}_new_seth.bioMod"
            biomech_pipeline.set_stop_frame(stop_frame, dlc_frame_idx, key, live_filter_method[key_counter].value != 0)
            biomech_pipeline.init_scapula_cluster(part)
            if live_filter_method[key_counter] == FilteringMethod.Kalman:
                init_kalman_filter_parameters(biomech_pipeline, key)
                biomech_pipeline.kalman_instance, biomech_pipeline.n_markers, biomech_pipeline.reordered_idx =None, None, None
            markers_rate = 60 if (live_filter_method[key_counter].value != 0 and "dlc" in key) else markers_rate
            biomech_pipeline.set_variable("markers_rate", markers_rate)
            biomech_pipeline.process_all_frames(markers_dic[key][1],
                                                compute_so=False,
                                                compute_id=True,
                                                live_filter_method=live_filter_method[key_counter],
                                                model_path=model_path,
                                                marker_names=markers_dic[key][0])
            key_counter += 1
            print("Done for source:", key)
        if save_data:
            biomech_pipeline.save(output_file, interpolate_dlc=interpolate_dlc)
        if plot:
            biomech_pipeline.plot_results(plot_by_cycle=False)


if __name__ == '__main__':
    participants = [f"P{i}" for i in range(9, 17)]
    source = ["depth", "minimal_vicon", "vicon", "dlc_0_8", "dlc_0_9", "dlc_1"]
    model_source = ["depth", "minimal_vicon", "vicon", "dlc_ribs", "dlc_ribs", "dlc_ribs"]
    filter_method = [FilteringMethod.Kalman, FilteringMethod.NONE, FilteringMethod.NONE, FilteringMethod.Kalman,
                     FilteringMethod.Kalman, FilteringMethod.Kalman]
    model_dir = prefix + "/Projet_hand_bike_markerless/RGBD"
    processed_data_path = prefix + "/Projet_hand_bike_markerless/RGBD"
    main(model_dir, participants, processed_data_path, save_data=False, stop_frame=2000,
        plot=True, source=source, model_source=model_source, live_filter_method=filter_method,
         interpolate_dlc=True)
