from processing_data.biomech_analysis.biomech_pipeline import BiomechPipeline
from processing_data.file_io import get_all_file, get_data_from_sources
from processing_data.biomech_analysis.enums import FilteringMethod
import os
try:
    from pyorerun import BiorbdModel, PhaseRerun
except ImportError:
    pass
from pyomeca import Markers
import numpy as np

prefix = "/mnt/shared" if os.name == "posix" else "Q:/"


def viz_rerun(results_dict, models):
    count = 0
    rerun_viz = None
    for key in results_dict.keys():
        dic_tmp = results_dict[key]
        q = dic_tmp["q"]["mean"]
        markers = dic_tmp["markers"]["mean"]
        t = np.linspace(0, q.shape[1] / 120, q.shape[1])
        # loading biorbd model
        biorbd_model = BiorbdModel(models[count])
        biorbd_model.options.mesh_color = (0, 0, 0) if key == "vicon" else biorbd_model.options.mesh_color
        rerun_viz = PhaseRerun(t) if rerun_viz is None else rerun_viz
        markers = Markers(data=markers, channels=list(biorbd_model.marker_names))
        rerun_viz.add_animated_model(biorbd_model, q, tracked_markers=markers)
        count += 1
    rerun_viz.rerun("Models")


def init_kalman_filter_parameters(biomech_pipeline, source):
    if "dlc" in source:
        # measurement_noise = [5] * 20
        # proc_noise = [1] * 20
        # measurement_noise[7:] = [1] * len(measurement_noise[7:])
        # proc_noise[7:] = [1] * len(measurement_noise[7:])

        # measurement_noise = [2] * 17
        # proc_noise = [1] * 17
        # measurement_noise[:8] = [5] * 8
        # proc_noise[:8] = [1e-1] * 8
        # measurement_noise[11:14] = [1] * 3
        # proc_noise[11:14] = [1] * 3

        measurement_noise = [1] * 20
        proc_noise = [10] * 20
        measurement_noise[:5] = [10] * 5
        proc_noise[:5] = [1] * 5
        measurement_noise[-3:] = [10] * 3
        proc_noise[-3:] = [1] * 3

        # measurement_noise = [1] * 20
        # proc_noise = [5] * 20
        # measurement_noise[:6] = [5] * 4
        # proc_noise[:6] = [1e-1] * 4
        # measurement_noise[-3:] = [10] * 3
        # proc_noise[-3:] = [0.2] * 3
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
        # measurement_noise[:4] = [20] * 4
        # proc_noise[:4] = [2] * 4
        # compute from cluster :
        # measurement_noise[10:13] = [1] * 3
        # proc_noise[10:13] = [1] * 3
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
    biomech_pipeline.set_variable("init_f_ext", f_ext)
    biomech_pipeline.set_variable("init_emg", emg)
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
    biomech_pipeline = BiomechPipeline()
    all_files, mapped_part = get_all_file(
        participants, processed_data_path, to_include=["gear"], to_exclude=["result", "less", "more"]
    )
    markers_rate = 120
    for part, file in zip(mapped_part, all_files):
        trial_short = file.split(os.sep)[-1].split("_")[0] + "_" + file.split(os.sep)[-1].split("_")[1]
        output_file = (
            prefix
            + f"/Projet_hand_bike_markerless/process_data/{part}/result_biomech_{trial_short}_with_technical_marker.bio"
        )

        markers_dic, forces, f_ext, emg, vicon_to_depth, peaks, rt, dlc_frame_idx = get_data_from_sources(
            part, trial_short, source, model_dir, model_source, live_filter_method, source_to_keep, output_file
        )
        init_participant(biomech_pipeline, part, forces, f_ext, emg, vicon_to_depth, peaks, rt, trial_short, model_dir)
        key_counter = 0
        model_path_final = []
        for key in markers_dic.keys():
            model_path = f"{model_dir}/{part}/model_scaled_{model_source[key_counter]}.bioMod"
            # if "dlc" in key:
            #
            #     # model_path = f"{model_dir}/P9/model_scaled_{model_source[key_counter]}_technical_marker.bioMod"
            #
            # elif key == "vicon":
            #     # model_path = f"{model_dir}/{part}/model_scaled_vicon_markerless.bioMod"
            #     model_path = f"{model_dir}/{part}/model_scaled_{model_source[key_counter]}_new_seth.bioMod"
            # else:
            #     model_path = f"{model_dir}/{part}/model_scaled_{model_source[key_counter]}_new_seth.bioMod"
            # if not os.path.exists(model_path):
            #     shutil.copy(f"{model_dir}/{part}/model_scaled_{model_source[key_counter]}_new_seth.bioMod", model_path)
            biomech_pipeline.set_stop_frame(
                stop_frame,
                dlc_frame_idx,
                key,
                live_filter_method[key_counter].value != 0,
                data_shape=markers_dic[key][1].shape[2],
            )
            if live_filter_method[key_counter].value != 0 and "dlc" in key:
                biomech_pipeline.fps = 60
                biomech_pipeline.f_ext = biomech_pipeline.f_ext[..., ::2]
                biomech_pipeline.emg = biomech_pipeline.emg[..., ::2]
            else:
                biomech_pipeline.fps = markers_rate
            biomech_pipeline.init_scapula_cluster(part,
                                                  measurements_dir_path=f"D:\Documents\Programmation\pose_estimation\data_collection_mesurement",
                                                  calibration_matrix_dir="D:\Documents\Programmation\pose_estimation\calibration_matrix")
            if live_filter_method[key_counter] == FilteringMethod.Kalman:
                init_kalman_filter_parameters(biomech_pipeline, key)
                biomech_pipeline.kalman_instance, biomech_pipeline.n_markers, biomech_pipeline.reordered_idx = (
                    None,
                    None,
                    None,
                )
            biomech_pipeline.process_all_frames(
                markers_dic[key][1],
                compute_so=False,
                compute_id=False,
                compute_jrf=False,
                live_filter_method=live_filter_method[key_counter],
                model_path=model_path,
                marker_names=markers_dic[key][0],
            )
            model_path_final.append(biomech_pipeline.msk_function.model.path().absolutePath().to_string())
            key_counter += 1
            print("Done for source:", key)
        if save_data:
            biomech_pipeline.save(output_file, interpolate_dlc=interpolate_dlc)

        if plot:
            biomech_pipeline.plot_results(plot_by_cycle=False)
        viz_rerun(biomech_pipeline.results_dict, model_path_final)


if __name__ == "__main__":
    participants = [f"P{i}" for i in range(9, 10)]
    #participants.pop(participants.index("P12"))
    source = [
        # "depth",
        "vicon",
        #"minimal_vicon",
        # , "dlc_0_8", "dlc_0_9",
        "dlc_1"
    ]
    model_source = [
        # "depth",
        "vicon_markerless",
        # "minimal_vicon",
        # , "dlc_ribs", "dlc_ribs",
        #"minimal_vicon",
        "dlc_technical_marker"
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
        save_data=False,
        stop_frame=1000,
        plot=True,
        source=source,
        model_source=model_source,
        live_filter_method=filter_method,
        interpolate_dlc=True,
    )
