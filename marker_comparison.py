import os
import numpy as np
import pandas as pd
from biosiglive import load, save, RealTimeProcessing
from pathlib import Path
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
try:
    from msk_utils import perform_biomechanical_pipeline
except:
    pass

from biosiglive import OfflineProcessing, OfflineProcessingMethod, load, MskFunctions, InverseKinematicsMethods
from biosiglive.processing.msk_utils import ExternalLoads

try:
    import bioviz
except:
    pass
try:
    import biorbd
    bio_pack = True
except:
    bio_pack = False
    pass
import glob
import csv


def _prepare_mot(output_file, n_rows, n_columns, columns_names):
    headers = [
        [output_file],
        ["version = 1"],
        [f"nRows = {n_rows}"],
        [f"nColumns = {n_columns + 1}"],
        ["inDegrees=yes"],
        ["endheader"]
    ]
    first_row = ["time", ]
    for i in range(len(columns_names)):
        first_row.append(columns_names[i])
    headers.append(first_row)
    return headers


def _get_vicon_to_depth_idx(names_depth=None, names_vicon=None):
    vicon_markers_names = [_convert_string(name) for name in names_vicon]
    depth_markers_names = [_convert_string(name) for name in names_depth]
    vicon_to_depth_idx = []
    for name in vicon_markers_names:
        if name in depth_markers_names:
            vicon_to_depth_idx.append(vicon_markers_names.index(name))
    return vicon_to_depth_idx


def _convert_string(string):
    return string.lower().replace("_", "")


def read_sto_mot_file(filename):
    """
    Read sto or mot file from Opensim
    ----------
    filename: path
        Path of the file witch have to be read
    Returns
    -------
    Data Dictionary with file informations
    """
    data = {}
    data_row = []
    first_line = ()
    end_header = False
    with open(f"{filename}", "rt") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) == 0:
                pass
            elif row[0][:9] == "endheader":
                end_header = True
                first_line = idx + 1
            elif end_header is True and row[0][:9] != "endheader":
                row_list = row[0].split("\t")
                if idx == first_line:
                    names = row_list
                else:
                    data_row.append(row_list)
    data_mat = np.zeros((len(names), len(data_row)))
    for r in range(len(data_row)):
        data_mat[:, r] = np.array(data_row[r], dtype=float)
    return data_mat, names


def write_sto_mot_file(all_paths, vicon_markers, depth_markers):
    all_data = []
    files = glob.glob(f"{all_paths['trial_dir']}Res*")
    with open(files[0], 'r') as file:
        csvreader = csv.reader(file, delimiter='\n')
        for row in csvreader:
            all_data.append(np.array(row[0].split("\t")))
    all_data = np.array(all_data, dtype=float).T
    data_index = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
    all_data = all_data[data_index, :]
    all_data = np.append(all_data, np.zeros((3, all_data.shape[1])), axis=0)

    source = ["vicon", "depth"]
    rate = [120, 60]
    interp_size = [vicon_markers.shape[2], depth_markers.shape[2]]
    for i in range(2):
        x = np.linspace(0, 100, all_data.shape[1])
        f = interp1d(x, all_data)
        x_new = np.linspace(0, 100, interp_size[i])
        all_data_int = f(x_new)
        dic_data = {
            "RFX": all_data_int[0, :],
            "RFY": all_data_int[1, :],
            "RFZ": all_data_int[2, :],
            "RMX": all_data_int[3, :],
            "RMY": all_data_int[4, :],
            "RMZ": all_data_int[5, :],
            "LFX": all_data_int[6, :],
            "LFY": all_data_int[7, :],
            "LFZ": all_data_int[8, :],
            "LMX": all_data_int[9, :],
            "LMY": all_data_int[10, :],
            "LMZ": all_data_int[11, :],
            "px": all_data_int[-1, :],
            "py": all_data_int[-1, :],
            "pz": all_data_int[-1, :]
        }
        # save(dic_data, f"{dir}/{participant}_{trial}_sensix_{source[i]}.bio")
        headers = _prepare_mot(f"{all_paths['trial_dir']}{participant}_{trial}_sensix_{source[i]}.mot",
                               all_data_int.shape[1], all_data_int.shape[0], list(dic_data.keys()))
        duration = all_data_int.shape[1] / rate[i]
        time = np.around(np.linspace(0, duration, all_data_int.shape[1]), decimals=3)
        for frame in range(all_data_int.shape[1]):
            row = [time[frame]]
            for j in range(all_data_int.shape[0]):
                row.append(all_data_int[j, frame])
            headers.append(row)
        with open(f"{all_paths['trial_dir']}{participant}_{trial}_sensix_{source[i]}.mot", 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(headers)

def rmse(data, data_ref):
    return np.sqrt(np.mean(((data - data_ref) ** 2), axis=-1))


def load_data(participant, trial, all_paths=None, cluster_dic=None):
    pass

def scale_model(participant, model_ordered_names, model_format_names, all_paths):
    trial = ["seated_anato", "standing_anato"]
    dir = []
    for i in range(len(trial)):
        dir.append(glob.glob(participant_dir + f"{trial[i]}*")[0] + os.sep)
    all_paths_tmp = all_paths.copy()
    all_paths_tmp["trial_dir"] = dir[0]
    (markers_depth, _, markers_depth_names, _, idx) = load_data(participant, "seated_anato", all_paths_tmp)
    all_paths_tmp["trial_dir"] = dir[1]
    (_, markers_vicon, _, markers_vicon_names, _) = load_data(participant, "standing_anato", all_paths_tmp)

    # markers_depth = order_markers_from_names(model_ordered_names[0], markers_depth_names, markers_depth)
    idx = idx[:-1]
    markers_depth = fill_with_nan(markers_depth, idx)
    markers_depth_filled = np.zeros((3, markers_depth.shape[1], markers_depth.shape[2]))
    for i in range(3):
        marker_depth_df = pd.DataFrame(markers_depth[i, :, :], markers_depth_names)
        markers_depth_filled[i, :, :] = marker_depth_df.interpolate(method='linear', axis=1)
    markers_depth_filtered = np.zeros((3, markers_depth_filled.shape[1], markers_depth_filled.shape[2]))
    for i in range(3):
        markers_depth_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(markers_depth_filled[i, :, :],
                                                                                    4,
                                                                                    60,
                                                                                    4)

    markers_depth = markers_depth_filtered
    markers_depth = interpolate_data(markers_vicon, markers_depth)
    # markers_depth = get_rotated_markers(markers_depth[:, :, :], markers_vicon[:, :, :], all_paths_tmp, vicon_to_depth_idx,
    #                                             from_file=False)
    source = ["depth", "vicon"]
    markers_names = [markers_depth_names, markers_vicon_names]
    rate = [120, 120]
    markers = [markers_depth, markers_vicon]
    for i in range(0, 2):
        all_paths_tmp["trial_dir"] = dir[i]
        marker = order_markers_from_names(model_ordered_names[i], markers_names[i], markers[i])
        # q_recons, _ = MskFunctions(model=f"{all_paths_tmp['participant_dir']}wu_bras_gauche_depth.bioMod",
        #                         data_buffer_size=50).compute_inverse_kinematics(marker[:, :, :50],
        #                                                                         method=InverseKinematicsMethods.BiorbdKalman)
        # import bioviz
        # b = bioviz.Viz(model_path=f"{all_paths_tmp['participant_dir']}wu_bras_gauche_depth.bioMod")
        # b.load_experimental_markers(marker[:, :, :50])
        # b.load_movement(q_recons)
        # b.exec()
        osim_model_path = f"{all_paths_tmp['main_dir']}wu_bras_gauche.osim"
        # model = biorbd.Model(f"{all_paths_tmp['participant_dir']}wu_bras_gauche_depth.bioMod")
        # model_name = [name.to_string() for name in model.markerNames()]
        write_trc(all_paths_tmp, participant, trial[i], marker, rate[i], source[i], model_format_names[i])
        # ---------- model scaling ------------ #
        model_output = (f"{all_paths_tmp['participant_dir']}" + Path(osim_model_path).stem + f"{source[i]}_scaled.osim")
        scaling_tool = f"{all_paths_tmp['participant_dir']}" + f"scaling_tool_{source[i]}.xml"
        trc_file = f"{all_paths_tmp['trial_dir']}{participant}_{trial[i]}_from_{source[i]}.trc"
        # pyosim.Scale(
        #     model_input=osim_model_path,
        #     model_output=model_output,
        #     xml_input=scaling_tool,
        #     xml_output=f"{all_paths_tmp['participant_dir']}" + f"scaling_tool_output_{source[i]}.xml",
        #     static_path=trc_file,
        #     mass=75,
        # )

    # plt.figure("markers_depth")
    # # markers_depth[:, 1, :] = markers_depth[:, 0, :]
    # for i in range(len(markers_depth_names)):
    #     plt.subplot(4, 4, i+1)
    #     for j in range(3):
    #         plt.plot(markers_depth[j, i, :]*1000, c='b')
    #         plt.plot(markers_vicon[j, vicon_to_depth_idx[i], :]*1000, c='r')
    #
    # plt.show()
    # fig = plt.figure("vicon")
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1, 1, 1])
    # end_plot = 1
    # for i in range(len(vicon_to_depth_idx)):
    #     ax.scatter(markers_vicon[0, vicon_to_depth_idx[i], :end_plot],
    #                markers_vicon[1, vicon_to_depth_idx[i], :end_plot],
    #                markers_vicon[2, vicon_to_depth_idx[i], :end_plot], c='r')
    #     ax.scatter(markers_depth[0, i, :end_plot], markers_depth[1, i, :end_plot], markers_depth[2, i, :end_plot], c='b')
    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')
    # plt.show()



def compute_error(markers_depth, markers_vicon, vicon_to_depth_idx, state_depth, state_vicon):
    n_markers_depth = markers_depth.shape[1]
    err_markers = np.zeros((n_markers_depth, 1))
    # new_markers_depth_int = OfflineProcessing().butter_lowpass_filter(new_markers_depth_int, 6, 120, 4)
    for i in range(len(vicon_to_depth_idx)):
        # ignore NaN values
        # if i not in [4, 5, 6]:
        nan_index = np.argwhere(np.isnan(markers_vicon[:, vicon_to_depth_idx[i], :]))
        new_markers_depth_tmp = np.delete(markers_depth[:, i, :], nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(markers_vicon[:, vicon_to_depth_idx[i], :], nan_index, axis=1)
        nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
        new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
        err_markers[i, 0] = np.median(np.sqrt(
            np.mean(((new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000) ** 2), axis=0)))

    err_q = []
    for i in range(state_depth.shape[0]):
        # if i not in [3, 4, 5, 8]:
        err_q.append(np.mean(np.sqrt(np.mean(((state_depth[i, :] - state_vicon[i, :]) ** 2), axis=0))))
    return err_markers, err_q


def compute_ik(participant, session, markers, source, method=InverseKinematicsMethods.BiorbdKalman,
               use_opensim=False,
               markers_names=None,
               all_paths=None,
               onset=None, ):
    if not use_opensim:
        model_path = f"data_files\{participant}_{session}\wu_bras_gauche_{source}.bioMod"
        funct = MskFunctions(model=model_path, data_buffer_size=100)
        q, _ = funct.compute_inverse_kinematics(markers[:, :, :100], method=method)
        q_mean = np.mean(q, axis=1).round(3)
        print(q_mean[3], q_mean[4], q_mean[5], " xyz ", q_mean[0], q_mean[1], q_mean[2])
        with open(model_path, 'r') as file:
            data = file.read()
        # replace the target string
        data = data.replace('\tRT 0 0 0 xyz 0 0 0 // thorax',
                            f'\tRT {q_mean[3]} {q_mean[4]} {q_mean[5]}'
                            f' xyz {q_mean[0]} {q_mean[1]} {q_mean[2]} // thorax\n'
                            f'\t\t//RT 0 0 0 xyz 0 0 0 // thorax')
        with open(model_path + "_tmp", 'w') as file:
            file.write(data)
        funct = MskFunctions(model=model_path + "_tmp", data_buffer_size=markers.shape[2])
        os.remove(model_path + "_tmp")
        q, q_dot = funct.compute_inverse_kinematics(markers, method=method)
    else:
        source = ["depth", "vicon"]
        rate = [120, 120]
        q = []
        for i in range(2):
            # osim_model_path = f"{all_paths['participant_dir']}wu_bras_gauche_{source[i]}_scaled.osim"
            osim_model_path = f"{all_paths['participant_dir']}wu_bras_gauche_depth_scaled_markers.osim"

            write_trc(all_paths, participant, trial, markers[i], rate[i], source[i], markers_names[i])
            trc_file = f"{all_paths['trial_dir']}{participant}_{trial}_from_{source[i]}.trc"
            if onset is not None:
                onsets = {f"{participant}_{trial}_from_{source[i]}": onset}
            else:
                onsets = None
            ik_input = f"{all_paths['main_dir']}" + f"ik_{source[i]}_template.xml"
            ik_output = f"{all_paths['trial_dir']}" + f"inverse_kin_{source[i]}_out.xml"
            mot_output = f"{all_paths['trial_dir']}" + f"ik_files_{source[i]}"
            # pyosim.InverseKinematics(osim_model_path, ik_input, ik_output, trc_file, mot_output, onsets=onsets)
            q_tmp, _ = read_sto_mot_file(f"{mot_output}{os.sep}{participant}_{trial}_from_{source[i]}.mot")
            q.append(q_tmp)
    return q


def compute_id():
    pass


def _interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(3):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def refine_synchro(depth_markers, vicon_markers, vicon_to_depth_idx):
    error = np.inf
    depth_markers_tmp = np.zeros((3, depth_markers.shape[1], depth_markers.shape[2]))
    vicon_markers_tmp = np.zeros((3, vicon_markers.shape[1], vicon_markers.shape[2]))
    idx = 0
    for i in range(30):
        vicon_markers_tmp = vicon_markers[:, :, :-i] if i!= 0 else vicon_markers
        depth_markers_tmp = _interpolate_data(depth_markers, vicon_markers_tmp.shape[2])
        error_markers, _ = compute_error(
            depth_markers_tmp, vicon_markers_tmp, vicon_to_depth_idx, np.zeros((1,1)), np.zeros((1,1))
        )
        error_tmp = np.mean(error_markers)
        # print(error_tmp, i)
        if error_tmp < error:
            error = error_tmp
            idx = i
        else:
            break
    return depth_markers_tmp, vicon_markers_tmp, idx


if __name__ == '__main__':
    model_dir = "models/"
    participants = ["P9"]#, "P10", "P11", "P12", "P13", "P14"]  # ,"P9", "P10",
    processed_data_path = "Q:\Projet_hand_bike_markerless/process_data"
    model_paths = [model_dir + "wu_bras_gauche_depth.bioMod", model_dir + "wu_bras_gauche_vicon.bioMod"]
    all_errors_mark, all_errors_q = [], []
    source = ["depth", "vicon"]
    for part in participants:
        all_files = os.listdir(f"{processed_data_path}/{part}")
        all_files = [file for file in all_files if "gear" in file]

        for file in all_files:
            data = load(f"{processed_data_path}/{part}/{file}")
            markers_depth = data["markers_depth_interpolated"][:, :, :150]
            markers_vicon = data["truncated_markers_vicon"][:, :, :150]
            sensix_data = data["sensix_data_interpolated"]
            depth_markers_names = data["depth_markers_names"]
            vicon_markers_names = data["vicon_markers_names"]
            vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)
            markers_depth, markers_vicon, idx = refine_synchro(markers_depth, markers_vicon, vicon_to_depth_idx)
            markers_depth_filtered = np.zeros((3, markers_depth.shape[1], markers_depth.shape[2]))
            for i in range(3):
                markers_depth_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(markers_depth[i, :, :],
                                                                          4, 120, 4)
            markers_from_source = [markers_depth, markers_vicon]
            markers_name_from_source = [depth_markers_names, vicon_markers_names]
            model_paths = [model_dir + "wu_bras_gauche_depth.bioMod",
                           model_dir + "wu_bras_gauche_vicon.bioMod"]
            forces = ExternalLoads()
            forces.add_external_load(
                point_of_application=[0, 0, 0],
                applied_on_body="radius_left_pro_sup_left",
                express_in_coordinate="ground",
                name="hand_pedal",
                load=np.zeros((6, 1)),
            )
            f_ext = np.array([sensix_data["RMY"],
                             -sensix_data["RMX"],
                             sensix_data["RMZ"],
                             sensix_data["RFY"],
                             -sensix_data["RFX"],
                             sensix_data["RFZ"]])
            # for s in range(len(source)):
            #
            # for s in range(0, 2):
            #     ik_function = MskFunctions(model=model_paths[s], data_buffer_size=6)
            #     for i in range(markers_vicon.shape[2]):
            #         tic = time.time()
            #         result_biomech = perform_biomechanical_pipeline(markers=markers_from_source[s][:, :-3, i],
            #                                                         msk_function=ik_function,
            #                                                         msk_function=ik_function,
            #                                                         frame_idx=i,
            #                                                         external_loads=forces,
            #                                                         kalman_freq=120,
            #                                                         f_ext=f_ext[:, i],
            #                                                         )
            #         # if result_biomech is not None:
            #             # result_biomech["time"]["process_time"] = time.time() - tic
            #             # result_biomech["markers_names"] = markers_name_from_source[s]
            #             # save(result_biomech,
            #             #      f"{processed_data_path}/{part}/result_biomech_{source[s]}.bio", add_data=True)
            result_data_depth = load(f"{processed_data_path}/{part}/result_biomech_{file}_{source[0]}.bio")
            result_data_vicon = load(f"{processed_data_path}/{part}/result_biomech_{file}_{source[1]}.bio")
            print("time to process depth:", np.mean(result_data_depth["time"]["process_time"]))
            print("time to process vicon:", np.mean(result_data_vicon["time"]["process_time"]))
            q_depth = result_data_depth["q"]
            q_vicon = result_data_vicon["q"]
            import bioviz
            b = bioviz.Viz(model_path=model_paths[0],
                           show_muscles=False,
                           show_local_ref_frame=False,
                           # show_global_ref_frame=False,
                           # background_color=[1, 1, 1],
                           mesh_opacity=1,
                           )
            b.load_movement(q_depth)
            model_bio = biorbd.Model(model_paths[0])
            # b.load_experimental_markers(markers_depth[:, :-3, :])
            f_ext_to_plot = np.zeros((1, 6, int(q_depth.shape[1])))
            for i in range(q_depth.shape[1]):
                A = [0, 0, 0, 1]
                B = [0, 0, 0, 1]
                all_jcs = model_bio.allGlobalJCS(q_depth[:, i])
                RT = all_jcs[-1 + 6].to_array()
                RT2 = all_jcs[-9 + 6].to_array()
                A = RT @ A
                B = RT2 @ B

                # f_ext[0, 3:, i] = (RT2 @ (np.array([dic_data["LFX"][i],  dic_data["LFY"][i], dic_data["LFZ"][i], 1])))[:3]
                f_ext_to_plot[0, 3:, i] = f_ext[3:, 0, i]

                # f_ext[0, :3, i] = model_bio.CoMbySegment(q[:, i])[-1].to_array()
                f_ext_to_plot[0, :3, i] = B[:3]
                # f_ext[:3, 0] + cross(vecteur_BA, f_ext[3:6, 0])

            b.load_experimental_forces(f_ext_to_plot, segments=["ground"], normalization_ratio=0.5)
            b.exec()
            error_markers, error_q = compute_error(markers_depth, markers_vicon, vicon_to_depth_idx,
                                                   q_depth, q_vicon)
            all_errors_mark.append(np.mean(error_markers))
            print("part:", part, "file", file, np.mean(error_markers))
            # t_d = np.linspace(0, 100, markers_depth.shape[2])
            # t_v = np.linspace(0, 100, markers_vicon.shape[2])

            # plt.figure()
            # for i in range(len(vicon_to_depth_idx)):
            #     plt.subplot(4, 4, i+1)
            #     for j in range(3):
            #         plt.plot(t_d, markers_depth_filtered[j, i, :], "b")
            #         plt.plot(t_d, markers_depth[j, i, :], "g")
            #         plt.plot(t_v, markers_vicon[j, vicon_to_depth_idx[i], :], "r")
            # plt.show()
    #         sources = ["depth", "depth filtered", "vicon_reduced"]
    #
    #         kalman = []
    #         if do_biomechanical_pipeline:
    #             markers_from_source = [ordered_markers_depth, ordered_markers_vicon, reduce_ordered_markers_vicon]
    #             ordered_names_from_source = [depth_model_format_names, vicon_model_format_names, depth_model_format_names]
    #             markers_from_source_raw = np.copy(markers_from_source[0])
    #             # markers_kalman = np.zeros_like(ordered_markers_depth_before_filtered)
    #             for j in range(0, 2):
    #                 if os.path.isfile(f"{trial_dir}result_biomech_{sources[j]}.bio"):
    #                     os.remove(f"{trial_dir}result_biomech_{sources[j]}.bio")
    #                 if j == 0:
    #                     marker_filter = [RealTimeProcessing(60, 5), RealTimeProcessing(60, 5), RealTimeProcessing(60, 5)]
    #                 ik_function = MskFunctions(model=model_paths[j], data_buffer_size=6)
    #                 emg_mat = np.zeros((8, markers_from_source[j].shape[2]))
    #                 for i in range(markers_from_source[j].shape[2]):
    #                     ratio = 18 if j == 1 else 36
    #                     emg_proc = emg_processing.process_emg(raw_emg[:, i:i + ratio],
    #                                                           band_pass_filter=True,
    #                                                           normalization=True,
    #                                                           mvc_list=mvc[:8],
    #                                                           moving_average_window=200
    #                                                           )[:, -1]
    #                     if j == 0:
    #                         for b in range(3):
    #                             markers_from_source[j][b, :, i] = marker_filter[b].process_emg(
    #                                 markers_from_source[j][b, :, i][:, np.newaxis],
    #                                 moving_average=True,
    #                                 band_pass_filter=False,
    #                                 centering=False,
    #                                 absolute_value=False,
    #                                 moving_average_window=5,
    #                             )[:, -1]
    #                         if markers_from_source[j][0, :, i].max() > 0:
    #                             l_collar_TS = cluster_config["l_collar_TS"]
    #                             l_pointer_TS = cluster_config["l_pointer_TS"]
    #                             l_pointer_IA = cluster_config["l_pointer_IA"]
    #                             l_collar_IA = cluster_config["l_collar_IA"]
    #                             angle_wand_ia = cluster_config["angle_wand_ia"]
    #                             l_wand_ia = cluster_config["l_wand_ia"]
    #                             calibration_matrix = cluster_config["calibration_matrix"]
    #                             anato_from_cluster_depth, _ = convert_cluster_to_anato(l_collar_TS, l_pointer_TS, l_pointer_IA,
    #                                                                                    l_collar_IA, angle_wand_ia, l_wand_ia,
    #                                                                                    calibration_matrix,
    #                                                                                    markers_depth[:, [3, 4, 5],
    #                                                                                    i:i + 1] * 1000)
    #                             markers_from_source[j][:, 4, i] = anato_from_cluster_depth[:3, 0, 0] / 1000
    #                             markers_from_source[j][:, 5, i] = anato_from_cluster_depth[:3, 2, 0] / 1000
    #                             markers_from_source[j][:, 6, i] = anato_from_cluster_depth[:3, 1, 0] / 1000
    #                         # markers_from_source[j][:, :, i] = markers_from_source_raw[:, :, i]
    #
    #                     if j == 0:
    #                         f_ext_tmp = f_ext[:, ::2][:, i][:, np.newaxis]
    #                     else:
    #                         f_ext_tmp = f_ext[:, i][:, np.newaxis]
    #                     tic = time.time()
    #                     # if i > 10:
    #                     freq = 120 if j == 1 else 60
    #                     result_biomech = perform_biomechanical_pipeline(markers=markers_from_source[j][:, :, i],
    #                                                                     msk_function=ik_function,
    #                                                                     frame_idx=i,
    #                                                                     external_loads=forces,
    #                                                                     kalman_freq=freq,
    #                                                                     f_ext=f_ext_tmp,
    #                                                                     )
    #                     if result_biomech is not None:
    #                         result_biomech["time"]["process_time"] = time.time() - tic
    #                         result_biomech["markers_names"] = ordered_names_from_source[j]
    #                         save(result_biomech, f"{trial_dir}result_biomech_{sources[j]}.bio", add_data=True)
    #     # result_biomech["frame_idx"] = camera.camera_frame_numbers[camera.frame_idx]
    #
    # # plt.figure()
    # # for i in range(markers_from_source[0].shape[1]):
    # #     plt.subplot(4, 4, i+1)
    # #     plt.plot(markers_from_source[0][1, i, :])
    # #     plt.plot(markers_from_source_raw[1, i, :])
    # # plt.show()
    # results = []
    # mus_torque = []
    # start_offset = 5
    # for k in range(2):
    #     results.append(load(f"{trial_dir}result_biomech_{sources[k]}.bio"))
    # for k in range(2):
    #     ratio = 1
    #     if k == 0:
    #         for key in results[k].keys():
    #             try:
    #                 results[k][key] = fil_and_interpolate(results[k][key][..., 1:], idx[6:],
    #                                                       results[k + 1][key][..., 20:].shape[-1])
    #                 results[k][key] = results[k][key][..., :]
    #             except:
    #                 pass
    #         start_offset = 0
    #     else:
    #         for key in results[k].keys():
    #             try:
    #                 results[k][key] = results[k][key][..., 20:]
    #             except:
    #                 pass
    #         start_offset = 0
    #     model = biorbd.Model(model_paths[k])
    #     tau, mus_act = results[k]["tau"][:, start_offset::ratio], results[k]["mus_act"][:, start_offset::ratio]
    #     q, q_dot = results[k]["q"][:, start_offset::ratio], results[k]["q_dot"][:, start_offset::ratio]
    #     # import bioviz
    #     # b = bioviz.Viz(model_paths[0])
    #     # b.load_movement(q[:, 6:])
    #     # b.load_experimental_markers(markers_from_source[0][:, :, 6+2:])
    #     # b.exec()
    #     mus_j_torque = np.zeros((tau.shape[0], tau.shape[1]))
    #     mus_name_list = [name.to_string() for name in model.muscleNames()]
    #     muscles_states = model.stateSet()
    #     for a in range(tau.shape[1]):
    #         for m in range(model.nbMuscles()):
    #             muscles_states[m].setActivation(mus_act[m, a])
    #         muscles_force = model.muscleForces(muscles_states, q[:, a], q_dot[:, a])
    #         mus_j_torque[:, a] = model.muscularJointTorque(muscles_force, q[:, a], q_dot[:, a]).to_array()
    #     mus_torque.append(mus_j_torque)
    #     from msk_utils import get_tracking_idx
    #
    #     track_idx = get_tracking_idx(model=biorbd.Model(model_paths[k]))
    #     plt.figure("q")
    #     for i in range(q.shape[0]):
    #         plt.subplot(4, 4, i + 1)
    #         plt.plot(results[k]["q"][i, start_offset::ratio])
    #         plt.plot(results[k]["q_raw"][i, start_offset::ratio])
    #     plt.figure("q_dot")
    #     for i in range(q_dot.shape[0]):
    #         plt.subplot(4, 4, i + 1)
    #         plt.plot(results[k]["q_dot"][i, start_offset::ratio])
    #     plt.figure("q_ddot")
    #     for i in range(q.shape[0]):
    #         plt.subplot(4, 4, i + 1)
    #         plt.plot(results[k]["q_ddot"][i, start_offset::ratio])
    #     plt.figure("tau")
    #     for i in range(tau.shape[0]):
    #         plt.subplot(4, 4, i + 1)
    #         # plt.plot(mus_j_torque[i, :], "--")
    #         plt.plot(results[k]["tau"][i, start_offset::ratio], "r")
    #         # plt.plot(results[k]["res_tau"][i, ::ratio], ".-")
    #         plt.plot(results[k]["res_tau"][i, start_offset::ratio] + mus_j_torque[i, :])
    #     plt.figure("mus_act")
    #     for i in range(mus_act.shape[0]):
    #         plt.subplot(6, 6, i + 1)
    #         if i in track_idx:
    #             plt.plot(results[k]["emg_proc"][track_idx.index(i), start_offset::ratio], "r")
    #         plt.plot(results[k]["mus_act"][i, start_offset::ratio])
    #
    # markers_depth_int = fil_and_interpolate(markers_depth, idx[:], markers_vicon.shape[-1])
    #
    # # plt.figure("markers_depth")
    # # for i in range(len(vicon_to_depth_idx)):
    # #     plt.subplot(4, 4, i+1)
    # #     for j in range(3):
    # #         plt.plot(markers_depth_int[j, i, :]*1000, c='b')
    # #         # plt.plot(new_markers_depth[j, i, :]*1000, c='g')
    # #         plt.plot(markers_vicon[j, vicon_to_depth_idx[i], :]*1000, c='r')
    # #         plt.plot(results[0]["markers"][j, i, :]*1000, c='g')
    # #         plt.plot(results[1]["markers"][j, vicon_to_depth_idx[i], :]*1000, c='y')
    # ordered_markers_depth_int = fil_and_interpolate(ordered_markers_depth[:, :, 5:], idx[5:],
    #                                                 results[0]["markers"].shape[-1])
    # plt.figure("markers_depth")
    # for i in range(ordered_markers_depth_int.shape[1]):
    #     plt.subplot(4, 4, i + 1)
    #     for j in range(3):
    #         plt.plot(ordered_markers_depth_int[j, i, :-5] * 1000, c='b')
    #         # plt.plot(new_markers_depth[j, i, :]*1000, c='g')
    #         # plt.plot(markers_vicon[j, vicon_to_depth_idx[i], :] * 1000, c='r')
    #         plt.plot(results[0]["markers"][j, i, 5:] * 1000, c='g')
    #         # plt.plot(results[1]["markers"][j, vicon_to_depth_idx[i], :] * 1000, c='y')
    #
    # error_markers, error_q = compute_error(markers_depth_int, markers_vicon, vicon_to_depth_idx,
    #                                        results[0]["q"][1:, :], results[1]["q"][1:, :])
    # print("mean_error_markers", np.mean(error_markers[:]))
    # print("mean_error_q", np.mean(error_q[:]) * 180 / np.pi)
    # print(f"process time depth: {np.mean(results[0]['time']['process_time'][1:])}")
    # print(f"process time vicon: {np.mean(results[1]['time']['process_time'][1:])}")
    # plt.show()
    # # if _scale_model:
    # #     scale_model(participant,
    # #                 [model_ordered_depth_names, model_ordered_vicon_names],
    # #                 [depth_model_format_names, vicon_model_format_names],
    # #                 all_paths
    # #                 )
    #
    # # q = compute_ik(participant,
    # #                                  session,
    # #                                  [markers_depth_int, markers_vicon],
    # #                                  source=["depth", "vicon"],
    # #                                  use_opensim=True,
    # #                                  markers_names=[depth_model_format_names, vicon_model_format_names],
    # #                                  all_paths=all_paths,
    # #                onset=[0, 1.2]
    # #                                  )
    # # final_idx = 1
    # # fig = plt.figure("markers")
    # # ax = fig.add_subplot(111, projection='3d')
    # # for i in range(markers_vicon.shape[1]):
    # #     if i < markers_depth.shape[1]:
    # #         ax.scatter(markers_depth_int[0, i, :final_idx], markers_depth_int[1, i, :final_idx], markers_depth_int[2, i, :final_idx], c='b')
    # #         # ax.scatter(new_markers_depth[0, i, :final_idx], new_markers_depth[1, i, :final_idx], new_markers_depth[2, i, :final_idx], c='g')
    # #
    # #     if i in vicon_to_depth_idx:
    # #         ax.scatter(markers_vicon[0, i, :final_idx], markers_vicon[1, i, :final_idx],
    # #                    markers_vicon[2, i, :final_idx], c='r')
    # # for i in range(3):
    # #     ax.scatter(anato_from_cluster_depth[0, i, :final_idx],
    # #                anato_from_cluster_depth[1, i, :final_idx],
    # #                anato_from_cluster_depth[2, i, :final_idx], c="g")
    # #     ax.scatter(anato_from_cluster_vicon[0, i, :final_idx],
    # #                anato_from_cluster_vicon[1, i, :final_idx],
    # #                anato_from_cluster_vicon[2, i, :final_idx], c="y")
    # # plt.show()
    #
    # # depth_ik, vicon_ik = q[0], q[1]
    # # for i in range(vicon_ik.shape[0]):
    # #     plt.subplot(4, 6, i + 1)
    # #     plt.plot(vicon_ik[i, :])
    # #     plt.plot(depth_ik[i, :])
    # # plt.show()
    # # plt.plot(markers_depth_filtered[2, 8, :])
    # # plt.show()
    # vicon_to_depth_idx = [vicon_to_depth_idx]
    # vicon_to_depth_idx.append([markers_vicon.shape[1],
    #                            markers_vicon.shape[1] + 1,
    #                            markers_vicon.shape[1] + 2])
    # vicon_to_depth_idx = sum(vicon_to_depth_idx, [])
    #
    # depth_ik = np.zeros((7, markers_depth_int.shape[2]))
    # vicon_ik = np.zeros((7, markers_depth_int.shape[2]))
    # markers_depth_int = np.append(markers_depth_int, anato_from_cluster_depth[:3, :, :], axis=1)
    # markers_vicon = np.append(markers_vicon, anato_from_cluster_vicon[:3, :, :], axis=1)
    #
    # error_markers, error_q = compute_error(markers_depth_int, markers_vicon, vicon_to_depth_idx,
    #                                        depth_ik[1:, :], vicon_ik[1:, :])
    # # error_markers_non_filter, error_q_bis = compute_error(new_markers_depth, markers_vicon, vicon_to_depth_idx,
    # #                               depth_ik[1:, :], vicon_ik[1:, :])
    # #
    # # print("mean_error_markers_non_filter", np.mean(error_markers_non_filter[:]))
    # print("mean_error_markers", np.mean(error_markers[:]))
    # print("mean_error_q", np.mean(error_q[:]) * 180 / np.pi)
    #
    # plt.show()
    # # fig = plt.figure("vicon")
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.set_box_aspect([1, 1, 1])
    # # end_plot = 500
    # # for i in range(len(vicon_to_depth_idx)):
    # #     ax.scatter(markers_vicon[0, vicon_to_depth_idx[i], :end_plot],
    # #                markers_vicon[1, vicon_to_depth_idx[i], :end_plot],
    # #                markers_vicon[2, vicon_to_depth_idx[i], :end_plot], c='r')
    # #     ax.scatter(markers_depth_int[0, i, :end_plot], markers_depth_int[1, i, :end_plot], markers_depth_int[2, i, :end_plot], c='b')
    # #     ax.set_xlabel('X Label')
    # #     ax.set_ylabel('Y Label')
    # #     ax.set_zlabel('Z Label')
    #
    # plt.figure("states")
    # for i in range(depth_ik[1:, :].shape[0] - 2):
    #     plt.subplot(4, 5, i + 1)
    #     plt.plot(depth_ik[i + 1, :], c='b')
    #     plt.plot(vicon_ik[i + 1, :], c='r')
    # plt.show()
