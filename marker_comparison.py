import os

import numpy as np
import pandas as pd
import json
import cv2

import biosiglive
from scapula_cluster.from_cluster_to_anato import ScapulaCluster
from biosiglive import load, save, ExternalLoads, RealTimeProcessing, RealTimeProcessingMethod
# from pyomeca import Markers, Analogs
from pathlib import Path
# from osim_to_biomod.converter import Converter
import time
import matplotlib.pyplot as plt

try:
    from msk_utils import perform_biomechanical_pipeline
except:
    pass

from biosiglive import OfflineProcessing, OfflineProcessingMethod, load, MskFunctions, InverseKinematicsMethods
try:
    import casadi as ca
except:
    pass
from scipy.interpolate import interp1d
# from C3DtoTRC import WriteTrcFromMarkersData
try:
    import bioviz
except:
    pass
try:
    import biorbd
    bio_pack=True
except:
    bio_pack = False
    pass

import glob
import csv
try:
    import pyosim
    import opensim
    osim_pack = True
except:
    osim_pack = False
    pass

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
        time = np.around(np.linspace(0, duration,  all_data_int.shape[1]), decimals=3)
        for frame in range(all_data_int.shape[1]):
            row = [time[frame]]
            for j in range(all_data_int.shape[0]):
                row.append(all_data_int[j, frame])
            headers.append(row)
        with open(f"{all_paths['trial_dir']}{participant}_{trial}_sensix_{source[i]}.mot", 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(headers)


def homogeneous_transform_optimization(points1, points2):
    assert len(points1) == len(points2), "Point sets must have the same number of points."

    num_points = points1.shape[1]
    # Create optimization variables
    x = ca.MX.sym("x", 12)  # [t_x, t_y, t_z, R11, R12, R13, R21, R22, R23, R31, R32, R33]
    # Extract translation and rotation components
    t = x[:3]
    R = ca.MX(3, 3)
    R[0, 0] = x[3]
    R[0, 1] = x[4]
    R[0, 2] = x[5]
    R[1, 0] = x[6]
    R[1, 1] = x[7]
    R[1, 2] = x[8]
    R[2, 0] = x[9]
    R[2, 1] = x[10]
    R[2, 2] = x[11]

    # Create objective function to minimize distance
    distance = 0
    for i in range(num_points):
        transformed_point = ca.mtimes(R, points1[:, i]) + t
        distance += ca.sumsqr(transformed_point[:] - points2[:, i])

    # Create optimization problem
    nlp = {'x': x, 'f': distance}
    opts = {'ipopt.print_level': 5, 'ipopt.tol': 1e-9}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve the optimization problem
    solution = solver()
    # Extract the optimal translation and rotation
    optimal_t = solution["x"][:3]
    optimal_R = np.ndarray((3, 3))
    optimal_R[0, 0] = solution["x"][3]
    optimal_R[0, 1] = solution["x"][4]
    optimal_R[0, 2] = solution["x"][5]
    optimal_R[1, 0] = solution["x"][6]
    optimal_R[1, 1] = solution["x"][7]
    optimal_R[1, 2] = solution["x"][8]
    optimal_R[2, 0] = solution["x"][9]
    optimal_R[2, 1] = solution["x"][10]
    optimal_R[2, 2] = solution["x"][11]
    return optimal_R, optimal_t


def rmse(data, data_ref):
    return np.sqrt(np.mean(((data - data_ref) ** 2), axis=-1))


def start_idx_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data["start_frame"]


def load_data(participant, trial, all_paths=None, cluster_dic=None):
    all_color_files_tmp = glob.glob(all_paths["trial_dir"] + f"color*.png")
    # vicon_data = load(fr"{all_paths['data_dir']}{participant}_{trial}_c3d.bio")
    if "gear" in trial:
        vicon_data = load(fr"{all_paths['trial_dir']}/pedalage_{trial.replace('_', '')}_processed.bio")
    else:
        vicon_data = load(fr"{all_paths['trial_dir']}/{trial}_processed.bio")
    idxs = [i for i in range(len(vicon_data["markers_names"])) if
            vicon_data["markers_names"][i].lower().replace("_", "") in ["scapaa", "scapia", "scapts"]]
    vicon_data["markers_names"] = [vicon_data["markers_names"][i] for i in range(len(vicon_data["markers_names"])) if i not in idxs]
    new_vicon_data = np.zeros((4, len(vicon_data["markers_names"]), vicon_data["markers"].shape[2]))
    count = 0
    for i in range(len(vicon_data["markers_names"]) + len(idxs)):
        if i not in idxs:
            new_vicon_data[:, count, :] = vicon_data["markers"][:, i, :]
            count += 1
    vicon_data["markers"] = new_vicon_data
    if os.path.isfile(fr"{all_paths['trial_dir']}tracking_config.json"):
        start_frame = start_idx_from_json(fr"{all_paths['trial_dir']}tracking_config.json")
    else:
        start_frame = 0
    try:
        depth_data = load(f"{all_paths['trial_dir']}markers_pos.bio", number_of_line=len(all_color_files_tmp)-start_frame)
    except:
        depth_data = {"markers_in_meters": np.zeros((3, 13, len(all_color_files_tmp))),
                      "markers_names": ["xiph", "ster", "clavsc", "M1",
                                 "M2", "M3", "Clavac","delt",
                                 "arm_l", "epic_l","larm_l", "styl_r", "styl_u"]}
    idx = []
    for file in all_color_files_tmp:
        idx.append(int(Path(file.split(os.sep)[-1].split("_")[-1]).stem))
    idx.sort()
    idx_files = idx
    idx = depth_data["camera_frame_idx"]
    len_frame = len(idx)
    # size = idx[start_frame] - idx[depth_data["frame_idx"][-len_frame]]
    start_idx_vicon = abs(idx_files[0] - idx[0]) * 2
    real_nb_frame = (depth_data["camera_frame_idx"][-1] - depth_data["camera_frame_idx"][0]) * 2
    # idx = idx[start_frame:start_frame + len_frame]
    # P5
    # offset = 20
    offset = 25 # P6 gear_20
    # start_offset = 5 # P6 gear_20
    offset = 20
    start_offset = 0 # P6 gear_15

    # trigger = vicon_data["trigger"].values
    # trigger_values = np.argwhere(trigger[0, ::18] > 1.5)
    # start_idx = int(trigger_values[0][0]) + (nb_frame * 2)
    # try:
    #     end_idx = int(trigger_values[int(np.argwhere(trigger_values > start_idx + 200)[0][0])])
    # except:
    #     end_idx = trigger[0, ::18].shape[0]
    # trigger_idx = [start_idx, end_idx]

    markers_vicon_names = vicon_data["markers_names"]
    if not isinstance(vicon_data["markers"], np.ndarray):
        vicon_data["markers"] = vicon_data["markers"].values
    markers_vicon = vicon_data["markers"][:3, :, :] * 0.001
    markers_vicon = markers_vicon[:, :, start_idx_vicon + start_offset: start_idx_vicon + start_offset + real_nb_frame + offset]
    markers_depth = depth_data["markers_in_meters"][:, :, -len_frame:]
    emg_proc = vicon_data["raw_emg"]
    mvc = vicon_data["mvc"]
    emg_names = vicon_data["emg_names"][:-1]
    l_collar_TS = cluster_dic["l_collar_TS"]
    l_pointer_TS = cluster_dic["l_pointer_TS"]
    l_pointer_IA = cluster_dic["l_pointer_IA"]
    l_collar_IA = cluster_dic["l_collar_IA"]
    angle_wand_ia = cluster_dic["angle_wand_ia"]
    l_wand_ia = cluster_dic["l_wand_ia"]
    calibration_matrix = cluster_dic["calibration_matrix"]
    anato_from_cluster_vicon, landmarks_dist = convert_cluster_to_anato(
        l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia,
                             calibration_matrix, markers_vicon[:, [14, 15, 16], :] * 1000)
    anato_from_cluster_depth, _ = convert_cluster_to_anato(l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia,
                             calibration_matrix, markers_depth[:, [3, 4, 5], :] * 1000)
    anato_from_cluster_vicon = anato_from_cluster_vicon / 1000
    anato_from_cluster_depth = anato_from_cluster_depth / 1000
    markers_depth = np.append(markers_depth, anato_from_cluster_depth[:3, :, :], axis=1)
    markers_vicon = np.append(markers_vicon, anato_from_cluster_vicon[:3, :, :], axis=1)

    return markers_depth,\
           markers_vicon, depth_data["markers_names"][:13], markers_vicon_names, idx, emg_proc, emg_names, mvc


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


def convert_model(in_path, out_path, viz=False):
    """
    Convert a model from OpenSim to BioMod format.

    Parameters
    ----------
    in_path : str
        Path of the model to convert
    out_path : str
        Path of the converted model
    viz : bool, optional
        If True, the model will be visualized using bioviz package. The default is None.
    """

    #  convert_model
    converter = Converter(
        out_path, in_path, ignore_clamped_dof_tag=False, ignore_muscle_applied_tag=True, print_warnings=False
    )
    converter.convert_file()
    if viz:
        b = bioviz.Viz(model_path=out_path)
        b.exec()


def interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(3):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


def interpolate_2d_data(data, shape):
    x = np.linspace(0, 100, data.shape[1])
    f_mark = interp1d(x, data)
    x_new = np.linspace(0, 100, shape)
    new_data = f_mark(x_new)
    return new_data


def compute_error(markers_depth, markers_vicon, vicon_to_depth_idx, state_depth, state_vicon):
    n_markers_depth = markers_depth.shape[1]
    err_markers = np.zeros((n_markers_depth, 1))
    # new_markers_depth_int = OfflineProcessing().butter_lowpass_filter(new_markers_depth_int, 6, 120, 4)
    for i in range(len(vicon_to_depth_idx)):
        # ignore NaN values
        nan_index = np.argwhere(np.isnan(markers_vicon[:, vicon_to_depth_idx[i], :]))
        new_markers_depth_tmp = np.delete(markers_depth[:, i, :], nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(markers_vicon[:, vicon_to_depth_idx[i], :], nan_index, axis=1)
        nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
        new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
        err_markers[i, 0] = np.median(np.sqrt(
            np.mean(((new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000) ** 2), axis=0)))

    err_q = []
    for i in range(state_depth.shape[0] - 2):
        if i not in [3, 4, 5, 8]:
            err_q.append(np.mean(np.sqrt(np.mean(((state_depth[i, :] - state_vicon[i, :]) ** 2), axis=0))))
    return err_markers, err_q


def compute_ik(participant, session, markers, source, method=InverseKinematicsMethods.BiorbdKalman,
               use_opensim=False,
               markers_names=None,
               all_paths=None,
               onset = None,):

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


def write_trc(all_paths, participant, trial, markers, rate, source, model_markers_names):
    WriteTrcFromMarkersData(output_file_path=f"{all_paths['trial_dir']}{participant}_{trial}_from_{source}.trc",
                            markers=np.round(markers, 5),
                            marker_names=model_markers_names,
                            data_rate=rate,
                            cam_rate=rate,
                            n_frames=markers.shape[2],
                            start_frame=1,
                            units="m").write()

def save_data():
    pass


def order_markers_from_names(ordered_name, data_name, data):
    if isinstance(data_name, np.ndarray):
        data_name = data_name.tolist()
    data_name = [name.lower().replace("_", "") for name in data_name]
    ordered_name = [name.lower().replace("_", "") for name in ordered_name]
    reordered_data = np.zeros((3, len(ordered_name), data.shape[2]))
    for i, name in enumerate(ordered_name):
        if not isinstance(name, str):
            name = name.to_string()
        reordered_data[:, i, :] = data[:, data_name.index(name), :]
    return reordered_data


def _load_optimal_rt(from_file=False, data_depth=None, data_vicon=None, all_paths=None):
    if not from_file:
        r, T = homogeneous_transform_optimization(data_depth, data_vicon)
    else:
        rt_matrix = load(fr"{all_paths['participant_dir']}RT_optimal.bio")
        r = rt_matrix["rotation_matrix"]
        T = rt_matrix["translation_matrix"]
    return r, T


def get_rotated_markers(markers_depth, markers_vicon, all_paths, vicon_to_depth_idx, from_file=False):
    markers_vicon_tmp = np.zeros((3, markers_depth.shape[1], markers_vicon.shape[2]))
    for i in range(len(vicon_to_depth_idx)):
        markers_vicon_tmp[:, i, :] = markers_vicon[:, vicon_to_depth_idx[i], :]

    list_nan_idx = np.sort(np.argwhere(np.isnan(markers_vicon_tmp[:, :, :]))[:, 2])
    if list_nan_idx.shape[0] != 0:
        first_non_nan_idx = [i for i in range(list_nan_idx[-1]) if i not in list_nan_idx][0]
    else:
        first_non_nan_idx = 0

    # list_idx = [0,1,2,6,7,8,9,10,11,12]
    # r, T = _load_optimal_rt(from_file, markers_depth[:, list_idx, 0],
    #                         markers_vicon_tmp[:, list_idx, 0], all_paths)
    r, T = _load_optimal_rt(from_file, markers_depth[:, :, first_non_nan_idx],
                            markers_vicon_tmp[:, :, first_non_nan_idx], all_paths)
    new_markers_depth = np.zeros((3, markers_depth.shape[1], markers_depth.shape[2]))
    count = 0
    for i in range(markers_depth.shape[2]):
        new_markers_depth[:, :, i] = np.dot(np.array(r),
                                            np.array(markers_depth[:, :, count])
                                            ) + np.array(T)
        count += 1
    return new_markers_depth, r, T


def get_all_names(all_paths):
    depth_model_path = f"{all_paths['model_dir']}wu_bras_gauche_seth_depth.bioMod"
    vicon_model_path = f"{all_paths['model_dir']}wu_bras_gauche_seth_vicon.bioMod"
    depth_model = biorbd.Model(depth_model_path)
    depth_model_format_names = [name.to_string() for name in depth_model.markerNames()]
    vicon_model = biorbd.Model(vicon_model_path)
    vicon_model_format_names = [name.to_string() for name in vicon_model.markerNames()]
    model_ordered_depth_names = ["xiph", "ster", "clavsc", "M1",
                                 "M2", "M3", "Clavac","delt",
                                 "arm_l", "epic_l","larm_l", "styl_r", "styl_u"]
    model_ordered_vicon_names = ['ster', 'xiph', 'C7', 'T5', 'clavsc', 'clavac', 'scapAA', 'scapTS',
       'scapIA', 'delt', 'arml', 'epic_m', 'epic_l', 'elbow', 'larml',
       'styl_r', 'styl_u', 'M1', 'M2', 'M3']
    # model_ordered_depth_names = ['C7', 'T5', 'RIBS_r',  'Clavsc', 'Acrom', 'Scap_AA', 'Scap_IA',  'delt',
    # 'epic_l', 'arm_l', 'styl_u',  'styl_r', 'larm_l']
    # model_ordered_vicon_names = ['ster', 'xiph', 'c7', 't5', 'ribs', 'clavsc', 'clavac', 'scapaa', 'scapts',
    #  'scapia', 'delt', 'epicl', 'epicm', 'arml', 'elb',  'stylu', 'stylr', 'larml']
    vicon_to_depth_idx = [1, 0, 4, 14, 15, 16, 5, 6, 7, 9, 11, 12, 13]
    return (model_ordered_depth_names, model_ordered_vicon_names, depth_model_format_names, vicon_model_format_names,
           vicon_to_depth_idx)


def fill_with_nan(markers, idx):
    size = idx[-1] - idx[0]
    if len(markers.shape) == 2:
        new_markers_depth = np.zeros((markers.shape[0], size))
        count = 0
        for i in range(size):
            if i + idx[0] in idx:
                new_markers_depth[:, i] = markers[:, count]
                count += 1
            else:
                new_markers_depth[:, i] = np.nan
        return new_markers_depth
    elif len(markers.shape) == 3:
        new_markers_depth = np.zeros((3, markers.shape[1], size))
        count = 0
        for i in range(size):
            if i + idx[0] in idx:
                new_markers_depth[:, :, i] = markers[:, :, count]
                count += 1
            else:
                new_markers_depth[:, :, i] = np.nan
        return new_markers_depth


def convert_cluster_to_anato(l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia,
                             calibration_matrix, data):
    new_cluster = ScapulaCluster(
        l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia, calibration_matrix
    )
    anato_pos = new_cluster.process(marker_cluster_positions=data, cluster_marker_names=["M1", "M2", "M3"], save_file=False)
    land_dist = new_cluster.get_landmarks_distance()
    return anato_pos, land_dist


def init_kalman(points):
    n_measures = 3
    n_states = 6
    kalman = cv2.KalmanFilter(n_states, n_measures)
    kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
    # kalman.processNoiseCov = np.eye(n_states, dtype = np.float32) * 1
    kalman.measurementNoiseCov = np.eye(n_measures, dtype=np.float32) * 5
    kalman.errorCovPost = 1. * np.eye(n_states, n_states, dtype=np.float32)
    
    kalman.measurementMatrix = np.zeros((n_measures, n_states), np.float32)
    Measurement_array = []
    dt_array = []
    for i in range(0, n_states, 6):
        Measurement_array.append(i)
        Measurement_array.append(i + 1)
        Measurement_array.append(i + 2)
    
    for i in range(0, n_states):
        if i not in Measurement_array:
            dt_array.append(i)
    
    kalman.transitionMatrix[0, 2] = 1
    kalman.transitionMatrix[1, 3] = 1
    kalman.transitionMatrix[2, 4] = 1
    for i in range(0, n_measures):
        kalman.measurementMatrix[i, Measurement_array[i]] = 1
    input_points = np.float32(np.ndarray.flatten(points))
    kalman.statePre = np.array([input_points[0], input_points[1], input_points[2], 0, 0, 0], dtype=np.float32)
    kalman.statePost = np.array([input_points[0], input_points[1], input_points[2], 0, 0, 0], dtype=np.float32)
    kalman.predict()
    return kalman


def fil_and_interpolate(data, idx, shape, names=None):
    data_nan = fill_with_nan(data, idx)
    names = [f"n_{i}" for i in range(data_nan.shape[-2])] if not names else names
    if len(data_nan.shape) == 2:
        data_df = pd.DataFrame(data_nan, names)
        data_filled_extr = data_df.interpolate(method='linear', axis=1)
        data_int = interpolate_2d_data(data_filled_extr, shape)
    elif len(data_nan.shape) == 3:
        data_filled_extr = np.zeros((3, data_nan.shape[1], data_nan.shape[2]))
        for i in range(3):
            data_df = pd.DataFrame(data_nan[i, :, :], names)
            data_filled_extr[i, :, :] = data_df.interpolate(method='linear', axis=1)
        data_int = interpolate_data(data_filled_extr, shape)
    else:
        raise ValueError("Data shape not supported")
    return data_int


if __name__ == '__main__':
    _scale_model = False
    participant = "P5"
    #P5
    l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia = 77, 27, 29, 153, 9, 50.5
    #P6
    # l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia = 85, 20, 29, 147, 9, 69
    calibration_matrix = "/home/amedeo/Documents/programmation/scapula_cluster/calibration_matrix/calibration_mat_left_reflective_markers.json"

    cluster_config = {
        "l_collar_TS": l_collar_TS,
        "l_pointer_TS": l_pointer_TS,
        "l_pointer_IA": l_pointer_IA,
        "l_collar_IA": l_collar_IA,
        "angle_wand_ia": angle_wand_ia,
        "l_wand_ia": l_wand_ia,
        "calibration_matrix": calibration_matrix,
    }

    trial = "gear_20"
    # session = "session2"
    data_dir = f"/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files/{participant}/"
    config_dir = f"/media/amedeo/Disque Jeux/Documents/Programmation/pose_estimation/data_files/{participant}/"
    model_dir = "models/"
    dirs = glob.glob(config_dir + "*")
    trial_dir = None
    do_biomechanical_pipeline = False

    for dir in dirs:
        if trial in dir and len(Path(dir).suffix) == 0:
            trial_dir = dir + os.sep
    all_paths = {"data_dir": data_dir, "config_dir": config_dir, "model_dir": model_dir,
                 "trial_dir": trial_dir}

    (model_ordered_depth_names, model_ordered_vicon_names, depth_model_format_names, vicon_model_format_names,
    vicon_to_depth_idx) = get_all_names(all_paths)

    (markers_depth, markers_vicon, markers_depth_names,
     markers_vicon_names, idx, raw_emg, emg_names, mvc) = load_data(participant, trial, all_paths, cluster_config)
    markers_depth_names = [markers_depth_names]
    markers_depth_names.append(["scapaa", "scapia", "scapts"])
    markers_depth_names = sum(markers_depth_names, [])
    markers_vicon_names = [markers_vicon_names]
    markers_vicon_names.append(["scapaa", "scapia", "scapts"])
    markers_vicon_names = sum(markers_vicon_names, [])
    vicon_to_depth_idx = [vicon_to_depth_idx]
    vicon_to_depth_idx.append([max(vicon_to_depth_idx[0]) + 1,
                               max(vicon_to_depth_idx[0]) + 2,
                               max(vicon_to_depth_idx[0]) + 3])
    vicon_to_depth_idx = sum(vicon_to_depth_idx, [])
    sensix_files = glob.glob(f"{trial_dir}sensix_Results**.bio")
    sensix_data = load(sensix_files[0])
    f_ext = np.array([sensix_data["RMY"],
                      -sensix_data["RMX"],
                      sensix_data["RMZ"],
                      sensix_data["RFY"],
                      -sensix_data["RFX"],
                      sensix_data["RFZ"]])
    f_ext = interpolate_2d_data(f_ext, int(120*f_ext.shape[1] / 100))
    # hand_pedal = load()
    # write_sto_mot_file(all_paths, markers_vicon, markers_depth)
    # markers_vicon = order_markers_from_names(model_ordered_vicon_names, markers_vicon_names, markers_vicon)
    # markers_depth = order_markers_from_names(model_ordered_depth_names, markers_depth_names, markers_depth)
    model_paths = [all_paths["model_dir"] + "wu_bras_gauche_seth_depth.bioMod",
    all_paths["model_dir"] + "wu_bras_gauche_seth_vicon.bioMod"]
    emg_processing = RealTimeProcessing(2160)
    forces = ExternalLoads()
    forces.add_external_load(point_of_application=[0, 0, 0],
                             applied_on_body="hand_left",
                             express_in_coordinate="ground",
                             name="hand_pedal",
                             load=np.zeros((6, 1)))
    scaling_factor = (100, 10)
    # markers_depth = fill_with_nan(markers_depth, idx)
    # markers_depth_filled = np.zeros((3, markers_depth.shape[1], markers_depth.shape[2]))
    # for i in range(3):
    #     marker_depth_df = pd.DataFrame(markers_depth[i, :, :], markers_depth_names)
    #     markers_depth_filled[i, :, :] = marker_depth_df.interpolate(method='linear', axis=1)
    # markers_depth_filtered = np.zeros((3, markers_depth_filled.shape[1], markers_depth_filled.shape[2]))
    # for i in range(3):
    #     markers_depth_filtered[i, :, :] = OfflineProcessing().butter_lowpass_filter(markers_depth_filled[i, :, :],
    #                                                                         4, 60, 4)
    # markers_depth = markers_depth_filtered
    # markers_depth_int = interpolate_data(markers_vicon, markers_depth)
    # markers_depth_before_filter = interpolate_data(markers_vicon, markers_depth_filled)
    markers_depth, r, t = get_rotated_markers(markers_depth, markers_vicon, all_paths, vicon_to_depth_idx,
                                                from_file=False)
    # markers_depth_before_filter, r, t = get_rotated_markers(markers_depth_before_filter, markers_vicon, all_paths, vicon_to_depth_idx,
    #                                             from_file=False)
    # markers_depth = markers_depth_int
    # write_sto_mot_file(all_paths, markers_vicon, markers_depth)
    ordered_markers_vicon = order_markers_from_names(vicon_model_format_names, markers_vicon_names, markers_vicon)
    ordered_markers_depth = order_markers_from_names(depth_model_format_names, markers_depth_names, markers_depth)
    # ordered_markers_depth_before_filtered = order_markers_from_names(depth_model_format_names,
    #                                                                  markers_depth_names,
    #                                                                  markers_depth_before_filter
    #                                                                  )
    from biosiglive import OfflineProcessing
    emg_proc_offline = OfflineProcessing(2160).process_emg(raw_emg[:, :markers_vicon.shape[2] * 18], band_pass_filter=True,
                                                         moving_average=True, low_pass_filter=False,
                                                         centering=False) / np.repeat(np.array(mvc)[:8, np.newaxis],
                                                                                      markers_vicon.shape[2] * 18, axis=1)

    # plt.figure()
    # for i in range(ordered_markers_depth.shape[1]):
    #     plt.subplot(4, 4, i+1)
    #     plt.plot(ordered_markers_depth[0, i, :])
    #     plt.plot(ordered_markers_depth[1, i, :])
    # plt.show()
    sources = ["depth", "vicon"]
    kalman = []
    if do_biomechanical_pipeline:
        markers_from_source = [ordered_markers_depth, ordered_markers_vicon]

        ordered_names_from_source = [depth_model_format_names, vicon_model_format_names]
        markers_from_source_raw = np.copy(markers_from_source[0])
        # markers_kalman = np.zeros_like(ordered_markers_depth_before_filtered)
        for j in range(1):
            if os.path.isfile(f"{trial_dir}result_biomech_{sources[j]}.bio"):
                os.remove(f"{trial_dir}result_biomech_{sources[j]}.bio")
            if j == 0:
                marker_filter = [RealTimeProcessing(60, 5), RealTimeProcessing(60, 5), RealTimeProcessing(60, 5)]
            ik_function = MskFunctions(model=model_paths[j], data_buffer_size=6)
            emg_mat = np.zeros((8, markers_from_source[j].shape[2]))
            for i in range(markers_from_source[j].shape[2]):
                ratio = 18 if j == 1 else 36
                emg_proc = emg_processing.process_emg(raw_emg[:, i:i+ratio],
                                                      band_pass_filter=True,
                                                      normalization=True,
                                                      mvc_list=mvc[:8],
                                                      moving_average_window=400
                                                      )[:, -1]
                if j == 0:
                    # for b in range(3):
                        # markers_from_source[j][b, :, i] = marker_filter[b].process_emg(
                        #                 markers_from_source[j][b, :, i][:, np.newaxis],
                        #                 moving_average=True,
                        #                 band_pass_filter=False,
                        #                 centering=False,
                        #                 absolute_value=False,
                        #                 moving_average_window=5,
                        #             )[:, -1]
                    markers_from_source[j][:, :, i] = markers_from_source_raw[:, :, i]

                emg_mat[:, i] = emg_proc
                emg_mat[:, i] = emg_proc_offline[:, ::ratio][:, i]
                if j == 0:
                    f_ext_tmp = f_ext[:, ::2][:, i][:, np.newaxis]
                else:
                    f_ext_tmp = f_ext[:, i][:, np.newaxis]
                tic = time.time()
                # if i > 10:
                freq = 120 if j == 1 else 60
                result_biomech = perform_biomechanical_pipeline(markers=markers_from_source[j][:, :, i],
                                                                msk_function=ik_function,
                                                                frame_idx=i,
                                                                external_loads=forces,
                                                                scaling_factor=scaling_factor,
                                                                emg=emg_mat[:, i],
                                                                kalman_freq=freq,
                                                                emg_names=emg_names,
                                                                f_ext=f_ext_tmp,
                                                                )
                if result_biomech is not None:
                    result_biomech["time"]["process_time"] = time.time() - tic
                    result_biomech["markers_names"] = ordered_names_from_source[j]
                    save(result_biomech, f"{trial_dir}result_biomech_{sources[j]}.bio", add_data=True)

        # result_biomech["frame_idx"] = camera.camera_frame_numbers[camera.frame_idx]

    # plt.figure()
    # for i in range(markers_from_source[0].shape[1]):
    #     plt.subplot(4, 4, i+1)
    #     plt.plot(markers_from_source[0][1, i, :])
    #     plt.plot(markers_from_source_raw[1, i, :])
    # plt.show()
    results = []
    mus_torque = []
    for k in range(2):
        results.append(load(f"{trial_dir}result_biomech_{sources[k]}.bio"))
    for k in range(2):
        ratio = 1
        if k == 0:
            for key in results[k].keys():
                try:
                    results[k][key] = fil_and_interpolate(results[k][key], idx[:], results[k+1][key].shape[-1])
                except:
                    pass
        model = biorbd.Model(model_paths[k])
        tau, mus_act = results[k]["tau"][:, ::ratio], results[k]["mus_act"][:, ::ratio]
        q, q_dot = results[k]["q"][:, ::ratio], results[k]["q_dot"][:, ::ratio]
        # import bioviz
        # b = bioviz.Viz(model_paths[0])
        # b.load_movement(q[:, 6:])
        # b.load_experimental_markers(markers_from_source[0][:, :, 6+2:])
        # b.exec()
        mus_j_torque = np.zeros((tau.shape[0], tau.shape[1]))
        mus_name_list = [name.to_string() for name in model.muscleNames()]
        muscles_states = model.stateSet()
        for a in range(tau.shape[1]):
            for m in range(model.nbMuscles()):
                muscles_states[m].setActivation(mus_act[m, a])
            muscles_force = model.muscleForces(muscles_states, q[:, a], q_dot[:, a])
            mus_j_torque[:, a] = model.muscularJointTorque(muscles_force, q[:, a], q_dot[:, a]).to_array()
        mus_torque.append(mus_j_torque)
        from msk_utils import get_tracking_idx
        track_idx = get_tracking_idx(model=biorbd.Model(model_paths[k]))

        plt.figure("q")
        for i in range(q.shape[0]):
            plt.subplot(4, 3, i+1)
            plt.plot(results[k]["q"][i, ::ratio])
            plt.plot(results[k]["q_raw"][i, ::ratio])
        plt.figure("q_dot")
        for i in range(q_dot.shape[0]):
            plt.subplot(4, 3, i+1)
            plt.plot(results[k]["q_dot"][i, ::ratio])
        plt.figure("q_ddot")
        for i in range(q.shape[0]):
            plt.subplot(4, 3, i+1)
            plt.plot(results[k]["q_ddot"][i, ::ratio])
        plt.figure("tau")
        for i in range(tau.shape[0]):
            plt.subplot(4, 3, i+1)
            plt.plot(mus_j_torque[i, :], "--")
            plt.plot(results[k]["tau"][i, ::ratio], "r")
            plt.plot(results[k]["res_tau"][i, ::ratio], ".-")
            plt.plot(results[k]["res_tau"][i, ::ratio] + mus_j_torque[i, :])
        plt.figure("mus_act")
        for i in range(mus_act.shape[0]):
            plt.subplot(6, 6, i+1)
            if i in track_idx:
                plt.plot(results[k]["emg_proc"][track_idx.index(i), ::ratio], "r")
            plt.plot(results[k]["mus_act"][i, ::ratio])
    plt.show()

    new_markers_depth = np.zeros((3, markers_depth_before_filter.shape[1], markers_depth_before_filter.shape[2]))
    count = 0
    for i in range(markers_depth_before_filter.shape[2]):
        new_markers_depth[:, :, i] = np.dot(np.array(r),
                                            np.array(markers_depth_before_filter[:, :, count])
                                            ) + np.array(t)
        count += 1
    anato_from_cluster_vicon, landmarks_dist = convert_cluster_to_anato(l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia,
                             calibration_matrix, markers_vicon[:, [14, 15, 16], :] * 1000)
    anato_from_cluster_depth, _ = convert_cluster_to_anato(l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia,
                             calibration_matrix, markers_depth_int[:, [3,4,5], :] * 1000)
    anato_from_cluster_vicon = anato_from_cluster_vicon / 1000
    anato_from_cluster_depth = anato_from_cluster_depth / 1000
    print("Landmark distance to use for scaling is : "
          f"\n\taa_ts : {landmarks_dist[0]}, "
          f"\n\taa_ia: {landmarks_dist[1]}, "
          f"\n\tts_ai: {landmarks_dist[2]}")
    # if _scale_model:
    #     scale_model(participant,
    #                 [model_ordered_depth_names, model_ordered_vicon_names],
    #                 [depth_model_format_names, vicon_model_format_names],
    #                 all_paths
    #                 )

    # q = compute_ik(participant,
    #                                  session,
    #                                  [markers_depth_int, markers_vicon],
    #                                  source=["depth", "vicon"],
    #                                  use_opensim=True,
    #                                  markers_names=[depth_model_format_names, vicon_model_format_names],
    #                                  all_paths=all_paths,
    #                onset=[0, 1.2]
    #                                  )
    final_idx = 1
    fig = plt.figure("markers")
    ax = fig.add_subplot(111, projection='3d')
    for i in range(markers_vicon.shape[1]):
        if i < markers_depth.shape[1]:
            ax.scatter(markers_depth_int[0, i, :final_idx], markers_depth_int[1, i, :final_idx], markers_depth_int[2, i, :final_idx], c='b')
            # ax.scatter(new_markers_depth[0, i, :final_idx], new_markers_depth[1, i, :final_idx], new_markers_depth[2, i, :final_idx], c='g')

        if i in vicon_to_depth_idx:
            ax.scatter(markers_vicon[0, i, :final_idx], markers_vicon[1, i, :final_idx],
                       markers_vicon[2, i, :final_idx], c='r')
    for i in range(3):
        ax.scatter(anato_from_cluster_depth[0, i, :final_idx],
                   anato_from_cluster_depth[1, i, :final_idx],
                   anato_from_cluster_depth[2, i, :final_idx], c="g")
        ax.scatter(anato_from_cluster_vicon[0, i, :final_idx],
                   anato_from_cluster_vicon[1, i, :final_idx],
                   anato_from_cluster_vicon[2, i, :final_idx], c="y")
    plt.show()

    # depth_ik, vicon_ik = q[0], q[1]
    # for i in range(vicon_ik.shape[0]):
    #     plt.subplot(4, 6, i + 1)
    #     plt.plot(vicon_ik[i, :])
    #     plt.plot(depth_ik[i, :])
    # plt.show()
    # plt.plot(markers_depth_filtered[2, 8, :])
    # plt.show()
    vicon_to_depth_idx = [vicon_to_depth_idx]
    vicon_to_depth_idx.append([markers_vicon.shape[1],
                               markers_vicon.shape[1] + 1,
                               markers_vicon.shape[1] + 2])
    vicon_to_depth_idx = sum(vicon_to_depth_idx, [])

    depth_ik = np.zeros((7, markers_depth_int.shape[2]))
    vicon_ik = np.zeros((7, markers_depth_int.shape[2]))
    markers_depth_int = np.append(markers_depth_int, anato_from_cluster_depth[:3, :, :], axis=1)
    markers_vicon = np.append(markers_vicon, anato_from_cluster_vicon[:3, :, :], axis=1)

    error_markers, error_q = compute_error(markers_depth_int, markers_vicon, vicon_to_depth_idx,
                                  depth_ik[1:, :], vicon_ik[1:, :])
    # error_markers_non_filter, error_q_bis = compute_error(new_markers_depth, markers_vicon, vicon_to_depth_idx,
    #                               depth_ik[1:, :], vicon_ik[1:, :])
    #
    # print("mean_error_markers_non_filter", np.mean(error_markers_non_filter[:]))
    print("mean_error_markers", np.mean(error_markers[:]))
    print("mean_error_q", np.mean(error_q[:]))
    plt.figure("markers_depth")
    for i in range(len(vicon_to_depth_idx)):
        plt.subplot(4, 4, i+1)
        for j in range(3):
            plt.plot(markers_depth_int[j, i, :]*1000, c='b')
            # plt.plot(new_markers_depth[j, i, :]*1000, c='g')
            plt.plot(markers_vicon[j, vicon_to_depth_idx[i], :]*1000, c='r')
    plt.show()
    # fig = plt.figure("vicon")
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1, 1, 1])
    # end_plot = 500
    # for i in range(len(vicon_to_depth_idx)):
    #     ax.scatter(markers_vicon[0, vicon_to_depth_idx[i], :end_plot],
    #                markers_vicon[1, vicon_to_depth_idx[i], :end_plot],
    #                markers_vicon[2, vicon_to_depth_idx[i], :end_plot], c='r')
    #     ax.scatter(markers_depth_int[0, i, :end_plot], markers_depth_int[1, i, :end_plot], markers_depth_int[2, i, :end_plot], c='b')
    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')

    plt.figure("states")
    for i in range(depth_ik[1:, :].shape[0] - 2):
        plt.subplot(4, 5, i+1)
        plt.plot(depth_ik[i+1, :], c='b')
        plt.plot(vicon_ik[i+1, :], c='r')
    plt.show()
