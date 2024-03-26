import os.path
from pathlib import Path
import shutil

import biorbd
import numpy as np
import time
from biosiglive import InverseKinematicsMethods, RealTimeProcessingMethod, RealTimeProcessing, save
from biosiglive.streaming.utils import dic_merger


def _map_activation(emg_proc, muscle_track_idx, model, emg_names, emg_init=None, mvc_normalized=True):
    # emg_names = ["PectoralisMajor",
    #              "BIC",
    #              "TRI",
    #              "LatissimusDorsi",
    #              'TrapeziusClav',
    #              "DeltoideusClavicle_A",
    #              'DeltoideusScapula_M',
    #              'DeltoideusScapula_P']
    emg_names = ["PECM",
                 "bic",
                 "tri",
                 "LAT",
                 'TRP1',
                 "DELT1",
                 'DELT2',
                 'DELT3']
    if mvc_normalized:
        emg_proc = emg_init
    act = np.zeros((len(muscle_track_idx), int(emg_proc.shape[1])))
    act_init = np.zeros((len(muscle_track_idx), int(emg_proc.shape[1])))
    init_count = 0
    for j, name in enumerate(emg_names):
        count = 0
        for i in range(model.nbMuscles()):
            if name in model.muscleNames()[i].to_string():
                count += 1
        act[list(range(init_count, init_count + count)), :] = emg_proc[j, :]
        if emg_init is not None:
            act_init[list(range(init_count, init_count + count)), :] = emg_init[j, :]
        init_count += count
    return act, act_init


def get_tracking_idx(model):
    muscle_list = []
    for i in range(model.nbMuscles()):
        muscle_list.append(model.muscleNames()[i].to_string())
    # emg_names = ["PectoralisMajor",
    #              "BIC",
    #              "TRI",
    #              "LatissimusDorsi",
    #              'TrapeziusClav',
    #              "DeltoideusClavicle_A",
    #              'DeltoideusScapula_M',
    #              'DeltoideusScapula_P']
    emg_names = ["PECM",
                 "bic",
                 "tri",
                 "LAT",
                 'TRP1',
                 "DELT1",
                 'DELT2',
                 'DELT3']

    muscle_track_idx = []
    for i in range(len(emg_names)):
        for j in range(len(muscle_list)):
            if emg_names[i] in muscle_list[j]:
                muscle_track_idx.append(j)
    return muscle_track_idx


def _compute_ik(msk_function, markers, frame_idx, kalman_freq=60, times=None, dic_to_save=None, file_path=None):
    tic_init = time.time()
    if frame_idx == 0:
        # q, q_dot = msk_function.compute_inverse_kinematics(markers[:, :, np.newaxis],
        #                                                    InverseKinematicsMethods.BiorbdLeastSquare, )
        # # import bioviz
        # # b = bioviz.Viz(loaded_model=msk_function.model)
        # # b.load_movement(np.repeat(q,  5, axis=1))
        # # b.load_experimental_markers(np.repeat(markers[:, :, np.newaxis], 5, axis=2))
        # # b.exec()
        # msk_function.clean_all_buffers()
        model_path = msk_function.model.path().absolutePath().to_string()
        with open(model_path, "r") as file:
            data = file.read()
        init_idx = data.find("SEGMENT DEFINITION")
        end_idx = data.find("translations xyz // thorax") + len("translations xyz // thorax") + 1
        data_to_insert = f"SEGMENT DEFINITION\n\tsegment thorax_parent\n\t\tparent base\n\t \tRTinMatrix\t0\n    \t\tRT 1.57 -1.57 0 xyz 0 0 0\n\tendsegment\n// Information about ground segment\n\tsegment thorax\n\t parent thorax_parent\n\t \tRTinMatrix\t0\n    \t\tRT 0 0 0 xyz 0 0 0 // thorax\n\t\trotations xyz // thorax\n\t\ttranslations xyz // thorax\n\t\tranges \n\t\t-3 3\n\t\t-3 3\n\t\t-3 3\n\t\t-0.1 0.1\n\t\t-0.1 0.1\n\t\t-0.1 0.1\n"
        data = data[:init_idx] + data_to_insert + data[end_idx:]
        new_model_path = _compute_new_model_path(file_path, model_path)
        with open(new_model_path, "w") as file:
            file.write(data)

        msk_function.model = biorbd.Model(new_model_path)
        q, q_dot = msk_function.compute_inverse_kinematics(markers[:, :, np.newaxis],
                                                           InverseKinematicsMethods.BiorbdLeastSquare, )
        # import bioviz
        # b = bioviz.Viz(loaded_model=msk_function.model)
        # b.load_movement(np.repeat(q,  5, axis=1))
        # b.load_experimental_markers(np.repeat(markers[:, :, np.newaxis], 5, axis=2))
        # b.exec()

        with open(new_model_path, "r") as file:
            data = file.read()
        data = data.replace(
           "RT 0 0 0 xyz 0 0 0 // thorax",
           f"RT {q[3, 0]} {q[4, 0]} {q[5, 0]} xyz {q[0, 0]} {q[1, 0]} {q[2, 0]} // thorax",
        )
        data = data.replace(
            "rotations xyz // thorax",
            f"//rotations xyz // thorax",
        )
        data = data.replace(
            "translations xyz // thorax",
            f"// translations xyz // thorax",
        )
        # new_model_path = _compute_new_model_path(file_path, model_path)
        with open(new_model_path, "w") as file:
            file.write(data)
        q = q[6:, :]
        # q[:3, :] = 0
        # idx_to_delete = [0, 1, 2]
        # q = np.delete(q, idx_to_delete, axis=0)
        # import bioviz
        # b = bioviz.Viz(loaded_model=msk_function.model)
        # b.load_movement(np.repeat(q,  5, axis=1))
        # b.load_experimental_markers(np.repeat(markers[:, :, np.newaxis], 5, axis=2))
        # b.exec()
        msk_function.model = biorbd.Model(new_model_path)
        msk_function.clean_all_buffers()
        # q = np.zeros_like(q)
        #q = np.delete(q, idx_to_remove, axis=0)
        #q = np.zeros_like(q)

        #q[:3, :] = 0
    else:
        q = msk_function.kin_buffer[0].copy()
    initial_guess = [q[:, -1], np.zeros_like(q)[:, 0], np.zeros_like(q)[:, 0]]
    msk_function.compute_inverse_kinematics(markers[:, :, np.newaxis],
                                            method=InverseKinematicsMethods.BiorbdKalman,
                                            kalman_freq=kalman_freq,
                                            initial_guess=initial_guess)
    q = msk_function.kin_buffer[0].copy()
    time_ik = time.time() - tic_init
    times["ik"] = time_ik
    dic_to_save["q_raw"] = q[:, -1:]
    dic_to_save["q_dot"] = msk_function.kin_buffer[1][:, -1:]
    return times, dic_to_save


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


def _compute_id(msk_function, f_ext, external_loads, times, dic_to_save):
    tic = time.time()
    B = [0, 0, 0, 1]
    f_ext_mat = np.zeros((6, 1))
    all_jcs = msk_function.model.allGlobalJCS(msk_function.kin_buffer[0][:, -1])
    RT = all_jcs[-1].to_array()
    B = RT @ B
    vecteur_OB = B[:3]
    f_ext_mat[:3, 0] = f_ext[:3] + np.cross(vecteur_OB, f_ext[3:6])
    f_ext_mat[3:, 0] = f_ext[3:]
    external_loads.update_external_load_value(-f_ext_mat, name="hand_pedal")
    tau = msk_function.compute_inverse_dynamics(positions_from_inverse_kinematics=True,
                                                state_idx_to_process=[0],
                                                windows_length=5,
                                                external_load=external_loads)
    time_id = time.time() - tic
    times["id"] = time_id
    dic_to_save["tau"] = tau[:, -1:]
    dic_to_save["q"] = msk_function.id_state_buffer[0][:, -1:]
    dic_to_save["q_dot"] = msk_function.id_state_buffer[1][:, -1:]
    dic_to_save["q_ddot"] = msk_function.id_state_buffer[2][:, -1:]
    return times, dic_to_save


def _compute_so(msk_function, emg, times, dic_to_save, scaling_factor, print_optimization_status=False):
    if msk_function.model.nbQ() > 12:
        msk_function.tau_buffer[:6, :] = np.zeros((6, msk_function.tau_buffer.shape[1]))

    tic = time.time()
    track_idx = get_tracking_idx(msk_function.model)
    if emg is not None:
        emg = _map_activation(emg[:, np.newaxis], track_idx, msk_function.model, emg_names=None, emg_init=emg[:, np.newaxis], mvc_normalized=True)[0]
    mus_act, res_tau = msk_function.compute_static_optimization(
        # q=q_df[:, -1:], q_dot=q_dot_df[:, -1:], tau=tau[:, -1:],
        scaling_factor=scaling_factor,
        data_from_inverse_dynamics=True,
        compile_only_first_call=True,
        emg=emg,
        muscle_track_idx=track_idx,
        weight={"tau": 1000000000, "act": 1000,
                "tracking_emg": 1000000000000000,
                "pas_tau": 10000000000},
        print_optimization_status=print_optimization_status,
        torque_tracking_as_objective=True,
    )
    mus_act = np.clip(mus_act, 0.0001, 0.999999)
    mus_force = _compute_muscle_force(mus_act, msk_function.model,  msk_function.id_state_buffer[0][:, -1:],
                                      msk_function.id_state_buffer[1][:, -1:]
                                      )
    time_so = time.time() - tic
    times["so"] = time_so
    dic_to_save["mus_act"] = mus_act[:, -1:]
    dic_to_save["res_tau"] = res_tau[:, -1:]
    dic_to_save["mus_force"] = mus_force[:, np.newaxis]
    if emg is not None:
        dic_to_save["emg_proc"] = [] if emg is None else emg[:, -1:]
    return times, dic_to_save


def _compute_muscle_force(mus_act, model, q, q_dot):
    muscles_states = model.stateSet()
    for k in range(model.nbMuscles()):
        muscles_states[k].setActivation(mus_act[k, 0])
    muscles_force = model.muscleForces(muscles_states, q[:, 0], q_dot[:, 0]).to_array()
    return muscles_force


def _compute_jrf(msk_function, times, dic_to_save, external_loads=None):
    q_df, q_dot_df, q_ddot_df = (msk_function.id_state_buffer[0][:, -1:],
                                 msk_function.id_state_buffer[1][:, -1:]
                                 , msk_function.id_state_buffer[2][:, -1:])
    mus_act = dic_to_save["mus_act"]
    mus_act = np.clip(mus_act, 0.0001, 0.999999)
    tic = time.time()
    jrf = msk_function.compute_joint_reaction_load(q_df,
                                                   q_dot_df,
                                                   q_ddot_df,
                                                   mus_act,
                                                   express_in_coordinate="scapula_left",
                                                   apply_on_segment="scapula_left",
                                                   application_point=[[0, 0, 0]],
                                                   from_distal=True,
                                                   external_loads=external_loads,
                                                   kinetics_from_inverse_dynamics=False,
                                                   act_from_static_optimisation=False,
                                                   )
    time_jrf = time.time() - tic
    times["jrf"] = time_jrf
    dic_to_save["jrf"] = jrf[:, :, -1:]
    return times, dic_to_save


def process_next_frame(markers, msk_function, frame_idx, external_loads=None,
                       scaling_factor=None, emg=None, kalman_freq=120, emg_names=None, f_ext=None,
                       compute_so=False, compute_id=False, compute_jrf=False, compute_ik=True, file=None,
                       print_optimization_status=False):
    times = {}
    dic_to_save = {"q": None, "q_dot": None, "q_ddot": None,
                   "q_raw": None,
                   "tau": None,
                   "mus_act": None,
                   "emg_proc": None,
                   "res_tau": None,
                   "jrf": None,
                   "time": None}

    if markers[0, 0].mean() != 0:
        if compute_ik:
            times, dic_to_save = _compute_ik(msk_function,
                                             markers,
                                             frame_idx,
                                             kalman_freq=kalman_freq, times=times, dic_to_save=dic_to_save, file_path=file)

        # import bioviz
        # b = bioviz.Viz(loaded_model=msk_function.model)
        # b.load_movement(np.repeat(msk_function.kin_buffer[0], 5, axis=1))
        # b.load_experimental_markers(np.repeat(markers[:, :, np.newaxis], 5, axis=2))
        # b.exec()
        if compute_id:
            if not compute_ik:
                raise ValueError("Inverse kinematics must be computed to compute inverse dynamics")
            times, dic_to_save = _compute_id(msk_function, f_ext, external_loads, times, dic_to_save)

        if compute_so:
            if not compute_id:
                raise ValueError("Inverse dynamics must be computed to compute static optimization")
            times, dic_to_save = _compute_so(msk_function, emg, times, dic_to_save, scaling_factor,
                                             print_optimization_status=print_optimization_status,
                                             )

        if compute_jrf:
            if not compute_so:
                raise ValueError("Static optimization must be computed to compute joint reaction forces")
            times, dic_to_save = _compute_jrf(msk_function, times, dic_to_save, external_loads)

        times["tot"] = sum(times.values())
        dic_to_save["time"] = times
        return dic_to_save
    else:
        return None


def process_all_frames(markers, msk_function, external_loads, scaling_factor, emg, f_ext,
                       compute_id=True, compute_so=True, compute_jrf=True,stop_frame=None, file=None,
                       print_optimization_status=False):
    final_dic = {}
    stop_frame = markers.shape[2] if stop_frame is None else stop_frame
    for i in range(stop_frame):
        emg_tmp = emg if emg is None else emg[:, i]
        dic_to_save = process_next_frame(markers[..., i], msk_function, i, external_loads,
                                         scaling_factor, emg_tmp, f_ext=f_ext[:, 0, i], kalman_freq=120,
                                         emg_names=None, compute_id=compute_id,
                                         compute_so=compute_so, compute_jrf=compute_jrf, file=file,
                                         print_optimization_status=print_optimization_status)
        final_dic = dic_merger(final_dic, dic_to_save)
    return final_dic


