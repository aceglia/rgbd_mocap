import numpy as np
import time
from biosiglive import InverseKinematicsMethods


def _map_activation(emg_proc, muscle_track_idx, model, emg_names, emg_init=None, mvc_normalized=True):
    emg_names = ["PectoralisMajor",
                 "BIC",
                 "TRI",
                 "LatissimusDorsi",
                 'TrapeziusClav',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                 'DeltoideusScapula_P']
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
    emg_names = ["PectoralisMajor",
                 "BIC",
                 "TRI",
                 "LatissimusDorsi",
                 'TrapeziusClav',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                 'DeltoideusScapula_P']

    muscle_track_idx = []
    for i in range(len(emg_names)):
        for j in range(len(muscle_list)):
            if emg_names[i] in muscle_list[j]:
                muscle_track_idx.append(j)
    return muscle_track_idx


def perform_biomechanical_pipeline(markers, msk_function, frame_idx, external_loads=None,
                                   scaling_factor=None, emg=None, kalman_freq=60, emg_names=None, f_ext=None,):
    initial_guess = None
    # if frame_idx == 5:
    #     q, q_dot = msk_function.compute_inverse_kinematics(markers[:, :, np.newaxis],
    #                                                        InverseKinematicsMethods.BiorbdLeastSquare, )
    #     msk_function.clean_all_buffers()
    #     initial_guess = [q[:, 0], np.zeros_like(q)[:, 0], np.zeros_like(q)[:, 0]]
    if markers[0, 0].mean() != 0:
        if frame_idx == 0:
            q, q_dot = msk_function.compute_inverse_kinematics(markers[:, :, np.newaxis],
                                                              InverseKinematicsMethods.BiorbdLeastSquare,)
            msk_function.clean_all_buffers()
            initial_guess = [q[:, 0], np.zeros_like(q)[:, 0], np.zeros_like(q)[:, 0]]
        tic_init = time.time()
        initial_guess = [msk_function.kin_buffer[0][:, -1],
                         np.zeros_like(msk_function.kin_buffer[0])[:, 0],
                         np.zeros_like(msk_function.kin_buffer[0])[:, 0]] if not initial_guess else initial_guess
        msk_function.compute_inverse_kinematics(markers[:, :, np.newaxis],
                                              method=InverseKinematicsMethods.BiorbdKalman,
                                              kalman_freq=kalman_freq,
                                              initial_guess=initial_guess)
        q = msk_function.kin_buffer[0].copy()
        # import bioviz
        # b = bioviz.Viz(loaded_model=msk_function.model)
        # b.load_movement(np.repeat(msk_function.kin_buffer[0], 5, axis=1))
        # b.load_experimental_markers(np.repeat(markers[:, :, np.newaxis], 5, axis=2))
        # b.exec()
        time_ik = time.time() - tic_init
        tic = time.time()
        B = [0, 0, 0, 1]
        f_ext_mat = np.zeros((6, 1))
        all_jcs = msk_function.model.allGlobalJCS(msk_function.kin_buffer[0][:, -1])
        RT = all_jcs[-1].to_array()
        B = RT @ B
        vecteur_OB = B[:3]
        f_ext_mat[:3, 0] = f_ext[:3, 0] + np.cross(vecteur_OB, f_ext[3:6, 0])
        external_loads.update_external_load_value(f_ext_mat, name="hand_pedal")
        tau = msk_function.compute_inverse_dynamics(positions_from_inverse_kinematics=True,
                                                   state_idx_to_process=[0],
                                                   windows_length=5,
                                                   external_load=external_loads)
        time_id = time.time() - tic
        # msk_function.tau_buffer[:6, :] = np.zeros((6, msk_function.tau_buffer.shape[1]))
        tic = time.time()
        q_df, q_dot_df, q_ddot_df = (msk_function.id_state_buffer[0][:, -1:],
                                     msk_function.id_state_buffer[1][:, -1:]
                                     , msk_function.id_state_buffer[2][:, -1:])
        # emg = _map_activation(emg[:, np.newaxis], get_tracking_idx(
        #         msk_function.model), msk_function.model, emg_names, emg_init=emg[:, np.newaxis], mvc_normalized=True)[0]
        # mus_act, res_tau = msk_function.compute_static_optimization(
        #     # q=q_df[:, -1:], q_dot=q_dot_df[:, -1:], tau=tau[:, -1:],
        #     scaling_factor=scaling_factor,
        #     data_from_inverse_dynamics=True,
        #     compile_only_first_call=True,
        #     emg=emg,
        #     muscle_track_idx=get_tracking_idx(
        #         msk_function.model),
        #     weight={"tau": 1000000000, "act": 1,
        #             "tracking_emg": 1000000000000,
        #             "pas_tau": 1000}
        # )
        # time_so = time.time() - tic
        # tic = time.time()
        # jrf = msk_function.compute_joint_reaction_load(q_df,
        #                                               q_dot_df,
        #                                               q_ddot_df,
        #                                               mus_act,
        #                                               express_in_coordinate="scap_glen",
        #                                               apply_on_segment="scapula_left",
        #                                               application_point=[[0, 0, 0]],
        #                                               from_distal=True,
        #                                               external_loads=external_loads,
        #                                               kinetics_from_inverse_dynamics=False,
        #                                               act_from_static_optimisation=False,
        #                                               )
        time_jrf = time.time() - tic
        time_tot = time.time() - tic_init
        times = {"ik": time_ik, "id": time_id,
                 # "so": time_so,
                 "jrf": time_jrf,
                 "tot": time_tot}
        return {"q": q_df[:, -1:], "q_dot": q_dot_df[:, -1:], "q_ddot": q_ddot_df[:, -1:],
                "q_raw": q[:, -1:],
                "tau": tau[:, -1:],
                # "mus_act": mus_act[:, -1:],
                "emg_proc": emg[:, -1:],
                # "res_tau": res_tau[:, -1:],
                # "jrf": jrf[:, :, -1:],
                "time": times}
    else:
        return None