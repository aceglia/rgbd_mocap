import biorbd
import numpy as np
import matplotlib.pyplot as plt
import bioviz
from collections import Counter


def _compute_muscle_forces(model, q, qdot, act, segment_names, segment_idx):
    # all_bodies_action = {}
    # for name in segment_names[1:]:
    #     all_bodies_action[name] = dict(muscle_idx=[], direction=[], muscle_force_action=[])
    # muscles_states = model.stateSet()
    # count = 0
    # for g, group in enumerate(model.muscleGroups()):
    #     for m in range(group.nbMuscles()):
    #         muscle = group.muscle(m)
    #         muscles_states[m].setActivation(act[m])
    #         muscle.length(model, q)
    #         # call len to update muscles
    #         points_in_global = muscle.musclesPointsInGlobal()
    #         list_via_point_parent = [group.origin().to_string()]
    #         for j in range(muscle.pathModifier().nbObjects()):
    #             list_via_point_parent.append(biorbd.ViaPoint(muscle.pathModifier().object(j)).parent().to_string())
    #         list_via_point_parent.append(group.insertion().to_string())
    #         # count how many body are concerned
    #         nb_bodies = len(Counter(list_via_point_parent).keys())
    #         nb_joints = nb_bodies - 1
    #         force_direction = []
    #         idx = []
    #         for key in Counter(list_via_point_parent).keys():
    #             idx.append(sum(np.argwhere(np.array(list_via_point_parent) == key).tolist(), []))
    #         for i in range(nb_joints):
    #             force_direction.append(points_in_global[idx[i+1][0]].to_array() - points_in_global[idx[i][-1]].to_array())
    #             force_direction[-1] = force_direction[-1] / np.linalg.norm(force_direction[-1])
    #         for k, key in enumerate(list(Counter(list_via_point_parent).keys())[1:]):
    #             all_bodies_action[key]["muscle_idx"].append(count)
    #             all_bodies_action[key]["direction"].append(force_direction[k])
    #         count += 1
    #
    # muscle_forces = model.muscleForces(muscles_states, q, qdot).to_array()
    # final_forces = []
    # for i in range(1, len(segment_names)):
    #     dic = all_bodies_action[segment_names[i]]
    #     dic["muscle_force_action"] = [muscle_forces[dic["muscle_idx"][j]] * dic["direction"][j] for j in range(len(dic["direction"]))]
    #     dic["muscle_force_action"] = np.sum(np.array(dic["muscle_force_action"]), axis=0)
    #     final_forces.append(dic["muscle_force_action"])
    muscles_states = model.stateSet()
    for m in range(model.nbMuscles()):
        muscles_states[m].setActivation(act[m])
    muscle_forces = model.muscleForces(muscles_states, q, qdot).to_array()
    muscle_moment_arm = model.muscularJointTorque(muscle_forces, q, qdot).to_array()
    idx_segment = []
    for k in range(model.nbSegment()):
        if model.segment(k).name().to_string() in segment_names[1:]:
            idx_segment.append(k)
    translational_in_local = []
    rotationnal_in_local = []
    for i in range(len(idx_segment)):
        translational_in_local.append(muscle_moment_arm[segment_idx[i][:3]])
        # translational_in_global.append(np.dot(model.globalJCS(idx_segment[i]).to_array(),
        #                                       np.array(local_forces.tolist() + [1])))
        rotationnal_in_local.append(muscle_moment_arm[segment_idx[i][3:]])
    return rotationnal_in_local, translational_in_local


def _compute_ligament_forces(model, q, qdot, segment_names, segment_idx):
    ligament_moment_arm = model.ligamentJointTorque(q, qdot)
    # idx_segment = []
    # for k in range(model.nbSegment()):
    #     if model.segment(k).name().to_string() in segment_names[1:]:
    #         idx_segment.append(k)
    translational_in_local = []
    rotationnal_in_local = []
    for i in range(len(segment_names)):
        translational_in_local.append(ligament_moment_arm[segment_idx[i][:3]])
        # translational_in_global.append(np.dot(model.globalJCS(idx_segment[i]).to_array(),
        #                                       np.array(local_forces.tolist() + [1])))
        rotationnal_in_local.append(ligament_moment_arm[segment_idx[i][3:]])
    return rotationnal_in_local, translational_in_local


def _create_simplified_model(model, segments_names, q, qdot, qddot, write_bioMod=False):
    model_empty = biorbd.Model()
    q_tot = None
    qdot_tot = None
    qddot_tot = None
    count = 0
    ordered_dof_prox_to_dist = []
    ordered_dof_prox_to_dist_idx = []
    nb_dof = 0
    for i in range(model.nbSegment()):
        if i == 0:
            ordered_dof_prox_to_dist = [model.segment(i).name().to_string()]
        if model.segment(i).name().to_string() in segments_names:
            rot = trans = "xyz"
            if q_tot is None:
                q_tot = np.zeros([6, q.shape[1]])
                qdot_tot = np.zeros([6, q.shape[1]])
                qddot_tot = np.zeros([6, q.shape[1]])
            else:
                q_tot = np.append(q_tot, np.zeros([6, q.shape[1]]), axis=0)
                qdot_tot = np.append(qdot_tot, np.zeros([6, q.shape[1]]), axis=0)
                qddot_tot = np.append(qddot_tot, np.zeros([6, q.shape[1]]), axis=0)
            ordered_dof_prox_to_dist.append(model.segment(i).name().to_string())
            ordered_dof_prox_to_dist_idx.append([nb_dof + k for k in range(6)])
            nb_dof += 6

        else:
            rot = model.segment(i).seqR().to_string()
            trans = model.segment(i).seqT().to_string()
            nb_dof += len(rot + trans)
            for j in range(len(rot + trans)):
                if q_tot is None:
                    q_tot = q[count, :][np.newaxis, :]
                    qdot_tot = qdot[count, :][np.newaxis, :]
                    qddot_tot = qddot[count, :][np.newaxis, :]
                else:
                    q_tot = np.append(q_tot, q[count, :][np.newaxis, :], axis=0)
                    qdot_tot = np.append(qdot_tot, qdot[count, :][np.newaxis, :], axis=0)
                    qddot_tot = np.append(qddot_tot, qddot[count, :][np.newaxis, :], axis=0)
                count += 1

        seg = model.segment(i)
        (name, parent_str, rot, trans, QRanges, QDotRanges, QDDotRanges, characteristics, RT) = (
            seg.name().to_string(),
            seg.parent().to_string(),
            rot,
            trans,
            [biorbd.Range(-3, 3)] * (len(rot) + len(trans)),
            [biorbd.Range(-3 * 10, 3 * 10)] * (len(rot) + len(trans)),
            [biorbd.Range(-3 * 100, 3 * 100)] * (len(rot) + len(trans)),
            seg.characteristics(),
            seg.localJCS(),
        )
        model_empty.AddSegment(name, parent_str, trans, rot, QRanges, QDotRanges, QDDotRanges, characteristics, RT)
    for i, group in enumerate(model.muscleGroups()):
        name, insertion, origin = group.name().to_string(), group.insertion().to_string(), group.origin().to_string()
        model_empty.addMuscleGroup(name, origin, insertion)
        for m in range(group.nbMuscles()):
            model_empty.muscleGroups()[-1].addMuscle(model.muscleGroup(i).muscle(m))

    if write_bioMod:
        biorbd.Writer.writeModel(model_empty, "model_simplified.bioMod")
    return model_empty, q_tot, qdot_tot, qddot_tot, ordered_dof_prox_to_dist, ordered_dof_prox_to_dist_idx


def compute_jrf(model, q, qdot, qddot, act):
    # update model kin
    non_virtual_segments = [seg.name().to_string() for seg in model.segments() if seg.characteristics().mass() > 1e-7]
    if isinstance(q, list):
        q = np.array(q)
    if isinstance(qdot, list):
        qdot = np.array(qdot)
    if isinstance(qddot, list):
        qddot = np.array(qddot)
    if len(q.shape) != 2:
        q = q[:, np.newaxis]
    if len(qdot.shape) != 2:
        qdot = qdot[:, np.newaxis]
    if len(qddot.shape) != 2:
        qddot = qddot[:, np.newaxis]
    model_simplified, q, qdot, qddot, ordered_seg, ordered_idx = _create_simplified_model(
        model, non_virtual_segments, q, qdot, qddot
    )
    all_trans = np.ndarray((len(ordered_seg) - 1, 3, q.shape[1]))
    all_rot = np.ndarray((len(ordered_seg) - 1, 3, q.shape[1]))
    # b = bioviz.Viz(loaded_model=model_simplified)
    # b.load_movement(q)
    # b.exec()
    for i in range(q.shape[1]):
        Tau = model_simplified.InverseDynamics(q[:, i], qdot[:, i], qddot[:, i]).to_array()
        # model_simplified.UpdateKinematicsCustom(q[:, i], qdot[:, i], qddot[:, i])
        rot_muscle_actions, trans_muscle_actions = _compute_muscle_forces(
            model_simplified, q[:, i], qdot[:, i], act[:, i], segment_names=ordered_seg, segment_idx=ordered_idx
        )
        translational_in_local = []
        translational_in_global = []
        rotationnal_in_local = []
        for j in range(len(ordered_seg[1:])):
            translational_in_local.append(Tau[ordered_idx[j][:3]])
            translational_in_global.append(
                np.dot(model.globalJCS(idx_segment[i]).to_array(), np.array(local_forces.tolist() + [1]))
            )
            rotationnal_in_local.append(Tau[ordered_idx[j][3:]])
        for k in range(len(ordered_seg[1:])):
            all_trans[k, :, i] = np.sum(np.array([trans_muscle_actions[k], translational_in_local[k]]), axis=0)
            all_rot[k, :, i] = np.sum(np.array([rot_muscle_actions[k], rotationnal_in_local[k]]), axis=0)
        # ligament_forces = _compute_muscle_forces(model, q_init[:, i], qdot_init[:, i], act[:, i])
        # add_residual_torque
        # add actuators
    return all_rot, all_trans


if __name__ == "__main__":
    from biosiglive import MskFunctions

    model = "pendulum.bioMod"
    model = biorbd.Model(model)
    msx_funct = MskFunctions(model, 1)
    # b = bioviz.Viz(loaded_model=model)
    # b.exec()
    q = np.zeros((model.nbQ(), 100))
    q[1, :] = np.linspace(0, 1.57, 100)
    qdot = np.zeros_like(q)
    qdot[:, 1:] = np.diff(q)
    qddot = np.zeros_like(q)
    qddot[:, 1:] = np.diff(qdot)
    act = np.zeros((model.nbMuscles(), 100)) + 0.1
    act[0:1, :] = np.zeros((1, 100)) + 0.1
    act[1:, :] = np.zeros((1, 100)) + 0.1
    all_rot, all_trans = np.zeros((1, 3, q.shape[1])), np.zeros((1, 3, q.shape[1]))
    for i in range(q.shape[1]):
        all_rot_tmp, all_trans_tmp = msx_funct.compute_joint_reaction_load(
            q[:, i],
            qdot[:, i],
            qddot[:, i],
            act[:, i],
            express_in_coordinate="Seg1",
            apply_on_segment="Seg0",
            application_point=[[0, 0, 0]],
        )
        all_rot[0, :, i : i + 1], all_trans[0, :, i : i + 1] = all_rot_tmp[0, :, :], all_trans_tmp[0, :, :]
    # all_rot, all_trans = compute_jrf(model, q, qdot, qddot, act)
    all_trans_plot = np.zeros((all_trans.shape[0], 6, all_trans.shape[2]))
    all_trans_plot[:, 3:, :] = all_trans
    # all_trans_plot[:, 3:, :] = np.ones((all_trans.shape[0], 3, all_trans.shape[2]))

    b = bioviz.Viz(loaded_model=model)
    b.load_movement(q)
    b.load_experimental_forces(all_trans_plot[:, :, :], ["Seg1"])
    b.exec()
