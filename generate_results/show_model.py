from biosiglive import load, MskFunctions, InverseKinematicsMethods
import bioviz
import biorbd
import numpy as np
from utils_old import load_all_data



def _convert_string(string):
    return string.lower().replace("_", "")


def reorder_markers(markers, model, names):
    model_marker_names = [_convert_string(model.markerNames()[i].to_string()) for i in range(model.nbMarkers())]
    assert len(model_marker_names) == len(names)
    assert len(model_marker_names) == markers.shape[1]
    count = 0
    reordered_markers = np.zeros((markers.shape[0], len(model_marker_names), markers.shape[2]))
    for i in range(len(names)):
        if names[i] == "elb":
            names[i] = "elbow"
        if _convert_string(names[i]) in model_marker_names:
            reordered_markers[:, model_marker_names.index(_convert_string(names[i])),
            :] = markers[:, count, :]
            count += 1
    return reordered_markers

def get_force_to_show(sensix_data, q, model_bio):
    # f_ext = np.array([sensix_data["RMY"],
    #                   -sensix_data["RMX"],
    #                   sensix_data["RMZ"],
    #                   sensix_data["RFY"],
    #                   -sensix_data["RFX"],
    #                   sensix_data["RFZ"]])
    f_ext = np.array([-sensix_data["RMY"],
                      sensix_data["RMX"],
                      sensix_data["RMZ"],
                      -sensix_data["RFY"],
                      sensix_data["RFX"],
                      sensix_data["RFZ"]])
    f_ext = f_ext[:, 0, :]
    f_ext_mat = np.zeros((1, 6, f_ext.shape[1]))
    for i in range(f_ext.shape[1]):
        B = [0, 0, 0, 1]
        all_jcs = model_bio.allGlobalJCS(q[:, i])
        RT = all_jcs[-1].to_array()
        B = RT @ B
        vecteur_OB = B[:3]
        f_ext_mat[0, :3, i] = vecteur_OB
        # f_ext_mat[0, :3, i] = f_ext[:3, i] + np.cross(vecteur_OB, f_ext[3:6, i])
        f_ext_mat[0, 3:, i] = f_ext[3:, i]

        # # f_ext[0, 3:, i] = (RT2 @ (np.array([dic_data["LFX"][i],  dic_data["LFY"][i], dic_data["LFZ"][i], 1])))[:3]
        # f_ext[0, 3:, i] = -((np.array([sensix_data["LFY"][0, i], -sensix_data["LFX"][0, i], sensix_data["LFZ"][0, i], 1])))[:3]
        #
        # f_ext[0, :3, i] = model_bio.CoMbySegment(q[:, i])[-1].to_array()
        # f_ext[0, :3, i] = B[:3]
        # f_ext[:3, 0] + cross(vecteur_BA, f_ext[3:6, 0])
    return f_ext_mat


if __name__ == '__main__':
    participants = ["P9"]
    trials = [["gear_10"]] * len(participants)
    all_data, trials = load_all_data(participants,
                                     "/mnt/shared/Projet_hand_bike_markerless/process_data", trials
                                     )
    key = ["markers"]
    n_key = all_data[participants[0]][list(all_data[participants[0]].keys())[0]]["markers_depth"].shape[1]
    means_file = np.ndarray((len(participants) * n_key))
    diffs_file = np.ndarray((len(participants) * n_key))
    all_colors = []
    for p, part in enumerate(all_data.keys()):
        for f, file in enumerate(all_data[part].keys()):
            markers = all_data[part][file]["truncated_markers_vicon"]
            names_from_source = all_data[part][file]["vicon_markers_names"]
            # put nan at idx where the marker is not visible
            idx_ts = names_from_source.index("scapts")
            idx_ai = names_from_source.index("scapia")
            names_from_source[idx_ts] = "scapia"
            names_from_source[idx_ai] = "scapts"
            # markers = all_data[part][file]["markers_depth"][:, :-3, :]
            msk_func = MskFunctions(model=f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/gear_10_processed_3_model_scaled_vicon.bioMod",
                                    data_buffer_size=markers.shape[2])
            markers_target = reorder_markers(markers[:, :-3, :],
                                             msk_func.model,
                                             names_from_source[:-3])
            markers = markers_target
            q, _, _ = msk_func.compute_inverse_kinematics(markers, method=InverseKinematicsMethods.BiorbdKalman)
            f_ext = get_force_to_show(all_data[part][file]["sensix_data_interpolated"], q, msk_func.model)

            b = bioviz.Viz(loaded_model=msk_func.model)
            b.load_movement(q[:, :500])
            b.load_experimental_forces(f_ext[:, :, :500], segments=["ground"], normalization_ratio=0.5)
            b.load_experimental_markers(markers[:, :, :500])
            b.exec()