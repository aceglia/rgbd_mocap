from biosiglive import save, load, OfflineProcessing
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv
import biorbd
import bioviz
from scipy.interpolate import interp1d
from utils_old import load_all_data
from biosiglive import MskFunctions, InverseKinematicsMethods
from scipy.signal import find_peaks


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
            reordered_markers[:, model_marker_names.index(_convert_string(names[i])), :] = markers[:, count, :]
            count += 1
    return reordered_markers


def read_sensix_files():
    pass


def express_forces_in_global(crank_angle, f_ext):
    crank_angle = crank_angle
    Roty = np.array(
        [[np.cos(crank_angle), 0, np.sin(crank_angle)], [0, 1, 0], [-np.sin(crank_angle), 0, np.cos(crank_angle)]]
    )
    return Roty @ f_ext


if __name__ == "__main__":
    participants = ["P16"]  # , "P11", "P12", "P13"]#, "P14", "P15", "P16"]
    trials = [["gear_15"]] * len(participants)
    all_data, trials = load_all_data(participants, "/mnt/shared/Projet_hand_bike_markerless/process_data", trials)

    for part in all_data.keys():
        for f, file in enumerate(all_data[part].keys()):
            markers = all_data[part][file]["truncated_markers_vicon"]
            names_from_source = all_data[part][file]["vicon_markers_names"]
            # put nan at idx where the marker is not visible
            idx_ts = names_from_source.index("scapts")
            idx_ai = names_from_source.index("scapia")
            names_from_source[idx_ts] = "scapia"
            names_from_source[idx_ai] = "scapts"
            # markers = all_data[part][file]["markers_depth"][:, :-3, :]
            msk_func = MskFunctions(
                model=f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/gear_10_processed_3_model_scaled_vicon.bioMod",
                data_buffer_size=markers.shape[2],
            )
            markers_target = reorder_markers(markers[:, :-3, :], msk_func.model, names_from_source[:-3])
            markers = markers_target
            q, _, _ = msk_func.compute_inverse_kinematics(markers, method=InverseKinematicsMethods.BiorbdKalman)
            dic_data = all_data[part][file]["sensix_data_interpolated"]
            peaks, _ = find_peaks(dic_data["crank_angle"][0, :])
            peaks = [peak for peak in peaks if dic_data["crank_angle"][0, peak] > 6]
            for key in dic_data.keys():
                if isinstance(dic_data[key], np.ndarray):
                    dic_data[key] = dic_data[key][0, peaks[0] :]
            all_data_int = dic_data["crank_angle"][np.newaxis, :].copy()

            f_x = dic_data["raw_LFX"].copy()
            start_cycle = False
            end_cycle = False
            for i in range(all_data_int.shape[1]):
                if dic_data["crank_angle"][i] < 0.1:
                    start_cycle = True
                    end_cycle = False
                elif dic_data["crank_angle"][i] > 6:
                    end_cycle = True
                    start_cycle = False
                if start_cycle and dic_data["crank_angle"][i] > 0.1 and dic_data["crank_angle"][i] < 6:
                    dic_data["crank_angle"][i] = 0
            start_cycle = False
            end_cycle = False
            for i in range(all_data_int.shape[1]):
                if dic_data["left_pedal_angle"][i] < 0.1:
                    start_cycle = True
                    end_cycle = False
                elif dic_data["left_pedal_angle"][i] > 6:
                    end_cycle = True
                    start_cycle = False
                if start_cycle and dic_data["left_pedal_angle"][i] > 0.1 and dic_data["left_pedal_angle"][i] < 6:
                    dic_data["left_pedal_angle"][i] = 0
            start_cycle = False
            end_cycle = False
            for i in range(all_data_int.shape[1]):
                if dic_data["right_pedal_angle"][i] < 0.1:
                    start_cycle = True
                    end_cycle = False
                elif dic_data["right_pedal_angle"][i] > 6:
                    end_cycle = True
                    start_cycle = False
                if start_cycle and dic_data["right_pedal_angle"][i] > 0.1 and dic_data["right_pedal_angle"][i] < 6:
                    dic_data["right_pedal_angle"][i] = 0

            for i in range(all_data_int.shape[1]):
                dic_data["crank_angle"][i] = dic_data["crank_angle"][i]  # 3.14
                # dic_data["crank_angle"][i] = dic_data["crank_angle"][i] - 1.57
                crank_angle = -dic_data["crank_angle"][i]
                left_angle = -dic_data["left_pedal_angle"][i]
                right_angle = -dic_data["right_pedal_angle"][i]
                force_vector_l = [
                    dic_data["raw_LFX_crank"][i],
                    dic_data["raw_LFY_crank"][i],
                    dic_data["raw_LFZ_crank"][i],
                ]
                force_vector_r = [
                    dic_data["raw_RFX_crank"][i],
                    dic_data["raw_RFY_crank"][i],
                    dic_data["raw_RFZ_crank"][i],
                ]

                force_vector_l = express_forces_in_global(crank_angle, force_vector_l)
                force_vector_r = express_forces_in_global(crank_angle, force_vector_r)
                # force_vector_l = express_forces_in_global(left_angle, force_vector_l)
                # force_vector_r = express_forces_in_global(right_angle, force_vector_r)
                dic_data["LFX"][i] = force_vector_l[0]
                dic_data["LFY"][i] = force_vector_l[1]
                dic_data["LFZ"][i] = force_vector_l[2]
                dic_data["RFX"][i] = force_vector_r[0]
                dic_data["RFY"][i] = force_vector_r[1]
                dic_data["RFZ"][i] = force_vector_r[2]
            # B = RT @ B
            # A = RT2 @ A
            com_pos = []
            for i in range(q.shape[1]):
                com_pos.append(msk_func.model.CoMbySegment(q[:, i])[-1].to_array()[1] * 3)

            # plt.figure()
            # for i in range(peaks[1], peaks[2]):
            #     # if i % 10 != 0:
            #     #     continue
            #     crank_vector = np.array([10, 0, 0])[:, np.newaxis]
            #     pedal_vector = np.array([0, 0, 10])[:, np.newaxis]
            #     dic_data["crank_angle"][i] = dic_data["crank_angle"][i]
            #     pedal_vector = express_forces_in_global(-dic_data["right_pedal_angle"][i], pedal_vector)
            #     pedal_vector = express_forces_in_global(dic_data["crank_angle"][i], pedal_vector)
            #
            #     crank_vector = express_forces_in_global(dic_data["crank_angle"][i], crank_vector)
            #     force_vector = np.array([dic_data["raw_LFX_crank"][i], dic_data["raw_LFX_crank"][i], dic_data["raw_LFX_crank"][i]])[:, np.newaxis]
            #     # force_vector = express_forces_in_global(-dic_data["right_pedal_angle"][i], force_vector)
            #     force_vector = express_forces_in_global(dic_data["crank_angle"][i], force_vector)
            #
            #     plt.quiver(0, 0, crank_vector[0], crank_vector[2], color="r")
            #     plt.quiver(crank_vector[0], crank_vector[2], pedal_vector[0], pedal_vector[2], color="g")
            #     plt.quiver(crank_vector[0], crank_vector[2], force_vector[0], force_vector[2], color="b")
            #     # plt.scatter(crank_vector[0], crank_vector[2], c="r")
            #     # plt.plot(force_vector_zeros)
            #     # plt.show()
            #     # plt.plot(dic_data["LFZ"], label="raw_LFX")
            #     plt.xlim(-20, 20)
            #     plt.ylim(-20, 20)
            #     plt.draw()
            #     plt.pause(0.8)
            #     plt.cla()
            if part in ["P10", "P11", "P12", "P13"]:
                f_ext = np.array(
                    [
                        dic_data["LMY"],
                        dic_data["LMX"],
                        dic_data["LMZ"],
                        dic_data["LFY"],
                        -dic_data["LFX"],
                        -dic_data["LFZ"],
                    ]
                )
            else:
                f_ext = np.array(
                    [
                        dic_data["LMY"],
                        dic_data["LMX"],
                        dic_data["LMZ"],
                        dic_data["LFY"],
                        dic_data["LFX"],
                        dic_data["LFZ"],
                    ]
                )
            f_ext_mat = np.zeros((1, 6, f_ext.shape[1]))
            for i in range(f_ext.shape[1]):
                B = [0, 0, 0, 1]
                all_jcs = msk_func.model.allGlobalJCS(q[:, i])
                RT = all_jcs[-1].to_array()
                B = RT @ B
                vecteur_OB = B[:3]
                f_ext_mat[0, :3, i] = vecteur_OB
                # f_ext_mat[0, :3, i] = f_ext[:3, i] + np.cross(vecteur_OB, f_ext[3:6, i])
                f_ext_mat[0, 3:, i] = f_ext[3:, i]
                # from numpy import cross
                # f_ext[:3, 0] + cross(vecteur_BA, f_ext[3:6, 0])
            b = bioviz.Viz(loaded_model=msk_func.model)
            b.load_movement(q[:, :600])
            b.load_experimental_forces(f_ext_mat[:, :, :600], segments=["ground"], normalization_ratio=0.5)
            b.exec()
            # #os.remove("data/P3_gear_20_sensix.bio")
            # save(dic_data, "data/passive_global_ref.bio")
            # plt.figure()
            # plt.plot(dic_data["time"], dic_data["crank_angle"], label="RFZ")
            # plt.show()
