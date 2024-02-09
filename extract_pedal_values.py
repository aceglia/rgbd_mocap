from biosiglive import save, load, OfflineProcessing
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv
import biorbd
import bioviz
from scipy.interpolate import interp1d


def read_sensix_files():
    pass

def express_forces_in_global(crank_angle, f_ext):
    crank_angle = crank_angle
    Roty = np.array([[np.cos(crank_angle), 0, np.sin(crank_angle)],
                        [0, 1, 0],
                        [-np.sin(crank_angle), 0, np.cos(crank_angle)]])
    return Roty @ f_ext


if __name__ == '__main__':
    all_data = []
    data_path = "data/Results-mvt_actif_2_001.lvm"
    with open(data_path, 'r') as file:
        csvreader = csv.reader(file, delimiter='\n')
        for row in csvreader:
            all_data.append(np.array(row[0].split("\t")))
    all_data = np.array(all_data, dtype=float).T


    q = load("data/P3_gear_20_kalman.bio")["q"][6:, 108:3737]

    # passif
    #all_data = all_data[:, 10226:int(q.shape[1] * (250/120)) + 10226]
    # actif
    all_data = all_data[:, 10032:int(q.shape[1] * (250/120)) + 10032]

    all_data_int = all_data
    all_data_int = np.zeros((all_data.shape[0], q.shape[1]))
    x_new = np.linspace(0, 1, q.shape[1])
    for d, data in enumerate(all_data):
        # filter before interpolation
        # data = OfflineProcessing().butter_lowpass_filter(data, 60, 250, 6)
        f = interp1d(np.linspace(0, 1, all_data.shape[1]), data)
        all_data_int[d, :] = f(x_new)
    dic_data = {"time": all_data_int[0, :],
                "RFX": all_data_int[1, :],
                "RFY": all_data_int[2, :],
                "RFZ": all_data_int[3, :],
                "RMX": all_data_int[4, :],
                "RMY": all_data_int[5, :],
                "RMZ": all_data_int[6, :],
                "LFX": all_data_int[9, :],
                "LFY": all_data_int[10, :],
                "LFZ": all_data_int[11, :],
                "LMX": all_data_int[12, :],
                "LMY": all_data_int[13, :],
                "LMZ": all_data_int[14, :],
                "crank_angle": all_data_int[19, :],
                "right_pedal_angle": all_data_int[17, :],
                "left_pedal_angle": all_data_int[18, :],
                }
    # dic_data = {"time": all_data_int[0, :],
    #             "RFX": all_data_int[21, :],
    #             "RFY": all_data_int[22, :],
    #             "RFZ": all_data_int[23, :],
    #             "RMX": all_data_int[24, :],
    #             "RMY": all_data_int[25, :],
    #             "RMZ": all_data_int[26, :],
    #             "LFX": all_data_int[10, :],
    #             "LFY": all_data_int[11, :],
    #             "LFZ": all_data_int[12, :],
    #             "LMX": all_data_int[31, :],
    #             "LMY": all_data_int[32, :],
    #             "LMZ": all_data_int[33, :],
    #             "crank_angle": all_data_int[19, :]}
    model_bio = biorbd.Model("models/wu_bras_gauche_seth.bioMod")
    b = bioviz.Viz("models/wu_bras_gauche_seth.bioMod")
    f_x = dic_data["LFX"].copy()
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
        if dic_data["right_pedal_angle"][i] < 0.08:
            start_cycle = True
            end_cycle = False
        elif dic_data["right_pedal_angle"][i] > 6:
            end_cycle = True
            start_cycle = False
        if start_cycle and dic_data["right_pedal_angle"][i] > 0.1 and dic_data["right_pedal_angle"][i] < 6:
            dic_data["right_pedal_angle"][i] = 0

    for i in range(all_data_int.shape[1]):
        crank_angle = dic_data["crank_angle"][i]
        left_angle = dic_data["left_pedal_angle"][i]
        right_angle = dic_data["right_pedal_angle"][i]
        force_vector_l = [dic_data["LFX"][i], dic_data["LFY"][i], dic_data["LFZ"][i]]
        force_vector_r = [dic_data["RFX"][i], dic_data["RFY"][i], dic_data["RFZ"][i]]
        force_vector_l = express_forces_in_global(-left_angle, force_vector_l)
        force_vector_r = express_forces_in_global(-right_angle, force_vector_r)
        force_vector_l = express_forces_in_global(crank_angle, force_vector_l)
        force_vector_r = express_forces_in_global(crank_angle, force_vector_r)
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
        com_pos.append(model_bio.CoMbySegment(q[:, i])[-1].to_array()[1] * 3)

    plt.figure()

    for i in range(q.shape[1]):
        crank_vector = np.array([10, 0, 0])
        pedal_vector = np.array([0, 0, 10])
        pedal_vector = express_forces_in_global(-dic_data["left_pedal_angle"][i], pedal_vector[:, np.newaxis])
        pedal_vector = express_forces_in_global(dic_data["crank_angle"][i], pedal_vector)
        crank_vector = express_forces_in_global(dic_data["crank_angle"][i], crank_vector[:, np.newaxis])
        force_vector = np.array([dic_data["LFX"][i], dic_data["LFY"][i], dic_data["LFZ"][i]])
        force_vector = express_forces_in_global(-dic_data["left_pedal_angle"][i], force_vector[:, np.newaxis])
        force_vector = express_forces_in_global(dic_data["crank_angle"][i], force_vector)
        # plt.quiver(0, 0, crank_vector[0], crank_vector[2])
        # plt.quiver(crank_vector[0], crank_vector[2], pedal_vector[0], pedal_vector[2])
        # plt.quiver(crank_vector[0], crank_vector[2], force_vector[0], force_vector[2])
        # #plt.scatter(crank_vector[0], crank_vector[2], c="r")
        # #plt.plot(force_vector_zeros)
        # #plt.show()
        # # plt.plot(dic_data["LFZ"], label="LFX")
        # plt.xlim(-20, 20)
        # plt.ylim(-20, 20)
        # plt.draw()
        # plt.pause(0.1)
        # plt.cla()
    # vecteur_BA = A[:3] - B[:3]
    plt.figure()
    plt.plot(dic_data["LFX"], label="LFX")
    plt.plot(dic_data["crank_angle"], label="crank")
    plt.plot(f_x, "--")
    plt.legend()
    plt.figure()
    plt.plot(dic_data["LFZ"], label="LFX")
    # plt.show()
    plt.figure()
    plt.plot(dic_data["LFY"], label="LFY")
    plt.figure()
    plt.plot(dic_data["crank_angle"], label="crank")
    plt.plot(dic_data["left_pedal_angle"], label="left")
    plt.plot(dic_data["right_pedal_angle"], label="LFY")
    plt.legend()
    plt.figure()
    plt.plot(dic_data["LFX"], label="crank")
    plt.plot(q[-2, :], label="LFY")
    #plt.show()


    # force_locale[:3] = f_ext[:3, 0] + cross(vecteur_BA, f_ext[3:6, 0])
    f_ext = np.zeros((1, 6, int(dic_data["RFX"].shape[0])))
    for i in range(q.shape[1]):
        A = [0, 0, 0, 1]
        B = [0, 0, 0, 1]
        all_jcs = model_bio.allGlobalJCS(q[:, i])
        RT = all_jcs[-1].to_array()
        RT2 = all_jcs[-9].to_array()
        A = RT @ A
        B = RT2 @ B

        # f_ext[0, 3:, i] = (RT2 @ (np.array([dic_data["LFX"][i],  dic_data["LFY"][i], dic_data["LFZ"][i], 1])))[:3]
        f_ext[0, 3:, i] = ((np.array([dic_data["LFY"][i], -dic_data["LFX"][i], dic_data["LFZ"][i], 1])))[:3]

        # f_ext[0, :3, i] = model_bio.CoMbySegment(q[:, i])[-1].to_array()
        # f_ext[0, :3, i] = B[:3]
        # f_ext[:3, 0] + cross(vecteur_BA, f_ext[3:6, 0])

    b.load_movement(q[:, :600])
    b.load_experimental_forces(f_ext[:, :, :600], segments=["ground"], normalization_ratio=0.5)
    b.exec()
    #os.remove("data/P3_gear_20_sensix.bio")
    # save(dic_data, "data/passive_global_ref.bio")
    # plt.figure()
    # plt.plot(dic_data["time"], dic_data["crank_angle"], label="RFZ")
    # plt.show()
