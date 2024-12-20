from processing_data.file_io import get_dlc_data, get_all_file
import numpy as np
import json
from biosiglive import save, load

def get_normal_vector(data, name):
    marker = data[1][:, data[0].index(name), :]
    M1 = marker[:, 0]
    M2 = marker[:, 20]
    M3 = marker[:, 40]
    first_axis_vector = M3 - M1
    second_axis_vector = M2 - M1
    third_axis_vector = np.cross(first_axis_vector, second_axis_vector, axis=0)[:, None]
    third_axis_vector = np.repeat(third_axis_vector, data[1].shape[2], axis=1)
    rt = np.zeros((4, 4, data[1].shape[2]))
    vertical_vector = data[1][:, data[0].index("ster"), :] - data[1][:, data[0].index("xiph"), :]
    antero_vector = np.cross(vertical_vector, third_axis_vector, axis=0)
    medial_vector = -np.cross(vertical_vector, antero_vector, axis=0)
    rt[:3, 0, :] = medial_vector / np.linalg.norm(medial_vector, axis=0)
    rt[:3, 1, :] = antero_vector / np.linalg.norm(antero_vector, axis=0)
    rt[:3, 2, :] = vertical_vector / np.linalg.norm(vertical_vector, axis=0)
    rt[:3, 3, :] = data[1][:, data[0].index("xiph"), :]
    rt[3, 3, :] = 1
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1, 1, 1])
    # ax.quiver(M1[0], M1[1], M1[2], third_axis_vector[0], third_axis_vector[1], third_axis_vector[2], color='b')
    # ax.quiver(data[1][:, data[0].index("xiph"), 0][0], data[1][:, data[0].index("xiph"), 0][1],
    #           data[1][:, data[0].index("xiph"), 0][2], third_axis_vector[0], third_axis_vector[1], third_axis_vector[2] , color='y',
    #           length=0.1, normalize=True)
    # ax.quiver(data[1][:, data[0].index("xiph"), 0][0], data[1][:, data[0].index("xiph"), 0][1],
    #           data[1][:, data[0].index("xiph"), 0][2], medial_vector[0], medial_vector[1], medial_vector[2], color='g',
    #           length=0.1, normalize=True)
    # ax.quiver(data[1][:, data[0].index("xiph"), 0][0], data[1][:, data[0].index("xiph"), 0][1],
    #           data[1][:, data[0].index("xiph"), 0][2], vertical_vector[0], vertical_vector[1], vertical_vector[2], color='m',
    #           length=0.1, normalize=True)
    # ax.quiver(data[1][:, data[0].index("xiph"), 0][0], data[1][:, data[0].index("xiph"), 0][1],
    #           data[1][:, data[0].index("xiph"), 0][2], antero_vector[0]*10, antero_vector[1]*10, antero_vector[2]*10, color='c',
    #           length=0.1, normalize=True)
    # ax.scatter(M1[0], M1[1], M1[2], color='r')
    # ax.scatter(M2[0], M2[1], M2[2], color='g')
    # ax.scatter(M3[0], M3[1], M3[2], color='b')
    # ax.scatter(marker_tec_1_glob[0], marker_tec_1_glob[1], marker_tec_1_glob[2], color='r')
    # ax.scatter(marker_tec_2_glob[0], marker_tec_2_glob[1], marker_tec_2_glob[2], color='g')
    # for mar in range(data[1].shape[1]):
    #     if mar == data[0].index("xiph"):
    #         color = "r"
    #     else:
    #         color = "k"
    #     ax.scatter(data[1][0, mar, 0], data[1][1, mar, 0], data[1][2, mar, 0], color=color)
    # plt.show()
    return rt

def get_measure(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["ster_C7"] * 0.001


def add_virtual_markers(data, rt, remove_marker=None, measurement_file=None):
    dist = np.linalg.norm(data[1][:, data[0].index("xiph"), :] - data[1][:, data[0].index("ster"), :], axis=0)
    marker_tec_2 = np.repeat(np.array([0, -0.2, 0, 1])[:, None],data[1].shape[2],  axis = 1)
    l = get_measure(measurement_file) if measurement_file is not None else 0.145
    marker_tec_1 = np.zeros((4, data[1].shape[2]))
    marker_tec_1[2, :] = dist
    marker_tec_1[1, :] = -l
    marker_tec_1[3, :] = 1

    new_markers = np.ndarray((3, 2, data[1].shape[2]))
    for i in range(data[1].shape[2]):
        new_markers[:, 0, i] = np.dot(rt[:, :, i], marker_tec_1[:, i])[:3]
        new_markers[:, 1, i] = np.dot(rt[:, :, i], marker_tec_2[:, i])[:3]

    names = data[0]
    if remove_marker is not None:
        data[1] = np.delete(data[1], data[0].index(remove_marker), axis=1)
        names.pop(data[0].index(remove_marker))
    data_out = np.concatenate((data[1][:, :data[0].index("xiph") + 1, :],
                               new_markers, data[1][:, data[0].index("xiph") + 1:, :]), axis=1)
    names = names[:names.index("xiph") + 1] + ["marker_tec_1", "marker_tec_2"] + names[names.index("xiph") + 1:]


    data_out = (names, data_out)
    # plot(data_out)
    return data_out

def plot(data): #, vect_0, vect_1, vect_2):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    # ax.quiver(data[1][:, data[0].index("xiph"), 0][0], data[1][:, data[0].index("xiph"), 0][1],
    #           data[1][:, data[0].index("xiph"), 0][2], vect_0[0], vect_0[1], vect_0[2], color='g',
    #           length=0.1, normalize=True)
    # ax.quiver(data[1][:, data[0].index("xiph"), 0][0], data[1][:, data[0].index("xiph"), 0][1],
    #           data[1][:, data[0].index("xiph"), 0][2], vect_2[0], vect_2[1], vect_2[2], color='m',
    #           length=0.1, normalize=True)
    # ax.quiver(data[1][:, data[0].index("xiph"), 0][0], data[1][:, data[0].index("xiph"), 0][1],
    #           data[1][:, data[0].index("xiph"), 0][2], vect_1[0]*10, vect_1[1]*10, vect_1[2]*10, color='c',
    #           length=0.1, normalize=True)
    for mar in range(data[1].shape[1]):
        if mar == data[0].index("marker_tec_1"):
            color = "r"
        elif mar == data[0].index("marker_tec_2"):
            color = "g"
        else:
            color = "k"
        ax.scatter(data[1][0, mar, 0], data[1][1, mar, 0], data[1][2, mar, 0], color=color)
    plt.show()



if __name__ == '__main__':
    ratio = 1
    root_dir = "/mnt/shared/Projet_hand_bike_markerless/RGBD"
    participants = [f"P{i}" for i in range(9, 10)]
    files, participants = get_all_file(participants,root_dir,  is_dir=True, to_include="gear")
    for directory, participant in zip(files, participants):
        # if "gear_20" not in directory : continue
        #dlc_data_path = f"{directory}/marker_pos_multi_proc_3_crops_normal_500_down_b1_ribs_and_cluster_{ratio}_with_model_pp_full.bio"
        dlc_data_path = f"{directory}/marker_pos_multi_proc_3_crops_normal_500_model_0_5_pp.bio"

        markers_dic = load(dlc_data_path)
        data_list = [list(markers_dic["markers_names"][:, 0]), markers_dic["markers_in_meters"]]
        data_list_dlc = [list(markers_dic["markers_names"][:, 0]), markers_dic["dlc_in_meters"][:3, ...]]
        rt = get_normal_vector(data_list, "styl_r")
        rt_dlc = get_normal_vector(data_list_dlc, "styl_r")
        data = add_virtual_markers(data_list, rt, remove_marker=None,
                                   measurement_file=f"/home/amedeoceglia/Documents/programmation/rgbd_mocap/data_collection_mesurement/measurements_{participant}.json")
        data_dlc = add_virtual_markers(data_list_dlc, rt_dlc, remove_marker=None,
                                   measurement_file=f"/home/amedeoceglia/Documents/programmation/rgbd_mocap/data_collection_mesurement/measurements_{participant}.json")

        markers_dic["markers_in_meters"] = data[1]
        markers_dic["dlc_in_meters"] = data_dlc[1]

        markers_dic["markers_names"] = np.repeat(np.array(data[0])[:, None], data[1].shape[2], axis=1)
        save(markers_dic, dlc_data_path.replace(".bio", "_technical.bio"), safe=False)
        print(f"File saved for trial : {directory} and participant : {participant}")
