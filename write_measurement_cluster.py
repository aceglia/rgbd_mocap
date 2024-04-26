import os
import json

if __name__ == '__main__':
    participant = ["P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    # calibration_matrix_path = "/home/amedeo/Documents/programmation/scapula_cluster/calibration_matrix/calibration_mat_left_reflective_markers.json"

    source = ["depth", "vicon"]
    calibration_matrix_depth = ["calibration_mat_left_RGBD_markers.json"] * 4 + \
                               ["calibration_mat_left_RGBD_screen.json"] * (len(participant) - 4)

    calibration_matrix_vicon = ["calibration_mat_left_reflective_markers.json"] * len(participant)

    measure_depth = [
        [77, 23, 29, 153, 9, 50.5],  # P5
        [85, 20, 29, 147, 9, 69],  # P6
        [83.5, 33.5, 25, 132, -9, 39],  # P7
        [74, 26, 28, 134, -9, 39],  # P8
        [86, 35, 38, 125.5, -9, 54],  # P9
        [86, 49, 52, 130, -9, 45],  # P10
        [100.5, 26, 36, 135.5, -9, 53.5],  # P11
        [86, 47, 48, 119, -9, 65],  # P12
        [92, 29, 26, 131, -9, 51],  # P13
        [93, 35, 48, 153, -9, 53],  # P14
        [96.5, 39, 24.5, 137, -9,  49.5],  # P15
        [95, 33, 32, 153, -9, 63.5],  # P16
    ]

    measure_vicon = [
        [77, 23, 29, 153, 9, 50.5],  # P5
        [85, 20, 29, 147, 9, 69],  # P6
        [83.5, 28, 19, 132, -9, 39],  # P7
        [74, 35, 38, 133.5, -9, 46],  # P8
        [86, 43, 52, 130.5, -9, 45],  # P9
        [93, 42, 36, 145, -9, 57],  # P10
        [85.5, 37, 48.5, 136.5, -9, 46.5],  # P11
        [84, 39, 46, 109, -9, 65],  # P12
        [91, 26.5, 38, 126, -9, 42],  # P13
        [80, 37, 55, 150, -9, 50],  # P14
        [90, 34, 28.5, 135.5, -9, 49.5],  # P15
        [103, 33, 36.5, 148.5, -9, 70],  # P16

    ]
    measures = [measure_depth, measure_vicon]
    calib_matrix = [calibration_matrix_depth, calibration_matrix_vicon]
    ster_C7 = [144,  # P5
               139,  # P6
               126,  # P7
               134,  # P8
               145,  # P9
               152,  # P10
               151,  # P11
               148,  # P12
               143,  # P13
               149,  # P14
               143,  # P15
               159,  # P16
               ]

    age = [22,  # P5
           23,  # P6
           21,  # P7
           21,  # P8
           21,  # P9
           22,  # P10
           22,  # P11
           21,  # P12
           20,  # P13
           21,  # P14
           58,  # P15
           22,  # P16
           ]

    height = [166,  # P5
              173,  # P6
              170,  # P7
              160,  # P8
              171,  # P9
              185,  # P10
              178,  # P11
              172,  # P12
              166,  # P13
              184,  # P14
              163,  # P15
              187,  # P16
              ]

    weight = [73,  # P5
              72,  # P6
              51,  # P7
              71,  # P8
              62,  # P9
              76,  # P10
              68,  # P11
              68,  # P12
              68,  # P13
              85,  # P14
              57,  # P15
              78,  # P16
              ]

    sexe = ["M",  # P5
            "M",  # P6
            "F",  # P7
            "F",  # P8
            "M",  # P9
            "M",  # P10
            "M",  # P11
            "F",  # P12
            "F",  # P13
            "M",  # P14
            "F",  # P15
            "M",  # P16

            ]


    if not os.path.isdir("data_collection_mesurement"):
        os.mkdir("data_collection_mesurement")
    for i in range(len(participant)):
        dic = {"participant": participant[i], "ster_C7": ster_C7[i],
               "measure_names": ["l_collar_TS",
                                 "l_pointer_TS", "l_pointer_IA", "l_collar_IA", "angle_wand_ia", "l_wand_ia"],
               "age": age[i], "height": height[i], "weight": weight[i], "sexe": sexe[i]}
        for s in range(len(source)):
            dic["with_" + source[s]] = {
                "calibration_matrix_name": calib_matrix[s][i],
                "measure": measures[s][i],
            }
        with open(f"data_collection_mesurement{os.sep}measurements_" + participant[i] + ".json", "w") as outfile:
            json.dump(dic, outfile, indent=4)
