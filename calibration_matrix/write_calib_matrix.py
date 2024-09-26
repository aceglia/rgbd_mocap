"""
This file is used to write the calibration matrix for the scapula cluster device.
It is intended to be used only if the device has been changed.
Refer to the documentation for more information about the angles and lengths.
Please modify this file carefully as it will be used to reconstruct the scapula anatomical frame.
"""
import json
import numpy as np

alpha_1 = 0 * (np.pi / 180)
beta = 45 * (np.pi / 180)
beta_g = -45 * np.pi / 180

# Length of part "pivot" to allow to consider the middle of the joint when using the pivot position.
# This length must be changed if the pivot part is changed.
length_pivot = 7.5

# Position of AI and TS markers in the cluster frame (depend on the position of the device).
# The real position are found when running the main code.
pos_AI_TS_r = [[5.69, 7.3 - 80, 14.25, 1], [-11.25, 7.3 - 80, 5.69, 1]]
pos_AI_TS_l = [[-5.69, 7.3 - 80, -16.25, 1], [-11.25, 7.3 - 80, -5.69, 1]]

# Please decomment the markers position corresponding to your configuration
# # ------- Reflective markers positions in Ra ------- #
# M_left_Ra = [[15.64, 37.4, 9.32, 1], [-65.64, 37.4, 9.32, 1], [-25, 24.89, 64.1, 1]]
# M_right_Ra = [[-65.64, 37.4, -9.32, 1], [15.64, 37.4, -9.32, 1], [-25, 24.89, -64.1, 1]]

# ------- RGBD markers positions in Ra ------- #
# M_left_Ra = [[20.7, 42.1, 6.4, 1], [-70.7, 42.1, 6.4, 1], [-25, 29.58, 69.96, 1]]
# M_right_Ra = [[-70.7, 42.1, -6.4, 1], [20.7, 42.1, -6.4, 1], [-25, 29.58, -69.96, 1]]


# ------- Oriented RGBD markers positions in Ra ------- #
# M_right_Ra = [[-72.55, 40.66, -7.29, 1], [19.35, 44.05, -7.18, 1], [-27.5, 29.58, -69.96, 1]]
# M_left_Ra = [[22.55, 40.66, 7.29, 1], [-69.35, 44.05, 7.18, 1], [-22.5, 29.58, 69.96, 1]]

# ------- Custom screen for the RGBD camera (10/01/2024) ------- #
# Found using vicon and small markers can be not optimal.
M_left_Ra = [[3.594721921768439, 32.881744183404706, 32.24294072573264, 1],
             [-9.639076750447344, 81.05386981261762, 52.15412298381773, 1],
             [-29.56340378537749, 39.22389305741463, 72.48601160493172, 1]]
M_right_Ra = None


def save_matrix(M1Ta, M2T1, M3T2, pos_ai_ts, markers_in_ra, suffix):
    dic = {
        "mat_R1toRa": M1Ta,
        "mat_R2toR1": M2T1,
        "mat_R3toR2": M3T2,
        "pos_AI_TS": pos_ai_ts,
        "markers_in_Ra": markers_in_ra,
        "length_pivot": length_pivot,
    }
    calibration_file = f"calibration_{suffix}.json"
    with open(calibration_file, "w") as f:
        json.dump(dic, f, indent=4)

if M_left_Ra:
    save_matrix(
        [
            [np.cos(alpha_1), -np.sin(alpha_1), 0, -31],
            [np.sin(alpha_1), np.cos(alpha_1), 0, 11.87],
            [0, 0, 1, -2.5],
            [0, 0, 0, 1],
        ],
        [
            [1, 0, 0, -10000],
            [0, np.cos(beta_g), -np.sin(beta_g), 14.21],
            [0, np.sin(beta_g), np.cos(beta_g), -2.78],
            [0, 0, 0, 1],
        ],
        [[-10000, 0, -10000, 0], [0, 1, 0, 0], [-10000, 0, -10000, 0], [0, 0, 0, 1]],
        pos_AI_TS_l,
        M_left_Ra,
        "mat_left_RGBD_screen",
    )

if M_right_Ra:
    save_matrix(
        [
            [np.cos(alpha_1), -np.sin(alpha_1), 0, -31],
            [np.sin(alpha_1), np.cos(alpha_1), 0, 11.87],
            [0, 0, 1, 2.5],
            [0, 0, 0, 1],
        ],
        [[1, 0, 0, -10000], [0, np.sin(beta), -np.cos(beta), 13.89], [0, np.cos(beta), np.sin(beta), 3.59], [0, 0, 0, 1]],
        [[-10000, 0, -10000, 0], [0, 1, 0, 0], [-10000, 0, -10000, 0], [0, 0, 0, 1]],
        M_right_Ra,
        pos_AI_TS_r,
        "calibration_mat_r_RGBD_screen",
    )
