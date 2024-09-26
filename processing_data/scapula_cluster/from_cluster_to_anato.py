import numpy as np
from pyomeca import Markers
import ezc3d
from pathlib import Path
import json
import pickle


class ScapulaCluster:
    def __init__(
        self, l_collar_ts, l_pointer_ts, l_pointer_ia, l_collar_ia, angle_wand_ia, l_wand_ia, calibration_matrix_path
    ):
        self.is_data_processed = False
        self.l_collar_ts = l_collar_ts
        self.l_pointer_ts = l_pointer_ts
        self.l_pointer_ia = l_pointer_ia
        self.l_collar_ia = l_collar_ia
        self.angle_wand_ia = angle_wand_ia
        self.l_wand_ia = l_wand_ia
        self.calibration_matrix = calibration_matrix_path
        self.save_file = False
        self.calibration_matrix_loaded = False

    def process(
        self,
        marker_cluster_positions: np.ndarray | list = None,
        c3d_files: str | list = None,
        marker_names: list = None,
        cluster_marker_names: list = None,
        frame_of_interest: np.ndarray = None,
        save_file: bool = False,
        save_in_picle: bool = False,
        file_path: str = None,
        data_rate: int = 100,
        units: str = "mm",
    ) -> list:
        """
        Process the data from the c3d files and return the scapula cluster position
        """
        if not self.calibration_matrix_loaded:
            self._load_calibration_matrix()
        if c3d_files and marker_cluster_positions:
            raise ValueError("You must provide either c3d_files or marker_cluster_positions")
        self.is_data_processed = True
        return self._load_markers(
            marker_cluster_positions,
            c3d_files,
            marker_names,
            cluster_marker_names,
            frame_of_interest,
            save_file,
            save_in_picle,
            file_path,
            data_rate,
            units,
        )

    def _load_calibration_matrix(self):
        calibration_matrix = json.load(open(self.calibration_matrix, "r"))
        cluster_in_Ra = np.array(calibration_matrix["markers_in_Ra"])[:, :, np.newaxis]
        pos_IA_TS = calibration_matrix["pos_AI_TS"]
        if "left" in self.calibration_matrix:
            pos_IA_TS[0][1] = pos_IA_TS[0][1] + self.l_pointer_ia
            pos_IA_TS[0][2] = pos_IA_TS[0][2] - self.l_collar_ia
            pos_IA_TS[1][0] = pos_IA_TS[1][0] - self.l_collar_ts
            pos_IA_TS[1][1] = pos_IA_TS[1][1] + self.l_pointer_ts
        elif "right" in self.calibration_matrix:
            pos_IA_TS[0][1] = pos_IA_TS[0][1] + self.l_pointer_ia
            pos_IA_TS[0][2] = pos_IA_TS[0][2] + self.l_collar_ia
            pos_IA_TS[1][0] = pos_IA_TS[1][0] - self.l_collar_ts
            pos_IA_TS[1][1] = pos_IA_TS[1][1] + self.l_pointer_ts
        else:
            raise ValueError("left or right must be in calibration matrix file name.")
        self.pos_ia_ts = pos_IA_TS
        self.l_wand_ia = self.l_wand_ia + calibration_matrix["length_pivot"]
        self.angle_wand_ia = self.angle_wand_ia * (np.pi / 180)
        rot_m_a = self._create_axis_coordinates(cluster_in_Ra[0, :3], cluster_in_Ra[1, :3], cluster_in_Ra[2, :3])[
            :, :, 0
        ]
        # rot_m_a = np.linalg.inv(rot_m_a)
        # import matplotlib.pyplot as plt
        # fig = plt.figure("bis")
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect([1, 1, 1])
        # x_y_z = rot_m_a[:3, 3]
        # vecx = rot_m_a[:3, 0]
        # vecy = rot_m_a[:3, 1]
        # vecz = rot_m_a[:3, 2]
        # ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecx[0], vecx[1], vecx[2], length=60, normalize=False, color="r")
        # ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecy[0], vecy[1], vecy[2], length=60, normalize=False, color="g")
        # ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecz[0], vecz[1], vecz[2], length=60, normalize=False, color="b")
        # ax.scatter(cluster_in_Ra[0, 0, ], cluster_in_Ra[ 0, 1,], cluster_in_Ra[ 0, 2,], c='r', marker='o')
        # ax.scatter(cluster_in_Ra[1, 0, ], cluster_in_Ra[ 1, 1,], cluster_in_Ra[ 1, 2,], c='g', marker='o')
        # ax.scatter(cluster_in_Ra[2, 0, ], cluster_in_Ra[ 2, 1,], cluster_in_Ra[ 2, 2,], c='b', marker='o')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()
        self.mat_ra_to_rm = np.linalg.inv(rot_m_a)
        self.calibration_matrix_loaded = True

    def get_landmarks_distance(self):
        """
        Return the distance between the landmarks
        """
        if not self.is_data_processed:
            raise RuntimeError("You must run the process function before using this function.")
        ts_in_ra, ia_in_ra = self._get_ts_ai_in_ra()
        aa_ts = np.linalg.norm(ts_in_ra)
        aa_ia = np.linalg.norm(ia_in_ra)
        ia_ts = np.linalg.norm((ts_in_ra - ia_in_ra))
        return [aa_ts, aa_ia, ia_ts]

    @staticmethod
    def _extract_transformation_matrix(l, angle, file_path, matrix_name):
        if "right" in file_path:
            right = True
        else:
            right = False
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                if matrix_name in data:
                    matrix = data[matrix_name]
                else:
                    raise RuntimeError(f"The matrix '{matrix_name}' was not found in the file.")
        except Exception as e:
            raise RuntimeError(f"An exception occured : {str(e)}")

        matrice = np.array(matrix).astype(dtype=np.float64)
        if matrix_name == "mat_R3toR2":
            if right:
                matrice[0, 0] = np.cos(angle)
                matrice[0, 2] = np.sin(angle)
                matrice[2, 0] = -np.sin(angle)
                matrice[2, 2] = np.cos(angle)
            else:
                matrice[0, 0] = np.cos(angle)
                matrice[0, 2] = -np.sin(angle)
                matrice[2, 0] = np.sin(angle)
                matrice[2, 2] = np.cos(angle)
        if matrix_name == "mat_R2toR1":
            matrice[0, 3] = -l
        return matrice

    @staticmethod
    def _save_in_file(
        marker_pos, initial_position, marker_names, save_path: str, fps, save_in_picle: bool = False, units: str = "mm"
    ):
        """
        Write data to a c3d file
        """
        all_position = np.concatenate((initial_position, marker_pos), axis=1)
        if not save_in_picle:
            c3d = ezc3d.c3d()
            # Fill it with random data
            c3d["parameters"]["POINT"]["RATE"]["value"] = [fps]
            c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_names
            c3d["data"]["points"] = all_position
            c3d["parameters"]["POINT"]["UNITS"]["value"] = [units]
            # Write the data
            c3d.write(save_path)
        else:
            with open(save_path, "wb") as file:
                pickle.dump(
                    {
                        "initial_marker_positions": initial_position,
                        "markers_names": marker_names[:-3],
                        "markers_from_cluster": {"positions": marker_pos, "names": marker_names[-3:]},
                        "data_rate": fps,
                        "units": units,
                    },
                    file,
                )

    @staticmethod
    def _create_axis_coordinates(M1, M2, M3):
        first_axis_vector = M3 - M1
        second_axis_vector = M2 - M1
        third_axis_vector = -np.cross(first_axis_vector, second_axis_vector, axis=0)
        # third_axis_vector = np.cross(first_axis_vector, second_axis_vector, axis=0)

        second_axis_vector = np.cross(third_axis_vector, first_axis_vector, axis=0)
        n_frames = first_axis_vector.shape[1]
        rt = np.zeros((4, 4, n_frames))
        rt[:3, 0, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
        rt[:3, 1, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
        rt[:3, 2, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
        rt[:3, 3, :] = M1
        rt[3, 3, :] = 1
        return rt

    def _get_ts_ai_in_ra(self):
        ai, ts = self.pos_ia_ts[0], self.pos_ia_ts[1]
        _1Ta = self._extract_transformation_matrix(
            self.l_wand_ia, self.angle_wand_ia, self.calibration_matrix, "mat_R1toRa"
        )
        _2T1 = self._extract_transformation_matrix(
            self.l_wand_ia, self.angle_wand_ia, self.calibration_matrix, "mat_R2toR1"
        )
        _3T2 = self._extract_transformation_matrix(
            self.l_wand_ia, self.angle_wand_ia, self.calibration_matrix, "mat_R3toR2"
        )
        return _1Ta.dot(ts), _1Ta.dot(_2T1.dot(_3T2.dot(ai)))

    def _from_local_to_global(self, marker):
        aa_in_ra = np.array([0, 0, 0, 1])
        ts_in_ra, ia_in_ra = self._get_ts_ai_in_ra()
        ia_in_rm, ts_in_rm, aa_in_rm = (
            self.mat_ra_to_rm.dot(ia_in_ra),
            self.mat_ra_to_rm.dot(ts_in_ra),
            self.mat_ra_to_rm.dot(np.array([0, 0, 0, 1])),
        )
        mat_rm_to_rg = self._create_axis_coordinates(marker[:3, 0, :], marker[:3, 1, :], marker[:3, 2, :])
        coord_gen = np.zeros((4, 3, mat_rm_to_rg.shape[2]))
        for i in range(mat_rm_to_rg.shape[2]):
            mat_rm_to_rg = self._create_axis_coordinates(marker[:3, 0, i:i+1], marker[:3, 1, i:i+1], marker[:3, 2, i:i+1])
            coord_gen[:, 1, i] = np.dot(mat_rm_to_rg[:, :, 0], ia_in_rm)
            coord_gen[:, 2, i] = np.dot(mat_rm_to_rg[:, :, 0], ts_in_rm)
            coord_gen[:, 0, i] = np.dot(mat_rm_to_rg[:, :, 0], aa_in_rm)
        return coord_gen

    def _load_markers(
        self,
        marker_positions,
        all_files,
        marker_names=None,
        cluster_marker_names=None,
        frame_of_interest=None,
        save_file=False,
        save_in_pickle=False,
        file_path=None,
        data_rate=100,
        units="mm",
    ):
        if marker_positions is not None:
            if not isinstance(marker_positions, list):
                if len(marker_positions.shape) != 3:
                    raise ValueError("marker_positions must be a list of 3D array or a single 3D array.")
                marker_positions = [marker_positions]

        len_data = len(marker_positions) if marker_positions is not None else len(all_files)

        if frame_of_interest:
            if len(frame_of_interest) != len_data:
                raise ValueError("frame_of_interest and all_files must have the same length if provided.")
            for f, frame in enumerate(frame_of_interest):
                if isinstance(frame, int):
                    frame_of_interest[f] = [frame, None]
                elif isinstance(frame, list):
                    if len(frame) != 2:
                        frame_of_interest[f] = [frame[0], None]
        else:
            frame_of_interest = [[0, None] for _ in range(len_data)]

        if marker_names and isinstance(marker_names, list):
            marker_names = np.array(marker_names)
            if len(marker_names.shape) == 1:
                marker_names = marker_names[np.newaxis, :]
                marker_names = marker_names.repeat(len_data, axis=0)
        if cluster_marker_names and isinstance(cluster_marker_names, list):
            cluster_marker_names = np.array(cluster_marker_names)
            if len(cluster_marker_names.shape) == 1:
                cluster_marker_names = cluster_marker_names[np.newaxis, :]
                cluster_marker_names = cluster_marker_names.repeat(len_data, axis=0)
        if cluster_marker_names is not None and cluster_marker_names.shape[1] != 3:
            raise ValueError("cluster_marker_names must have a length of 3.")
        if (
            marker_names is not None
            and marker_names.shape[0] != len_data
            or cluster_marker_names is not None
            and cluster_marker_names.shape[0] != len_data
        ):
            raise ValueError("marker_names and all_files must have the same length if provided.")

        global_coordinates = []
        count = 0
        for i in range(len_data):
            # print("Processing file %s" % all_files[i])
            s, e = frame_of_interest[i][0], frame_of_interest[i][1]
            if all_files:
                list_all_markers_data = Markers.from_c3d(all_files[i])
                if e:
                    all_markers_data = list_all_markers_data.values[:, :, s:e]
                else:
                    all_markers_data = list_all_markers_data.values[:, :, s:]
                data_rate = list_all_markers_data.rate
                units = list_all_markers_data.units if units is None else units
                marker_names_tmp = list_all_markers_data.channel.values if marker_names is None else marker_names[i]
                if file_path:
                    file_path_tmp = file_path[i]
                else:
                    file_path_tmp = all_files[i][:-4] + "_processed.c3d"
            else:
                if marker_positions[i].shape[0] != 4:
                    marker_positions[i] = np.append(( marker_positions[i]), np.ones_like(marker_positions[i][0:1, :, :]), axis=0)
                all_markers_data = marker_positions[i][:, :, s:e]
                if file_path:
                    file_path_tmp = file_path[i]
                else:
                    file_path_tmp = "markers_processed.pkl"
                marker_names_tmp = [] if marker_names is None else marker_names[i]
                units = units if units is not None else "mm"
            cluster = np.zeros((4, 3, all_markers_data.shape[2]))
            name = None
            for m in range(all_markers_data.shape[1]):
                if count == 3:
                    break
                if len(marker_names_tmp) > 0:
                    name = marker_names_tmp[m]
                if len(marker_names_tmp) > 0:
                    if cluster_marker_names is not None and name is not None and name in cluster_marker_names[i]:
                        cluster[:, cluster_marker_names[i].tolist().index(name), :] = all_markers_data[:, m, :]
                        count += 1
                else:
                    if m >= 3:
                        raise ValueError(
                            "marker_names must be provided if more than 3 markers are present in the file."
                        )
                    cluster[:, m, :] = all_markers_data[:, m, :]
                    count += 1

            if count != 3:
                print("Warning: not all markers were found in the file.")
            count = 0
            global_coordinates.append(self._from_local_to_global(cluster))
            if isinstance(marker_names_tmp, np.ndarray):
                marker_names_tmp = marker_names_tmp.tolist()
            marker_names_tmp.append("scap_aa_from_cluster")
            marker_names_tmp.append("scap_ia_from_cluster")
            marker_names_tmp.append("scap_ts_from_cluster")
            if save_file:
                self._save_in_file(
                    global_coordinates[-1],
                    all_markers_data,
                    marker_names_tmp,
                    file_path_tmp,
                    data_rate,
                    save_in_pickle,
                    units,
                )
        if len(global_coordinates) == 1:
            global_coordinates = global_coordinates[0]
        return global_coordinates
