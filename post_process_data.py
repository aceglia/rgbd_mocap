import os.path
import json
from biosiglive import load, save, OfflineProcessing
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import pandas as pd
from scipy.interpolate import interp1d
from pyomeca import Analogs, Markers
import glob
from scapula_cluster.from_cluster_to_anato import ScapulaCluster
from pathlib import Path
import csv
import casadi as ca


class ProcessData:
    def __init__(self,
                 measurements_dir_path=None,
                 calibration_matrix_dir=None,
                 emg_names=None,
                 depth_markers_names=None,
                 vicon_markers_names=None, ):
        self.emg_rate = 2160
        self.vicon_rate = 120
        self.sensix_rate = 100
        self.rgbd_rate = 60
        self.emg_names = ['pec.IM EMG1',
                          'bic.IM EMG2',
                          'tri.IM EMG3', 'lat.IM EMG4', 'trap.IM EMG5', 'delt_ant.IM EMG6',
                          'delt_med.IM EMG7', 'delt_post.IM EMG8']

        self.depth_markers_names = ['ster', 'xiph', 'clavsc', 'clavac',
                                    'delt', 'arml', 'epicl', 'larml', 'stylr', 'stylu', 'm1', 'm2', 'm3']
        self.vicon_markers_names = ['ster', 'xiph', 'c7', 't10', 'clavsc', 'clavac',
                                    'delt', 'arml', 'epicm', 'epicl', 'elbow', 'larml', 'stylr', 'stylu',
                                    'm1', 'm2', 'm3']

        self.measurements_dir_path = "data_collection_mesurement"
        self.calibration_matrix_dir = "../scapula_cluster/calibration_matrix"
        self.emg_names = emg_names if emg_names else self.emg_names
        self.calibration_matrix_dir = calibration_matrix_dir if calibration_matrix_dir else self.calibration_matrix_dir
        self.depth_markers_names = depth_markers_names if depth_markers_names else self.depth_markers_names
        self.vicon_markers_names = vicon_markers_names if vicon_markers_names else self.vicon_markers_names
        self.measurements_dir_path = measurements_dir_path if measurements_dir_path else self.measurements_dir_path
        self.data_path = None
        self.process_file_path = None
        self.participant = None
        self.mvc_files = None
        self.mvc = []
        self.emg = []
        self.emg_proc = []
        self.is_emg = False
        self.truncated_emg = None

    def _get_vicon_to_depth_idx(self, names_depth=None, names_vicon=None):
        names_depth = names_depth if names_depth else self.depth_markers_names
        names_vicon = names_vicon if names_vicon else self.vicon_markers_names
        vicon_markers_names = [self._convert_string(name) for name in names_vicon]
        depth_markers_names = [self._convert_string(name) for name in names_depth]
        vicon_to_depth_idx = []
        for name in vicon_markers_names:
            if name in depth_markers_names:
                vicon_to_depth_idx.append(vicon_markers_names.index(name))
        return vicon_to_depth_idx

    @staticmethod
    def _express_forces_in_global(crank_angle, f_ext):
        crank_angle = crank_angle
        roty = np.array([[np.cos(crank_angle), 0, np.sin(crank_angle)],
                         [0, 1, 0],
                         [-np.sin(crank_angle), 0, np.cos(crank_angle)]])
        return roty @ f_ext

    def _find_corresponding_sensix_file(self, file):
        final_sensix_file = None
        for sensix_file in self.sensix_files:
            if self._convert_string(Path(file).stem) in self._convert_string(Path(sensix_file).stem):
                final_sensix_file = sensix_file
        if final_sensix_file is None:
            print(f"sensix file not found for file {file}")
        return final_sensix_file

    def _sensix_from_file(self, file_path):
        sensix_file = self._find_corresponding_sensix_file(file_path)
        if sensix_file is None:
            self.sensix_data = {}
            return

        all_data = []
        with open(sensix_file, 'r') as file:
            csvreader = csv.reader(file, delimiter='\n')
            for row in csvreader:
                all_data.append(np.array(row[0].split("\t")))
        all_data_int = np.array(all_data, dtype=float).T
        process_delay_frame = int(all_data_int[0, 0] * 100) - 1
        all_data_int = np.concatenate((np.zeros_like(all_data_int)[:, :process_delay_frame], all_data_int), axis=1)
        if self.first_trigger != self.trigger_idx[0]:
            dif = self.trigger_idx[0] - self.first_trigger
            nb_frame = ceil(100 * dif / 2160)
        else:
            nb_frame = 0
        all_data_int = all_data_int[:, nb_frame + ceil(self.delay_from_depth_image * 100/60)
                                       + ceil(self.delay_for_synchro * 100/120):100 * 120]
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

        for key in dic_data.keys():
            dic_data[key] = self._smooth_sensix_angle(dic_data[key]) if "angle" in key else dic_data[key]

        for i in range(all_data_int.shape[1]):
            crank_angle = dic_data["crank_angle"][i]
            left_angle = dic_data["left_pedal_angle"][i]
            right_angle = dic_data["right_pedal_angle"][i]
            force_vector_l = [dic_data["LFX"][i], dic_data["LFY"][i], dic_data["LFZ"][i]]
            force_vector_r = [dic_data["RFX"][i], dic_data["RFY"][i], dic_data["RFZ"][i]]
            force_vector_l = self._express_forces_in_global(-left_angle, force_vector_l)
            force_vector_r = self._express_forces_in_global(-right_angle, force_vector_r)
            force_vector_l = self._express_forces_in_global(crank_angle, force_vector_l)
            force_vector_r = self._express_forces_in_global(crank_angle, force_vector_r)
            dic_data["LFX"][i] = force_vector_l[0]
            dic_data["LFY"][i] = force_vector_l[1]
            dic_data["LFZ"][i] = force_vector_l[2]
            dic_data["RFX"][i] = force_vector_r[0]
            dic_data["RFY"][i] = force_vector_r[1]
            dic_data["RFZ"][i] = force_vector_r[2]
        self.sensix_data = dic_data

    @staticmethod
    def _fill_with_nan(markers, idx):
        size = idx[-1] - idx[0]
        if len(markers.shape) == 2:
            new_markers_depth = np.zeros((markers.shape[0], size))
            count = 0
            for i in range(size):
                if i + idx[0] in idx:
                    new_markers_depth[:, i] = markers[:, count]
                    count += 1
                else:
                    new_markers_depth[:, i] = np.nan
            return new_markers_depth
        elif len(markers.shape) == 3:
            new_markers_depth = np.zeros((3, markers.shape[1], size))
            count = 0
            for i in range(size):
                if i + idx[0] in idx:
                    new_markers_depth[:, :, i] = markers[:, :, count]
                    count += 1
                else:
                    new_markers_depth[:, :, i] = np.nan
            return new_markers_depth

    @staticmethod
    def _interpolate_data(markers_depth, shape):
        new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
        for i in range(3):
            x = np.linspace(0, 100, markers_depth.shape[2])
            f_mark = interp1d(x, markers_depth[i, :, :])
            x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
            new_markers_depth_int[i, :, :] = f_mark(x_new)
        return new_markers_depth_int

    @staticmethod
    def _interpolate_2d_data(data, shape):
        x = np.linspace(0, 100, data.shape[1])
        f_mark = interp1d(x, data)
        x_new = np.linspace(0, 100, shape)
        new_data = f_mark(x_new)
        return new_data

    def _fill_and_interpolate(self, data, shape, idx=None, names=None, fill=True):
        data_nan = self._fill_with_nan(data, idx) if fill else data
        if len(data_nan.shape) == 1:
            data_nan = data_nan.reshape((1, data_nan.shape[0]))
        names = [f"n_{i}" for i in range(data_nan.shape[-2])] if not names else names
        if len(data_nan.shape) == 2:
            data_df = pd.DataFrame(data_nan, names)
            data_filled_extr = data_df.interpolate(method='linear', axis=1)
            data_int = self._interpolate_2d_data(data_filled_extr, shape)
        elif len(data_nan.shape) == 3:
            data_filled_extr = np.zeros((3, data_nan.shape[1], data_nan.shape[2]))
            for i in range(3):
                data_df = pd.DataFrame(data_nan[i, :, :], names)
                data_filled_extr[i, :, :] = data_df.interpolate(method='linear', axis=1)
            data_int = self._interpolate_data(data_filled_extr, shape)
        else:
            raise ValueError("Data shape not supported")
        return data_int

    @staticmethod
    def _smooth_sensix_angle(data):
        start_cycle = False
        for i in range(data.shape[0]):
            if data[i] < 0.08:
                start_cycle = True
            elif data[i] > 6:
                start_cycle = False
            if start_cycle and 0.1 < data[i] < 6:
                data[i] = 0
        return data

    def _find_corresponding_depth_file(self, file):
        final_depth_file = None
        for depth_file in self.depth_files:
            if self._convert_string(Path(file).stem) in self._convert_string(Path(depth_file).stem):
                final_depth_file = depth_file
        if final_depth_file is None:
            print(f"depth file not found for file {file}")
        return final_depth_file

    def _reorder_markers_from_names(self, markers_data, markers_names, ordered_markers_names):
        count = 0
        reordered_markers = np.zeros((markers_data.shape[0], len(ordered_markers_names), markers_data.shape[2]))
        for i in range(len(markers_names)):
            if markers_names[i] == "elb":
                markers_names[i] = "elbow"
            if self._convert_string(markers_names[i]) in ordered_markers_names:
                reordered_markers[:, ordered_markers_names.index(self._convert_string(markers_names[i])),
                :] = markers_data[:, count, :]
                count += 1
        return reordered_markers

    def _rotate_markers(self, markers_depth, markers_vicon):
        markers_vicon_tmp = np.zeros((3, markers_depth.shape[1], markers_vicon.shape[2]))
        vicon_to_depth_idx = self._get_vicon_to_depth_idx()
        for i in range(len(vicon_to_depth_idx)):
            markers_vicon_tmp[:, i, :] = markers_vicon[:3, vicon_to_depth_idx[i], :]
        list_nan_idx = list(np.unique(np.argwhere(np.isnan(markers_vicon_tmp))[:, 1]))
        #list_nan_idx += [0, 10, 11, 12]
        list_zeros = list(np.unique(np.argwhere(markers_vicon_tmp == 0)[:, 1]))

        list_union = list(np.unique(np.array(list_zeros + list_nan_idx)))
        list_union = [i for i in range(markers_vicon_tmp.shape[1]) if i not in list_union]
        r, T, f, c = [], [], np.inf, 0
        for i in range(0, 100):
            r_tmp, T_tmp, f_tmp = self._homogeneous_transform_optimization(
                markers_depth[:, list_union, 0],
                markers_vicon_tmp[:, list_union, self.delay_for_synchro + i]
            )
            if f_tmp < f:
                r, T, f = r_tmp, T_tmp, f_tmp
                c = i
            else:
                break
        if c >= 99:
            print("waring: optimisation iteration exceeded")
        else:
            print("number of iteration needed: ", c)
        #self.delay_from_depth_image += c
        self.delay_for_synchro += c
        new_markers_depth = np.zeros((3, markers_depth.shape[1], markers_depth.shape[2]))
        count = 0
        for i in range(markers_depth.shape[2]):
            new_markers_depth[:, :, i] = np.dot(np.array(r),
                                                np.array(markers_depth[:, :, count])
                                                ) + np.array(T)
            count += 1
        return new_markers_depth

    def _load_depth_markers(self, file):
        corresponding_depth_file = self._find_corresponding_depth_file(file)
        depth_markers_data = load(corresponding_depth_file + os.sep + "markers_pos.bio")
        self.img_idx = depth_markers_data["camera_frame_idx"]
        self.delay_from_depth_image = self.img_idx[0] - self.depth_first_image_idx[
            self.depth_files.index(corresponding_depth_file)]
        depth_markers = depth_markers_data["markers_in_meters"]
        depth_markers_names = depth_markers_data["markers_names"][:13]
        reordered_markers_depth = self._reorder_markers_from_names(depth_markers, depth_markers_names,
                                                                   self.depth_markers_names)
        return reordered_markers_depth

    def _load_vicon_markers(self, file):
        markers_data = Markers.from_c3d(filename=file)
        markers_names_tmp = markers_data.channel.values
        markers_data = markers_data.values * 0.001
        if "ster" in markers_names_tmp:
            final_markers_names = self.vicon_markers_names
        else:
            print(f"error while loading markers on file {file}.")
            return
        # reorder_markers by names
        reordered_markers = self._reorder_markers_from_names(markers_data, markers_names_tmp, final_markers_names)
        return reordered_markers

    def _markers_from_file(self, file):
        measurement_file_path = self.measurements_dir_path + os.sep + f"measurements_{self.participant}.json"
        sources = ["with_depth", "with_depth"]
        reordered_markers_depth = self._load_depth_markers(file)
        reordered_markers_vicon = self._load_vicon_markers(file)
        reordered_markers_list = [reordered_markers_depth, reordered_markers_vicon,
                                  reordered_markers_vicon]

        markers_names_list = [self.depth_markers_names, self.vicon_markers_names]
        measurement_data = json.load(open(measurement_file_path))
        #self.delay_from_depth_image = 105
        end_file_idx = self.all_idx_img[self.depth_files.index(self._find_corresponding_depth_file(file))][-1]
        start_idx_file = self.depth_first_image_idx[self.depth_files.index(self._find_corresponding_depth_file(file))]
        start_idx_file = start_idx_file + self.delay_from_depth_image
        end = (end_file_idx - start_idx_file) * 2
        start = int(self.trigger_idx[0] / 18) + self.delay_from_depth_image * 2
        reordered_markers_list[2] = reordered_markers_list[2][:, :, start:start+end]
        self.delay_for_synchro = 0
        self.rotated_depth = self._rotate_markers(reordered_markers_list[0], reordered_markers_list[2])
        reordered_markers_list[2] = reordered_markers_list[1][:, :, start+self.delay_for_synchro:start+end+self.delay_for_synchro]

        reordered_markers_list[0] = self.rotated_depth
        # fig = plt.figure("test")
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect([1, 1, 1])
        end_plot = 500
        vicon_to_depth_idx = self._get_vicon_to_depth_idx() #names_depth=self.depth_markers_names_post,
                                                          #names_vicon=self.vicon_markers_names_post)

        # plt.figure()
        # t_d = np.linspace(0, 100, reordered_markers_list[0].shape[2])
        # t_v = np.linspace(0, 100, reordered_markers_list[2].shape[2])
        # for i in range(len(vicon_to_depth_idx)):
        #     plt.subplot(4, 4, i + 1)
        #     for j in range(3):
        #         # plt.plot(markers_depth_filtered[j, i, :], "b")
        #         plt.plot(t_d, reordered_markers_list[0][j, i, :])
        #         plt.plot(t_v, reordered_markers_list[2][j, vicon_to_depth_idx[i], :], "r")
        # plt.show()

        # for i in range(len(vicon_to_depth_idx)):
        #     ax.scatter(reordered_markers_list[2][0, vicon_to_depth_idx[i], :end_plot],
        #                reordered_markers_list[2][1, vicon_to_depth_idx[i], :end_plot],
        #                reordered_markers_list[2][2, vicon_to_depth_idx[i], :end_plot], c='r')
        #     ax.scatter(reordered_markers_list[0][0, i, :end_plot],
        #                reordered_markers_list[0][1, i, :end_plot],
        #                reordered_markers_list[0][2, i, :end_plot], c='b')
        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')
        #     ax.set_zlabel('Z Label')
        # plt.show()
        for s, source in enumerate(sources):
            measurements = measurement_data[f"{source}"]["measure"]
            calibration_matrix = self.calibration_matrix_dir + os.sep + measurement_data[f"{source}"][
                "calibration_matrix_name"]
            anato_from_cluster, landmarks_dist = self._convert_cluster_to_anato(measurements,
                                                                                calibration_matrix,
                                                                                reordered_markers_list[s][:,
                                                                                -3:, :] * 1000)
            first_idx = markers_names_list[s].index("clavac")
            reordered_markers_list[s] = np.concatenate((reordered_markers_list[s][:3, :first_idx+1, :],
                                                        anato_from_cluster[:3, :, :] * 0.001,
                                                        reordered_markers_list[s][:3, first_idx+1:, :]),
                                                       axis=1)
        reordered_markers_list[2] = reordered_markers_list[1][:, :,
        ceil(self.trigger_idx[0] / 18) + self.delay_from_depth_image * 2:ceil(
            self.trigger_idx[1] / 18)]

        self.depth_markers_names_post = ['ster', 'xiph', 'clavsc', 'clavac', 'scapaa', 'scapts', 'scapia',
                                    'delt', 'arml', 'epicl', 'larml', 'stylr', 'stylu', 'm1', 'm2', 'm3']
        self.vicon_markers_names_post = ['ster', 'xiph', 'c7', 't10', 'clavsc', 'clavac', 'scapaa', 'scapts', 'scapia',
                                    'delt', 'arml', 'epicm', 'epicl', 'elbow', 'larml', 'stylr', 'stylu',
                                    'm1', 'm2', 'm3']
        self.reordered_markers_list = reordered_markers_list

    @staticmethod
    def _convert_cluster_to_anato(measurements,
                                  calibration_matrix, data):

        new_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                                     measurements[4], measurements[5], calibration_matrix)

        anato_pos = new_cluster.process(marker_cluster_positions=data, cluster_marker_names=["M1", "M2", "M3"],
                                        save_file=False)
        anato_pos_ordered = np.zeros_like(anato_pos)
        anato_pos_ordered[:, 0, :] = anato_pos[:, 0, :]
        anato_pos_ordered[:, 1, :] = anato_pos[:, 2, :]
        anato_pos_ordered[:, 2, :] = anato_pos[:, 1, :]
        land_dist = new_cluster.get_landmarks_distance()
        return anato_pos, land_dist

    def _emg_from_file(self):
        if not self.is_emg:
            self.emg_proc = []
            self.truncated_emg = []
            return
        emg = self.analog_data[:-1, :]
        emg_proc = OfflineProcessing(data_rate=2160).process_emg(emg,
                                                                 moving_average=False,
                                                                 low_pass_filter=True,
                                                                 normalization=True,
                                                                 mvc_list=self.mvc)
        self.emg_proc = emg_proc
        self.emg = emg
        self.truncated_emg = emg[
                             :,
                             self.trigger_idx[0] + self.delay_from_depth_image * 36 + self.delay_for_synchro * 18
                             :self.trigger_idx[1] + self.delay_for_synchro * 18]

    def _trigger_from_file(self, file):
        try:
            self.analog_data = Analogs.from_c3d(filename=file, usecols=self.emg_names + ["Electric Current.1"]).values
            self.is_emg = True
        except:
            self.analog_data = Analogs.from_c3d(filename=file, usecols=["Electric Current.1"]).values
            print(f"WARNING: No emg data found for file {file}")
            self.is_emg = False
        trigger = self.analog_data[-1, :]
        self.trigger = trigger
        trigger_values = np.argwhere(trigger > 1.5)
        start_limit = 50000
        if len(trigger_values) == 0:
            start_idx = 0
            end_idx = trigger.shape[0]
        else:
            triggers = [int(trig) for trig in trigger_values if trig < start_limit]
            self.first_trigger = triggers[0] if triggers[-1] - triggers[0] > 1000 else triggers[-1]
            start_idx = triggers[-1]
            try:
                end_idx = int(trigger_values[int(np.argwhere(trigger_values > start_idx + 3600)[-1, 0])])
            except:
                end_idx = trigger.shape[0]
        self.trigger_idx = [start_idx, end_idx]

    def _compute_mvc(self):
        if len(self.mvc_files) == 0:
            self.mvc = []
            return
        mvc_data = [Analogs.from_c3d(filename=file, usecols=self.emg_names).values for file in self.mvc_files]
        mvc_mat = np.append(mvc_data[0], mvc_data[1], axis=1)
        mvc = list(OfflineProcessing.compute_mvc(mvc_data[0].shape[0], mvc_trials=mvc_mat, window_size=2160))
        self.mvc = mvc

    def _get_files(self):
        vicon_files = glob.glob(self.vicon_path + f"{self.participant}/Session_1/**.c3d")
        if len(vicon_files) == 0:
            vicon_files = glob.glob(self.vicon_path + f"{self.participant}/**.c3d")
        if len(vicon_files) == 0:
            vicon_files = glob.glob(self.vicon_path + f"{self.participant}/session_1/**.c3d")
        if len(vicon_files) == 0:
            raise ValueError(f"no c3d files found for participant {self.participant}")
        depth_files = os.listdir(self.depth_path + f"{self.participant}")
        self.depth_files = [self.depth_path + f"{self.participant}" + os.sep + file for file in depth_files if
                            "gear" in file and len(os.listdir(self.depth_path + f"{self.participant}" + os.sep + file)) >= 10000
                            #or "anato" in file
                            ]
        self.depth_first_image_idx = []
        self.all_idx_img = []
        for depth_files in self.depth_files:
            all_color_files_tmp = glob.glob(depth_files + os.sep + "color_*.png")
            idx = []
            for file in all_color_files_tmp:
                idx.append(int(Path(file.split(os.sep)[-1].split("_")[-1]).stem))
            idx.sort()
            self.all_idx_img.append(idx)
            self.depth_first_image_idx.append(idx[0])
        self.mvc_files = [file for file in vicon_files if "sprint" in file and "process" not in file
]
        self.vicon_files = [file for file in vicon_files if "gear_20" in file
                            # or "anato" in file and "cluster" not in file
                            and "process" not in file

                            ]
        sensix_files = os.listdir(self.sensix_path + f"{self.participant}")
        self.sensix_files = [self.sensix_path + f"{self.participant}" + os.sep + file for file in sensix_files if
                             "gear" in file and "result" in self._convert_string(file)]

    def _interpolate_all_data(self, file):
        self.emg_interp = []
        self.sensix_interp = {}
        self.depth_interp = self._fill_and_interpolate(data=self.reordered_markers_list[0],
                                                       idx=self.img_idx,
                                                       shape=self.reordered_markers_list[2].shape[2],
                                                       fill=True)
        #rotated_depth = self._rotate_markers(self.depth_interp, self.reordered_markers_list[2])

        if self.is_emg:
            self.emg_interp = self._fill_and_interpolate(data=self.truncated_emg,
                                                         shape=self.reordered_markers_list[2].shape[2],
                                                         fill=False)

        if self.sensix_data != {}:
            for key in self.sensix_data.keys():
                self.sensix_interp[key] = self._fill_and_interpolate(data=self.sensix_data[key],
                                                                     shape=self.reordered_markers_list[2].shape[2],
                                                                     fill=False)

        fig = plt.figure("vicon")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        end_plot = 200
        vicon_to_depth_idx = self._get_vicon_to_depth_idx(names_depth=self.depth_markers_names_post,
                                                          names_vicon=self.vicon_markers_names_post)
        for i in range(len(vicon_to_depth_idx)):
            if i not in [4,5,6]:
                ax.scatter(self.reordered_markers_list[2][0, vicon_to_depth_idx[i], :end_plot],
                           self.reordered_markers_list[2][1, vicon_to_depth_idx[i], :end_plot],
                           self.reordered_markers_list[2][2, vicon_to_depth_idx[i], :end_plot], c='r')
                ax.scatter( self.depth_interp[0, i, :end_plot],  self.depth_interp[1, i, :end_plot],
                            self.depth_interp[2, i, :end_plot], c='b')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
        plt.show()
        # plt.savefig(f"{self.participant}_{Path(file).stem}.png")

    @staticmethod
    def _convert_string(string):
        return string.lower().replace("_", "")

    @staticmethod
    def _homogeneous_transform_optimization(points1, points2):
        assert len(points1) == len(points2), "Point sets must have the same number of points."

        num_points = points1.shape[1]
        # Create optimization variables
        x = ca.MX.sym("x", 12)  # [t_x, t_y, t_z, R11, R12, R13, R21, R22, R23, R31, R32, R33]
        # Extract translation and rotation components
        t = x[:3]
        R = ca.MX(3, 3)
        R[0, 0] = x[3]
        R[0, 1] = x[4]
        R[0, 2] = x[5]
        R[1, 0] = x[6]
        R[1, 1] = x[7]
        R[1, 2] = x[8]
        R[2, 0] = x[9]
        R[2, 1] = x[10]
        R[2, 2] = x[11]

        # Create objective function to minimize distance
        distance = 0
        for i in range(num_points):
            transformed_point = ca.mtimes(R, points1[:, i]) + t
            distance += ca.sumsqr(transformed_point[:] - points2[:, i])

        # Create optimization problem
        nlp = {'x': x, 'f': distance}
        opts = {'ipopt.print_level': 0, 'ipopt.tol': 1e-13}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve the optimization problem
        solution = solver()
        # Extract the optimal translation and rotation
        optimal_t = solution["x"][:3]
        optimal_R = np.ndarray((3, 3))
        f = float(solution["f"])
        optimal_R[0, 0] = solution["x"][3]
        optimal_R[0, 1] = solution["x"][4]
        optimal_R[0, 2] = solution["x"][5]
        optimal_R[1, 0] = solution["x"][6]
        optimal_R[1, 1] = solution["x"][7]
        optimal_R[1, 2] = solution["x"][8]
        optimal_R[2, 0] = solution["x"][9]
        optimal_R[2, 1] = solution["x"][10]
        optimal_R[2, 2] = solution["x"][11]
        return optimal_R, optimal_t, f

    def _save_file(self, file):
        dic_to_save = {"file_name": file,
                       "markers_depth": self.reordered_markers_list[0],
                       "markers_depth_interpolated": self.depth_interp,
                       "markers_vicon": self.reordered_markers_list[1],
                       "truncated_markers_vicon": self.reordered_markers_list[2],
                       "truncated_emg_proc": self.truncated_emg,
                       "emg_proc": self.emg_proc,
                       "emg_proc_interpolated": self.emg_interp,
                       "depth_markers_names": self.depth_markers_names_post,
                       "vicon_markers_names": self.vicon_markers_names_post,
                       "emg_names": self.emg_names,
                       "vicon_rate": self.vicon_rate, "emg_rate": self.emg_rate,
                       "sensix_rate": self.sensix_rate, "rgbd_rate": self.rgbd_rate,
                       "mvc": self.mvc,
                       "sensix_data": self.sensix_data,
                       "sensix_data_interpolated": self.sensix_interp,
                       "trigger_idx": self.trigger_idx}

        if os.path.isfile(f"{processed_data_path}/{self.participant}/{Path(file).stem}_processed.bio"):
            os.remove(f"{processed_data_path}/{self.participant}/{Path(file).stem}_processed.bio")
        if not os.path.isdir(f"{processed_data_path}/{self.participant}"):
            os.mkdir(f"{processed_data_path}/{self.participant}")
        save(dic_to_save, f"{processed_data_path}/{self.participant}/{Path(file).stem}_processed.bio", add_data=False)
        print(f"file {self.participant}/{Path(file).stem}_processed.bio saved")

    def process(self, output_data_path, participant, vicon_path, rgbd_path, sensix_path):
        self.depth_path = rgbd_path
        self.vicon_path = vicon_path
        self.sensix_path = sensix_path
        self.participant = participant
        self.process_file_path = output_data_path
        self._get_files()
        self._compute_mvc()
        for f, file in enumerate(self.vicon_files):
            print("processing trial: ", file)
            self._trigger_from_file(file)
            self._markers_from_file(file)
            self._sensix_from_file(file)
            if len(self.mvc) != 0:
                self._emg_from_file()
            else:
                self.is_emg = False
            self._interpolate_all_data(file)
            self._save_file(file)

def main(participants, processed_data_path, vicon_path, rgbd_path, sensix_path):
    process_class = ProcessData()
    for p, part in enumerate(participants):
        print(f"processing participant {part}")
        process_class.process(processed_data_path, part, vicon_path, rgbd_path, sensix_path)


if __name__ == '__main__':
    participants = ["P10", "P11", "P10", "P11", "P12", "P13", "P14"]  #,"P9", "P10",
    processed_data_path = "Q:\Projet_hand_bike_markerless/process_data"
    vicon_data_files = "Q:\Projet_hand_bike_markerless/vicon/"
    depth_data_files = "Q:\Projet_hand_bike_markerless/RGBD/"
    sensix_path = "Q:\Projet_hand_bike_markerless/sensix/"
    main(participants, processed_data_path, vicon_data_files, depth_data_files, sensix_path)
