import os
from math import ceil

import numpy as np
import biorbd
import json

from lxml.html.defs import frame_tags

from processing_data.biomech_analysis.msk_utils import compute_cor, run_ik, run_so, run_jrf, run_id, get_map_activation_idx, get_tracking_idx
from rgbd_mocap.tracking.kalman import Kalman
import time
from biosiglive import MskFunctions, save, RealTimeProcessing
from biosiglive.file_io.save_and_load import dic_merger
from processing_data.biomech_analysis.enums import FilteringMethod
from processing_data.scapula_cluster.from_cluster_to_anato import ScapulaCluster
from processing_data.data_processing_helper import convert_cluster_to_anato, reorder_markers_from_names, process_cycles, fill_and_interpolate, refine_synchro

prefix = "/mnt/shared" if os.name == "posix" else "Q:/"

class BiomechPipeline:
    def __init__(self, stop_frame=None):
        self.cycles_computed = False
        self.model_dir = None
        self.markers_rate = None
        self.marker_names = None
        self.idx_cluster = None
        self.kalman_params = None
        self.moving_window = None
        self.rt_processing_list = None
        self.live_filter_method = None
        self.reordered_idx = None
        self.scaling_factor = (1, 1)
        self.proc_noise = None
        self.measurement_noise = None
        self.fps = 120
        self.scapula_cluster = None
        self.kalman_instance = None
        self.n_markers = None
        self.processed_markers = None
        self.stop_frame = stop_frame
        self.range_frame = None
        self.msk_function = None
        self.forces = None
        self.frame_idx = None
        self.key = None
        self.rt_matrix = None
        self.range_idx = None
        self.external_loads = None
        self.emg = None
        self.peaks = None
        self.vicon_to_depth_idx = None
        self.trial_name = None
        self.f_ext = None
        self.emg_names = ["PectoralisMajorThorax_M",
                 "BIC",
                 "TRI_lat",
                 "LatissimusDorsi_S",
                 'TrapeziusScapula_S',
                 "DeltoideusClavicle_A",
                 'DeltoideusScapula_M',
                 'DeltoideusScapula_P']
        # self.emg_names = ["PECM",
        #          "bic",
        #          "tri",
        #          "LAT",
        #          'TRP1',
        #          "DELT1",
        #          'DELT2',
        #          'DELT3']
        self.results_dict = {}
        self.compute_so = None
        self.compute_jrf = None
        self.compute_id = None
        self.compute_ik = None
        self.print_optimization_status = None
        self.filter_function = None
        self.emg_track_idx = None
        self.muscle_map_idx = None
        self.frame_count = 0
        self.current_frame = 0
        self.dlc_frame_count = 0
        self.current_frame = 0

    def set_stop_frame(self, stop_frame, frame_idx, key, live_filter, data_shape=None):
        self.frame_idx = frame_idx
        self.key = key
        if stop_frame is None and data_shape:
            stop_frame = data_shape

        if frame_idx is not None:
            if stop_frame is None:
                stop_frame = (frame_idx[-1] - frame_idx[0]) * 2
            if (frame_idx[-1] - frame_idx[0]) * 2 < stop_frame:
                stop_frame = (frame_idx[-1] - frame_idx[0]) * 2
            if stop_frame % 2 != 0:
                stop_frame -= 1
        stop_frame_tmp = int(stop_frame // 2) if live_filter and "dlc" in key else stop_frame
        if "dlc" not in key or not live_filter:
            range_frame = range(stop_frame_tmp)
        else:
            range_frame = range(frame_idx[0], frame_idx[-1])
        self.range_frame = range_frame
        self.stop_frame = stop_frame_tmp

    def init_scapula_cluster(self, participant, measurements_dir_path=None, calibration_matrix_dir=None, config="with_depth"):
        measurements_dir_path = "/home/amedeoceglia/Documents/programmation/rgbd_mocap/data_collection_mesurement"
        calibration_matrix_dir = "/home/amedeoceglia/Documents/programmation/rgbd_mocap/calibration_matrix"
        measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{participant}.json"))
        measurements = measurement_data[config]["measure"]
        calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[config][
            "calibration_matrix_name"]
        self.scapula_cluster = ScapulaCluster(measurements[0], measurements[1], measurements[2], measurements[3],
                                     measurements[4], measurements[5], calibration_matrix)

    def _get_next_frame_from_kalman(self, markers_data=None, forward=0, rotate=False, compute_from_cluster=True):
        if rotate and (self.rt_matrix is not None and markers_data is not None):
            markers_dlc_hom = np.ones((4, markers_data.shape[1], 1))
            markers_dlc_hom[:3, :, 0] = markers_data[..., 0]
            markers_data = np.dot(np.array(self.rt_matrix), markers_dlc_hom[:, :, 0])[:3, :, None]
        # next_frame = markers_data
        if self.n_markers is None:
            if self.kalman_instance is not None:
                self.n_markers = len(self.kalman_instance)
            elif markers_data is not None:
                self.n_markers = markers_data.shape[1]
            else:
                raise ValueError("Impossible to know how many markers there are.")
        next_frame = np.zeros((3, self.n_markers, 1))
        self.kalman_instance = [None] * self.n_markers if self.kalman_instance is None else self.kalman_instance
        for k in range(self.n_markers):
            if self.kalman_instance[k] is None and markers_data is not None:
                measurement_noise_factor = self.kalman_params[:int(markers_data.shape[1])][k]
                process_noise_factor = self.kalman_params[int(markers_data.shape[1]):int(markers_data.shape[1] * 2)][k]
                self.kalman_instance[k] = Kalman(markers_data[:, k, 0], n_measures=3, n_diff=2, fps=self.markers_rate,
                                            measurement_noise_factor=measurement_noise_factor,
                                            process_noise_factor=process_noise_factor,
                                            error_cov_post_factor=0,
                                            error_cov_pre_factor=0
                                            )
                next_frame[:, k, 0] = self.kalman_instance[k].predict()
            elif self.kalman_instance[k] is not None:
                next_frame[:, k, 0] = self.kalman_instance[k].predict()
                if markers_data is not None:
                    next_frame[:, k, 0] = self.kalman_instance[k].correct(markers_data[:, k, 0])
                if forward != 0:
                    next_frame[:, k, 0] = self.kalman_instance[k].get_future_pose(dt=forward)
            else:
                raise ValueError("Unexpected error.")
        if compute_from_cluster:
            anato_from_cluster = convert_cluster_to_anato(next_frame[:, -3:, :],
                                                          scapula_cluster=self.scapula_cluster)
            next_frame = np.concatenate(
                (next_frame[:, :self.idx_cluster + 1, :], anato_from_cluster[:3, ...],
                 next_frame[:, self.idx_cluster + 1:, :]),
                axis=1)
        return next_frame[..., 0]

    def set_variable(self, variable, value):
        setattr(self, variable, value)

    def _filter_markers(self, markers):
        self.processed_markers = np.zeros_like(markers)
            # if self.processed_markers is None else self.processed_markers
        # if self.processed_markers.shape != markers.shape:
        #     self.processed_markers = np.zeros_like(markers)
        for i in range(3):
            self.processed_markers[i, :, :] = self.rt_processing_list[i].process_emg(markers[i, :, :],
                                                                                           band_pass_filter=False,
                                                                                           centering=False,
                                                                                           absolute_value=False,
                                                                                        moving_average=True,
                                                                                           moving_average_window=self.moving_window,
                                                                                           )[:, -1:]
        return self.processed_markers

    def get_filter_function(self, **kwargs):
        if self.live_filter_method == FilteringMethod.NONE:
            self.moving_window = 0
            return lambda x: x

        elif self.live_filter_method == FilteringMethod.MovingAverage:
            self.moving_window = 14
            self.rt_processing_list = [RealTimeProcessing(120, self.moving_window),
                               RealTimeProcessing(120, self.moving_window),
                               RealTimeProcessing(120, self.moving_window)]
            return lambda x: self._filter_markers(x)

        elif self.live_filter_method == FilteringMethod.Kalman:
            self.moving_window = 0
            self.kalman_params = self.measurement_noise + self.proc_noise
            return lambda x: self._get_next_frame_from_kalman(x, **kwargs)
        else:
            raise ValueError("Invalid filtering method")

    def get_filtered_markers(self, markers, live_filter_method, compute_from_cluster=True):
        if live_filter_method == FilteringMethod.NONE:
            markers_tmp = markers[..., self.frame_count:self.frame_count + 1]
            if self.reordered_idx is not None and self.frame_count != 0:
                markers_tmp = markers_tmp[:, self.reordered_idx, None]
            else:
                model_names = [self.msk_function.model.markerNames()[i].to_string() for i in
                               range(self.msk_function.model.nbMarkers())]

                markers_tmp, self.reordered_idx = reorder_markers_from_names(markers_tmp[:, :-3, None], model_names,
                                                             self.marker_names[:-3])
                self.marker_names = model_names
            return markers_tmp[..., 0]
        if "dlc" not in self.key:
                # self.marker_names = self.marker_names[:self.idx_cluster + 1] + ["scapaa", "scapts", "scapia"] + self.marker_names[self.idx_cluster + 4:]
            markers_tmp = markers[..., self.frame_count:self.frame_count + 1]
            if compute_from_cluster:
                if self.frame_count == self.moving_window:
                    self.idx_cluster = self.marker_names.index("clavac")
                markers_tmp = np.delete(markers_tmp, [self.idx_cluster + 1, self.idx_cluster + 2, self.idx_cluster + 3], axis=1)
            if self.reordered_idx is not None and self.frame_count != 0:
                markers_tmp = markers_tmp[:, self.reordered_idx, None]
            else:
                model_names = [self.msk_function.model.markerNames()[i].to_string() for i in
                               range(self.msk_function.model.nbMarkers())]

                markers_tmp, self.reordered_idx = reorder_markers_from_names(markers_tmp[:, :-3, None], model_names,
                                                         self.marker_names[:-3])
                self.marker_names = model_names
            markers_tmp = self.filter_function(markers_tmp[..., 0])

            return markers_tmp[...]

        elif "dlc" in self.key:
            count_dlc = self.dlc_frame_count if live_filter_method != FilteringMethod.NONE else self.frame_count
            if live_filter_method == 0 or (live_filter_method != 0 and self.current_frame in self.frame_idx):
                markers_tmp = markers[..., count_dlc:count_dlc + 1]
                self.dlc_frame_count += 1
            else:
                markers_tmp = None
            if self.frame_count == self.moving_window:
                if self.frame_count == self.moving_window:
                    self.idx_cluster = self.marker_names.index("clavac")
                self.marker_names = self.marker_names[:self.idx_cluster + 1] + ["scapaa", "scapia", "scapts"] + self.marker_names[self.idx_cluster + 1:]

            markers_tmp = self.filter_function(markers_tmp)
            if self.reordered_idx is not None and self.frame_count != 0:
                markers_tmp = markers_tmp[:, self.reordered_idx, None]
            else:
                model_names = [self.msk_function.model.markerNames()[i].to_string() for i in
                               range(self.msk_function.model.nbMarkers())]

                markers_tmp, self.reordered_idx = reorder_markers_from_names(markers_tmp[:, :-3, None], model_names,
                                                             self.marker_names[:-3])
                self.marker_names = model_names
            markers_tmp = markers_tmp[:, :, 0]
            return markers_tmp

    def process_all_frames(self, markers: np.ndarray, model_path: str, live_filter_method: FilteringMethod = FilteringMethod.NONE,
                           compute_id: bool=True, compute_so: bool=True, compute_jrf:bool =False,
                           print_optimization_status: bool =False, compute_ik: bool =True,
                           marker_names: list =None, **kwargs):
        self.frame_count = 0
        self.current_frame = 0
        self.dlc_frame_count = 0
        self.compute_so = compute_so
        self.compute_jrf = compute_jrf
        self.compute_id = compute_id
        self.compute_ik = compute_ik
        self.print_optimization_status = print_optimization_status
        self.msk_function = MskFunctions(model=model_path, data_buffer_size=20, system_rate=self.markers_rate)
        self.live_filter_method = live_filter_method
        self.marker_names = marker_names
        final_dic = {}
        compute_from_cluster = True if "dlc" in self.key else False
        self.filter_function = self.get_filter_function(rotate="dlc" in self.key, forward=0,
                                                        compute_from_cluster=compute_from_cluster) #"dlc" in self.key)
        self.emg_track_idx = get_tracking_idx(self.msk_function.model, self.emg_names)
        self.muscle_map_idx = get_map_activation_idx(self.msk_function.model, self.emg_names)
        for i in self.range_frame:
            self.current_frame = i
            tic = time.time()
            markers_tmp = self.get_filtered_markers(markers,
                                                    live_filter_method,
                                                    compute_from_cluster=compute_from_cluster
                                                    )
            if not self.frame_count >= self.moving_window:
                print("Waiting for enough frames to compute inverse kinematics. Still needs: ",
                      self.moving_window - self.frame_count)
                self.frame_count += 1
                continue
            dic_to_save = self.process_next_frame(markers_tmp, **kwargs)
            tim_to_get_frame = time.time() - tic
            dic_to_save["time"]["time_to_get_frame"] = tim_to_get_frame
            dic_to_save["markers"] = markers_tmp[:, :, None] if markers_tmp.ndim == 2 else markers_tmp
            if dic_to_save is not None:
                final_dic = dic_merger(final_dic, dic_to_save)
            self.frame_count += 1
            if self.stop_frame is not None and self.frame_count >= self.stop_frame:
                break
        final_dic["marker_names"] = self.marker_names
        # import matplotlib.pyplot as plt
        # for i in range(final_dic["markers"].shape[1]):
        #     plt.subplot(4, 4, i + 1)
        #     for j in range(markers.shape[0]):
        #         plt.plot(markers[j, i, :self.stop_frame], label=f"marker {j}", c="r")
        #         plt.plot(final_dic["markers"][j, i, :], label=f"filtered marker {j}", c="b")
        # plt.show()

        final_dic["center_of_rot"] = compute_cor(final_dic["q"], self.msk_function.model)
        if self.key == "minimal":
            import bioviz
            b = bioviz.Viz("/mnt/shared/Projet_hand_bike_markerless/RGBD/P9/output_models/gear_5_model_scaled_minimal_vicon_new_seth.bioMod")
            b.load_movement(final_dic["q"])
            b.load_experimental_markers(final_dic["markers"][:, :, :])
            b.exec()
        self.results_dict[self.key] = final_dic
        return self.results_dict[self.key]

    def process_next_frame(self, markers, **kwargs) -> dict:
        times = {}
        dic_to_save = {"q": None, "q_dot": None, "q_ddot": None,
                       "tau": None,
                       "mus_act": None,
                       "emg_proc": None,
                       "res_tau": None,
                       "jrf": None,
                       "time": None,
                       "markers": None}
        if self.compute_ik and self.frame_count >= self.moving_window:
            init_ik = True if self.frame_count == self.moving_window else False
            if self.key =="vicon":
                pass
            times, dic_to_save, self.msk_function = run_ik(self.msk_function,
                                             markers, times=times, dic_to_save=dic_to_save, init_ik=init_ik,
                                        kalman_freq=self.markers_rate, model_prefix=self.trial_name)
            if self.compute_id:
                if not self.compute_ik:
                    raise ValueError("Inverse kinematics must be computed to compute inverse dynamics")
                times, dic_to_save = run_id(self.msk_function, self.f_ext[..., self.frame_count],
                                            self.external_loads, times, dic_to_save)

            if self.compute_so:
                if not self.compute_id:
                    raise ValueError("Inverse dynamics must be computed to compute static optimization")
                emg = self.emg[..., self.frame_count] if self.emg is not None else None
                times, dic_to_save = run_so(self.msk_function, emg, times, dic_to_save,
                                            self.scaling_factor,
                                                 print_optimization_status=self.print_optimization_status,
                                                 emg_names=self.emg_names, track_idx=self.emg_track_idx,
                                            map_emg_idx=self.muscle_map_idx, **kwargs)

            if self.compute_jrf:
                if not self.compute_so:
                    raise ValueError("Static optimization must be computed to compute joint reaction forces")
                times, dic_to_save = run_jrf(self.msk_function, times, dic_to_save, self.external_loads)

            times["tot"] = sum(times.values())
            dic_to_save["time"] = times
            return dic_to_save
        else:
            print("Waiting for enough frames to compute inverse kinematics. Still needed : ", self.moving_window - self.frame_count)
            return None

    def _handle_dlc_before_saving(self, interpolate_dlc=True):
        for key in self.results_dict.keys():
            data_dic_tmp = self.results_dict[key]
            if "dlc" in key:
                tmp1 = data_dic_tmp["markers"][:, 7, :].copy()
                tmp2 = data_dic_tmp["markers"][:, 6, :].copy()
                data_dic_tmp["markers"][:, 6, :] = tmp1
                data_dic_tmp["markers"][:, 7, :] = tmp2
                data_dic_tmp["marker_names"][6], data_dic_tmp["marker_names"][7] = \
                data_dic_tmp["marker_names"][7], data_dic_tmp["marker_names"][6]
                dlc_mark_tmp = data_dic_tmp["markers"][:, :, :].copy()
                dlc_mark_tmp = np.delete(dlc_mark_tmp, data_dic_tmp["marker_names"].index("ribs"), axis=1)
                dlc_mark, idx = refine_synchro(self.results_dict["minimal_vicon"]["markers"][:, :, :],
                                               dlc_mark_tmp, plot_fig=False)
                for key_2 in data_dic_tmp.keys():
                    data_dic_tmp_2 = data_dic_tmp[key_2]
                    if isinstance(data_dic_tmp_2, np.ndarray):
                        data_dic_tmp_2 = data_dic_tmp_2[..., :-idx] if idx != 0 else data_dic_tmp_2[..., :]
                        if interpolate_dlc and self.live_filter_method != 0:
                            data_dic_tmp_2 = fill_and_interpolate(
                                data_dic_tmp_2,
                                fill=False,
                                shape=self.results_dict["depth"]["q"].shape[1])
                    data_dic_tmp[key_2] = data_dic_tmp_2
                self.results_dict[key] = data_dic_tmp

    def save(self, output_file, interpolate_dlc=False):
        dic_keys = self.results_dict.keys()
        is_dlc = len(["dlc" in key for key in dic_keys]) > 0
        if self.live_filter_method.value != 0 and is_dlc:
            self._handle_dlc_before_saving(interpolate_dlc=interpolate_dlc)
        self.results_dict = process_cycles(self.results_dict, self.peaks, n_peaks=None)
        self.cycles_computed = True
        self.results_dict["shared"] = {
            "emg": self.emg,
            "f_ext": self.f_ext,
            "peaks": self.peaks,
            "rt_matrix": self.rt_matrix,
            "vicon_to_depth": self.vicon_to_depth_idx
        }
        save(self.results_dict, output_file, safe=False)
        print(f"The file ({output_file}) has been saved.")

    def _compute_mean_cycle(self, plot_by_cycle=False, n_cycle=None):
        if not self.cycles_computed:
            dic_keys = self.results_dict.keys()
            is_dlc = len(["dlc" in key for key in dic_keys]) > 0
            if self.live_filter_method.value != 0 and is_dlc:
                self._handle_dlc_before_saving(interpolate_dlc=True)
            self.results_dict = process_cycles(self.results_dict, self.peaks, n_peaks=None)
        for key in self.results_dict.keys():
            if key == "shared":
                continue
            dic_tmp = {}
            if not isinstance(self.results_dict[key], dict):
                continue
            for key_2 in self.results_dict[key].keys():
                dic_tmp[key_2] = {}
                if isinstance(self.results_dict[key][key_2], np.ndarray):
                    if plot_by_cycle:
                        if n_cycle:
                            dic_tmp[key_2]["mean"] = np.mean(self.results_dict[key][key_2][:n_cycle, ...], axis=0)
                            dic_tmp[key_2]["std"] = np.std(self.results_dict[key][key_2][:n_cycle, ...], axis=0)
                        else:
                            dic_tmp[key_2]["mean"] = np.mean(self.results_dict[key][key_2][:, ...], axis=0)
                            dic_tmp[key_2]["std"] = np.std(self.results_dict[key][key_2][:, ...], axis=0)
                    else:
                        dic_tmp[key_2]["mean"] = self.results_dict[key][key_2]
                        dic_tmp[key_2]["std"] = self.results_dict[key][key_2]
                    # dic_tmp[key] = result[key][0, ...]
                else:
                    dic_tmp[key_2] = self.results_dict[key][key_2]
            self.results_dict[key] = dic_tmp

    def plot_results(self, plot_by_cycle=False, n_cycle=None):
        self._compute_mean_cycle(plot_by_cycle=plot_by_cycle, n_cycle=n_cycle)
        import matplotlib.pyplot as plt
        nb_source = len(self.results_dict.keys())
        colors = plt.cm.get_cmap('tab10', nb_source)
        count_source = 0
        count_key = 0
        keys_to_plot = ["markers", "q", "q_dot", "q_ddot", "tau", "mus_force"]
        for source in self.results_dict.keys():
            if source == "shared":
                continue
            # if "dlc" in source:
            #     continue
            plt.figure("legend")
            plt.plot([], c=colors(count_source), label=source)
            plt.legend()
            for key in self.results_dict[source].keys():
                if key not in keys_to_plot or isinstance(self.results_dict[source][key], list):
                    continue
                if key == "markers":
                    if source == "vicon":
                        continue
                    plt.figure("markers")
                    count = 0
                    for i in range(13):
                        plt.subplot(ceil(13/4), 4, i + 1)
                        if self.results_dict[source]["marker_names"][i] == "ribs":
                            count += 1
                        for j in range(3):
                            plt.plot(self.results_dict[source][key]["mean"][j, count, :], c=colors(count_source))
                        count += 1
                    continue

                plt.figure(key)
                for i in range(self.results_dict[source][key]["mean"].shape[0]):
                    plt.subplot(ceil(self.results_dict[source][key]["mean"].shape[0]/4), 4, i + 1)
                    if plot_by_cycle:
                        plt.plot(self.results_dict[source][key]["mean"][i, :n_cycle], c=colors(count_source))
                        plt.fill_between(np.arange(n_cycle),
                                         self.results_dict[source][key]["mean"][i, :n_cycle] - self.results_dict[source][key]["std"][i, :n_cycle],
                                         self.results_dict[source][key]["mean"][i, :n_cycle] + self.results_dict[source][key]["std"][i, :n_cycle],
                                         alpha=0.2, color=colors(count_source))
                    else:
                        plt.plot(self.results_dict[source][key]["mean"][i, :], c=colors(count_source))
            count_source += 1
        plt.show()
