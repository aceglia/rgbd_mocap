import C3DtoTRC
from biosiglive import load, OfflineProcessing
import os
import biorbd
from pathlib import Path
import matplotlib.pyplot as plt

from utils import _reorder_markers_from_names


def _convert_string(string):
    return string.lower().replace("_", "")


def _get_vicon_to_depth_idx(names_depth=None, names_vicon=None):
    vicon_markers_names = [_convert_string(name) for name in names_vicon]
    depth_markers_names = [_convert_string(name) for name in names_depth]
    vicon_to_depth_idx = []
    for name in vicon_markers_names:
        if name in depth_markers_names:
            vicon_to_depth_idx.append(vicon_markers_names.index(name))
    return vicon_to_depth_idx


def get_model_markers_names(model, names=None):
    if names:
        markers = []
        for name in names:
            for i in range(model.nbMarkers()):
                if model.markerNames()[i].to_string().lower().replace("_", "") == name.lower().replace("_", ""):
                    markers.append(model.markerNames()[i].to_string())
    else:
        markers = []
        for i in range(model.nbMarkers()):
            markers.append(model.markerNames()[i].to_string())
    return markers


if __name__ == "__main__":
    participants = [f"P{i}" for i in range(10, 11)]
    # participants.pop(participants.index("P14"))

    data_path = "Q://Projet_hand_bike_markerless/RGBD"
    model_source = ["dlc_ribs", "vicon"]
    sources = ["dlc_1"]  # , "vicon"]
    for participant in participants:
        all_files = os.listdir(rf"{data_path}\{participant}")
        all_files = [file for file in all_files if "gear" in file]
        for file in all_files:
            model = "normal_500_down_b1"
            filt = "filtered"
            file_tmp = f"Q://Projet_hand_bike_markerless/process_data/{participant}/result_biomech_{file.split('_')[0]}_{file.split('_')[1]}_{model}_no_root.bio"
            if not os.path.isfile(file_tmp):
                continue
            print("processing file : ", file_tmp)
            data = load(file_tmp)
            rate = 120
            for s, source in enumerate(sources):
                if source not in data.keys():
                    raise ValueError(f"source {source} not in data keys")
                model_path = f"Q://Projet_hand_bike_markerless/RGBD/P10/model_scaled_{model_source[s]}.bioMod"
                import numpy as np

                markers = data[source]["tracked_markers"][..., :1000]
                if source == "vicon":
                    markers = np.nan_to_num(markers, nan=0.0)
                model = biorbd.Model(model_path)
                marker_names = get_model_markers_names(model)
                output_path = f"Q://Projet_hand_bike_markerless/RGBD\{participant}/{file}/{Path(file).stem}_{source}_ribs_and_cluster.trc"
                if os.path.exists(output_path):
                    os.remove(output_path)
                ordered_names = [_convert_string(name) for name in marker_names]
                markers_ordered = _reorder_markers_from_names(markers, ordered_names, data[source]["marker_names"])

                C3DtoTRC.WriteTrcFromMarkersData(
                    output_file_path=output_path,
                    markers=markers_ordered,
                    marker_names=marker_names,
                    data_rate=rate,
                    cam_rate=rate,
                    n_frames=markers.shape[2],
                    start_frame=1,
                    units="m",
                ).write()
                print(f"File {output_path} written")
