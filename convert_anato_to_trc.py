import C3DtoTRC
from biosiglive import load, OfflineProcessing
import os
import biorbd
from pathlib import Path
import matplotlib.pyplot as plt

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

def get_model_markers_names(model, names):
    markers = []
    for name in names:
        for i in range(model.nbMarkers()):
            if model.markerNames()[i].to_string().lower().replace("_", "") == name.lower().replace("_", ""):
                markers.append(model.markerNames()[i].to_string())
    return markers


if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    # data_path = "/mnt/shared/Projet_hand_bike_markerless/process_data"
    data_path = "D:\Documents\Programmation\pose_estimation\data_files"
    source = ["depth", "vicon", "minimal_vicon"]
    markers_keys = ["markers_depth_interpolated", "truncated_markers_vicon", "truncated_markers_vicon"]
    names_keys = ["depth_markers_names", "vicon_markers_names", "depth_markers_names"]
    for participant in participants:
        file = fr"{data_path}/{participant}/anato_processed_3_crops.bio"

        if not os.path.isfile(file):
            continue
        print("processing file : ", file)
        load_markers = load(file)
        depth_markers_names = load_markers["depth_markers_names"]
        # idx_ts = depth_markers_names.index("scapts")
        # idx_ai = depth_markers_names.index("scapia")
        # depth_markers_names[idx_ts] = "scapia"
        # depth_markers_names[idx_ai] = "scapts"
        vicon_markers_names = load_markers["vicon_markers_names"]
        # idx_ts = vicon_markers_names.index("scapts")
        # idx_ai = vicon_markers_names.index("scapia")
        # vicon_markers_names[idx_ts] = "scapia"
        # vicon_markers_names[idx_ai] = "scapts"
        vicon_to_depth_idx = _get_vicon_to_depth_idx(depth_markers_names, vicon_markers_names)
        models = ["wu_bras_gauche_depth.bioMod", "wu_bras_gauche_vicon.bioMod", "wu_bras_gauche_depth.bioMod"]
        rate = [120, 120, 120]
        for i in range(len(source)):
            marker_names = load_markers[names_keys[i]]
            markers = load_markers[markers_keys[i]]
            if i == 0:
                markers = OfflineProcessing().butter_lowpass_filter(markers, 4, 120, 4)
            if i == 2:
                markers = markers[:, vicon_to_depth_idx, :]
            markers = markers[:, :-3, :]
            model = biorbd.Model(f"models/{models[i]}")
            marker_names = get_model_markers_names(model, marker_names)
            ia_idx = marker_names.index("SCAP_IA")
            ts_idx = marker_names.index("SCAP_TS")
            marker_names[ia_idx] = "SCAP_TS"
            marker_names[ts_idx] = "SCAP_IA"
            C3DtoTRC.WriteTrcFromMarkersData(
                output_file_path= f"D:\Documents\Programmation\pose_estimation\data_files/{participant}/{Path(file).stem}_{source[i]}.trc",
                markers=markers,
                marker_names=marker_names,
                data_rate=rate[i],
                cam_rate=rate[i],
                n_frames=markers.shape[2],
                start_frame=1,
                units="m",
            ).write()
