import biorbd
import pickle
try:
    import bioviz
    from pyomeca import Markers
except:
    pass
import time
import numpy as np
from c3dtotrc import WriteTrcFromMarkersData
from biosiglive import MskFunctions, InverseKinematicsMethods, PlotType, LivePlot, load

suffix = "06-07-2023_18_17_59"

# marker_names=["C7", "Scap_AA", "Scap_IA", "Acrom", "Clav_AC", "delt", "arm_l", "epic_l", "styl_u", "styl_r", "h_up", "h_down"]
file_name = "markers_gear_20_09-08-2023_17_07_42.bio"
participant = "P2_session2"
data = load(f"data_files\{participant}\{file_name}")
markers_pos = data["markers_in_meters"]
markers_names = data["markers_names"]
# markers_names = ['test:T5', 'test:C7', 'test:Ribs', 'test:clav_ac',
#        'test:scapaa','test:scapia', 'test:acrom', 'test:delt', 'test:larm', 'test:epicl',
#        'test:larm_l', 'test:styl_r', 'test:styl_l']
# suffix = "06-07-2023_18_17_59"
# markers_vicon = Markers.from_c3d(filename=f"Pedalage_{suffix}.c3d", usecols=markers_names)
# markers_vicon = markers_vicon.values[:, :, 70:-80] * 0.001
#
markers_names = ["T5", "C7",
                 'RIBS_r',
                 "CLAV_AC",
                  # 'RIBS_l',
                 "SCAP_AA",
                 "SCAP_IA",
                 "Acrom",
                 "DELT",
                 "ARMl",
                 "EPICl",
                 "larm_l",
                 "STYLr",
                 "STYLu"]
#
WriteTrcFromMarkersData(output_file_path =f"data_files\{participant}\{file_name[:-4]}.trc",
                        markers = markers_pos,
                        marker_names=markers_names,
                        data_rate=60,
                        cam_rate=60,
                        n_frames=markers_pos.shape[2],
                        start_frame=1,
                        units="m").write()

# WriteTrcFromMarkersData(output_file_path =f"markers_vicon_{suffix}.trc",
#                         markers=markers_vicon,
#                         marker_names=markers_names,
#                         data_rate=100,
#                         cam_rate=100,
#                         n_frames=markers_vicon.shape[2],
#                         start_frame=1,
#                         units="m").write()

raise ValueError
# markers_in_meters = np.asarray(np.concatenate((
#     #markers_pos["mark_pos_in_meters"][:, :1],
#                                                 markers_pos["markers_in_meters"][:, :],
#                                                 markers_pos["markers_in_meters"][:, :],
#                                                 markers_pos["markers_in_meters"][:, :]),
#                                               axis = 1)[:, :, np.newaxis], dtype=np.float64)
markers_in_meters = markers_pos
markers_in_meters_ordered = markers_in_meters.copy()
markers_in_meters_ordered[:, 10, :] = markers_in_meters[:, 12, :]
markers_in_meters_ordered[:, 11, :] = markers_in_meters[:, 10, :]
markers_in_meters_ordered[:, 12, :] = markers_in_meters[:, 11, :]

#repeat the same marker 3 times to have a 3D marker
# markers_in_meters = np.repeat(markers_in_meters, 20, axis=2)
#
#
# marker_plot = LivePlot(name="markers", plot_type=PlotType.Scatter3D)
# marker_plot.init()
# while True:
#     marker_plot.update(markers_in_meters[:, :, -1].T, size=0.03)
#     time.sleep(0.001)
# biomod_model = "kinematic_model_07-06-2023_14_47_31.bioMod"
biomod_model = "wu_left_est_pos.bioMod"
funct = MskFunctions(model=biomod_model, data_buffer_size=markers_in_meters_ordered.shape[2])
q_recons, _ = funct.compute_inverse_kinematics(markers_in_meters_ordered[:, :, :], method=InverseKinematicsMethods.BiorbdLeastSquare, kalman_freq=100)
q_mean = q_recons[:, 0]
print(q_mean[3], q_mean[4], q_mean[5], " xyz ", q_mean[0], q_mean[1], q_mean[2])
# with open(biomod_model, 'r') as file:
#     data = file.read()
# # replace the target string
# data = data.replace('shoulder\n\tRT -0.000 0.000 -0.000 xyz 0.000 0.000 0.000',
#                     f'shoulder\n\tRT {q_mean[3, 0]} {q_mean[4, 0]} {q_mean[5, 0]} xyz {q_mean[0, 0]} {q_mean[1, 0]} {q_mean[2, 0]}')
# with open(biomod_model, 'w') as file:
#     file.write(data)

b = bioviz.Viz(model_path=biomod_model, show_floor=False)
b.load_movement(q_recons)  # Q from kalman array(nq, nframes)
b.load_experimental_markers(markers_in_meters_ordered)  # experimental markers array(3, nmarkers, nframes)
b.exec()

