import biorbd
import pickle
import bioviz
import time
import numpy as np
from biosiglive import MskFunctions, InverseKinematicsMethods, PlotType, LivePlot

marker_names=["C7", "Scap_AA", "Scap_IA", "Acrom", "Clav_AC", "delt", "arm_l", "epic_l"]
with open("markers_pos.pkl", "rb") as f:
    markers_pos = pickle.load(f)

markers_in_meters = np.asarray(np.concatenate((
    #markers_pos["mark_pos_in_meters"][:, :1],
                                                markers_pos["mark_pos_in_meters"][:, 4:5],
                                                markers_pos["mark_pos_in_meters"][:, 1:4],
                                                markers_pos["mark_pos_in_meters"][:, 6:9]),
                                              axis = 1)[:, :, np.newaxis], dtype=np.float64)
markers_in_meters_ordered = markers_in_meters.copy()
markers_in_meters_ordered[:, 0, :] = markers_in_meters[:, 1, :]

# markers_in_meters = np.asarray(markers_pos["mark_pos_in_meters"][:, :, np.newaxis], dtype=np.float64)
# #repeat the same marker 3 times to have a 3D marker
markers_in_meters = np.repeat(markers_in_meters, 20, axis=2)
#
#
# marker_plot = LivePlot(name="markers", plot_type=PlotType.Scatter3D)
# marker_plot.init()
# while True:
#     marker_plot.update(markers_in_meters[:, :, -1].T, size=0.03)
#     time.sleep(0.001)
biomod_model = "wu.bioMod"
funct = MskFunctions(model="wu.bioMod")
q_recons, _ = funct.compute_inverse_kinematics(markers_in_meters, method=InverseKinematicsMethods.BiorbdLeastSquare)
q_mean = q_recons.mean(axis=1)
print(q_mean[3], q_mean[4], q_mean[5], " xyz ", q_mean[0], q_mean[1], q_mean[2])
b = bioviz.Viz(model_path=biomod_model, show_floor=False)
b.load_movement(q_recons)  # Q from kalman array(nq, nframes)
b.load_experimental_markers(markers_in_meters)  # experimental markers array(3, nmarkers, nframes)
b.exec()

