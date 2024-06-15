import rerun as rr
import numpy as np
from pyorerun import ModelUpdater, DisplayModelOptions
from biosiglive import load

if __name__ == '__main__':
    data_path = r"Q:\Projet_hand_bike_markerless\RGBD\P10\gear_5_15-01-2024_10_15_46\result_offline_gear_5_15-01-2024_10_15_46_normal_alone.bio"
    data = load(data_path)
    q = data["dlc"]["q"]
    model_path = "Q:\Projet_hand_bike_markerless\RGBD\P10/model_scaled_depth_new_seth.bioMod"
    model = ModelUpdater.from_file(model_path)
    display_option = DisplayModelOptions()
    display_option.mesh_color = (77, 77, 255)
    model.options = display_option

    rr.init("my_thing", spawn=True)
    rr.set_time_sequence(timeline="step", sequence=0)
    for i in range(q.shape[1]):
        model.to_rerun(q[:, i])
        # rr.log("anything", rr.Anything())