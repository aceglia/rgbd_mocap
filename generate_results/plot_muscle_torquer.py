import numpy as np
import matplotlib.pyplot as plt
import biorbd
from utils import load_results
from math import ceil


if __name__ == '__main__':
    participants = ["P10"]
    trials = [["gear_20"]] * len(participants)
    all_data, _ = load_results(participants,
                            "/mnt/shared/Projet_hand_bike_markerless/process_data",
                            trials, file_name="_seth"
                               # , to_exclude="full"
                               )
    all_errors_minimal = []
    all_errors_vicon = []
    count = 0
    cycle = False
    for part in all_data.keys():
        for f, file in enumerate(all_data[part].keys()):
            joints_names = ["Pro/retraction", "Depression/Elevation",
                            "Pro/retraction", "Lateral/medial rotation", "Tilt",
                            "Plane of elevation", "Elevation", "Axial rotation",
                            "Flexion/extension", "Pronation/supination"]
            trial_name = trials[f][0]
            all_results = all_data[part][file]
            sources = []
            results_from_sources = []
            for key in all_results.keys():
                sources.append(key)
                results_from_sources.append(all_results[key]) if not cycle else results_from_sources.append(all_results[key]["cycles"])
                print(f"mean time for source: {key} ", np.mean(all_results[key]["time"]["tot"][1:]))

            models = [f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_depth_seth.bioMod",
                      f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_vicon_seth.bioMod",
                      f"/mnt/shared/Projet_hand_bike_markerless/process_data/{part}/models/{trial_name}_processed_3_model_scaled_minimal_vicon_seth.bioMod"]
            bio_models = [biorbd.Model(models[0]), biorbd.Model(models[1]), biorbd.Model(models[2])]
            all_mjt = []
            color = ["b", "r", "g"]
            line = ["-", "-", "-"]

            def muscle_joint_torque(model, activations_fct, q_fct, qdot_fct):
                muscles_states = model.stateSet()
                for k in range(model.nbMuscles()):
                    muscles_states[k].setActivation(activations_fct[k])
                return model.muscularJointTorque(muscles_states, q_fct, qdot_fct).to_array()

            for k in range(len(results_from_sources)):
                mjt_tmp = np.zeros((results_from_sources[k]["q"].shape[0], results_from_sources[k]["q"].shape[1]))
                for i in range(results_from_sources[k]["q"].shape[1]):
                    mjt_tmp[:, i] = muscle_joint_torque(bio_models[k], results_from_sources[k]["mus_act"][:, i],
                                                        results_from_sources[k]["q"][:, i],
                                                        results_from_sources[k]["q_dot"][:, i])
                all_mjt.append(mjt_tmp)

            # plot muscle activations
            plt.figure("muscle torque")
            for i in range(results_from_sources[0]["tau"].shape[0]):
                plt.subplot(4, ceil(results_from_sources[0]["tau"].shape[0] / 4), i + 1)
                for k in range(len(results_from_sources)):
                    plt.plot(all_mjt[k][i, :] + results_from_sources[k]["res_tau"][i, :], line[k], color=color[k])
                    plt.plot(results_from_sources[k]["tau"][i, :], "--", color=color[k])
                plt.legend(sources)
        plt.show()