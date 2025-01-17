from biosiglive import load
import bioviz
import numpy as np


def get_all_file(participants, data_dir, trial_names=None, to_include=(), to_exclude=()):
    all_path = []
    parts = []
    if trial_names and len(trial_names) != len(participants):
        trial_names = [trial_names for _ in participants]
    for p, part in enumerate(participants):
        try:
            all_files = os.listdir(f"{data_dir}{os.sep}{part}")
        except FileNotFoundError:
            print(f"Participant {part} not found in {data_dir}")
            continue
        if trial_names:
            to_include += trial_names[p] if isinstance(trial_names[p], list) else trial_names
        all_files = [
            file
            for file in all_files
            if any([ext in file for ext in to_include]) and not any([ext in file for ext in to_exclude])
        ]
        final_files = [f"{data_dir}{os.sep}{part}{os.sep}{file}" for file in all_files]
        parts.append([part for _ in final_files])
        all_path.append(final_files)
    return sum(all_path, []), sum(parts, [])


if __name__ == "__main__":
    source = ["dlc_technical_marker"]  # , "vicon_markerless"]#, "vicon", "minimal_vicon"]
    participants = [f"P{i}" for i in range(10, 13)]  # , "P15", "P16"]#, "P14", "P15", "P16"]
    participant = participants[0]
    s = source[0]
    # participants.pop(participants.index("P12"))
    import os

    file_path = (
        f"/mnt/shared/Projet_hand_bike_markerless/process_data/{participant}"
        + f"/result_biomech_gear_20_with_technical_marker.bio"
    )

    data = load(file_path)
    end_idx = 1000
    q = data["dlc_1"]["q"][..., :end_idx]
    # q = data["vicon"]["q"][..., :end_idx]
    f_ext = data["shared"]["f_ext"][..., :end_idx]
    prefix = "/mnt/shared/" if os.name == "posix" else r"Q:/"
    # data_dir = f"{prefix}Projet_hand_bike_markerless/optim_params/reference_data"
    model_dir = f"{prefix}Projet_hand_bike_markerless/RGBD/"
    # files, part = get_all_file(participants, data_dir, to_include=["reference_torque_gear_20_with_technical_marker"])
    # data = load(files[0])
    # end_idx = 1000
    # q = data["q_ocp"][..., :end_idx]
    # q_dot = data["q_dot_ocp"][..., :end_idx]
    # tau = data["tau_ocp"][..., :end_idx]
    # f_ext = data["f_ext_ocp"][..., :end_idx]
    model_path = f"{model_dir}{participant}/output_models/gear_20_model_scaled_dlc_technical_marker_params.bioMod"
    import biorbd

    model = biorbd.Model(model_path)
    b = bioviz.Viz(loaded_model=model)
    b.load_movement(q)
    f_ext_mat = np.zeros((1, 6, q.shape[1]))
    for i in range(q.shape[1]):
        B = [0, 0, 0, 1]
        all_jcs = model.allGlobalJCS(q[:, i])
        RT = all_jcs[-1].to_array()
        B = RT @ B
        vecteur_OB = B[:3]
        # f_ext_mat[0, :, i] = f_ext[:3, i] + np.cross(vecteur_OB, f_ext[3:6, i])

        f_ext_mat[0, :3, i] = vecteur_OB
        f_ext_mat[0, 3:, i] = f_ext[3:, i]
        # f_ext_mat[0, 3:, i] = f_ext[:3, i] + np.cross(vecteur_OB, f_ext[3:6, i])
    b.load_experimental_forces(f_ext_mat, segments="ground", normalization_ratio=0.8)
    b.exec()
