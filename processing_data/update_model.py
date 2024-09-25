
def thorax_to_copy(path):
    old = path
    with open(old, "r") as file:
        data_old = file.read()
    to_copy = data_old[:data_old.find("mass")]
    return to_copy


def _fix_model(data, seth_model):
    with open(seth_model, "r") as file:
        data_seth = file.read()
    # update inertia
    first_idx_clav = data.find("inertia\n", data.find("segment clavicle_left\n"))
    end_idx_clav = data.find("com\t", data.find("segment clavicle_left\n"))
    first_idx_seth_clav = data_seth.find("inertia\n", data_seth.find("segment clavicle_left\n"))
    end_idx_seth_clav = data_seth.find("com\t", data_seth.find("segment clavicle_left\n"))
    data = data[:first_idx_clav] + data_seth[first_idx_seth_clav:end_idx_seth_clav] + data[end_idx_clav:]

    first_idx_scap = data.find("inertia\n", data.find("segment scapula_left\n"))
    end_idx_scap = data.find("com\t", data.find("segment scapula_left\n"))
    first_idx_seth_scap = data_seth.find("inertia\n", data_seth.find("segment scapula_left\n"))
    end_idx_seth_scap = data_seth.find("com\t", data_seth.find("segment scapula_left\n"))
    data = data[:first_idx_scap] + data_seth[first_idx_seth_scap:end_idx_seth_scap] + data[end_idx_scap:]

    # update_com
    first_idx_clav = data.find("com\t", data.find("segment clavicle_left\n"))
    end_idx_clav = data.find("meshfile", data.find("segment clavicle_left\n"))
    value = data[first_idx_clav:end_idx_clav].split(" ")[-1][:-len("\n\t\t")]
    value_float = float(value)
    new_value = str(value_float * -1)
    data = data.replace(value, new_value)
    first_idx_scap = data.find("com\t", data.find("segment scapula_left\n"))
    end_idx_scap = data.find("meshfile", data.find("segment scapula_left\n"))
    value = data[first_idx_scap:end_idx_scap].split(" ")[-1][:-len("\n\t\t")]
    value_float = float(value)
    new_value = str(value_float * -1)
    data = data.replace(value, new_value)
    return data


if __name__ == '__main__':
    muscle = False
    thorax = True
    fix_model = True
    participants = [f"P{i}" for i in range(10, 11)]
    # participants.pop(participants.index("P12"))
    source = ["dlc"]#, "vicon", "minimal_vicon"]
    seth_model = "Q:\Projet_hand_bike_markerless\RGBD\wu_bras_gauche_seth_for_cycle.bioMod"
    with open(seth_model, "r") as file:
        data_seth = file.read()
    start_idx = data_seth.find("// MUSCLE DEFINIION")
    muscle_to_copy = data_seth[start_idx:]
    for participant in participants:
        for s in source:
            old_model_path = f"Q:\Projet_hand_bike_markerless\RGBD\{participant}/model_scaled_{s}_test_ribs.bioMod"
            with open(old_model_path, "r") as file:
                data = file.read()
            if thorax:
                data_to_copy = thorax_to_copy(seth_model)
                data = data_to_copy + data[data.find("mass"):]
            if muscle:
                init_idx = data.find("// MUSCLE DEFINIION")
                data = data[:init_idx] + muscle_to_copy
            if fix_model:
                data = _fix_model(data, seth_model)
            with open(old_model_path[:-7] + "_new_seth_param.bioMod", "w") as file:
                file.write(data)