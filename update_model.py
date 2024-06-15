
def thorax_to_copy(path):
    old = path
    with open(old, "r") as file:
        data_old = file.read()
    to_copy = data_old[:data_old.find("mass")]
    return to_copy


if __name__ == '__main__':
    muscle = False
    thorax = True
    participants = [f"P{i}" for i in range(15, 16)]
    # participants.pop(participants.index("P12"))
    source = ["dlc"]#, "vicon", "minimal_vicon"]
    seth_model = "F:\markerless_project\P9/model_scaled_minimal_vicon_seth.bioMod"
    with open(seth_model, "r") as file:
        data_seth = file.read()
    start_idx = data_seth.find("// MUSCLE DEFINIION")
    muscle_to_copy = data_seth[start_idx:]
    for participant in participants:
        for s in source:
            old_model_path = f"Q:\Projet_hand_bike_markerless\RGBD\{participant}/model_scaled_{s}.bioMod"
            with open(old_model_path, "r") as file:
                data = file.read()
            if thorax:
                data_to_copy = thorax_to_copy(seth_model)
                data = data_to_copy + data[data.find("mass"):]
            if muscle:
                init_idx = data.find("// MUSCLE DEFINIION")
                data = data[:init_idx] + muscle_to_copy
            with open(old_model_path[:-7] + "_new_seth.bioMod", "w") as file:
                file.write(data)