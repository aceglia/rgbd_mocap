

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    source = ["depth", "vicon", "minimal_vicon"]
    seth_model = "/mnt/shared/Projet_hand_bike_markerless/RGBD/model_scaled_minimal_vicon_seth.bioMod"
    with open(seth_model, "r") as file:
        data_seth = file.read()
    start_idx = data_seth.find("// MUSCLE DEFINIION")
    data_to_copy = data_seth[start_idx:]
    for participant in participants:
        for s in source:
            old_model_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{participant}/model_scaled_{s}.bioMod"
            with open(old_model_path, "r") as file:
                data = file.read()
            init_idx = data.find("// MUSCLE DEFINIION")
            data = data[:init_idx] + data_to_copy
            with open(old_model_path[:-7] + "_seth.bioMod", "w") as file:
                file.write(data)