import json
import numpy as np

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12",  "P13", "P14", "P15", "P16"]
    data_path = "../data_collection_mesurement"
    all_age = []
    all_height = []
    all_weight = []
    for participant in participants:
        with open(f"{data_path}/measurements_{participant}.json") as f:
            data = json.load(f)
            all_age.append(data["age"])
            all_height.append(data["height"])
            all_weight.append(data["weight"])
    print("mean age", np.round(np.mean(all_age), 1), "pm", np.round(np.std(all_age), 1))
    print("mean height", np.round(np.mean(all_height), 1), "pm", np.round(np.std(all_height), 1))
    print("mean weight", np.round(np.mean(all_weight), 1), "pm", np.round(np.std(all_weight), 1))
