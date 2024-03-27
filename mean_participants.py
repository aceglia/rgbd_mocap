import json
import numpy as np

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12",  "P13", "P14", "P15", "P16"]
    data_path = "data_collection_mesurement"
    all_age = []
    all_height = []
    all_weight = []
    for participant in participants:
        with open(f"{data_path}/measurements_{participant}.json") as f:
            data = json.load(f)
            all_age.append(data["age"])
            all_height.append(data["height"])
            all_weight.append(data["weight"])
    print("mean age", np.mean(all_age), "pm", np.std(all_age))
    print("mean height", np.mean(all_height), "pm", np.std(all_height))
    print("mean weight", np.mean(all_weight), "pm", np.std(all_weight))
