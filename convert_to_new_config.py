import json
import os
import glob
from pathlib import Path

if __name__ == '__main__':
    participants = ["P10" ] #, "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    data_files = r"D:\Documents\Programmation\pose_estimation\data_files"
    for participant in participants:
        files = os.listdir(f"{data_files}{os.sep}{participant}")
        # files = [file for file in files if file[:7] == "gear_15"]
        files = [file for file in files if "gear" in file and os.path.isdir(f"{data_files}{os.sep}{participant}{os.sep}" + file)
                 ]
        for file in files:
            path = f"{data_files}{os.sep}{participant}{os.sep}{file}{os.sep}tracking_config.json"
            if os.path.isfile(path):
                with open(path, "r") as f:
                    try:
                        conf = json.load(f)
                    except json.decoder.JSONDecodeError:
                        # in case of empty file
                        continue
            else:
                continue

            all_files = glob.glob(f"{data_files}{os.sep}{participant}{os.sep}{file}{os.sep}color**.png")
            # if len(all_files) < 50:
            #     continue
            # check where there is a gap in the numbering
            idx = []
            for f in all_files:
                idx.append(int(f.split("\\")[-1].split("_")[-1].removesuffix(".png")))
            idx.sort()
            dic = {'directory': f"{data_files}{os.sep}{participant}{os.sep}{file}",
            'start_index': idx[conf["start_frame"]],
            'end_index': idx[-1],
            'crops': [],}
            for i in range(len(conf["start_crop"])):
                if conf["mask_params"][i]["min_dist"] == 0:
                    conf["mask_params"][i]["min_dist"] = 0.01
                dic[f"crops"].append(
                    {
                        "name": f"crop_{i}",
                        "area": [
                            conf["start_crop"][0][i],
                            conf["start_crop"][1][i],
                            conf["end_crop"][0][i],
                            conf["end_crop"][1][i]
                        ],
                        "filters": {
                            "blend": 100,
                            "white_range": [
                                conf["mask_params"][i]["min_threshold"],
                                conf["mask_params"][i]["max_threshold"]
                            ],
                            "blob_area": [
                                conf["mask_params"][i]["min_area"],
                                conf["mask_params"][i]["max_area"]
                            ],
                            "convexity": int(conf["mask_params"][i]["convexity"] * 100),
                            "circularity": int(conf["mask_params"][i]["circularity"] * 100),
                            "distance_between_blobs": 1,
                            "distance_in_centimeters": [
                                conf["mask_params"][i]["min_dist"] * 100,
                                conf["mask_params"][i]["clipping_distance_in_meters"] * 100
                            ],
                            "clahe_clip_limit": conf["mask_params"][i]["clahe_clip_limit"],
                            "clahe_grid_size": conf["mask_params"][i]["clahe_autre"],
                            # "gaussian_blur": conf["mask_params"][i]["blur"],
                            "gaussian_blur": 0,

                            "use_contour": conf["mask_params"][i]["use_contour"],
                            "mask": None,
                            "white_option": True,
                            "blob_option": True,
                            "clahe_option": True,
                            "distance_option": True,
                            "masks_option": False
                            ,
                        },
                    }
                )
                markers = []
                for key in conf["first_frame_markers"][i].keys():
                    markers.append(
                        {
                            "name": key,
                            "pos": [conf["first_frame_markers"][i][key][0][1],
                                    conf["first_frame_markers"][i][key][0][0]],
                        }
                    )
                dic[f"crops"][-1]["markers"] = markers
            with open(path[:-5] + "_gui.json", "w") as f:
                json.dump(dic, f, indent=4)
            print(f"File {file} has been converted")




