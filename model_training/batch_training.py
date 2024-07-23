import deeplabcut
import os
import yaml
import shutil
from pathlib import Path
import glob
import numpy as np


def modify_config_file(tmp_config, project_path):
    with open(tmp_config) as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    #data["project_path"] = r'\\10.89.24.15\q' + os.sep + project_path[3:]
    data["project_path"] = project_path
    # list_past_iter = glob.glob(project_path + r"\dlc-models\iteration-0\testApr23-trainset95shuffle1\train\**.index")
    # list_idx = []
    # for file in list_past_iter:
    #     list_idx.append(int(Path(file).stem.split("-")[-1]))
    # data["snapshotindex"] = -1 if len(list_idx) < 1 else np.sort(list_idx)[-1]
    with open(project_path + '\config.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    return project_path + '\config.yaml'


def modify_pose_cfg(pose_path, from_last_iter=False, max_iter_init=1000000):
    with open(pose_path) as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if from_last_iter:
        data_step = [5000, 100000, 200000, 300000]
        train_path = str(Path(pose_path).parent)
        list_past_iter = glob.glob(train_path + "\**.index")
        list_idx = []
        for file in list_past_iter:
            list_idx.append(int(Path(file).stem.split("-")[-1]))
        last_iter = np.sort(list_idx)[-1]
        data["init_weights"] = str(Path(glob.glob(train_path + f"\**{last_iter}.index")[0]))[:-len(".index")]
        max_iter = max_iter_init - last_iter
    else:
        data["convolution"]["embossratio"] = 0
        data["contrast"]["clahe"] = False
        data["crop_size"] = [400, 400]
        data["crop_sampling"] = "density"
        data["pos_dist_thresh"] = 9
        data["contrast"]["histeq"] = False
        data["batch_size"] = 1
        data["global_scale"] = 1
        data["covering"] = False
        data["elastic_transform"] = False
        data["motion_blur"] = False
        data["scale_jitter_lo"] = 0.8
        data["scale_jitter_up"] = 1.1
        max_iter = max_iter_init
        #data_step = [10000, 430000, 730000, 1030000]
        data_step = [5000, 100000, 200000, 300000]
    # data["rotratio"] = 0.0
    # data["rotation"] = False
    # data["cropratio"] = 0

    # data["apply_prob"] = 0
    #data["multi_step"] = [[0.005, data_step[0]], [0.02, data_step[1]], [0.002, data_step[2]], [0.001, data_step[3]]]
    with open(pose_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    return max_iter


if __name__ == '__main__':
    import time
    main_path = "Q:\Projet_hand_bike_markerless\RGBD\Training_data"
    local_path = r"C:\Users\User\Documents\Amedeo"
    config_tmp_path = "Q:\Projet_hand_bike_markerless\RGBD\Training_data\config_tmp.yaml"
    for participant in [f"P{idx}" for idx in list(range(16, 17))]:
        file_names = ["_excluded_normal_500_down_b1"]
        for file_name in file_names:
            tic = time.time()
            project_name = participant + file_name
            project_path = local_path + f"\DLC_projects" + os.sep + project_name
            # if os.path.isdir(project_path):
            #    shutil.rmtree(project_path, ignore_errors=True)
            if not os.path.isdir(project_path):
                os.makedirs(project_path)
            if len(os.listdir(project_path)) == 0:
                os.mkdir(project_path + "\dlc-models")
                os.mkdir(project_path + "\labeled-data")
                os.mkdir(project_path + r"\training-datasets")
                os.mkdir(project_path + r"\videos")
            # don't edit these:
            video_file_path = [project_path + os.sep + project_name + '/videos/']
            path_config_file = modify_config_file(config_tmp_path, project_path)
            if len(os.listdir(project_path + r"\labeled-data")) == 0:
                shutil.copytree(main_path + f"\{participant}{file_name[:-3]}", project_path + r"\labeled-data\data_test")
            if not os.path.exists(project_path + r"\labeled-data\annotated_images_aug"):
                os.mkdir(project_path + r"\labeled-data\annotated_images_aug")
                dist = project_path + r"\labeled-data\annotated_images_aug\CollectedData_Ame.csv"
                src = project_path + r"\labeled-data\data_test\CollectedData_Ame.csv"
                shutil.copy2(src, dist)
            if len(glob.glob(project_path + r"\labeled-data\annotated_images_aug\**.h5")) == 0:
                deeplabcut.convertcsv2h5(path_config_file, userfeedback=False)
                # src = project_path + r"\labeled-data\annotated_images_aug\CollectedData_ame.h5"
                # dist = project_path + r"\labeled-data\data_test\CollectedData_ame.h5"
                # shutil.copy2(src, dist)
            if not os.path.exists(project_path + "\dlc-models\iteration-0"):
                deeplabcut.create_training_dataset(path_config_file, net_type='mobilenet_v2_0.35', augmenter_type='imgaug', )
                from_last_iter = False
            else:
                from_last_iter = True
            pose_config_path = glob.glob(project_path + "\dlc-models\iteration-0\**")[0] + r"\train\pose_cfg.yaml"
            max_iter = modify_pose_cfg(pose_config_path, from_last_iter, max_iter_init=600000)
            deeplabcut.train_network(path_config_file, shuffle=1, displayiters=1000, saveiters=50000, maxiters=max_iter)
            print("time to train the network was:", time.time() - tic)
