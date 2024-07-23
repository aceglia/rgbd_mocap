import shutil
import deeplabcut
import os

if __name__ == '__main__':
    main_path = "Q:\Projet_hand_bike_markerless\RGBD\Training_data"
    main_path =  r"C:\Users\User\Documents\Amedeo"
    #config_tmp_path = "Q:\Projet_hand_bike_markerless\RGBD\Training_data\config_tmp.yaml"
    for participant in ["P10"]:#"P10", "P11"]: #, "P14", "P15"]:
        names = ["_excluded_normal_500_down_b1"]#, "_excluded_hist_eq", "_excluded_hist_eq_sharp"]
        for n in names:
            project_name = participant + n
            project_path = main_path + f"\DLC_projects" + os.sep + project_name
            path_config_file = project_path + '\config.yaml'
            if os.path.exists(project_path + "\exported-models"):
                shutil.rmtree(project_path + "\exported-models", ignore_errors=True)
            deeplabcut.export_model(path_config_file, iteration=None, shuffle=1, trainingsetindex=0, snapshotindex=None,
                                    TFGPUinference=True, overwrite=True, make_tar=True)
            #deeplabcut.evaluate_network(path_config_file, plotting=True)