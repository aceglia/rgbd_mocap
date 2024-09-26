import numpy as np

from utils_old import *
from biosiglive import save, load

if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    reload_data = False
    if reload_data:
        all_data, trials = load_results(participants,
                                "/mnt/shared/Projet_hand_bike_markerless/process_data",
                                file_name="normal_500_down_b1_no_root.bio", recompute_cycles=False,
                                        to_exclude=["live_filt"])
        save(all_data, "_all_data_tmp.bio", safe=False)

    else:
        all_data = load("_all_data_tmp.bio")

    source_key = ["dlc_0_8", "dlc_0_9", "dlc_1"]
    all_time_ik = []
    all_time_tracking = []
    all_time_tot = []
    for key in source_key:
        all_colors = []
        time_ik = []
        std_ik = []
        time_tracking = []
        std_tracking = []

        for p, part in enumerate(all_data.keys()):
            for f, file in enumerate(all_data[part].keys()):
                time_ik.append(np.mean(all_data[part][file][key]["time"]["ik"][1:]))
                std_ik.append(np.std(all_data[part][file][key]["time"]["ik"][1:]))
                time_tracking.append(np.mean(all_data[part][file][key]['time']["time_to_get_markers"][1:]))
                std_tracking.append(np.std(all_data[part][file][key]['time']["time_to_get_markers"][1:]))
        all_time_ik.append([np.round(np.mean(time_ik) * 1000, 2) , np.round(np.mean(std_ik) * 1000, 2)])
        all_time_tracking.append([np.round(np.mean(time_tracking) * 1000, 2), np.round(np.mean(std_tracking) * 1000, 2)])
        all_time_tot.append([np.round((np.mean(time_ik) + np.mean(time_tracking))  * 1000, 2),
                             np.round((np.mean(std_ik) + np.mean(std_tracking))*1000, 2)])

    print(r"""
    \begin{table}[h]
    \caption{Mean and standard deviation (SD) of the time taken by each main step.}
    \centering
    \begin{tabular}{lccc}
    \hline
         & & Mean (ms) & SD (ms)\\
         \hline
         """
          "\multirow{3}*{Bony landmarks extraction} & 0.8&" + f" {all_time_tracking[0][0]: .2f}& {all_time_tracking[0][1]: .2f} "  + r"\\" + "\n"                                                                                                           
            " & 0.9&" + f" {all_time_tracking[1][0]: .2f}& {all_time_tracking[1][1]: .2f} "  + r"\\" + "\n"
             "& 1.0   &" + f" {all_time_tracking[2][0]: .2f}& {all_time_tracking[2][1]: .2f}  " + r"\\" + "\n" + r" \hdashline" + "\n"
           "Pose estimation" + "& all&" + f"{all_time_ik[0][0]: .2f}& {all_time_ik[0][1]: .2f}"  + r"\\" + "\n" + r" \hline" + "\n"
           "\multirow{3}*{Total} "
           "& 0.8&" + r"\textbf{" + f"{all_time_tot[0][0]: .2f}" + r"} & \textbf{" + f"{all_time_tot[0][1]: .2f}" + "}" + r"\\" + "\n"
           "& 0.9  &" + r"\textbf{" + f"{all_time_tot[1][0]: .2f}" + r"} & \textbf{" + f"{all_time_tot[1][1]: .2f}" + "}" + r"\\" + "\n"
           "& 1.0    &" +  r"\textbf{" + f"{all_time_tot[2][0]: .2f}" + r"} & \textbf{" + f"{all_time_tot[2][1]: .2f}" + "}" + r"\\" + "\n" + r" \hline" + "\n"
           r"""\end{tabular}
           \label{tab:time}
           \end{table}""")

