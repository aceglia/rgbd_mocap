import numpy as np


def recursive_find_scale(model_data):
    scale_list = []
    start_index = 0
    delta_start = len("scale\t")
    while True:
        data_idx = model_data[start_index:].find("scale\t")
        data_start = data_idx+ delta_start
        if data_idx == -1:
            break
        end_data = model_data[start_index + data_start:].find("\n")
        global_end_data = start_index + data_start + end_data
        scale_factors = model_data[start_index + data_start:global_end_data].split(" ")
        scale_factors = [np.round(float(scale), 2) for scale in scale_factors]
        scale_list.append(scale_factors)
        start_index = global_end_data
    return scale_list


if __name__ == '__main__':
    participants = ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
    sources = ["depth", "vicon", "minimal_vicon"]
    model_dir = "/mnt/shared/Projet_hand_bike_markerless/RGBD"
    scale_factors_total = []
    for s, source in enumerate(sources):
        finale_scale = [0, 0, 0, 0, 0, 0]
        for participant in participants:
            model_file = f"{model_dir}/{participant}/model_scaled_{source}.bioMod"
            with open(model_file, "r") as file:
                model_data = file.read()
            scale_factors = recursive_find_scale(model_data)
            s_factors = [np.mean(scale) for scale in scale_factors]
            for i in range(len(finale_scale)):
                finale_scale[i] += s_factors[i]
        scale_factors_total.append([finale_scale[i]/len(participants) for i in range(len(finale_scale))] )
    # latex table:
    print(r"""
    \begin{table}
    \caption
    {Scaling factor(mean along all axis) by segment for the three methods.}
    \centering
    \begin
    {tabular}[c]
    {l | c | c | c}
    &  \multicolumn{3}{c}{Scaling factor} \\
     &  \multirow{2} * {RGBD} & \multicolumn{2}{c}{Vicon} \\
    Segment & & redundant & minimal \\ \hline
"""
f"Thorax & {np.round(scale_factors_total[0][0], 2)} & {np.round(scale_factors_total[1][0], 2)} & {np.round(scale_factors_total[2][0], 2)} "
          r" \\ \hline "
f"\nClavicle & {np.round(scale_factors_total[0][1], 2)} & {np.round(scale_factors_total[1][1], 2)} & {np.round(scale_factors_total[2][1], 2)} "
          r"\\ \hline" 
f"\nScapula & {np.round(scale_factors_total[0][2], 2)} & {np.round(scale_factors_total[1][2], 2)} & {np.round(scale_factors_total[2][2], 2)}"
          r"\\ \hline"
f"\nHumerus & {np.round(scale_factors_total[0][3], 2)} & {np.round(scale_factors_total[1][3], 2)} & {np.round(scale_factors_total[2][3], 2)}"
          r"\\ \hline"
f"\nForearm & {np.round(scale_factors_total[0][4], 2)} & {np.round(scale_factors_total[1][4], 2)} & {np.round(scale_factors_total[2][4], 2)} "
          r"\\ \hline"
r"""
\end
{tabular}
\label
{tab_delay}
\end
{table}
    """)



