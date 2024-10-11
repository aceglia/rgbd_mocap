try:
    import matplotlib.pyplot as plt
except:
    pass
import scipy.signal as signal
from scipy.interpolate import interp1d
from biosiglive.processing.data_processing import OfflineProcessing, RealTimeProcessing
from biosiglive import load
from data_processing.post_process_data import ProcessData
from kalma_dlc import KalmanFilterPredictor
from rgbd_mocap.tracking.kalman import Kalman
import utils_old
import numpy as np


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def _convert_cluster_to_anato(new_cluster, data):
    anato_pos = new_cluster.process(
        marker_cluster_positions=data, cluster_marker_names=["M1", "M2", "M3"], save_file=False
    )
    # anato_pos_ordered = np.zeros_like(anato_pos)
    # anato_pos_ordered[:, 0, :] = anato_pos[:, 0, :]
    # anato_pos_ordered[:, 1, :] = anato_pos[:, 2, :]
    # anato_pos_ordered[:, 2, :] = anato_pos[:, 1, :]
    return anato_pos


def _interpolate_data(markers_depth, shape):
    new_markers_depth_int = np.zeros((3, markers_depth.shape[1], shape))
    for i in range(3):
        x = np.linspace(0, 100, markers_depth.shape[2])
        f_mark = interp1d(x, markers_depth[i, :, :])
        x_new = np.linspace(0, 100, int(new_markers_depth_int.shape[2]))
        new_markers_depth_int[i, :, :] = f_mark(x_new)
    return new_markers_depth_int


if __name__ == "__main__":
    participants = ["P9"]
    trials = [["gear_10"]]
    # all_data, trials = load_all_data(participants,
    #                 "/mnt/shared/Projet_hand_bike_markerless/process_data",
    #                                  trials
    #                                 )
    import json, os
    from scapula_cluster.from_cluster_to_anato import ScapulaCluster

    measurements_dir_path = "../data_collection_mesurement"
    calibration_matrix_dir = "../../scapula_cluster/calibration_matrix"
    measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_P9.json"))
    measurements = measurement_data[f"with_depth"]["measure"]
    calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[f"with_depth"]["calibration_matrix_name"]
    new_cluster = ScapulaCluster(
        measurements[0],
        measurements[1],
        measurements[2],
        measurements[3],
        measurements[4],
        measurements[5],
        calibration_matrix,
    )
    new_cluster_dlc = ScapulaCluster(
        measurements[0],
        measurements[1],
        measurements[2],
        measurements[3],
        measurements[4],
        measurements[5],
        calibration_matrix,
    )
    new_cluster_vicon = ScapulaCluster(
        measurements[0],
        measurements[1],
        measurements[2],
        measurements[3],
        measurements[4],
        measurements[5],
        calibration_matrix,
    )

    n_window = 7
    for participant in participants:
        for trial in trials:
            # markers = all_data[participant][trial]["markers_depth_interpolated"]
            part = "P14"
            trial = "gear_20_22-01-2024_16_29_55"
            image_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial}"

            # model =
            dlc_data_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial}{os.sep}marker_pos_multi_proc_3_crops_normal_500_down_b1_ribs_and_cluster_1_pp.bio"
            dlc_data, _, names, frame_idx = utils.load_data_from_dlc(None, dlc_data_path, part)
            names = (
                names[: names.index("clavac") + 1] + ["scapaa", "scapia", "scapts"] + names[names.index("clavac") + 1 :]
            )
            trials = [f"/mnt/shared/Projet_hand_bike_markerless/RGBD/{part}/{trial}/reoriented_dlc_markers.bio"]
            trials = [
                "/mnt/shared/Projet_hand_bike_markerless/process_data/P14/result_biomech_gear_20_normal_times_three_filtered_all_dofs.bio"
            ]
            data = load(trials[0])
            n_final = 1000
            dlc_data = dlc_data[..., :n_final]
            frame_idx = frame_idx[:n_final]
            anato_from_cluster = _convert_cluster_to_anato(new_cluster, dlc_data[:, -3:, :] * 1000) * 0.001
            anato_tmp = anato_from_cluster.copy()
            idx_cluster = names.index("clavac")
            dlc_data_tmp = np.concatenate(
                (dlc_data[:, : idx_cluster + 1, :], anato_from_cluster[:3, ...], dlc_data[:, idx_cluster + 1 :, :]),
                axis=1,
            )
            new_markers_dlc = ProcessData()._fill_and_interpolate(
                data=dlc_data_tmp, idx=frame_idx, shape=frame_idx[-1] - frame_idx[0], fill=True
            )
            # new_markers_dlc_filtered = new_markers_dlc
            new_markers_dlc_filtered = np.zeros((3, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
            off_line_processing = [
                RealTimeProcessing(60, 150),
                RealTimeProcessing(60, 150),
                RealTimeProcessing(60, 150),
            ]
            for i in range(3):
                new_markers_dlc_filtered[i, :8, :] = OfflineProcessing().butter_lowpass_filter(
                    new_markers_dlc[i, :8, :], 3, 60, 2
                )
                new_markers_dlc_filtered[i, 8:, :] = OfflineProcessing().butter_lowpass_filter(
                    new_markers_dlc[i, 8:, :], 10, 60, 2
                )

            def signaltonoise_dB(a, axis=0, ddof=0):
                a = np.asanyarray(a)
                m = a.mean(axis)
                sd = a.std(axis=axis, ddof=ddof)
                return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))

            snr = signaltonoise(new_markers_dlc_filtered)
            rate = 120
            all_kalman = None
            dlc_kalman = np.zeros((dlc_data.shape[0], dlc_data.shape[1] + 3, dlc_data.shape[2] + 1))
            dlc_kalman_origin = np.zeros(
                (new_markers_dlc.shape[0], new_markers_dlc.shape[1], new_markers_dlc.shape[2] + 1)
            )
            dlc_filt_filt = np.zeros((new_markers_dlc.shape[0], new_markers_dlc.shape[1], new_markers_dlc.shape[2] + 1))

            count = 0
            count_bis = 0
            import time

            markers_dlc_hom = np.ones((4, dlc_kalman.shape[1], 1))
            rt = data["rt_matrix"]
            all_time = []
            kalman_dlc_predictor = KalmanFilterPredictor(fps=60, adapt=False, forward=1 / 60)
            dlc_kalman_origin[:2, :14, 0] = kalman_dlc_predictor.process(dlc_data[:2, :14, 0].T).T
            idx = None
            for i in range(frame_idx[0], frame_idx[-1]):
                if count_bis == n_final:
                    break
                tic = time.time()
                if i in frame_idx:
                    dlc_data_tmp = dlc_data[:, :, count : count + 1]
                    # if count_bis == 0:
                    #     mark, idx = _reorder_markers_from_names(dlc_data_tmp, names, names)
                    # else:
                    #     dlc_data_tmp = dlc_data_tmp[:, idx, :]
                    # anato_from_cluster = _convert_cluster_to_anato(new_cluster_dlc, dlc_data[:, -3:, count:count+1] * 1000) * 0.001
                    # anato_tmp = anato_from_cluster.copy()
                    # anato_from_cluster[:, 0, :] = anato_tmp[:, 0, :]
                    # anato_from_cluster[:, 1, :] = anato_tmp[:, 2, :]
                    # anato_from_cluster[:, 2, :] = anato_tmp[:, 1, :]
                    # dlc_data_tmp = np.concatenate((dlc_data[..., count:count+1], anato_from_cluster[:3, ...]), axis=1)
                else:
                    dlc_data_tmp = None

                def get_next_frame_from_kalman(
                    kalman_instance=None,
                    markers_data=None,
                    scapula_cluster=None,
                    params=None,
                    measurement_noise_factor=100,
                    process_noise_factor=5,
                    error_cov_post_factor=0,
                    error_cov_pre_factor=0,
                    rt_matrix=None,
                    n_markers=None,
                    idx_cluster=None,
                ):
                    dt_forward = params[-1]
                    if n_markers is None:
                        if kalman_instance is not None:
                            n_markers = len(kalman_instance)
                        elif markers_data is not None:
                            n_markers = markers_data.shape[1]
                        else:
                            raise ValueError("Impossible to know how many markers there are.")

                    next_frame = np.zeros((3, n_markers + 3, 1))
                    kalman_instance = [None] * n_markers if kalman_instance is None else kalman_instance
                    if markers_data is not None:
                        anato_from_cluster = (
                            _convert_cluster_to_anato(scapula_cluster, markers_data[:, -3:, :] * 1000) * 0.001
                        )
                        markers_data = np.concatenate(
                            (
                                markers_data[:, : idx_cluster + 1, :],
                                anato_from_cluster[:3, ...],
                                markers_data[:, idx_cluster + 1 :, :],
                            ),
                            axis=1,
                        )
                    for k in range(n_markers):
                        if kalman_instance[k] is None and markers_data is not None:
                            if params:
                                measurement_noise_factor = params[: int(markers_data.shape[1])][k]
                                process_noise_factor = params[
                                    int(markers_data.shape[1]) : int(markers_data.shape[1] * 2)
                                ][k]
                                # error_cov_post_factor = \
                                # params[int(markers_data.shape[1] * 2):int(markers_data.shape[1] * 3)][k]
                                # error_cov_pre_factor = params[int(markers_data.shape[1] * 3):-1][k]
                            kalman_instance[k] = Kalman(
                                markers_data[:, k, 0],
                                n_measures=3,
                                n_states=6,
                                fps=60,
                                measurement_noise_factor=measurement_noise_factor,
                                process_noise_factor=process_noise_factor,
                                # error_cov_post_factor=error_cov_post_factor,
                                # error_cov_pre_factor=error_cov_pre_factor
                            )
                            next_frame[:, k, 0] = kalman_instance[k].get_future_pose(dt=dt_forward)
                        elif kalman_instance[k] is not None:

                            if markers_data is not None:
                                kalman_instance[k].correct(markers_data[:, k, 0])
                            next_frame[:, k, 0] = kalman_instance[k].predict()
                            next_frame[:, k, 0] = kalman_instance[k].get_future_pose(dt=dt_forward)

                        else:
                            raise ValueError("Unexpected error.")

                    if rt_matrix is not None:
                        markers_dlc_hom = np.ones((4, next_frame.shape[1], 1))
                        markers_dlc_hom[:3, :, 0] = next_frame[..., 0]
                        next_frame = np.dot(np.array(rt_matrix), markers_dlc_hom[:, :, 0])
                    return next_frame[..., 0], kalman_instance

                # if all_kalman is None:
                #     dlc_kalman[:, :, count] = np.concatenate((dlc_data_tmp[:, :, 0], np.zeros((3, 3))), axis=1)
                # dlc_kalman[:, :, count_bis + 1], all_kalman = get_next_frame_from_kalman(
                #     all_kalman, dlc_data_tmp, new_cluster_dlc, 15,
                #     0.3, 1, 0, None)
                params = [
                    1.638819672361114,
                    1.9736804302698414,
                    1.9995932412272837,
                    1.997539532071001,
                    1.9633889310240709,
                    1.947695851912196,
                    1.9983647882075208,
                    1.998880629129683,
                    1.0807187795620514,
                    4.207208092362829e-09,
                    3.5408032064211595e-08,
                    2.123146633494158e-07,
                    2.0224394293459974e-08,
                    2.0433951366906184e-07,
                    0.3279000418749911,
                    0.20710480248051769,
                    0.8314056671836689,
                    0.9645683465216266,
                    0.330460439129379,
                    0.31702080456872317,
                    0.3089735874774765,
                    0.31183420065408335,
                    0.5073711499437568,
                    0.07494298104933172,
                    0.08953478141229149,
                    1.8994140162349553,
                    1.9487702387110173,
                    1.9915625624652324,
                    1.9450477411601077,
                    1.9947866648465944,
                    1.9988631829827175,
                    1.9248515992949242,
                    1.9997968625673925,
                    1.384130678267083,
                    0.5557883657741526,
                    0.5989681804784023,
                    2.24320551786683e-06,
                    0.009263235812780353,
                    1.0413250682013571e-07,
                    5.233418799794384e-05,
                    7.510003165591523e-07,
                    7.16208633911263e-08,
                    6.0177453360661156e-05,
                    0.6549354227360592,
                    0.8089013075729085,
                    0.5166585170062755,
                    0.2317465900941644,
                    0.4731827570928632,
                    0.038686491506018564,
                    0.00023974017702647945,
                    1.081763942496813,
                    1.202737507528403,
                    0.05013168718178318,
                    0.1372350579551961,
                    0.1903270100841814,
                    0.3308146651182463,
                    0.48928150602486353,
                    0.8026742677662071,
                    0.7029133518222641,
                    0.24114616678663592,
                    0.3673173711175879,
                    0.6638955719628095,
                    0.02663505418121219,
                    0.8095724772390148,
                    0.8763711023774112,
                    0.3050450802169265,
                    0.04490876818500625,
                    0.034099402450974634,
                ]
                # params = [12.848504744221934, 19.54452815665691, 19.993855273177015, 19.90980778022618, 19.9647255742759, 11.21557311096887, 19.98741689072562, 19.981109715898764, 10.543264530044311, 0.0016181979434673737, 1.2293594080510868e-06, 6.709486399127274e-05, 1.0294361782483558e-06, 6.071015239677332e-07, 2.1543750072777677, 1.3713098090284643, 8.408698531878704, 14.000787364573707, 3.256566956066174, 3.166404208534545, 3.0141050483394936, 3.0434487341913616, 2.947229140926736, 0.754309656770216, 0.8943254515387007, 18.595960329470937, 18.830464091391658, 19.998749169234802, 19.97940041319094, 19.984867790062616, 19.489903227491034, 12.519810494033761, 13.84779283464996, 13.230964795618668, 6.3659665356270585, 3.3128232502232047, 1.752775720172151e-05, 4.528645512547785, 8.58189939481435, 0.014370689290391703, 0.00014820835921192975, 2.2744628820246773e-06, 0.06195523237266917, 1.5293844429517294, 4.862714400732811, 1.0808993361884691, 2.488376976916697, 9.189164377826199, 8.736095927937575, 3.2706580286686875, 10.573897327220754, 5.666187448159939, 4.009811496640581, 0.47445945124871447, 13.758106874962685, 8.276486327604921, 6.847974793114729, 3.9200161841580603, 8.895776399510734, 6.446357179044716, 7.922023455192179, 8.101922835305556, 14.91878589740736, 1.9538585746927335, 4.76621118512418, 14.464610006091796, 6.810444336909947, 9.768015985793358, 7.351653855487542e-06]
                # params = [7.2088807264297365, 98.83915935054323, 99.99815240770162, 87.92847233456804, 94.41712615664048, 83.03695404214402, 99.98822821923929, 99.93591613544206, 89.36623479583442, 99.83924232905095, 99.51172165097037, 5.455742200101308, 0.011148598463011809, 0.5147018658716686, 25.82048110740258, 24.248115016235054, 24.42504806869668, 46.8588088590316, 2.603192005274989, 1.7333435167673517, 5.07293021026804, 13.982999977840523, 9.271802523422062, 3.5146810775835973, 3.7379830247780994, 71.15983603468794, 91.68357841589805, 26.74217672430608, 98.99656669452933, 97.58067798995248, 99.31607287148019, 46.39443880599283, 58.4306633904294, 39.30064854588915, 7.521160836988941, 8.211372577045282, 19.982464435841564, 81.19780396983585, 57.366656283660106, 96.51375783118162, 0.6732920458350473, 0.43592024681659614, 56.55637102686359, 6.82759878530538, 85.12775437329489, 15.715625645163545, 0.22051922292224269, 8.371739584477403, 35.04981098597891, 25.692538790650243, 22.56218936942404, 68.01880412837636, 43.62591959938588, 20.889534357752964, 23.425779995589117, 69.06206851518198, 10.575321629747375, 66.3414538888442, 18.42761567957521, 33.822485465160305, 63.65938170634681, 50.87904042580305, 37.98184662437191, 58.16954820763894, 60.21597285528973, 37.83357562306635, 99.76073070079846, 40.3451998047217, 3.007149525829531e-05]
                # params = [11.36676473615454, 95.06188630757623, 88.759661937489, 99.84017256600632, 99.97925367226914, 97.54315279221592, 99.88638838456319, 96.42504731294963, 97.87652680134624, 91.08421544803406, 94.54734005775886, 5.501107793005635, 3.6829497841113725, 11.513059939770724, 41.51196569994983, 33.661102870224276, 30.847652808921982, 92.44349245220722, 1.7229878471203715, 1.7141615242951005, 5.550639446539979, 14.495508084953146, 13.31495878740553, 3.548605788121904, 3.519109139293608, 73.86225817214697, 84.51994651454602, 25.800286645113037, 99.66464427260182, 99.50642522990658, 52.01134047109205, 43.672212739286806, 55.765885481541716, 39.88570917083517, 17.400855361972596, 42.89082544981169, 28.16813848182207, 6.774657099864965, 43.89289682954902, 14.033352043443534, 2.828033086851557, 6.128170377922171, 24.58619633316033, 16.144695271771777, 97.85502558143881, 23.38085334294328, 20.639036752409453, 39.35869711326733, 53.5349888501498, 22.22187204941233, 59.73514336078824, 10.104180489291181, 29.4222126366859, 20.24549722616661, 31.566309700716104, 7.631033158373171, 9.839226499552876, 25.010032389571187, 19.208790465359545, 50.67715319910566, 70.64937308302038, 0.5318751071245016, 32.91572305470233, 4.368523567298568, 16.81685183077225, 33.254193106317395, 50.50038652364253, 98.55815381616138, 2.7733288213081426e-06]
                # params = [33.854932901724865, 197.7562812942389, 199.93930880586052, 191.95792974021674, 197.97717938417674, 199.5130811408595, 197.82220758604234, 199.28326345674344, 195.4406278790149, 199.9810605737064, 199.66869887308272, 0.005686808993401228, 0.3705752906699691, 0.003185363531327173, 64.29170907693764, 47.201566295621404, 86.82672260187417, 135.753299677399, 2.2824916634976464, 1.7542331637223625, 5.340361580895508, 16.826256853013987, 12.94998305317464, 4.793811941481637, 4.840886318204323, 59.74281921540489, 46.31000203221055, 47.094969067927224, 170.15785900624394, 194.17044792496927, 196.82706199721383, 87.42902292192277, 142.08156872668295, 113.44527016352379, 45.865738048337924, 24.55382173100384, 57.3498264421611, 52.074253957784805, 80.24941809928191, 3.4783512052007435, 3.1914902010982225, 3.94431244070897, 24.970269078800314, 68.6252696016883, 199.94077380904093, 117.4451587247961, 72.16796625341426, 62.62788673269098, 26.83341320021353, 97.70306303939348, 50.29650501019472, 25.87917902861797, 171.0175469412674, 108.11970704594938, 29.156171859837347, 11.33030125964347, 30.74678348480026, 11.096760891059505, 40.989508173482015, 2.147907321690937, 32.724156170891554, 149.44546361622034, 158.58512857439393, 77.29024007885553, 80.77755977928494, 74.35814185930617, 12.706579158838624, 78.32293914502515]
                # params = [32.546491701294194, 18.099473043853912, 296.5812172771352, 221.7086879476543, 153.3042511499134, 97.22143624430095, 278.8646441279165, 289.25404893835, 146.87020502737013, 191.63020079674155, 299.99800492521234, 0.009659563587920151, 1.7992938247708077, 14.173575529648247, 199.54134609172314, 183.00688318437298, 78.00640825007129, 29.53377671182712, 298.186701350624, 1.8608423219172807, 5.077737686971215, 25.701352656247927, 18.595158048589706, 0.03826101541096754, 0.1777185998537597, 48.1164190208401, 56.234311310540974, 157.7525916612472, 205.40222417743655, 239.06261557916895, 269.3691263242472, 148.85717746871956, 97.78590222632894, 67.41366733394563, 18.769787333667388, 29.735095148008106, 41.64402157539401, 162.89543318364446, 89.82210536040002, 60.83450846272936, 298.724509159146, 31.94422263938015, 96.92032886166155, 77.2279949488636, 193.69226730935537, 189.28879811375975, 299.2364116655526, 104.21303039816222, 154.3931630447259, 27.48870524998653, 3.435019384263182, 79.00999843390022, 28.84980277387342, 184.17626465406033, 131.28977748821836, 50.72718266983804, 67.056773541977, 49.47432777691724, 6.039309074681116, 83.9343220593972, 181.1197959989522, 110.31123506805021, 74.45466850512405, 68.18829777818667, 90.61669412036154, 119.4765205540369, 93.44275276537911, 106.97269234372853]
                # params = [32.945014820598445, 396.2850200015831, 471.6167607541645, 159.08032141457156, 326.3914789470757, 0.9331162220316125, 208.70821964321146, 228.01742997914116, 0.08051942451170563, 0.0025840265899828535, 7.492520958095834e-07, 0.0009699955302241832, 0.00397324752929568, 0.00022095441968609923, 0.2227903802203025, 0.2501793403298956, 0.0006654494516046275, 0.8201991025832527, 364.27457334827744, 143.8803519000366, 185.76309850605696, 498.83019481387385, 444.27846681290754, 312.3666802095113, 368.2002637892474, 497.08283224408785, 462.5806280970104, 0.08901954623426023, 482.65938300405736, 466.17783984608997, 371.04097062071776, 443.27065729110336, 406.7889669948889, 359.7501059885032, 36.36271445200217, 40.804795713295285, 21.019417152429035, 52.5233724961798, 31.39764052273477, 117.47656141120028, 93.08587896236718, 27.03006704845077, 277.06850837589013, 307.34315775569576, 495.83905959274745, 335.0772738245678, 3.029026746476008, 23.829373641212683, 191.2071004876135, 31.94702669949018, 239.59883871170584, 63.39137712697093, 136.8198147893944, 424.55272283572253, 357.2520471728541, 137.45444143984687, 234.1626649173425, 19.998427986368878, 76.86449360799594, 227.32157192347216, 4.791920173628462, 274.44741767443884, 162.36863709680625, 470.9702964095069, 434.99903957466677, 354.68079025396855, 348.1320769231661, 254.4948031943731]
                # params = [0.0006099731497041457, 49.99846429206464, 44.543828753118, 35.36601907894371, 48.47994640933031, 29.619941820204332, 49.864971107061265, 47.36558787971047, 28.452799351192326, 1.5906436618320037, 0.2683725045287058, 0.0729744099476417, 0.3864596832694519, 0.740449001350447, 10.45239728215285, 33.309473291330576, 19.27917343009507, 33.32633891674841, 2.4478723974313996, 0.00011175390469730248, 0.6840216263396441, 11.47713270792376, 12.080935867852634, 0.0762885782098951, 0.0017728092095065317, 34.98535844647836, 40.382426146828294, 37.98404394312681, 49.60791700207538, 49.62187856810797, 49.64970893840519, 26.36647120828151, 47.16670583975327, 17.05570950313224, 13.883373042623997, 48.3086999066288, 23.86596266801848, 18.68864319200013, 21.48879870968154, 45.08413383696092, 2.415200587572492, 3.3853165377135244, 13.58803949610763, 5.9241717545133685, 13.07242123739899, 42.194023009978544, 6.6059310234020945, 35.90596638500213, 43.53084669814217, 27.019705877985885, 39.108052212390476, 15.556029326313029, 6.580869338308206, 11.08464617402659, 18.314479465822245, 3.5538669210452194, 6.861004799112441, 7.33107785807088, 41.222829652187094, 1.6943894327765263, 27.46594819059979, 14.82841546795879, 3.6540091231453253, 19.701215254222692, 46.81699767854484, 24.03129830584026, 46.36047278566231, 0.1634313385211762, 1.5660029353459024]
                # params = [323.02564003467364, 497.4884957302963, 498.5234737692179, 499.42973438247924, 484.7952623350464, 483.065385942473, 499.3912405551837, 499.9415600315498, 138.81589551628846, 84.67883347587515, 0.0054002823618976075, 35.21469544175374, 21.82288082096287, 19.715030727098192, 126.41863477253398, 138.81657342510863, 233.9427745701995, 1.8099571840397601, 115.44788993984277, 77.79195773370316, 123.57129519479508, 200.42687063929046, 309.86879456260175, 110.20045464076962, 112.10216985487438, 499.73880525424164, 498.3443875419508, 0.04985997430902939, 497.50942432219483, 499.64108343391297, 493.2573023081828, 489.58716548925736, 499.49887759787293, 499.8338318035646, 291.95266261797303, 3.6107595914503694, 0.0037567539567409367, 1.2314561217019335, 2.296660364945937, 4.9388365165149875, 9.164316394333841e-05, 0.007863202437531747, 0.3590152518026579, 5.496440488927848, 499.99937083353933, 487.6724129422138, 497.98527888828556, 499.1037477887209, 263.9410900827098, 154.25172680018312, 75.65752449100265, 437.2965609047667, 47.354950556074684, 174.49600561115756, 457.3177836978258, 337.5717607897817, 485.68202191752107, 111.28674429930827, 391.58766828960427, 216.8157764615354, 422.4142015815646, 113.39785094610772, 41.69578149806659, 380.2522037218806, 480.0476988042707, 190.01566792602532, 299.01321896441647, 49.15348904942076]
                params = [
                    np.float64(472.2654813238043),
                    np.float64(756.2779949201798),
                    np.float64(735.7860415491991),
                    np.float64(928.257097263313),
                    np.float64(812.3799212070905),
                    np.float64(624.6674124294418),
                    np.float64(320.7942179798739),
                    np.float64(301.97788341855954),
                    np.float64(150.1271863662071),
                    np.float64(0.0),
                    np.float64(2.3502986537575554e-07),
                    np.float64(0.0),
                    np.float64(0.0),
                    np.float64(0.0),
                    np.float64(8.085662969304732),
                    np.float64(365.01600884838007),
                    np.float64(187.13848345862235),
                    np.float64(634.5588267260474),
                    np.float64(581.5482308984457),
                    np.float64(667.260408273373),
                    np.float64(518.012710069234),
                    np.float64(376.08358042113787),
                    np.float64(597.5116838219593),
                    np.float64(49.92763599728841),
                    np.float64(51.75065413899098),
                    np.float64(876.8450036999939),
                    np.float64(346.36507845434517),
                    np.float64(974.6280672835907),
                    np.float64(760.9912663133331),
                    np.float64(329.7660030544987),
                    np.float64(626.8214488193189),
                    np.float64(725.0076998770165),
                    np.float64(591.166289491217),
                    np.float64(357.96908186961497),
                ]

                params += [0]
                # params = params + [0]
                dlc_kalman[:, :, count_bis + 1], all_kalman = get_next_frame_from_kalman(
                    all_kalman, dlc_data_tmp, new_cluster_dlc, params=params, idx_cluster=names.index("clavac")
                )
                to_dlc_origin = dlc_data_tmp[:2, :, 0].T if dlc_data_tmp is not None else None
                dlc_kalman_origin[:2, :14, count_bis + 1] = kalman_dlc_predictor.process(to_dlc_origin).T
                # tic = time.time()
                # for n in range(3):
                #     off_line_processing[n].lp_butter_order = 4
                #     off_line_processing[n].lpf_lcut = 15
                #     filt_tmp = off_line_processing[n].process_emg(dlc_kalman[n, :, count_bis + 1][..., None].copy(),
                #                                                                             band_pass_filter=False,
                #                                                                             normalization=False,
                #                                                                             centering=False,
                #                                                                             absolute_value=False,
                #                                                                             low_pass_filter=True,
                #                                                                             moving_average=False,
                #                                                                             )
                #     if count_bis > 10:
                #         dlc_filt_filt[n, :, count_bis + 1] = filt_tmp[:, -2]
                # print(time.time() - tic)
                #     # new_markers_dlc_filtered[i, 8:, :] = OfflineProcessing().butter_lowpass_filter(
                #     #     new_markers_dlc[i, 8:, :],
                #     #     10, 60, 2)
                # if i in frame_idx and count_bis>10:
                #     im = cv2.imread(image_path + os.sep + f"color_{i}.png")
                #     for m in range(dlc_kalman.shape[1]):
                #         cv2.circle(im, (int(dlc_kalman[0, m, count_bis + 1]), int(dlc_kalman[1, m, count_bis + 1])), 5, (0, 255, 0), -1)
                #     for m in range(new_markers_dlc_filtered.shape[1]):
                #         cv2.circle(im, (int(new_markers_dlc_filtered[0, m, count_bis]), int(new_markers_dlc_filtered[1, m, count_bis])), 8, (255, 0, 0), 1)
                #     for m in range(dlc_data.shape[1]):
                #         cv2.circle(im, (int(dlc_data[0, m, count]), int(dlc_data[1, m, count])), 5, (0, 0, 255), 1)
                #     cv2.namedWindow("im", cv2.WINDOW_NORMAL)
                #     cv2.imshow("im", im)
                #     cv2.waitKey(1)
                # for k in range(dlc_data.shape[1]):
                #     if all_kalman[k] is None and dlc_data_tmp is not None:
                #         all_kalman[k] = Kalman(dlc_data_tmp[:, k, 0], n_measures=3, n_states=6, fps=60)
                #         all_kalman[k].measurement_noise_factor = 100
                #         all_kalman[k].process_noise_factor = 5
                #         all_kalman[k].error_cov_post_factor = 0
                #         all_kalman[k].error_cov_pre_factor = 0
                #         all_kalman[k].init_kalman(dlc_data_tmp[:, k, 0])
                #         dlc_kalman[:, k, count_bis] = dlc_data_tmp[:, k, 0]
                #         dlc_kalman[:, k, count_bis + 1] = all_kalman[k].last_predicted_pos
                #     elif all_kalman[k] is not None:
                #         if dlc_data_tmp is not None:
                #             all_kalman[k].correct(dlc_data_tmp[:, k, 0])
                #         dlc_kalman[:, k, count_bis + 1] = all_kalman[k].predict()
                #
                # if count_bis > 120:
                #     for i in range(3):
                #         dlc_filt_filt[i, :, count_bis] =
                # anato_from_cluster = _convert_cluster_to_anato(new_cluster_dlc, dlc_kalman[:, -6:-3, count_bis:count_bis+1] * 1000) * 0.001
                # anato_tmp = anato_from_cluster.copy()
                # anato_from_cluster[:, 0, :] = anato_tmp[:, 0, :]
                # anato_from_cluster[:, 1, :] = anato_tmp[:, 2, :]
                # anato_from_cluster[:, 2, :] = anato_tmp[:, 1, :]
                # dlc_kalman[:, -3:, count_bis + 1] = anato_from_cluster[:3, :, 0]
                # markers_dlc_hom[:3, :, 0] = dlc_kalman[:, :, count_bis + 1]
                # _ = np.dot(np.array(rt), markers_dlc_hom[:, :, 0])
                # for j in range(1, 4):
                #     if count > 2:
                #         if all_kalman[-j] is None:
                #             all_kalman[-j] = Kalman(dlc_kalman[:, -j, count_bis:count_bis+1], n_measures=3, n_states=6, fps=60)
                #             all_kalman[-j].measurement_noise_factor = 10
                #             all_kalman[-j].process_noise_factor = 5
                #             all_kalman[-j].error_cov_post_factor = 0
                #             all_kalman[-j].error_cov_pre_factor = 0
                #             all_kalman[-j].init_kalman(dlc_kalman[:, -j, count_bis:count_bis + 1])
                #             dlc_kalman[:, -j, count_bis + 1] = all_kalman[-j].last_predicted_pos
                #         elif all_kalman[-j] is not None:
                #             all_kalman[-j].correct(dlc_kalman[:, -j, count_bis:count_bis + 1])
                #             dlc_kalman[:, -j, count_bis + 1] = all_kalman[-j].predict()

                count_bis += 1
                if i in frame_idx:
                    count += 1
                if count != 0:
                    all_time.append(time.time() - tic)
            dlc_reproc = np.zeros_like(dlc_kalman)
            for i in range(3):
                dlc_reproc[i, :8, :] = OfflineProcessing().butter_lowpass_filter(dlc_kalman[i, :8, :], 1, 60, 2)
                dlc_reproc[i, 8:, :] = OfflineProcessing().butter_lowpass_filter(dlc_kalman[i, 8:, :], 3, 60, 2)
            for i in range(dlc_kalman.shape[1]):
                for j in range(3):
                    x = dlc_kalman[j, i, :]
                    y = new_markers_dlc_filtered[j, i, :]
                    correlation = signal.correlate(x, y, mode="full")
                    lags = signal.correlation_lags(x.size, y.size, mode="full")
                    lag = lags[np.argmax(correlation)]
                    # print(i, lag)
            print(np.mean(all_time))
            dlc_data = np.concatenate((dlc_data, np.zeros((3, 3, dlc_data.shape[2]))), axis=1)
            count = 3
            for i in range(dlc_kalman.shape[1]):
                plt.subplot(3, dlc_kalman.shape[1] // 3 + 1, i + 1)
                plt.title(names[i])
                for j in range(3):
                    # if i in [5,6,7]:
                    #     plt.plot(dlc_kalman[j, -count, :], "k", alpha=1)
                    # plt.plot(marker_to_process[j, i, :], "r", alpha=0.5)
                    # plt.plot( markers[j, i, :], "g", alpha=0.5)
                    plt.plot(dlc_kalman[j, i, 1:], "r", alpha=1)
                    # plt.plot(dlc_kalman_origin[j, i, 1:], "y", alpha=1)
                    # plt.plot(dlc_data[j, i, :], "r", alpha=0.8)
                    plt.plot(new_markers_dlc[j, i, :], "--", color="g", alpha=1)
                    plt.plot(new_markers_dlc_filtered[j, i, :], "r", alpha=0.3)

                    # plt.plot( markers_vicon[j, i, :], "y", alpha=0.5)
                if i in [5, 6, 7]:
                    count -= 1

            plt.show()
