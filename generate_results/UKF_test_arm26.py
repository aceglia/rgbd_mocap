import numpy as np
import opensim as osim
from UKF import JointMarkerUKF, MarkerNoise


if __name__ == "__main__":
    model_path = r"D:\Documents\Programmation\OpenSim 4.5\models\Models\Arm26\arm26.osim"
    num_frames = 200
    model = osim.Model(model_path)
    q = np.linspace(0, np.pi/2, num_frames) 
    q = np.concatenate((np.zeros((1, num_frames)), q[None, :]), axis=0)  # Example joint angles
    markers_positions = np.zeros((3, model.getMarkerSet().getSize(), num_frames))
    state = model.initSystem()
    markers = model.getMarkerSet()
    for i in range(q.shape[1]) : # Example joint angles
        _ = [model.getCoordinateSet().get(c).setValue(state, q[c, i]) for c in range(model.getCoordinateSet().getSize())]
        model.realizePosition(state)
        markers_positions[:, :, i] = np.array([mark.getLocationInGround(state).to_numpy() for mark in model.getMarkerSet()]).T
    # add some white noise to the markers positions
    noise = np.random.normal(0, 0.006, markers_positions.shape)
    markers_positions += noise
    with_markers = False
    ukf = JointMarkerUKF(model, 60, with_markers=with_markers, type='constant_acceleration')
    first_marker_frame = markers_positions[:, :, 0].flatten()
    ukf.initialize(markers_positions[..., 0].flatten(), n_times=10, marker_noise_lvl=MarkerNoise.HIGH)
    # ukf.set_joint_angles(q[:, 0])
    # ukf.ukf.x[:ukf.N_JOINTS] = q[:, 0]

    q_est = None
    time_for_ukf = 0
    import time
    start_time = time.time()
    markers_est = np.zeros_like(markers_positions)
    markers_est_kalman = np.zeros_like(markers_positions)

    for i in range(markers_positions.shape[2]):
        if i == 500:
            break
        # marker_frame = markers_positions[:, :, i].flatten()
        marker_frame = markers_positions[:, :, i].flatten()
        if i < 10:
            print(ukf.ukf.x[:3])
        tic = time.time()
        theta_est, m_est = ukf.step(marker_frame)
        if with_markers:
            markers_est_kalman[:, :, i] = m_est
        toc = time.time()
        time_for_ukf += toc - tic
        q_est = theta_est[:, None] if q_est is None else np.concatenate((q_est, theta_est[:, None]), axis=1)
        markers_est[:, :, i] = np.array(ukf.hx(ukf.ukf.x)).reshape(3, -1)


    print(f"Time for UKF: {time_for_ukf:.2f} seconds")

    from pyorerun import OsimModel, DisplayModelOptions, PhaseRerun
    from pyomeca import Markers
    t_span = np.linspace(0, q_est.shape[1]/60, q_est.shape[1])
    display_options = DisplayModelOptions()
    display_options.mesh_path = "Geometry_cleaned"
    prr_model = OsimModel.from_osim_object(model, options=display_options)
    markers = Markers(data=markers_positions[:3, :, :q_est.shape[1]], channels=list(prr_model.marker_names))
    viz = PhaseRerun(t_span)
    viz.add_animated_model(prr_model, q_est, display_q=False, tracked_markers=markers)
    viz.rerun("msk_model")

    import matplotlib.pyplot as plt
    plt.figure("Joint angles")
    plt.plot(q[1, :], c='r')
    plt.plot(q_est[1, :])

    plt.figure("Markers")
    for j in range(markers_positions.shape[1]):
        plt.subplot(markers_positions.shape[1] // 3 + 1, 3, j + 1)
        for i in range(markers_positions.shape[0]):
            plt.plot(markers_positions[i, j, :], c='r')
            plt.plot(markers_est[i, j, :])
            if with_markers:
                plt.plot(markers_est_kalman[i, j, :], c='g')

    plt.show()
