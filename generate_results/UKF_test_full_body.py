import numpy as np
import opensim as osim
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from biosiglive import load
import json
import math

import pyorerun


class BoundedMerweSigmaPoints(MerweScaledSigmaPoints):
    def __init__(self, n, theta_bounds, **kwargs):
        super().__init__(n=n, **kwargs)
        self.theta_bounds = theta_bounds

    def sigma_points(self, x, P):
        sigmas = super().sigma_points(x, P)
        for i in range(sigmas.shape[0]):
            for j in range(len(self.theta_bounds)):
                min_j, max_j = self.theta_bounds[j]
                sigmas[i, j] = np.clip(sigmas[i, j], min_j, max_j)
        return sigmas


class JointMarkerUKF:
    def __init__(self, model, data_rate=100, with_markers=False):
        self.model = model if isinstance(model, osim.Model) else osim.Model(model)
        self.state = self.model.initSystem()
        self.dt = 1.0 / data_rate
        self.n_diff = 2
        self.with_markers = with_markers
        self.dof_names = tuple(s.toString() for s in self.model.getCoordinateSet())
        self.coordinates = [self.model.getCoordinateSet().get(coord) for coord in self.dof_names]
        self.markers = [self.model.getMarkerSet().get(i) for i in range(self.model.getNumMarkers())]
        
        self.N_JOINTS = self.model.getCoordinateSet().getSize()
        self.N_MARKERS = self.model.getMarkerSet().getSize()

        self.joint_mins = np.array([self.model.getCoordinateSet().get(i).getRangeMin()
                                     for i in range(self.N_JOINTS)])
        self.joint_maxs = np.array([self.model.getCoordinateSet().get(i).getRangeMax()
                                     for i in range(self.N_JOINTS)])

        self.theta_bounds = list(zip(self.joint_mins, self.joint_maxs))

        self.dim_x = (self.n_diff + 1) * self.N_JOINTS
        if self.with_markers:
            self.dim_x += (self.n_diff + 1) * 3 * self.N_MARKERS

        self.dim_z = 3 * self.N_MARKERS

        self.points = BoundedMerweSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0,
                                              theta_bounds=self.theta_bounds)

        self.ukf = UKF(dim_x=self.dim_x, dim_z=self.dim_z, dt=self.dt,
                       fx=self.fx, hx=self.hx, points=self.points)

        self.ukf.x = np.zeros(self.dim_x)
        self._init_ukf(update_q=True)


    def set_joint_angles(self, theta):
        # self.coordinates.setStateValues(theta)
        [coordinate.setValue(self.state, theta[i], enforceContraints=False) for i, coordinate in enumerate(self.coordinates)]
        self.model.realizePosition(self.state)
        # self.model.assemble(self.state)

    def _get_transition_matrix(self):
        A = np.eye(self.dim_x)
        for i in range(self.N_JOINTS):
            A[i, self.N_JOINTS + i] = self.dt
            if self.n_diff == 2:
                A[i, 2 * self.N_JOINTS + i] = 0.5 * self.dt**2
                A[self.N_JOINTS + i, 2 * self.N_JOINTS + i] = self.dt
        if self.with_markers:
            for i in range(self.N_JOINTS * 2, self.N_JOINTS * 2 + self.N_MARKERS * 3):
                A[i, self.N_JOINTS * 2 + i] = self.dt
                if self.n_diff == 2:
                    A[i, 2 * self.N_JOINTS * 2 + i] = 0.5 * self.dt**2
                    A[self.N_JOINTS * 2, 2 * self.N_JOINTS * 2 + i] = self.dt
        return A

    # def fx(self, x, dt):

    #     #create transistion matrix for constants velocity model
    #     x_new = np.dot(self._get_transition_matrix(), x)
    #     x_new[:self.N_JOINTS] = np.clip(x_new[:self.N_JOINTS], self.joint_mins, self.joint_maxs)        
    #     return x_new
    
    def fx(self, x, dt):
        x_new = self.transition_matrix @ x
        # === Enforce joint limits ===
        x_new[:self.N_JOINTS] = np.clip(x_new[:self.N_JOINTS], self.joint_mins, self.joint_maxs)

        # return np.concatenate([theta_new, theta_dot, m])
        return x_new


    def hx(self, x):
        theta = x[:self.N_JOINTS]
        # m_est = x[2 * self.N_JOINTS:]

        self.set_joint_angles(theta)
        markers_pos = np.zeros((3, self.N_MARKERS))
        for i in range(self.N_MARKERS):
            pos = self.markers[i].getLocationInGround(self.state)
            markers_pos[:, i] = [pos.get(j) for j in range(3)]
        if self.with_markers:
            markers_kalman = x[2 * self.N_JOINTS:2 * self.N_JOINTS + self.dim_z]
            return (markers_pos.flatten() + markers_kalman) / 2
        return markers_pos.flatten()

    def initialize(self, first_marker_frame, n_times=10):
        # default values from osim model
        # model = self.model
        # run the initialization step several times to stabilize the filter
        # self._init_ukf(update_q=True)
        # for _ in range(n_times):
        #     self.step(first_marker_frame)
        # self._init_ukf(update_q=False)
        # self.set_joint_angles(self.ukf.x[:self.N_JOINTS])
        self.ukf.update(first_marker_frame)
        for _ in range(n_times):
            self.step(first_marker_frame)
        self.set_joint_angles(self.ukf.x[:self.N_JOINTS])
        self._init_ukf()

    def _init_ukf(self, update_q=True):
        # self.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.01)
        self.ukf.P *= 1e-2
        self.ukf.Q *= 1e-3
        self.ukf.R *= 1e-2
        self.state.updQ()
        self.transition_matrix = self._get_transition_matrix()
        for i in range(self.N_JOINTS):
            # if update_q:
            #      self.ukf.x[i] = self.model.getCoordinateSet().get(i).getValue(self.state)
            self.ukf.x[self.N_JOINTS + i] = 0.0

        if self.with_markers:
            self.ukf.x[2 * self.N_JOINTS:2 * self.N_JOINTS + self.dim_z] = first_marker_frame

    def step(self, marker_frame):
        self.ukf.predict()
        self.ukf.update(marker_frame)

        theta_est = self.ukf.x[:self.N_JOINTS]
        if self.with_markers:
            marker_est = self.ukf.x[2 * self.N_JOINTS:2 * self.N_JOINTS + self.dim_z].reshape(3, -1)
            return theta_est , marker_est
        return theta_est , None

    def run(self, markers):
        states = np.empty((self.dim_x, markers.shape[-1]))
        self.initialize(markers[:, :, 0].flatten())
        for i in range(markers.shape[-1]):
            self.step(markers[:, :, i].flatten())
            states[:, i] = self.ukf.x
        return states
    
    # @property
    # def joint_angle(self):
    #     return self.ukf.x[:self.N_JOINTS]

    # @property
    # def joint_velocity(self):
    #     return self.ukf.x[self.N_JOINTS:self.N_JOINTS * 2]

    # @property
    # def markers(self):
    #     if self.with_markers:
    #         marker_est = self.ukf.x[2 * self.N_JOINTS:2 * self.N_JOINTS + self.dim_z].reshape(3, -1)
    #         return theta_est , marker_est
    #     return None
    
    # @property
    # def joint_acceleration(self):
    #     if self.n_diff == 2:
    #         return self.ukf.x[self.N_JOINTS * 2:self.N_JOINTS * 3]
    #     return None


if __name__ == "__main__":
    trc_file = "path_to_trc_file.trc"
    model_path = r"D:\Documents\Programmation\OpenSim 4.5\models\Models\Arm26\arm26.osim"
    model_path = r"D:\Documents\Programmation\osim_to_biomod\example\Models\Model_Pose2Sim.osim"
    file_name = "20250422_162024"
    markers_file = fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons\{file_name}.bio"
    marker_data = load(markers_file)
    markers_names = marker_data["key_point_reduced_names"]
    connections = marker_data["key_points_connections"]
    markers_3d = marker_data["key_points_filtered"]
    data_rate = 60  # Hz
    num_frames = 1000  # Number of frames to process
    model = osim.Model(model_path)
    mot_file = fr"F:\CIME_LOC\tmp_videos\{file_name}_output_jsons\ik_mot.mot"
    from pyorerun import OsimTimeSeries
    q = OsimTimeSeries(mot_file, model).q_in_radian
    # q = np.linspace(0, np.pi/2, num_frames) 
    # q = np.concatenate((np.zeros((1, num_frames)), q[None, :]), axis=0)  # Example joint angles
    # markers_positions = np.zeros((3, model.getMarkerSet().getSize(), num_frames))
    # markers = model.getMarkerSet()
    # for i in range(q.shape[1]) : # Example joint angles
    #     _ = [model.getCoordinateSet().get(c).setValue(state, q[c, i]) for c in range(model.getCoordinateSet().getSize())]
    #     model.realizePosition(state)
    #     markers_positions[:, :, i] = np.array([mark.getLocationInGround(state).to_numpy() for mark in model.getMarkerSet()]).T
    # # add some white noise to the markers positions
    # noise = np.random.normal(0, 0.005, markers_positions.shape)
    # markers_positions += noise
    marker_model_names = [name for name in tuple([s.getName() for s in model.getMarkerSet()])]
    markers_names[markers_names.index("MidHip")] = "CHip"
    markers_idx = [markers_names.index(name) for name in marker_model_names]
    markers_3d = markers_3d[:, markers_idx, :]
    with_markers = False
    ukf = JointMarkerUKF(model, data_rate, with_markers=with_markers)
    first_marker_frame = markers_3d[:, :, 0].flatten()
    ukf.initialize(markers_3d[..., 0].flatten(), n_times=30)
    # ukf.set_joint_angles(q[:, 0])
    # ukf.ukf.x[:ukf.N_JOINTS] = q[:, 0]

    q_est = None
    time_for_ukf = 0
    import time
    start_time = time.time()
    markers_est = np.zeros_like(markers_3d)
    markers_est_kalman = np.zeros_like(markers_3d)

    for i in range(markers_3d.shape[2]):
        if i == 500:
            break
        # marker_frame = markers_positions[:, :, i].flatten()
        marker_frame = markers_3d[:, :, i].flatten()
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

    import opensim
    # from pyorerun import OsimModel, OsimTimeSeries, DisplayModelOptions, PhaseRerun
    # from pyomeca import Markers
    # t_span = np.linspace(0, q_est.shape[1]/60, q_est.shape[1])
    # display_options = DisplayModelOptions()
    # display_options.mesh_path = "Geometry_cleaned"
    # prr_model = OsimModel.from_osim_object(model, options=display_options)
    # markers = Markers(data=markers_3d[:3, :, :q_est.shape[1]], channels=list(prr_model.marker_names))
    # viz = PhaseRerun(t_span)
    # viz.add_animated_model(prr_model, q_est, display_q=False, tracked_markers=markers)
    # viz.rerun("msk_model")

    import matplotlib.pyplot as plt
    plt.figure("Joint angles")
    plt.plot(q[1, :], c='r')
    plt.plot(q_est[1, :])

    plt.figure("Markers")
    for j in range(markers_3d.shape[1]):
        plt.subplot(markers_3d.shape[1] // 3 + 1, 3, j + 1)
        for i in range(markers_3d.shape[0]):
            plt.plot(markers_3d[i, j, :], c='r')
            plt.plot(markers_est[i, j, :])
            if with_markers:
                plt.plot(markers_est_kalman[i, j, :], c='g')

    plt.show()
