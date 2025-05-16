import numpy as np
import opensim as osim
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import biorbd
from enum import IntEnum

class MarkerNoise(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


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

class OsimModel:
    def __init__(self, model):
        self.model = osim.Model(model) if isinstance(model, str) else model
        self.state = self.model.initSystem()
        
    @property
    def n_dofs(self):
        return self.model.getCoordinateSet().getSize()

    @property
    def n_markers(self):
        return self.model.getMarkerSet().getSize()
    
    def update_kinematics(self, q):
        [coordinate.setValue(self.state, q[i], enforceContraints=False) for i, coordinate in enumerate(self.coordinates)]
        self.model.realizePosition(self.state)


class BiorbdModel:
    def __init__(self, model):
        self.model = biorbd.Model(model) if isinstance(model, str) else model
        



class ModelInterface:
    def __init__(self, model):
        self.state = None
        self.model, self.model_type = self.get_model_type(model)
        self.coordinates = [self.model.getCoordinateSet().get(coord) for coord in self.dof_names]
        self.markers = [self.model.getMarkerSet().get(i) for i in range(self.model.getNumMarkers())]
        

    def get_model_type(self, model):
        if isinstance(model, str):
            if model.endswith("bioMod"):
                return BiorbdModel(model), "biorbd_model"
            elif model.endswith("osim"):
                model = OsimModel(model)
                return model, "osim_model"
            else: 
                raise RuntimeError("Model type not recognize")
        if isinstance(model, biorbd.Model):
            return BiorbdModel(model), "biorbd_model"
        elif isinstance(model, osim.Model):
            return OsimModel(model), "osim_model"
        else:
            raise RuntimeError("Model type not recognize")
    

class JointMarkerUKF:
    def __init__(self, model, data_rate=100, with_markers=False, type='constant_velocity'):
        self.model = model if isinstance(model, osim.Model) else osim.Model(model)
        self.state = self.model.initSystem()
    
        self.dt = 1.0 / data_rate
        if type == 'constant_velocity':
            self.n_diff = 1
        elif type == 'constant_acceleration':
            self.n_diff = 2
        else:
            raise RuntimeError("Type is not a valid type")
        self.with_markers = with_markers
        self.dof_names = tuple(s.toString() for s in self.model.getCoordinateSet())
        self.coordinates = [self.model.getCoordinateSet().get(coord) for coord in self.dof_names]
        self.markers = [self.model.getMarkerSet().get(i) for i in range(self.model.getNumMarkers())]
        self.initial_states = self.model.getStateVariableValues(self.state).to_numpy()
        
        self.N_JOINTS = self.model.getCoordinateSet().getSize()
        self.N_MARKERS = self.model.getMarkerSet().getSize()

        self.joint_mins = np.array([self.model.getCoordinateSet().get(i).getRangeMin()
                                     for i in range(self.N_JOINTS)])
        self.joint_maxs = np.array([self.model.getCoordinateSet().get(i).getRangeMax()
                                     for i in range(self.N_JOINTS)])

        self.theta_bounds = list(zip(self.joint_mins, self.joint_maxs))

        self.dim_x = (self.n_diff + 1) * self.N_JOINTS
        if self.with_markers:
            self.dim_x += 3 * self.N_MARKERS * 2

        self.dim_z = 3 * self.N_MARKERS

        self.points = BoundedMerweSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0,
                                              theta_bounds=self.theta_bounds)

        self.ukf = UKF(dim_x=self.dim_x, dim_z=self.dim_z, dt=self.dt,
                       fx=self.fx, hx=self.hx, points=self.points)

        self.ukf.x = np.zeros(self.dim_x)
        # self._init_ukf(first_marker_frame, update_q=True)


    def set_joint_angles(self, theta):
        map_q = np.array([[q, 0] for q in theta]).flatten()
        self.initial_states[:self.N_JOINTS * 2] = map_q
        self.model.setStateVariableValues(self.state, osim.Vector(self.initial_states))
        # self.model.realizePosition(self.state)
        self.model.assemble(self.state)

    def _get_transition_matrix(self):
        A = np.eye(self.dim_x)
        num_joints, num_diff, time_step, dim_z = self.N_JOINTS, self.n_diff, self.dt, self.dim_z
        with_markers = self.with_markers
        
        for i in range(num_joints):
            A[i, num_joints + i] = time_step
            if num_diff == 2:
                A[i, num_diff * num_joints + i] = 0.5 * time_step**2
                A[num_joints + i, (num_joints * num_diff) + i] = time_step
        
        if with_markers:
            start_index_markers = num_joints * (num_diff + 1)
            for i in range(dim_z):
                A[start_index_markers + i, start_index_markers + dim_z + i] = time_step
    
        return A

    # def fx(self, x, dt):

    #     #create transistion matrix for constants velocity model
    #     x_new = np.dot(self._get_transition_matrix(), x)
    #     x_new[:self.N_JOINTS] = np.clip(x_new[:self.N_JOINTS], self.joint_mins, self.joint_maxs)        
    #     return x_new
    
    def fx(self, x, dt):
        x_new = self.transition_matrix @ x
        x_new[:self.N_JOINTS] = np.clip(x_new[:self.N_JOINTS], self.joint_mins, self.joint_maxs)
        return x_new


    def hx(self, x):
        theta = x[:self.N_JOINTS]
        # m_est = x[2 * self.N_JOINTS:]

        self.set_joint_angles(theta)
        markers_pos = np.zeros((3, self.N_MARKERS))
        for i in range(self.N_MARKERS):
            pos = self.markers[i].getLocationInGround(self.state)
            markers_pos[:, i] = [pos.get(j) for j in range(3)]
        # if self.with_markers:
        #     markers_pos = x[2 * self.N_JOINTS:2 * self.N_JOINTS + self.dim_z]
            # return (markers_pos.flatten() + markers_kalman) / 2
        return markers_pos.flatten()

    def initialize(self, first_marker_frame, n_times=10, marker_noise_lvl: MarkerNoise = MarkerNoise.LOW):
        self.marker_noise_lvl = marker_noise_lvl
        self._get_kalman_matrix()
        self._init_ukf(first_marker_frame)
        self.ukf.update(first_marker_frame)
        for _ in range(n_times):
            self.step(first_marker_frame)
        self.set_joint_angles(self.ukf.x[:self.N_JOINTS])
        self._init_ukf(first_marker_frame)

    def _get_kalman_matrix(self):
        if self.marker_noise_lvl == MarkerNoise.NONE:
            self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-5
            self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-6
            self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-6
        elif self.marker_noise_lvl == MarkerNoise.LOW:
            self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-3
            self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-4
            self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-4
        elif self.marker_noise_lvl == MarkerNoise.MEDIUM:
            self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-2
            self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-3
            self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-3
        elif self.marker_noise_lvl == MarkerNoise.HIGH:
            self.ukf.P = np.eye(self.ukf.P.shape[0]) * 1e-1
            self.ukf.Q = np.eye(self.ukf.Q.shape[0]) * 1e-2
            self.ukf.R = np.eye(self.ukf.R.shape[0]) * 1e-2
        else:
            raise RuntimeError("Marker noise level not recognized")

    def set_custom_matrix(self, P, Q, R):
        self.ukf.P = P
        self.ukf.Q = Q
        self.ukf.R = R

    def _init_ukf(self, first_marker_frame, update_q=True):
        self.transition_matrix = self._get_transition_matrix()
        self.ukf.x[:self.N_JOINTS] = self.model.getStateVariableValues(self.state).to_numpy()[:self.N_JOINTS*2][::2]
        self.ukf.x[self.N_JOINTS:self.N_JOINTS * (self.n_diff+1)] = 0
        if self.with_markers:
            self.ukf.x[self.n_diff + 1 * self.N_JOINTS: self.n_diff + 1 * self.N_JOINTS + self.dim_z] = first_marker_frame

    def step(self, marker_frame):
        self.ukf.predict()
        self.ukf.update(marker_frame)

        theta_est = self.ukf.x[:self.N_JOINTS]
        if self.with_markers:
            marker_est = self.ukf.x[self.n_diff + 1 * self.N_JOINTS:self.n_diff + 1 * self.N_JOINTS + self.dim_z].reshape(3, -1)
            return theta_est , marker_est
        return theta_est , None

    def run(self, markers):
        states = np.empty((self.dim_x, markers.shape[-1]))
        self.initialize(markers[:, :, 0].flatten())
        for i in range(markers.shape[-1]):
            self.step(markers[:, :, i].flatten())
            states[:, i] = self.ukf.x
        return states