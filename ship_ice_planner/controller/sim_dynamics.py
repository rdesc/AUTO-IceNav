from typing import List

import numpy as np
import pandas as pd

from ship_ice_planner.controller.NRC_supply import NrcSupply
from ship_ice_planner.controller.supply import Supply
from ship_ice_planner.controller.trajectory_setpoint import TrajectorySetpoint
from ship_ice_planner.utils.storage import Storage
from ship_ice_planner.geometry.utils import Rxy

VESSEL_MODELS = [
    'Fossen_supply',  # full scale model of an offshore supply vessel from https://github.com/cybergalactic/PythonVehicleSimulator
    'NRC_supply',     # 1:45 scale model of a vessel designed and built by the NRC
]
STATE_HISTORY_FILE_NAME = 'state_history.txt'


class State:

    def __init__(self, x, y, psi, u, v, r, dt):
        """Follows notation from Fossen 2021"""
        # eta
        self.x = x      # global x position (m)
        self.y = y      # global y position (m)
        self.psi = psi  # heading (rad)

        # nu
        self.u = u      # surge velocity (m/s)
        self.v = v      # sway velocity  (m/s)
        self.r = r      # yaw rate (rad/s)

        # propeller speed -- either RPS or RPM
        self.u_actual = []   # actual propeller speed
        self.u_control = []  # computed input vector

        self.dt = dt

    @property
    def eta(self):
        return [self.x, self.y, self.psi]

    @eta.setter
    def eta(self, eta):
        x, y, psi = eta
        self.x = x
        self.y = y
        self.psi = psi

    @property
    def nu(self):
        return [self.u, self.v, self.r]

    @nu.setter
    def nu(self, nu):
        u, v, r = nu
        self.u = u
        self.v = v
        self.r = r

    def get_vessel_speed(self):
        # speed in horizontal plane is U = sqrt(u^2 + v^2) -- Fossen 2021
        return np.sqrt(self.u**2 + self.v**2)

    def get_global_velocity(self):
        return Rxy(self.psi) @ [self.u, self.v]

    def integrate(self) -> List:
        # rotate surge and sway velocities into global frame
        u_g, v_g = self.get_global_velocity()
        self.x += self.dt * u_g
        self.y += self.dt * v_g
        self.psi = (self.psi + self.dt * self.r) % (2 * np.pi)

        return [self.x, self.y, self.psi]


class SimShipDynamics:
    def __init__(self,
                 vessel_model: str,
                 eta: List, nu: List,
                 dt: float,
                 target_speed: float,
                 max_acceleration: float,
                 output_dir=None,
                 control_mode='DPcontrol',
                 track_time=None,
                 **kwargs):
        """
        :param vessel_model: vessel model to use, available options are in VESSEL_MODELS
        :param eta: initial position and heading [x, y, psi]
        :param nu: initial velocity [u, v, r]
        :param dt: sample time (s)
        :param target_speed: target speed (m/s)
        :param max_acceleration: maximum acceleration (m/s^2)
        :param output_dir: directory to store simulation data
        :param control_mode: control mode for the vessel model
        :param track_time: time to track the trajectory in seconds before triggering a replan
        """
        assert vessel_model in VESSEL_MODELS
        self.dt = dt

        # for trajectory tracking
        self.target_speed = target_speed
        self.max_acceleration = max_acceleration
        self.iters_per_replan = int(track_time / dt) if track_time else None

        # for keeping track of vehicle state including ship pose and generalized velocity
        self.state = State(*eta, *nu, dt)
        self.state_storage = Storage(output_dir, file=STATE_HISTORY_FILE_NAME)  # log states at each time step

        self.vessel_model_name = vessel_model
        if vessel_model == 'NRC_supply':
            self.vessel_model = NrcSupply()
            self.state.u_actual = np.array([0, 0, 0], float)  # propeller speed in RPS
        elif vessel_model == 'Fossen_supply':
            self.vessel_model = Supply(controlSystem=control_mode,  # for debugging can do 'stepInput'
                                       V_current=0,    # ocean current speed m/s
                                       beta_current=0  # ocean current direction deg
                                       )
            self.state.u_actual = self.vessel_model.u_actual  # propeller speed in RPM

        # for trajectory tracking
        self.setpoint_generator = None
        self.setpoint = []

        self.sim_time = 0
        self.iteration = 0

    def control(self):
        [x, y, psi] = self.state.eta
        [u, v, r] = self.state.nu

        # compute the control input
        if self.vessel_model_name == 'NRC_supply':
            self.state.u_control = self.vessel_model.DPcontrol(self.state.eta, self.setpoint, self.dt)

        elif self.vessel_model_name == 'Fossen_supply':
            if self.vessel_model.controlMode == 'DPcontrol':
                self.vessel_model.set_setpoint(*self.setpoint)
                # [x, y, z, phi, theta, psi] -- 6 DOF described in Fossen 2021
                eta = np.array([x, y, 0, 0, 0, psi], float)  # ignore z, phi, theta
                # [u, v, w, p, q, r] -- generalized velocity
                nu = np.array([u, v, 0, 0, 0, r], float)  # ignore w, p, q

                self.state.u_control = self.vessel_model.DPcontrol(eta, nu, self.dt)

            elif self.vessel_model.controlMode == 'stepInput':
                self.state.u_control = self.vessel_model.stepInput(self.sim_time)

    def sim_step(self):
        [x, y, psi] = self.state.eta
        [u, v, r] = self.state.nu

        # propagate vehicle dynamics
        if self.vessel_model_name == 'NRC_supply':
            [u, v, r] = self.vessel_model.dynamics(u, v, np.rad2deg(r), self.state.u_control)
            self.state.nu = [u, v, np.deg2rad(r)]
            self.state.u_actual = self.state.u_control  # u_actual and u_control are the same for NRC_supply

        elif self.vessel_model_name == 'Fossen_supply':
            eta = np.array([x, y, 0, 0, 0, psi], float)
            nu = np.array([u, v, 0, 0, 0, r], float)

            [nu, u_actual] = self.vessel_model.dynamics(eta, nu, self.state.u_actual, self.state.u_control, self.dt)

            self.state.nu = [nu[0], nu[1], nu[5]]  # updated velocity
            self.state.u_actual = u_actual

        self.sim_time += self.dt
        self.iteration += 1
        self.setpoint = self.setpoint_generator.update(self.dt)

    def init_trajectory_tracking(self, path):
        self.setpoint_generator = TrajectorySetpoint(path=path,
                                                     eta=self.state.eta,
                                                     current_ship_speed=self.state.get_vessel_speed(),
                                                     target_speed=self.target_speed,
                                                     max_acceleration=self.max_acceleration,
                                                     wn=self.vessel_model.wn,
                                                     dist_to_ref_stuck=self.vessel_model.L)
        self.setpoint = self.setpoint_generator.update(self.dt)

    def check_trigger_replan(self):
        if self.iters_per_replan is None:
            return True
        return self.iteration % self.iters_per_replan == 0

    def get_state_history(self):
        data = self.state_storage.get_history()
        return pd.DataFrame(data)

    def log_step(self, **kwargs):
        """This should only be called once per simulation step!"""
        self.state_storage.put_scalars(time     =self.sim_time,
                                       x        =self.state.x,
                                       y        =self.state.y,
                                       psi      =self.state.psi,
                                       u        =self.state.u,
                                       v        =self.state.v,
                                       r        =self.state.r,
                                       u_control=list(self.state.u_control),
                                       u_actual =list(self.state.u_actual),
                                       setpoint =self.setpoint,
                                       tau      =list(self.vessel_model.compute_force(self.state.u_actual)),
                                       energy_use=self.vessel_model.compute_energy_use(self.state.u_actual,
                                                                                       self.state.nu,
                                                                                       self.dt),

                                       **kwargs)

        self.state_storage.step()

    def vessel_start_up(self, offset, final_y):
        """
        Sends propeller commands to the vessel model to get it moving
        """
        self.change_to_step_input()
        self.state.y -= offset

        while self.state.y < final_y:
            self.control()
            self.sim_step()
            self.state.integrate()

        self.sim_time = 0
        self.iteration = 0
        self.change_to_DP_control()

    def change_to_step_input(self):
        if self.vessel_model_name == 'NRC_supply':
            raise NotImplementedError

        elif self.vessel_model_name == 'Fossen_supply':
            self.vessel_model.controlMode = 'stepInput'

    def change_to_DP_control(self):
        if self.vessel_model_name == 'NRC_supply':
            raise NotImplementedError

        elif self.vessel_model_name == 'Fossen_supply':
            self.vessel_model.controlMode = 'DPcontrol'

    @staticmethod
    def get_min_turning_radius(u, r):
        """
        :param u: surge velocity (m/s)
        :param r: yaw rate (deg/s)

        :return: minimum turning radius (m)
        """
        return 180 * u / (np.pi * r)

    @staticmethod
    def get_max_yaw_rate(u, min_R):
        """
        :param u: surge velocity (m/s)
        :param R: minimum turning radius (m)

        At full scale at cruising speed (6-8 m/s), acceptable max yaw rate is 30 deg/min
        At low speeds (<= 2 m/s), acceptable max yaw rate is 45 deg/min

        :return: yaw rate (deg/s)
        """
        return 180 / ((np.pi * min_R) / u)
