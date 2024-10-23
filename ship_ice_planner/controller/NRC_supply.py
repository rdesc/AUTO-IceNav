from typing import Union

import numpy as np

from ship_ice_planner.geometry.utils import Rxy


class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.sum_error = 0
        self.prev_error = None

    def __call__(self, err, dt):
        d_err = (err - (self.prev_error or err)) / dt
        self.sum_error += err * dt
        self.prev_error = err

        return self.Kp * err + self.Ki * self.sum_error + self.Kd * d_err


class NrcSupply:
    def __init__(self):
        """
        Fully decoupled 3 DOF displacement vessel using first order Nomoto model
        Linear model for the 1:45 scale PSV from the NRC
        """
        self.A = np.diag([
            0.9980,  # u -- surge (m/s)
            0.9953,  # v -- sway (m/s)
            0.9909   # r -- yaw rate (deg/s)
        ])  # discretized dynamics at dt = 1/50 --> x_[k+1] = Ax_k + Bu_k
        self.B = np.array([
            4.321e-5,  # main prop (rps)
            1.730e-4,  # side tunnel (rps)
            0.04515    # differential fore/aft (rps)
        ])  # assume pure motions, i.e. no thrust allocation matrix

        # To calculate yaw rate given min turn radius R and surge u --> 360 / ((2 * pi * R) / u)
        self.input_lims = [
            0.5,   # m/s
            0.05,  # m/s
            8.6    # deg/s
        ]

        # initialize PD for each of surge u, sway v, and yaw rate r
        self.pid_list = [PID(10, 0, 20),  # too high of gains will make ship swing around too much
                         PID(10, 0, 20),
                         PID(5, 0, 5)]

        # for plotting purposes
        self.controls = [
            "#1 Main propeller (RPS)",
            "#2 Tunnel thruster (RPS)",
            "#3 Differential fore/aft (RPS)",
        ]

        self.L = 2.0    # length (m)
        self.mass = 90  # mass (kg)
        self.wn = np.diag([0.3, 0.3, 0.3])  # picked a bit arbitrarily

    def dynamics(self, u, v, r, u_control):
        [u, v, r] = self.A @ [u, v, r] + self.B * u_control

        # impose limits
        if abs(u) > self.input_lims[0]:
            u = self.input_lims[0] * np.sign(u)
        if abs(v) > self.input_lims[1]:
            v = self.input_lims[1] * np.sign(v)
        if abs(r) > self.input_lims[2]:
            r = self.input_lims[2] * np.sign(r)

        return [u, v, r]

    def DPcontrol(self, pose, setpoint, dt):
        x, y, psi = pose

        # get global error in heading, x, and y position
        e_x, e_y, e_psi = np.asarray(setpoint) - np.asarray([x, y, psi])

        # rotate error into body frame of vessel
        # need the inverse of the rotation matrix
        [e_x, e_y] = Rxy(psi).T @ [e_x, e_y]

        # update PD for each surge u, sway v, and yaw rate r
        u_control = [pid(error, dt) for error, pid in zip([e_x, e_y, e_psi], self.pid_list)]

        return np.asarray(u_control)

    def set_environment_force(self, force):
        """
        Force should be a 3D vector [X, Y, N]
        """
        pass  # do nothing for now... need to implement this

    def compute_force(self, u_control) -> np.ndarray:
        return u_control

    def compute_energy_use(self, u_control, nu, sampleTime) -> Union[float, np.ndarray]:
        if len(u_control.shape) == 1:
            return np.abs(self.compute_force(u_control)) @ np.abs(nu) * sampleTime

        energy_use = np.zeros(len(u_control))
        for i in range(len(u_control)):
            energy_use[i] = np.abs(self.compute_force(u_control[i])) @ np.abs(nu[i]) * sampleTime

        return energy_use
