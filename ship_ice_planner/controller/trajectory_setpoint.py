from typing import List

import numpy as np

from ship_ice_planner.geometry.utils import euclid_dist
from ship_ice_planner.utils.utils import compute_path_length


class TrajectorySetpoint:
    """
    Generates a smooth signal for the time varying references x_d(t), y_d(t), psi_d(t)
    This is necessary for good controller behavior
    """
    def __init__(self,
                 path: np.ndarray,           # reference path
                 eta: List,                  # current position and heading
                 current_ship_speed: float,
                 target_speed: float,
                 max_acceleration: float,
                 wn: np.ndarray,             # natural frequencies of the system
                 dist_to_ref_stuck: float,   # distance to reference before ship is considered stuck
                 T_factor: float = 0.5       # time constant factor for low pass filter
                 ):
        self.cx, self.cy, self.ch = path.T
        self.path_length = np.asarray(
            compute_path_length(np.asarray([self.cx, self.cy]).T, cumsum=True)
        )
        assert self.path_length.shape[0] == path.shape[0]
        self.s_ref = 0

        self.current_target_speed = current_ship_speed
        self.target_speed = target_speed

        self.max_acceleration = max_acceleration

        # initialize setpoints
        x, y, psi = eta
        self.x_d = x
        self.y_d = y
        self.psi_d = psi
        self.prev_psi_ref = psi

        # compute time constants based on natural frequencies
        self.T = T_factor / np.diag(wn)  # tracking is poor if T is too high
        assert self.T.shape == (3,)

        # get waypoint closest to position (this should be the first waypoint)
        ind = self.search_target_index(x, y)
        self.s_ref = self.path_length[ind]  # arc length s

        self.dist_to_ref_stuck = dist_to_ref_stuck  # half the ship length seems to be a reasonable distance

    def update(self, dt):
        """
        Get updated setpoints
        """
        self.current_target_speed += self.max_acceleration * dt
        self.current_target_speed = min(self.current_target_speed, self.target_speed)

        # advance along the reference path at the specified speed by updating the reference arc length s
        self.s_ref += self.current_target_speed * dt

        ind = np.sum(self.path_length <= self.s_ref) - 1
        x_ref, y_ref, psi_ref = self.cx[ind], self.cy[ind], self.ch[ind]

        # unwrap psi_ref
        psi_ref_unwrapped = np.unwrap([self.prev_psi_ref, psi_ref])[1]
        self.prev_psi_ref = psi_ref_unwrapped

        # smooth the signal using first order low pass filter
        self.x_d += dt * (x_ref - self.x_d) / self.T[0]
        self.y_d += dt * (y_ref - self.y_d) / self.T[1]
        self.psi_d += dt * (psi_ref_unwrapped - self.psi_d) / self.T[2]

        return [self.x_d, self.y_d, self.psi_d]

    def replan_update(self, current_ship_speed, current_ship_position, new_path):
        """
        Update the reference path and speed profile based on the current speed of the ship
        """
        self.current_target_speed = current_ship_speed
        ind = np.sum(self.path_length <= self.s_ref) - 1
        ref_x, ref_y = self.cx[ind], self.cy[ind]
        x, y = current_ship_position

        # update reference path
        self.cx, self.cy, self.ch = new_path.T
        self.path_length = np.asarray(
            compute_path_length(np.asarray([self.cx, self.cy]).T, cumsum=True)
        )

        # this is a hack to prevent the ship from getting stuck
        # if the ship is too far from the reference because it is unable
        # to track trajectory at target speed, then reset the reference to current ship position
        if euclid_dist([ref_x, ref_y], [x, y]) > self.dist_to_ref_stuck:
            ind = self.search_target_index(x, y)
            self.s_ref = self.path_length[ind]

        else:
            # get waypoint closest to ref
            ind = self.search_target_index(ref_x, ref_y)
            self.s_ref = self.path_length[ind]

    def search_target_index(self, x, y) -> int:
        # search nearest point index
        return np.argmin(np.linalg.norm(np.array([self.cx, self.cy]).T - np.array([x, y]), axis=1))
