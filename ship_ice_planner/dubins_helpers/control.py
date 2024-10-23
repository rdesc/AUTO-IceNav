from typing import Tuple

import dubins
import numpy as np


def compute_turn_rate_control(path, speed=None):
    assert path.shape[0] == 3

    # compute the time derivative of heading psi
    arc_length = np.linalg.norm(np.diff(path[:2, :], axis=1), axis=0)
    psi = np.unwrap(path[2, :])
    psi_dot = np.diff(psi) / arc_length  # psi_dot = d psi / ds

    if speed is not None:
        psi_dot = psi_dot * speed  # psi_dot = d psi / dt

    return psi_dot


def apply_control(start, V, U, dt, rk4=False):
    X = np.zeros([3, len(U) + 1])
    X[:, 0] = start

    f = lambda x, u: np.asarray([V * np.cos(x[2]), V * np.sin(x[2]), u])  # dx/dt = f(x,u)

    for i in range(len(U)):
        if rk4:
            k1 = f(X[:, i], U[i])
            k2 = f(X[:, i] + dt / 2 * k1, U[i])
            k3 = f(X[:, i] + dt / 2 * k2, U[i])
            k4 = f(X[:, i] + dt * k3, U[i])
            x_next = X[:, i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            x_next = X[:, i] + dt * f(X[:, i], U[i])

        X[:, i + 1] = x_next

    return X


def get_points_on_dubins_path(q0: Tuple, q1: Tuple, turning_radius: float, eps: float = 0, step_size: float = 0.5):
    dubins_path = dubins.shortest_path(q0, q1,
                                       turning_radius - eps)
    configurations, ts = dubins_path.sample_many(step_size)

    # there's a bug in the dubins library where the last point in the path is only sometimes included!
    # this inconsistency may cause issues downstream....
    # FIX -> remove the last point if the distance between the last point and q1 is less than step_size / 2
    if np.linalg.norm(configurations[-1][:2] - np.asarray(q1[:2])) < step_size / 2:
        configurations = configurations[:-1]

    return configurations, dubins_path.path_length()
