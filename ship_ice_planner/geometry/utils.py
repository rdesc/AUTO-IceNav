from typing import List

import numpy as np

M__2_PI = 2 * np.pi


def Rxy_3d(theta) -> np.ndarray:
    return np.asarray([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def Rxy(theta) -> np.ndarray:
    return np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])


def get_global_obs_coords(vertices: List[np.ndarray], positions: np.ndarray, angles: np.ndarray) -> List[np.ndarray]:
    return [
        (Rxy(angle) @ vert.T).T + position
        for vert, position, angle in zip(vertices, positions, angles)
    ]


def euclid_dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
