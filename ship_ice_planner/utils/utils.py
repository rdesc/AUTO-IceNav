import logging
import os
from typing import Tuple

import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from yaml import Loader

from ship_ice_planner import LOG_FILE


def resample_path(path, step_size, linear=False, plot=False):
    orig_path = np.copy(path)
    orig_path_shape = orig_path.shape
    assert orig_path_shape[1] == 3 or orig_path_shape[1] == 2, 'path must be of shape N x 3 or N x 2'
    # edge case
    if len(path) == 1:
        return path

    s = compute_path_length(path[:, :2], cumsum=True)

    if linear:
        f_x = interp1d(s, path[:, 0])
        f_y = interp1d(s, path[:, 1])
    else:
        f_x = CubicSpline(s, path[:, 0])
        f_y = CubicSpline(s, path[:, 1])

    t_new = np.arange(s[0], s[-1], step_size)
    path = np.asarray([f_x(t_new), f_y(t_new)]).T

    if orig_path_shape[1] == 3:
        # compute heading from the resampled path
        psi = np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))

        # resampled path
        path = np.c_[path[:-1], psi]

        t_new = t_new[:-1]

    # for debugging
    if plot:
        f, ax = plt.subplots(orig_path_shape[1], 1, sharex=True)
        ax[0].plot(t_new, path.T[0], label='resampled')
        ax[0].plot(s, orig_path.T[0], 'r.', label='original')
        ax[0].legend()
        ax[0].set_title('position x')
        ax[1].plot(t_new, path.T[1])
        ax[1].plot(s, orig_path.T[1], 'r.')
        ax[1].set_title('position y')
        ax[-1].set_xlabel('arc length s')

        if orig_path_shape[1] == 3:
            ax[2].plot(t_new, path.T[2])
            ax[2].plot(s, orig_path.T[2], 'r.')
            ax[2].set_title('heading psi')
            plt.show()

    return path


def compute_path_length(path, cumsum=False):
    assert path.shape[1] == 2, 'path shape should be N x 2 where each row are x,y coordinates'
    p = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    if cumsum:
        return [0, *p.cumsum()]  # prepend 0 to the cumsum so that the list is the same length as the original path
    return p.sum()


def tracking_error(pose, path):
    assert path.shape[1] == 3, 'path shape should be N x 3'
    if len(np.shape(pose)) < 1:
        pose = [pose]

    start_pose = pose[0]
    start_idx = np.argmin(np.linalg.norm(path[:, :2] - start_pose[:2], axis=1))
    track_error = []

    e_x, e_y = path[start_idx][:2] - start_pose[:2]
    e_psi = np.diff(np.unwrap([path[start_idx][2], start_pose[2]])).item()
    track_error.append([e_x, e_y, e_psi])

    for i in range(1, len(pose)):
        next_pose = pose[i]
        next_idx = np.argmin(np.linalg.norm(path[:, :2] - next_pose[:2], axis=1))

        e_x, e_y = path[next_idx][:2] - next_pose[:2]
        e_psi = np.diff(np.unwrap([path[next_idx][2], next_pose[2]])).item()

        track_error.append([e_x, e_y, e_psi])

    track_error = np.asarray(track_error)
    cross_track_error = np.mean(np.linalg.norm(track_error[:, :2], axis=1))
    heading_error = np.mean(np.abs(track_error[:, 2]))

    return cross_track_error, heading_error


def figure_8_path(start: Tuple, straight_path_length: float, turning_radius: float, step_size: float) -> np.ndarray:
    """
    Useful for testing the ship dynamics and controller
    Can easily be called from the dynamic_positioning_demo.py script
    """
    assert len(start) == 3, 'start must be a tuple of length 3 for x, y, psi'
    x, y, psi = start

    angles = np.arange(0, 2 * np.pi, 0.01)
    left_turn = np.asarray([
        x - turning_radius * np.sin(psi) + turning_radius * np.cos(angles),
        y + straight_path_length + turning_radius * np.cos(psi) + turning_radius * np.sin(angles),
        angles + psi
    ]).T
    left_turn = resample_path(left_turn, step_size)

    angles = np.arange(-2 * np.pi, 0, 0.01)[::-1]
    right_turn = np.asarray([
        x + turning_radius * np.sin(psi) - turning_radius * np.cos(angles),
        y + straight_path_length - turning_radius * np.cos(psi) - turning_radius * np.sin(angles),
        angles + psi
    ]).T
    right_turn = resample_path(right_turn, step_size)

    # straight path from start to ship_pos
    straight_path = np.asarray([
        [x, y, np.pi / 2],
        [x, straight_path_length + y, np.pi / 2]
    ])

    straight_path = resample_path(straight_path, step_size)

    return np.concatenate((straight_path, left_turn, right_turn), axis=0)


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    Class used to store configuration parameters
    """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, attr):
        if attr not in self.keys():
            raise AttributeError

        return self.get(attr)

    @staticmethod
    def to_dict(d):
        return {
            k: DotDict.to_dict(d[k]) if type(v) is DotDict else v
            for k, v in d.items()
        }

    @staticmethod
    def to_dot_dict(d):
        return DotDict({
            k: DotDict.to_dot_dict(d[k]) if type(v) is dict else v
            for k, v in d.items()
        })

    @staticmethod
    def load_from_file(fp):
        with open(fp, 'r') as fd:
            cfg = yaml.load(fd, Loader=Loader)

        # convert to a DotDict
        return DotDict.to_dot_dict(cfg)


def setup_logger(output=None, console=True, name="", level=logging.DEBUG, prefix=""):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create formatter
    formatter = logging.Formatter(
        prefix + "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if output:
        fh = logging.StreamHandler(open(os.path.join(output, LOG_FILE), 'w'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
