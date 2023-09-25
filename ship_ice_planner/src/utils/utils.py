import logging
import os
from operator import ge, le
from typing import Union

import matplotlib.ticker as tkr
import numpy as np
import yaml
from scipy.interpolate import interp1d
from yaml import Loader

M__2_PI = 2 * np.pi


def heading_to_world_frame(heading: int, theta_0: float, num_headings: int):
    """
    :param heading: ordinal or cardinal heading from ships frame of reference
    :param theta_0: angle between ship and fixed/world coordinates
    :param num_headings: number of headings in the discretized heading space
    """
    return (heading * M__2_PI / num_headings + theta_0) % M__2_PI


def resample_path(path, step_size):
    # edge case, path is of shape N x 3
    if len(path) == 1:
        return path

    # resample
    t = [0, *np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))]
    f_x = interp1d(t, path[:, 0])
    f_y = interp1d(t, path[:, 1])

    t_new = np.arange(t[0], t[-1], step_size)
    path = np.asarray([f_x(t_new), f_y(t_new)]).T

    # compute the heading along path
    theta = np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))

    # transform path to original scaling and then return
    return np.c_[path[:-1], theta]


class Path:
    def __init__(self, path: np.ndarray = None, swath: np.ndarray = None):
        self.path = path  # shape is 3 x n and order is start -> end
        self.swath = swath

    def update(self, path, swath, costmap, ship_pos_y, threshold_dist=None, threshold_cost=0.95):
        if self.path is None:
            self.path = path
            self.swath = swath
            return True

        if not threshold_dist or (self.path[1][-1] - ship_pos_y) < threshold_dist:
            self.path = path
            self.swath = swath
            return True

        # compute cost for new path up to old path max y
        new_swath = swath.copy()
        new_swath[int(self.path[1][-1]):] = 0
        new_swath[:int(ship_pos_y)] = 0
        new_cost = costmap[new_swath].sum()  # + path_length(self.clip_path(path, self.path[1][-1], le)[:2].T)

        # compute cost of old path starting from current ship pos y
        old_swath = self.swath.copy()
        old_swath[int(self.path[1][-1]):] = 0
        old_swath[:int(ship_pos_y)] = 0
        old_cost = costmap[old_swath].sum()  # + path_length(self.clip_path(self.path, ship_pos_y, ge)[:2].T)

        # check if the new path is better than the old path by some threshold
        # from experiments we get better performance if we only consider
        # the swath cost in the path cost comparison rather than swath cost + path length
        if new_cost < old_cost * threshold_cost:
            self.path = path
            self.swath = swath
            return True

        return False

    @staticmethod
    def clip_path(path, ship_pos_y: float, op: Union[ge, le] = ge):
        # clip points along path that are less/greater than ship y position
        return path[..., op(path[1], ship_pos_y)]


def rotation_matrix(theta) -> np.ndarray:
    return np.asarray([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


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
        fh = logging.StreamHandler(open(os.path.join(output, "log.txt"), 'w'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def scale_axis_labels(axes, scale):
    # divide axis labels by scale
    # thank you, https://stackoverflow.com/a/27575514/13937378
    def numfmt(x, pos):
        s = '{}'.format(x / scale)
        return s

    yfmt = tkr.FuncFormatter(numfmt)  # create your custom formatter function

    for ax in axes:
        ax.yaxis.set_major_formatter(yfmt)
        ax.xaxis.set_major_formatter(yfmt)
