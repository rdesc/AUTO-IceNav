import os
import pickle

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from ship_ice_planner.dubins_helpers.control import get_points_on_dubins_path
from ship_ice_planner.geometry.utils import Rxy_3d, Rxy, M__2_PI

CACHED_PRIM_PATHS = '.prim_paths.pkl'  # save to disk primitive paths so no need to regenerate every time


# ---------------- PRIMITIVES ---------------- #
"""
Motion primitives were generated using the method described in the paper
'Computing a Minimal Set of t-Spanning Motion Primitives for Lattice Planners'

Configurations are of the form (x, y, theta) and assume Dubins car kinematics

Papers that provide details on the generation of motion primitives for ship navigation:
- "An Optimization-Based Motion Planner for Autonomous Maneuvering of Marine Vessels in Complex Environments"
   https://ieeexplore.ieee.org/abstract/document/9303746
- "Two-Stage Optimized Trajectory Planning for ASVs Under Polygonal Obstacle Constraints: Theory and Experiments"
   https://ieeexplore.ieee.org/abstract/document/9246499
"""
PRIM_DICT = dict(
    # 8 HEADINGS
    PRIM_8H_1=[{  # turn radius = 2 l.u. t-error = 1.2
        (0, 0, 0): [(1, 0, 0), (3, 0, 0), (6, 0, 0), (2, 1, 1), (2, -1, 7), (2, 2, 2), (2, -2, 6), (3, 1, 0), (3, -1, 0), (3, 3, 1), (3, -3, 7), (4, 0, 1), (4, 0, 7), (4, 3, 0), (4, -3, 0)],
        (0, 0, 1): [(0, 3, 3), (0, 4, 2), (0, 4, 3), (1, 1, 1), (2, 2, 1), (3, 3, 1), (1, 2, 2), (1, 4, 1), (1, 5, 1), (2, 1, 0), (2, 3, 1), (3, 0, 7), (3, 2, 1), (3, 3, 0), (3, 3, 2), (4, 0, 0), (4, 0, 7), (4, 1, 1), (5, 1, 1)]
    }, 8, 2],  # list of primitives, number of headings, minimum turning radius
    PRIM_8H_2=[{  # min turn radius = 5 l.u. (lattice units) t-error = 1.05
        (0, 0, 0): [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (5, 1, 0), (5, -1, 0), (5, 2, 1), (5, -2, 7), (5, 3, 1), (5, -3, 7), (5, 5, 2), (5, -5, 6), (5, 10, 2), (5, -10, 6), (6, 2, 0), (6, -2, 0), (6, 5, 1), (6, 5, 2), (6, -5, 6), (6, -5, 7), (6, 6, 2), (6, -6, 6), (7, 1, 1), (7, -1, 7), (7, 5, 2), (7, -5, 6), (7, 7, 1), (7, -7, 7), (7, 8, 1), (7, -8, 7), (7, 9, 1), (7, -9, 7), (8, 3, 0), (8, -3, 0), (8, 4, 0), (8, -4, 0), (9, 0, 1), (9, 0, 7), (9, 5, 0), (9, -5, 0), (10, 7, 0), (10, -7, 0), (10, 8, 0), (10, -8, 0), (10, 9, 0), (10, -9, 0), (10, 13, 0), (10, -13, 0), (10, 14, 0), (10, -14, 0), (11, -1, 1), (11, 1, 7), (12, -2, 1), (12, 2, 7), (12, -3, 1), (12, 3, 7)],
        (0, 0, 1): [(-3, 12, 2), (-2, 12, 2), (-1, 8, 3), (-1, 11, 2), (0, 8, 3), (0, 9, 2), (0, 9, 3), (0, 15, 1), (1, 1, 1), (1, 7, 2), (1, 8, 3), (1, 13, 1), (1, 14, 1), (1, 15, 1), (2, 5, 2), (2, 12, 1), (2, 13, 1), (3, 4, 1), (3, 5, 1), (3, 5, 2), (3, 6, 1), (3, 7, 1), (3, 8, 1), (3, 9, 1), (3, 10, 1), (3, 11, 1), (4, 3, 1), (5, 2, 0), (5, 3, 0), (5, 3, 1), (5, 6, 2), (6, 3, 1), (6, 5, 0), (7, 1, 0), (7, 3, 1), (7, 7, 0), (7, 7, 2), (7, 8, 0), (7, 9, 0), (8, 0, 7), (8, 1, 7), (8, -1, 7), (8, 3, 1), (8, 7, 2), (9, 0, 0), (9, 0, 7), (9, 3, 1), (9, 7, 2), (10, 3, 1), (11, -1, 0), (11, 3, 1), (12, -2, 0), (12, 2, 1), (12, -3, 0), (13, 1, 1), (13, 2, 1), (13, 9, 2), (14, 1, 1), (15, 0, 1), (15, 1, 1), (17, -3, 1)]
    }, 8, 5],
    PRIM_8H_3=[{  # min turn radius = 5 l.u. t-error = 1.2
        (0, 0, 0): [(1, 0, 0), (5, 1, 0), (5, -1, 0), (5, 2, 1), (5, -2, 7), (5, 3, 1), (5, -3, 7), (5, 5, 2), (5, -5, 6), (5, 6, 2), (5, -6, 6), (6, 2, 0), (6, -2, 0), (6, 5, 1), (6, -5, 7), (7, 1, 1), (7, -1, 7), (7, 7, 1), (7, -7, 7), (7, 8, 1), (7, -8, 7), (7, 9, 1), (7, -9, 7), (8, 3, 0), (8, -3, 0), (8, 4, 0), (8, -4, 0), (9, 0, 1), (9, 0, 7), (9, 5, 0), (9, -5, 0), (10, 7, 0), (10, -7, 0), (10, 8, 0), (10, -8, 0), (10, 9, 0), (10, -9, 0)],
        (0, 0, 1): [(-1, 8, 3), (0, 8, 3), (0, 9, 2), (1, 1, 1), (1, 7, 2), (1, 8, 3), (2, 5, 2), (3, 4, 1), (3, 5, 1), (3, 5, 2), (3, 6, 1), (3, 7, 1), (3, 8, 1), (3, 9, 1), (3, 10, 1), (4, 3, 1), (5, 2, 0), (5, 3, 0), (5, 3, 1), (5, 6, 2), (6, 3, 1), (6, 5, 0), (7, 1, 0), (7, 3, 1), (7, 7, 0), (7, 7, 2), (7, 8, 0), (7, 9, 0), (8, 0, 7), (8, 1, 7), (8, -1, 7), (8, 3, 1), (8, 7, 2), (9, 0, 0), (9, 3, 1), (9, 7, 2), (10, 3, 1)]
    }, 8, 5],
    PRIM_8H_4=[{  # min turn radius = 5 l.u.
        (0, 0, 0): [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (5, 1, 0), (5, -1, 0), (5, 2, 1), (5, -2, 7), (5, 3, 1), (5, -3, 7), (5, 5, 2), (5, -5, 6), (6, 2, 0), (6, -2, 0), (6, 5, 1), (6, 5, 2), (6, -5, 6), (6, -5, 7), (6, 6, 2), (6, -6, 6), (7, 1, 1), (7, -1, 7), (7, 5, 2), (7, -5, 6), (7, 7, 1), (7, -7, 7), (7, 8, 1), (7, -8, 7), (7, 9, 1), (7, -9, 7), (8, 3, 0), (8, -3, 0), (8, 4, 0), (8, -4, 0), (9, 0, 1), (9, 0, 7), (9, 5, 0), (9, -5, 0), (11, -1, 1), (11, 1, 7), (12, -2, 1), (12, 2, 7), (12, -3, 1), (12, 3, 7)],
        (0, 0, 1): [(-1, 8, 3), (-1, 11, 2), (0, 8, 3), (0, 9, 2), (0, 9, 3), (1, 1, 1), (1, 7, 2), (1, 8, 3), (2, 5, 2), (3, 4, 1), (3, 5, 1), (3, 5, 2), (3, 6, 1), (3, 7, 1), (3, 8, 1), (3, 9, 1), (3, 10, 1), (3, 11, 1), (4, 3, 1), (5, 2, 0), (5, 3, 0), (5, 3, 1), (5, 6, 2), (6, 3, 1), (6, 5, 0), (7, 1, 0), (7, 3, 1), (7, 7, 0), (7, 7, 2), (7, 8, 0), (7, 9, 0), (8, 0, 7), (8, 1, 7), (8, -1, 7), (8, 3, 1), (8, 7, 2), (9, 0, 0), (9, 0, 7), (9, 3, 1), (9, 7, 2), (10, 3, 1), (11, -1, 0), (11, 3, 1)]
    }, 8, 5],

    # MORE PRIMITIVES CAN BE ADDED HERE

    # ---------- #

    # 16 HEADINGS
    PRIM_16H_1=[{  # min turn radius = 1 l.u.
        (0, 0, 0): [(1, 0, 0), (1, 0, 1), (1, 0, 15), (1, 1, 4), (1, -1, 12), (2, 0, 2), (2, 0, 14), (2, 1, 0), (2, -1, 0), (2, 1, 1), (2, -1, 15), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4), (2, -2, 12), (2, -2, 13), (2, -2, 14), (2, -2, 15), (2, -3, 0), (2, 3, 1), (2, 3, 2), (2, -3, 14), (2, -3, 15), (2, -4, 0), (3, 0, 3), (3, 0, 13), (3, -1, 2), (3, 1, 14), (3, -3, 1), (3, 3, 15), (4, -1, 3), (4, 1, 13)],
        (0, 0, 1): [(-1, 4, 4), (0, 3, 3), (0, 3, 4), (0, 3, 5), (0, 3, 6), (0, 4, 2), (1, 0, 0), (1, 0, 15), (1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 2, 3), (1, 3, 1), (1, 3, 2), (2, 0, 14), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 1, 15), (2, -1, 15), (2, 2, 0), (2, 2, 4), (2, -2, 12), (2, 3, 0), (3, 0, 3), (3, 0, 13), (3, -2, 1)],
        (0, 0, 2): [(-1, 3, 4), (-1, 4, 2), (0, 2, 4), (0, 2, 5), (0, 2, 6), (0, 3, 2), (0, 3, 6), (0, 3, 7), (0, 4, 1), (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), (1, 3, 2), (2, 0, 0), (2, 0, 14), (2, 0, 15), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 0), (2, 2, 4), (2, 2, 5), (2, 2, 15), (3, 0, 2), (3, 0, 13), (3, 0, 14), (3, -1, 0), (3, 1, 2), (3, 1, 3), (4, 0, 3), (4, -1, 2)],
        (0, 0, 3): [(-2, 3, 3), (-1, 2, 5), (0, 1, 4), (0, 1, 5), (0, 2, 6), (0, 3, 1), (0, 3, 7), (1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 4), (3, 0, 0), (3, 0, 1), (3, 0, 14), (3, 0, 15), (3, 1, 2), (3, 1, 3), (3, 2, 4), (4, 0, 2), (4, -1, 0)]
    }, 16, 1],

    # MORE PRIMITIVES CAN BE ADDED HERE

    # DEBUGGING PRIMITIVES
    PRIM_GO_STRAIGHT=[{
        (0, 0, 0): [(1, 0, 0)],
    }, 8, 5],

)
# -------------------------------------------- #


def heading_to_world_frame(heading: int, theta_0: float, num_headings: int):
    """
    :param heading: the heading index in the discretized heading space
    :param theta_0: angle between ship and fixed/world coordinates
    :param num_headings: number of headings in the discretized heading space
    """
    return (heading * M__2_PI / num_headings + theta_0) % M__2_PI


class Primitives:
    """
    Class which stores the finite set of ship actions or motion primitives
    """

    def __init__(
            self,
            prim_name: str,
            scale: float = 1,
            step_size: float = 1,
            cache=True,
    ):
        """
        :param prim_name: name of the primitive set
        :param scale: scale factor for the primitives -- the scaling of the costmap needs to be considered here
        :param step_size: step size for sampling points on the path
        :param cache: whether to cache the primitive paths to disk
        """
        self.scale = scale
        self.step_size = step_size

        assert prim_name in PRIM_DICT, 'Primitives "{}" not found in primitive dictionary! Options are: {}'.format(
            prim_name, list(PRIM_DICT.keys())   # list of primitive sets
        )
        self.prim_name = prim_name
        edge_set_dict, self.num_headings, self.turning_radius = PRIM_DICT[self.prim_name]

        self.turning_radius *= self.scale

        # scale the edge sets and turn them into npy arrays
        self.edge_set_dict = {}
        for k, v in edge_set_dict.items():
            self.edge_set_dict[k] = [(i[0] * self.scale, i[1] * self.scale, i[2]) for i in v]

        self.num_base_h = len(self.edge_set_dict.keys())
        self.paths, self.path_lengths = self.get_dubins_paths(cache)
        self.max_prim = self.get_max_prim()

        # compute the spacing between base headings
        self.spacing = 2 * np.pi / self.num_headings

        # for debugging purposes
        self.prim_count = {k1: {k2: 0 for k2 in self.edge_set_dict[k1]} for k1 in self.edge_set_dict}

    def plot(self, theta: float = 0, all_headings=False, ship_vertices=None):
        """ plots all the primitives in the edge set dict """
        start_headings = np.asarray([0, 1, 2, 3]) * np.pi / 2 if all_headings else [0]

        fig, ax = plt.subplots(1, self.num_headings // 4, figsize=(10, 10), sharex=True, sharey=True)

        for idx, (origin, edge_set) in enumerate(self.edge_set_dict.items()):
            ax[idx].set_title('Prim name: {}\nTheta: {} rad\nStart heading: {}'.format(
                self.prim_name, round(theta, 2), origin[2])
            )
            color = 'r'

            for start_h in start_headings:
                R = Rxy_3d(theta + start_h)
                for edge in edge_set:
                    path = self.paths[(origin, tuple(edge))]
                    x, y, _ = R @ path
                    ax[idx].plot(x, y, color)
                    ax[idx].plot(x[-1], y[-1], 'k.', zorder=10)

                    if start_h == 0 and ship_vertices is not None:
                        # add ship footprint
                        R2 = Rxy(theta + idx * self.spacing)
                        # plot ship as polygon
                        rot_vertices = ship_vertices @ R2.T
                        ax[idx].add_patch(patches.Polygon(rot_vertices, True, fill=False, ec='k', zorder=1, alpha=0.5, linewidth='0.5'))

                    ax[idx].set_aspect('equal')

        ax[0].set_xlabel('x (m)')
        ax[0].set_ylabel('y (m)')

        plt.show()

    def get_max_prim(self):
        return int(
            round(max(self.path_lengths.values()))
        )

    def get_dubins_paths(self, cache=True):
        """ sample points to get a dubins path for each primitive """
        if os.path.isfile(CACHED_PRIM_PATHS) and cache:
            print('LOADING CACHED PRIM PATHS! Confirm, this is expected behaviour...')
            return pickle.load(open(CACHED_PRIM_PATHS, 'rb'))

        paths = {}
        path_lengths = {}
        for origin, edge_set in self.edge_set_dict.items():
            for edge in edge_set:
                q0 = (origin[0], origin[1],
                      heading_to_world_frame(origin[2], 0, self.num_headings))
                q1 = (edge[0], edge[1],
                      heading_to_world_frame(edge[2], 0, self.num_headings))
                configurations, path_length = get_points_on_dubins_path(
                    q0=q0, q1=q1,
                    turning_radius=self.turning_radius,
                    step_size=self.step_size, eps=1e-10  # eps is needed to fix bug in dubins library!
                )
                paths[(origin, edge)] = np.asarray(configurations).T
                path_lengths[(origin, edge)] = path_length

        if cache:
            with open(CACHED_PRIM_PATHS, 'wb') as file:
                pickle.dump((paths, path_lengths), file)

        return paths, path_lengths

    def update_prim_count(self, prim_count):
        for k1, v1 in prim_count.items():
            for k2, v2 in v1.items():
                self.prim_count[k1][k2] += v2

    @staticmethod
    def rotate_path(path, theta: float):
        R = Rxy_3d(theta)
        x, y, _ = R @ path
        t = [(i + theta) % (2 * np.pi) for i in path[2]]
        return np.asarray([x, y, t])

    @staticmethod
    def rotate_prim(theta: float, edge_set_dict):
        # note the headings are left untouched
        R = Rxy_3d(theta)
        new_edge_set = {}
        for k, v in edge_set_dict.items():
            new_v = v @ R.T
            new_edge_set[k] = new_v

        return new_edge_set


if __name__ == '__main__':
    # for testing purposes
    from ship_ice_planner.ship import FULL_SCALE_PSV_VERTICES
    for prim_name in PRIM_DICT:
        p = Primitives(prim_name=prim_name, scale=30, step_size=0.1, cache=False)
        p.plot(0, all_headings=False, ship_vertices=FULL_SCALE_PSV_VERTICES)
