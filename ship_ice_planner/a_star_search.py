""" A* search algorithm for finding a path by searching a graph of nodes connected by primitives """
import logging
import queue
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from ship_ice_planner import *
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.dubins_helpers.heuristic import dubins_heuristic
from ship_ice_planner.primitives import Primitives, heading_to_world_frame
from ship_ice_planner.ship import Ship
from ship_ice_planner.swath import Swath, rotate_swath
from ship_ice_planner.utils.hashmap import HashMap
from ship_ice_planner.utils.priority_queue import PriorityQueue
from ship_ice_planner.geometry.utils import Rxy_3d, M__2_PI


class AStar:
    def __init__(self,
                 full_costmap: CostMap,
                 prim: Primitives,
                 ship: Ship,
                 swath_dict: Swath,
                 swath_shape: Tuple,
                 weight: float = 1,
                 **kwargs):
        self.weight = weight  # static weighting for heuristic
        self.full_costmap = full_costmap
        self.prim = prim
        self.ship = ship
        self.orig_swath_dict = swath_dict

        # this is used to transform the swath array to the global frame
        self.swath_array_centre = swath_shape[0] // 2

        self.logger = logging.getLogger(__name__)

        # initialize member vars that are updated and used during planning
        self.subset_costmap = None
        self.swath_dict = None
        self.swath_arg_dict = None
        self.rotated_prims = None
        self.obstacles_only_heuristic = None

        # variable used to diagnose planning failures
        self.diagnostics: dict = None

        # numba setup
        if NUMBA_IS_AVAILABLE:
            self.concat = nb.njit()(self._concat)
            self.get_swath_cost = nb.njit()(self._get_swath_cost)
        else:
            self.concat = self._concat
            self.get_swath_cost = self._get_swath_cost

        # the way we represent nodes on lattice is odd since we use global coordinates
        # for x and y while for heading we use the lattice heading units

    def search(self, start: tuple, goal_y: float):
        """
        :param start: tuple of the form (x, y, theta) where x and y are in global coordinates
        :param goal_y: goal y position in global coordinates
        """
        self.diagnostics = None

        self.swath_dict = {}      # this stores the binary swath arrays
        self.swath_arg_dict = {}  # this stores the indices of the swath array that are 1
        theta_0 = start[2] % M__2_PI
        R = Rxy_3d(theta_0)
        # reset rotated primitives
        self.rotated_prims = {}

        # get subset of costmap based on finite horizon
        upper_lim = min(self.full_costmap.shape[0], int(goal_y) + self.swath_array_centre)
        self.subset_costmap = self.full_costmap.cost_map[: upper_lim]

        lower_lim = int(start[1] - self.ship.length)
        # extend the costmap if necessary to account for start being too close to the bottom edge of the costmap
        if lower_lim < 0:
            lower_padding = np.zeros((-lower_lim, self.subset_costmap.shape[1]))
            self.subset_costmap = np.vstack((lower_padding, self.subset_costmap))
        else:
            self.subset_costmap = self.subset_costmap[lower_lim:]

        # transform start and goal nodes
        start = (start[0], start[1] - lower_lim, 0)
        goal_y = goal_y - lower_lim

        # compute lookup for the obstacles only heuristic
        self.obstacles_only_heuristic = self.compute_obstacles_only_heuristic(self.subset_costmap, int(self.ship.width))

        # use custom spatial hash maps since we are using the finite precision
        # coordinates of the nodes as the keys to the map
        open_set = HashMap.from_points(cell_size=10, scale=10e3, points=[start])
        closed_set = HashMap(cell_size=10, scale=10e3)

        # dicts to keep track of all the relevant path information
        # these maps are used to build the optimal path at the end of search
        came_from = {start: None}
        came_from_by_edge = {start: None}
        g_score = {start: 0}
        f_score = {start: self.weight * self.heuristic(start, goal_y, theta_0)}
        path_length = {start: 0}
        generation = {start: 0}  # keep track of depth (or generation) of the search
        # these two variables are used to address the swath overlap issue
        # where double counting of costs in the costmap can occur from overlapping swaths
        swath_overlap = {start: None}
        swath_overlap_mask = np.zeros_like(self.subset_costmap, dtype=bool)

        # priority queue based on lowest f score of open set nodes
        open_set_queue = PriorityQueue(item=(f_score[start], start))

        while len(open_set) != 0:
            try:
                node = open_set_queue.get_item()
            except queue.Empty:
                self.logger.error('Open set is empty!')
                break

            node = node[1]

            if node[1] >= goal_y:  # we don't care about the final heading at the goal
                self.logger.info('Found path! node {} goal {} generations {}'.format(node, goal_y, generation[node]))

                # build path goal ~> start
                goal = node
                node_path = [node]
                node_path_length = [path_length[node]]

                while node != start:
                    pred = came_from[node]
                    node = pred
                    node_path.append(node)
                    node_path_length.append(path_length[node])

                node_path.reverse()  # we want start ~> goal
                full_path, full_swath, prim_count, edge_seq = self.build_path(node_path, came_from_by_edge, start, theta_0)
                self.prim.update_prim_count(prim_count)
                swath_cost = self.subset_costmap[full_swath].sum()

                # convert nodes in the node path to world coords
                w_node_path = []
                for node in node_path:
                    # convert theta
                    theta = heading_to_world_frame(node[2], full_path[2][0], self.prim.num_headings)
                    w_node_path.append([node[0], node[1], theta])
                node_path = w_node_path

                # transform to global frame
                full_path[1] += lower_lim
                temp_array = np.zeros_like(self.full_costmap.cost_map, dtype=bool)
                if lower_lim < 0:
                    temp_array[:upper_lim] = full_swath[-lower_lim:]
                else:
                    temp_array[lower_lim: upper_lim] = full_swath
                full_swath = temp_array  # better solution would be to return costmap subset along with swath
                node_path = np.asarray(node_path).T
                node_path[1] += lower_lim
                closed_set = {
                    k: (v[0][0], v[0][1] + lower_lim, v[0][2])
                    for k, v in closed_set.to_dict().items()
                }

                self.logger.info('path length {}'.format(sum(node_path_length)))
                self.logger.info('g_score at goal {}'.format(g_score[goal]))

                # return a big tuple with all the relevant information
                return (
                    full_path,               # planned path
                    full_swath,              # swath boolean array of the planned path
                    node_path,               # coordinates of the state lattice nodes in the planned path
                    closed_set,              # expanded nodes
                    g_score[goal],           # cost of the path
                    swath_cost,              # swath cost of the path
                    sum(node_path_length),   # path length
                    edge_seq                 # edges used in the path
                )

            open_set.pop(node)
            closed_set.add(node)

            # find the base heading
            base_heading = node[2] % self.prim.num_base_h
            origin = (0, 0, base_heading)

            # get the edge set based on the current node heading
            edge_set = self.prim.edge_set_dict[origin]

            for e in edge_set:
                if e not in self.rotated_prims:
                    self.rotated_prims[e] = R @ e
                neighbour = self.concat(node, self.rotated_prims[e], base_heading, self.prim.spacing)

                # check if neighbour is in closed set
                if neighbour in closed_set:
                    continue

                if (
                        # check if neighbouring lattice node is within environment boundaries
                        0 < neighbour[0] < self.subset_costmap.shape[1] and
                        0 < neighbour[1] < self.subset_costmap.shape[0]
                ):
                    # get swath and swath cost
                    key = (e, int(node[2]))
                    if key not in self.swath_dict:
                        self.swath_dict[key] = rotate_swath(self.orig_swath_dict[key], theta_0)  # takes a big chunk of time in algorithm
                        self.swath_arg_dict[key] = np.argwhere(self.swath_dict[key])

                    if node is not start:
                        swath_overlap_mask[:] = 0
                        swath_overlap_mask[swath_overlap[node]] = 1
                    swath_cost = self.get_swath_cost(node,
                                                     self.swath_arg_dict[key],
                                                     self.subset_costmap,
                                                     self.swath_array_centre,
                                                     swath_overlap_mask)
                    assert swath_cost >= 0, 'swath cost is negative! {}'.format(swath_cost)

                    temp_path_length = self.prim.path_lengths[(origin, e)]
                    temp_g_score = g_score[node] + swath_cost + temp_path_length

                    # check if neighbour has already been added to open set
                    neighbour_in_open_set = False
                    if neighbour in open_set:
                        neighbour_in_open_set = True
                        neighbour = open_set.query(neighbour)[0]  # get the key the first time neighbour was added

                    if temp_g_score < g_score.get(neighbour, np.inf):
                        # this path to neighbor is better than any previous one. Record it!
                        came_from[neighbour] = node
                        came_from_by_edge[neighbour] = (origin, e)
                        path_length[neighbour] = temp_path_length
                        g_score[neighbour] = temp_g_score
                        new_f_score = g_score[neighbour] + (
                            self.weight * self.heuristic(neighbour, goal_y, theta_0) if self.weight else 0)
                        generation[neighbour] = generation[node] + 1
                        rr, cc = (self.swath_arg_dict[key] + [[int(round(node[1])) - self.swath_array_centre,
                                                               int(round(node[0])) - self.swath_array_centre]]).T
                        swath_overlap[neighbour] = (rr, cc)

                        if not neighbour_in_open_set:
                            open_set.add(neighbour)
                            f_score[neighbour] = new_f_score  # add a new entry
                            open_set_queue.put((new_f_score, neighbour))

                        else:
                            old_f_score = f_score[neighbour]
                            open_set_queue.update(orig_item=(old_f_score, neighbour),
                                                  new_item=(new_f_score, neighbour))
                            f_score[neighbour] = new_f_score  # update an existing entry

                        # plt.figure()
                        # plt.imshow(self.subset_costmap, origin='lower')  # helpful plots for debugging
                        # plt.plot(node[0], node[1], 'xg')
                        # plt.plot(neighbour[0], neighbour[1], 'xr')
                        # plt.show()

        self.logger.warning('Failed to find a path! Expanded {} nodes'.format(len(closed_set)))
        self.diagnostics = {'start': start,
                            'goal': goal_y,
                            'limits': (lower_lim, upper_lim),
                            'expanded': closed_set,
                            'cost_map': self.subset_costmap}
        return False

    def build_path(self, path, came_from_by_edge, start, theta_0) -> Tuple[np.ndarray, np.ndarray, dict, list]:
        """
        Path returned from graph search only consists of nodes between edges
        Need to construct the path via the primitive paths from these nodes
        """
        full_path = []
        full_swath = np.zeros_like(self.subset_costmap, dtype=bool)
        pt_a = start
        edge_seq = []
        prim_count = {k: {} for k in self.prim.edge_set_dict}
        for pt_b in path[1:]:
            key = came_from_by_edge[pt_b]
            edge_seq.append(key)
            path_ab = self.prim.paths[key]
            origin, edge = key
            theta = heading_to_world_frame(pt_a[2] - origin[2], theta_0, self.prim.num_headings)

            # rotate
            rot_path_ab = self.prim.rotate_path(path_ab, theta)

            # add start point
            rot_path_ab[0] += pt_a[0]
            rot_path_ab[1] += pt_a[1]
            full_path.append(rot_path_ab)

            # add swath
            swath = self.get_swath(pt_a,
                                   self.swath_dict[(edge, int(pt_a[2]))],
                                   self.subset_costmap.shape,
                                   self.swath_array_centre)

            # aggregating swaths themselves is fine since
            # the swath array is of type bool
            full_swath += swath

            # for debugging purposes, keep track of the prims picked
            prim_count[origin][edge] = prim_count[origin][edge] + 1 if key in prim_count[origin] else 1

            # update start point
            pt_a = pt_b

        return np.hstack(full_path), full_swath, prim_count, edge_seq

    @staticmethod
    def _get_swath_cost(start_pos, swath, cost_map, swath_array_centre, swath_overlap_mask) -> float:
        cost = 0

        for i in swath:
            # indices cannot be negative or be greater than the cost map size
            ind1 = int(round(start_pos[1])) + i[0] - swath_array_centre
            ind2 = int(round(start_pos[0])) + i[1] - swath_array_centre
            if ind1 < 0 or ind2 < 0 or ind1 >= cost_map.shape[0] or ind2 >= cost_map.shape[1]:
                return np.inf
            if not swath_overlap_mask[ind1, ind2]:
                cost += cost_map[ind1, ind2]

        return cost

    @staticmethod
    def get_swath(start_pos, raw_swath, cost_map_shape, swath_array_centre) -> np.ndarray:
        # swath mask has starting node at the centre and want to put at the starting node of currently expanded node
        # in the costmap, need to remove the extra columns/rows of the swath mask
        swath_size = raw_swath.shape[0]
        min_y = int(round(start_pos[1])) - swath_array_centre
        max_y = int(round(start_pos[1])) + swath_array_centre + 1
        min_x = int(round(start_pos[0])) - swath_array_centre
        max_x = int(round(start_pos[0])) + swath_array_centre + 1

        # Too close to the bottom
        a0 = 0
        if min_y < 0:
            a0 = abs(min_y)
            min_y = 0

        # Too close to the top
        b0 = swath_size
        if max_y >= cost_map_shape[0]:
            b0 = swath_size - (max_y - (cost_map_shape[0] - 1))
            max_y = cost_map_shape[0] - 1

        # Too far to the left
        a1 = 0
        if min_x < 0:
            a1 = abs(min_x)
            min_x = 0

        # Too far to the right
        b1 = swath_size
        if max_x >= cost_map_shape[1]:
            b1 = swath_size - (max_x - (cost_map_shape[1] - 1))
            max_x = cost_map_shape[1] - 1

        # fit raw swath onto costmap centred at start_pos
        swath = np.zeros(cost_map_shape, dtype=bool)
        swath[min_y:max_y, min_x:max_x] = raw_swath[a0:b0, a1:b1]

        # plt.imshow(raw_swath, origin='lower')
        # plt.show()
        # plt.imshow(swath, origin='lower')
        # plt.show()

        # compute cost
        return swath

    def heuristic(self, p0: Tuple, goal_y: float, theta_0: float) -> float:
        theta_0 = heading_to_world_frame(p0[2], theta_0, self.prim.num_headings)
        return (
                # heuristic for the path length
                dubins_heuristic((p0[0], p0[1], theta_0), goal_y,
                                 self.prim.turning_radius,
                                 (0, self.subset_costmap.shape[1] - 1))[0]
                +
                # heuristic for the swath cost
                self.obstacles_only_heuristic[int(p0[1])]
        )

    @staticmethod
    def _concat(x: Tuple, y: Tuple, base_heading: int, spacing: float) -> Tuple:
        # find the position and heading of the two points
        p1_theta = x[2] * spacing - spacing * base_heading  # starting heading
        p2_theta = y[2] * spacing  # edge heading

        result = [x[0] + (np.cos(p1_theta) * y[0] - np.sin(p1_theta) * y[1]),
                  x[1] + (np.sin(p1_theta) * y[0] + np.cos(p1_theta) * y[1])]

        # compute the final heading after concatenating x and y
        heading = (p2_theta + p1_theta) % M__2_PI

        return result[0], result[1], int(heading / spacing + 1e-6)

    @staticmethod
    def compute_obstacles_only_heuristic(costmap: np.ndarray, ship_width: int):
        """
        Find a lower bound for the swath cost given the costmap and the
        width (in costmap grid units) 'w' of the ship

        A lower bound is the minium sum of 'w' consecutive cells in each row
        of the costmap
        """
        # use sliding_window_view to create a sliding window view of the array
        windows = np.lib.stride_tricks.sliding_window_view(costmap, window_shape=(ship_width,), axis=1)

        # sum along the last axis
        window_sums = windows.sum(axis=-1)

        # find the minimum sum for each row
        min_sums = window_sums.min(axis=-1)

        # need to reverse order before doing cumsum since we
        # want the sum from a given row j to the last row
        return min_sums[::-1].cumsum()[::-1]
