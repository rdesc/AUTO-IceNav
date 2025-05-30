import numpy as np

from ship_ice_planner.utils.utils import compute_path_length
from ship_ice_planner.swath import compute_swath_cost


class Path:
    """
    Class to store path and swath information and compare new path to old path
    """
    def __init__(self,
                 path: np.ndarray = None,
                 swath: np.ndarray = None,
                 threshold_path_progress: float = 0,
                 threshold_dist_to_path: float = 0,
                 threshold_cost_diff: float = 0,
                 ship_vertices: np.ndarray = None,
                 costmap=None):
        self.path = path  # shape is 3 x n and order is start -> end
        self.swath = swath
        # threshold for how much progress the ship has to make before we update the path
        # parameter is passed in as a fraction [0, 1] of the path length
        self.threshold_path_progress_frac = threshold_path_progress
        self.path_length = None
        # threshold for how far the ship has to be from the path before we update the path
        # parameter is passed in as a distance in the same units as the path/swath
        self.threshold_dist_to_path = threshold_dist_to_path
        # threshold for how much the cost of the new path has to differ from the old path
        # before we update the path
        self.threshold_cost_diff = threshold_cost_diff
        self.ship_vertices = ship_vertices  # for computing swath cost
        self.costmap = costmap  # for computing swath cost

    def update(self,
               path: np.ndarray,
               swath: np.ndarray,
               ship_pos: np.ndarray) -> bool:

        if self.path is None:
            self.path = path
            self.swath = swath
            self.path_length = compute_path_length(path[:2].T, cumsum=True)
            return True  # return True to indicate that the path was updated

        # if no threshold is defined, just update the path
        if not (self.threshold_path_progress_frac or self.threshold_dist_to_path or self.threshold_cost_diff):
            self.path = path
            self.swath = swath
            return True

        # find index closest to ship position
        dist = np.linalg.norm(self.path[:2] - ship_pos[:2, None], axis=0)
        trunc_ind = np.argmin(dist)
        self.path = self.path[:, trunc_ind:]
        self.path_length = self.path_length[trunc_ind:]
        path_to_ship_dist = dist[trunc_ind]

        # check if the path progress fraction is defined and if the ship has made enough progress along the path
        path_progress_condition = (
                self.threshold_path_progress_frac and
                self.path_length[0] > self.threshold_path_progress_frac * self.path_length[-1]
        )

        # check if the distance to path threshold is defined and if the ship's distance to the path is greater than the threshold
        distance_to_path_condition = (
                self.threshold_dist_to_path and
                (path_to_ship_dist > self.threshold_dist_to_path)
        )

        # check if the cost difference threshold is defined and if the new path's cost is greater than the old path's cost
        _, prev_path_cost = compute_swath_cost(cost_map=self.costmap.cost_map,
                                               path=self.path.T,
                                               ship_vertices=self.ship_vertices,
                                               resample=False,
                                               compute_cumsum=False)
        max_y = self.path[1, -1]
        temp_swath = np.copy(swath)
        temp_swath[int(max_y):, :] = False  # zero out the swath above the max y of the path
        new_path_cost = self.costmap.cost_map[temp_swath].sum()

        cost_diff_condition = (
                self.threshold_cost_diff and
                (prev_path_cost - new_path_cost) / prev_path_cost > self.threshold_cost_diff
        )

        # if either condition is met, update the path
        if path_progress_condition or distance_to_path_condition or cost_diff_condition:
            self.path = path
            self.swath = swath
            self.path_length = compute_path_length(path[:2].T, cumsum=True)
            return True

        return False
