import numpy as np

from ship_ice_planner.utils.utils import compute_path_length


class Path:
    """
    Class to store path and swath information and compare new path to old path
    """
    def __init__(self,
                 path: np.ndarray = None,
                 swath: np.ndarray = None,
                 threshold_path_progress: float = 0,
                 threshold_dist_to_path: float = 0):
        self.path = path  # shape is 3 x n and order is start -> end
        self.swath = swath
        # threshold for how much progress the ship has to make before we update the path
        # parameter is passed in as a fraction [0, 1] of the path length
        self.threshold_path_progress_frac = threshold_path_progress
        self.path_length = None
        # threshold for how far the ship has to be from the path before we update the path
        # parameter is passed in as a distance in the same units as the path/swath
        self.threshold_dist_to_path = threshold_dist_to_path

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
        if not (self.threshold_path_progress_frac or self.threshold_dist_to_path):
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

        # if either condition is met, update the path
        if path_progress_condition or distance_to_path_condition:
            self.path = path
            self.swath = swath
            self.path_length = compute_path_length(path[:2].T, cumsum=True)
            return True

        return False

# TODO: add option for comparing old cost to new cost, no need to recompute swath