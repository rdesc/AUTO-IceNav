import os
import pickle
from typing import Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy import ndimage
from skimage.draw import draw

from ship_ice_planner.geometry.utils import Rxy
from ship_ice_planner.utils.utils import resample_path

Swath = Dict[Tuple, np.ndarray]
CACHED_SWATH = '.swath.pkl'  # save to disk swaths so no need to regenerate every time


def generate_swath(ship, prim, cache=True) -> Union[Swath, Tuple]:
    """
    Generate swath for each motion primitive given the ship footprint
    For each edge we generate the swath centered on a square array
    This makes it easy to rotate the swath about the image centre

    The resolution of the swath (i.e. the size of a grid cell) is
    the same as the resolution of the costmap, i.e. they are 1:1
    """
    if os.path.isfile(CACHED_SWATH) and cache:
        print('LOADING CACHED SWATH! Confirm, this is expected behaviour...')
        return pickle.load(open(CACHED_SWATH, 'rb'))

    # swath is a square boolean array with the ship positioned at the centre
    start_pos_xy = prim.max_prim + int(np.ceil(ship.max_ship_length / 2)) + 1
    swath_shape = (start_pos_xy * 2 + 1, start_pos_xy * 2 + 1)  # needs to be odd
    assert start_pos_xy * 2 + 1 == swath_shape[0] == swath_shape[1]

    swath_dict = {}
    # generate the swaths for each 0, 90, 180, 270 degrees
    # since our headings are uniformly discretized along the unit circle
    for i, h in enumerate(range(0, prim.num_headings, prim.num_headings // 4)):
        for origin, edge_set in prim.edge_set_dict.items():
            start = [start_pos_xy, start_pos_xy, origin[2]]

            for edge in edge_set:
                array = np.zeros(swath_shape, dtype=bool)
                path = prim.paths[(origin, edge)]  # assumes path is sampled finely enough to get the swath
                path = prim.rotate_path(path, np.pi / 2 * i)

                for p in path.T:
                    x, y, theta = p
                    x, y = x + start[0], y + start[1]
                    rot_vi = np.array([[x], [y]]) + Rxy(theta) @ ship.vertices.T
                    rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :])
                    array[rr, cc] = True  # if this fails then the swath shape needs to be made bigger by increasing start_pos_xy

                swath_dict[edge, h + origin[2]] = array

    if cache:
        with open(CACHED_SWATH, 'wb') as file:
            pickle.dump((swath_dict, swath_shape), file)

    return swath_dict, swath_shape


def view_all_swaths(swath_dict: Swath) -> None:
    running = True
    def on_close(event):
        nonlocal running
        if event.key == 'escape':
            running = False
            plt.close(plt.gcf())

    for key, _ in swath_dict.items():
        if not running:
            break

        f, ax = plt.subplots()
        f.canvas.mpl_connect('key_press_event', on_close)
        shape = swath_dict[key].shape
        ax.plot(shape[0] // 2, shape[0] // 2, 'bx')
        ax.plot(shape[0] // 2 + key[0][0], shape[0] // 2 + key[0][1], 'bx')
        ax.imshow(swath_dict[key], origin='lower')
        ax.set_title('Swath key {}\nPress "escape" to stop viewing swaths'.format(str(key)))
        plt.show()


def rotate_swath(swath, theta: float):
    return ndimage.rotate(swath, - theta * 180 / np.pi, order=0, reshape=False, prefilter=False)


def compute_swath_cost(cost_map: np.ndarray,
                       path: np.ndarray,
                       ship_vertices: np.ndarray,
                       resample: Union[bool, float] = False,
                       compute_cumsum=False,
                       plot=False
                       ) -> Tuple[np.ndarray, Union[float, list]]:
    """
    Generate the swath given a costmap, path, and ship footprint. This swath will be a little
    different from the one generated by A* search since in A* we use the processed swath dict.
    This method is useful when generating the swath on the final smoothed path.
    NOTE, this is a very expensive operation and should only be used for plotting/debugging purposes!

    This also assumes the path is sampled at a high enough frequency to capture the swath
    """
    # keep track of swath
    if resample:
        path = resample_path(path, resample)

    swath = np.zeros_like(cost_map, dtype=bool)
    cumsum = []
    for x, y, theta in path:  # full path is of shape n x 3
        R = Rxy(theta)

        # rotate/translate vertices of ship from origin to sampled point with heading = theta
        rot_vi = np.array([[x], [y]]) + R @ ship_vertices.T

        # draw rotated ship polygon and put occupied cells into a mask
        rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :], shape=cost_map.shape)
        swath[rr, cc] = True

        if compute_cumsum:
            cumsum.append(cost_map[swath].sum())

    cost = cost_map[swath].sum()

    if plot:
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(cost_map, origin='lower')
        ax[0].plot(path[:, 0], path[:, 1], 'r')
        swath_im = np.zeros(swath.shape + (4,))  # init RGBA array
        swath_im[:] = colors.to_rgba('m')
        swath_im[:, :, 3] = swath  # set pixel transparency to 0 if pixel value is 0
        # plot the full swath
        ax[0].imshow(swath_im, origin='lower', alpha=0.3, zorder=10)
        ax[1].plot(cumsum)
        ax[1].set_title('Cumulative Sum = ' + str(round(cumsum[-1])) if compute_cumsum else 'Cumulative Sum Disabled')
        plt.show()

    if compute_cumsum:
        return swath, cumsum

    return swath, cost
