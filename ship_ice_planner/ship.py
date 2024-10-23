from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from ship_ice_planner.geometry.utils import Rxy, euclid_dist

# ship vertices requirements:
#   - defined in the local frame of the ship
#   - centroid at (0,0)
#   - symmetric about the x-axis
#   - ship is facing to the right i.e. heading is 0
OEB_PSV_VERTICES = [  # from NRC
     [1., -0.],
     [0.9, 0.10],
     [0.5, 0.25],
     [-1., 0.25],
     [-1., -0.25],
     [0.5, -0.25],
     [0.9, -0.10]
]
FULL_SCALE_PSV_VERTICES = [
    # ship bow from "GPU-Event-Mechanics Evaluation of Ice Impact Load Statistics"
    # length and width from https://github.com/cybergalactic/MSS/tree/master/VESSELS
    [43.1, 0.0],
    [35.4, 4.0],
    [27.5, 6.5],
    [11.6, 9.0],
    [-33.1, 9.0],
    [-33.1, -9.0],
    [11.6, -9.0],
    [27.5, -6.5],
    [35.4, -4.0],
]


class Ship:
    def __init__(self, scale: float, vertices: Union[List, np.ndarray], padding: float = 0., mass: float = None, **kwargs):
        """
        :param scale: scale factor for vertices
        :param vertices: list of vertices that define the ship
        :param padding: padding to add to the vertices
        :param mass: mass of the ship in kg
        """
        self.vertices = np.asarray(
            [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in vertices]
        ) * scale
        self.max_ship_length = np.ceil(max(euclid_dist(a, b) for a in self.vertices for b in self.vertices))
        assert self.max_ship_length != 0, 'ship length cannot be 0'
        self.mass = mass
        self.width = self.vertices[:, 1].max() - self.vertices[:, 1].min()  # aka ship beam
        self.length = self.vertices[:, 0].max() - self.vertices[:, 0].min()

    def plot(self, turning_radius, heading=np.pi / 2, title=None):
        start_pos = (0, 0, heading)

        x, y, psi = start_pos
        R = Rxy(psi)

        f, ax = plt.subplots()
        # plot ship as polygon
        rot_vertices = self.vertices @ R.T + start_pos[:2]
        ax.add_patch(patches.Polygon(rot_vertices, True, fill=True, zorder=10, alpha=0.5))

        # plot the vertices
        for v in rot_vertices:
            ax.plot(v[0], v[1], 'k.', markersize=2)
            # add text to show vertex coordinates
            # ax.text(v[0], v[1], f'({v[0]:.2f}, {v[1]:.2f})')

        # plot line to show initial heading
        ax.plot([x, x + self.max_ship_length * np.cos(psi)],
                [y, y + self.max_ship_length * np.sin(psi)], 'r')

        # plot lines to show minimal turning radius
        a = np.arange(0, 2 * np.pi, 0.01)
        ax.plot((x - turning_radius * np.sin(psi) + turning_radius * np.cos(a)).tolist(),
                (y + turning_radius * np.cos(psi) + turning_radius * np.sin(a)).tolist(), 'g')
        ax.plot((x + turning_radius * np.sin(psi) + turning_radius * np.cos(a)).tolist(),
                (y - turning_radius * np.cos(psi) + turning_radius * np.sin(a)).tolist(), 'g')

        # plot the radius
        ax.plot([x, x - turning_radius * np.sin(psi)],
                [y, y + turning_radius * np.cos(psi)], 'b')

        ax.plot([x, x + turning_radius * np.sin(psi)],
                [y, y - turning_radius * np.cos(psi)], 'b')

        # plot the width
        ax.plot([x - self.width / 2 * np.cos(psi + np.pi / 2), x + self.width / 2 * np.cos(psi + np.pi / 2)],
                [y - self.width / 2 * np.sin(psi + np.pi / 2), y + self.width / 2 * np.sin(psi + np.pi / 2)],
                'k--', linewidth=2, zorder=20)

        # title
        if title is not None:
            ax.set_title(title)

        ax.set_aspect('equal')

        plt.show()


if __name__ == '__main__':
    # run main to test ship config
    ship = Ship(
        padding=0.,
        scale=1,
        vertices=FULL_SCALE_PSV_VERTICES
    )
    print('ship length = {:.2f} m, ship beam = {:.2f} m'.format(ship.length, ship.width))
    ship.plot(turning_radius=150)
