import logging
import random
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from skimage import draw

from ship_ice_planner.geometry.polygon import *
from ship_ice_planner.geometry.utils import Rxy_3d, Rxy, euclid_dist
from ship_ice_planner.image_process.ice_concentration import compute_sic
from ship_ice_planner.utils.plot import Plot

# define an arbitrary max cost applied to a cell in the costmap
MAX_COST = 1e10


class CostMap:
    """
    Discretizes the environment into a 2D map and assigns a cost to each grid cell.
    The center of a grid cell is considered the costmap coordinate.

    E.g. for a map environment of width 3 meters and height 1 meter we have a 2x4 costmap with a scale of 1.
    So we have a total of 8 grid cells as shown below. The X's represent the center of the grid cell
    and below the X's are the corresponding world coordinates (x,y) for this map. Indexing the costmap
    is done in the following way: cost_map[y, x] where y is the row and x is the column.
    -------------------------------------
    |        |        |        |        |
    |    X   |    X   |    X   |    X   |
    | (0,1)  | (1,1)  | (2,1)  | (3,1)  |
    -------------------------------------
    |        |        |        |        |
    |    X   |    X   |    X   |    X   |
    | (0,0)  | (1,0)  | (2,0)  | (3,0)  |
    -------------------------------------
    """
    def __init__(self, scale: float, length: int, width: int, collision_cost_weight: float = None,
                 obs_cost_buffer_factor: float = 0, ship_mass: float = None, boundary_margin: int = 0, padding: int = 0,
                 ice_resistance_weight: float = 0, ice_thickness: float = None, ice_density: float = None,
                 sic_kernel: Tuple[int, int] = None
                 ):
        """
        :param scale: the scaling factor for the costmap, divide by scale to get world units i.e. meters
        :param length: the height in meters of the channel
        :param width: the width in meters of the channel
        :param collision_cost_weight: weight for the collision cost term
        :param obs_cost_buffer_factor: factor to scale up the polygons for collision cost
        :param ship_mass: mass of the ship in kg
        :param boundary_margin: margin in metres on either side of the channel to apply a max cost
        :param padding: padding to add to the costmap
        :param ice_resistance_weight: weight for the ice resistance term
        :param ice_thickness: mean ice thickness in metres
        :param ice_density: mean ice density in kg/m^3
        :param sic_kernel: kernel size in costmap grid/pixel units for computing sea ice concentration
        """
        self.scale = scale  # scales everything by this factor
        self.padding = padding
        width, height = int(width * scale + 1), int(length * scale + 1) + self.padding
        self.cost_map = np.zeros((height, width))
        self.bin_occ_map = np.zeros((height, width), dtype=bool)
        self.collision_cost_weight = collision_cost_weight
        self.obs_cost_buffer_factor = obs_cost_buffer_factor
        self.ship_mass = ship_mass
        self.ice_resistance_weight = ice_resistance_weight
        self.ice_thickness = ice_thickness
        self.ice_density = ice_density
        self.sic_kernel = sic_kernel

        self.obstacles = []  # contains only the obstacles considered for building costmap
        self.all_obstacles = []
        self.margin = int(boundary_margin * scale)

        self.logger = logging.getLogger(__name__)

        if not self.collision_cost_weight and not self.ice_resistance_weight:
            self.logger.warning('\033[93mNo cost weights specified, costmap will be zero!\033[0m')

        if self.collision_cost_weight:
            assert self.ship_mass

        if self.ice_resistance_weight:
            assert self.sic_kernel

    @property
    def shape(self):
        return self.cost_map.shape

    def boundary_cost(self) -> None:
        if not self.margin:
            return
        self.cost_map[:, :self.margin] = MAX_COST
        self.cost_map[:, -self.margin:] = MAX_COST

    def clip_boundary_cost(self) -> None:
        if not self.margin:
            return
        # option to clip boundary cost to max costmap cost
        costmap_max = self.get_costmap_max()
        self.cost_map[:, :self.margin] = costmap_max
        self.cost_map[:, -self.margin:] = costmap_max

    def get_costmap_max(self):
        if not self.margin:
            return self.cost_map[:].max()
        return self.cost_map[:, self.margin:-self.margin].max()

    @staticmethod
    def populate_costmap(cost_map, centre, radius, pixels, constant_term, max_cost) -> None:
        rr, cc = pixels
        centre_x, centre_y = centre
        dists = euclid_dist((rr, cc), (centre_y, centre_x))
        cost_map[rr, cc] = np.maximum(
            cost_map[rr, cc],
            (
                    constant_term *
                    # 0.5 (or half a pixel size) is added to the radius to ensure we still have a cost at the edge of the polygon
                    ((radius + 0.5) ** 2 - dists ** 2) / (radius + 0.5) ** 2  # should be between 0 and 1
            )
        ).clip(0, max_cost)

    def update(self,
               obs_vertices: List[Any],
               obs_masses: List[float] = None,
               ship_pos_y: float = 0,
               ship_speed: float = 0,
               goal: float = None) -> None:
        """ Updates the costmap with the new obstacles """
        # clear costmap and obstacles
        self.cost_map[:] = 0
        self.bin_occ_map[:] = 0
        self.obstacles = []
        self.all_obstacles = []

        # update obstacles based on new positions
        for idx, ob_vert in enumerate(obs_vertices):
            # scale vertices to the costmap scale
            ob_vert = np.asarray(ob_vert) * self.scale

            self.all_obstacles.append(ob_vert)

            # quickly discard obstacles not part of horizon
            if goal and np.all(ob_vert[:, 1] > goal) or np.all(ob_vert[:, 1] < ship_pos_y):
                continue

            # scale up the obstacle
            if self.obs_cost_buffer_factor:
                ob_vert_swelled = shrink_or_swell_polygon(ob_vert, swell=True, factor=self.obs_cost_buffer_factor)
            else:
                ob_vert_swelled = ob_vert

            # get pixel coordinates on costmap that are contained inside obstacle/polygon
            rr, cc = draw.polygon(ob_vert_swelled[:, 1], ob_vert_swelled[:, 0], shape=self.cost_map.shape)

            # skip if 0 area
            if len(rr) == 0 or len(cc) == 0:
                continue

            # compute centroid of polygon
            # https://en.wikipedia.org/wiki/Centroid#Of_a_finite_set_of_points
            # centre = sum(cc) / len(cc), sum(rr) / len(rr)  # not as accurate
            centre = poly_centroid_shapely(ob_vert)  # should be the same as the centroid of the scaled up polygon

            # compute the obstacle radius
            radius = poly_radius(ob_vert_swelled, centre)

            # get ice mass
            mass_ice = obs_masses[idx] if obs_masses is not None else None
            if mass_ice is None:
                mass_ice = self.ice_thickness * self.ice_density * poly_area(ob_vert / self.scale)  # kg

            # add to list of obstacles
            self.obstacles.append({
                'vertices': ob_vert,
                'centre': centre,
                'radius': radius,
                'mass': mass_ice,
                'pixels': (rr, cc)
            })

            if self.collision_cost_weight and ship_speed and mass_ice:
                assert mass_ice is not None
                constant_term = (
                        self.collision_cost_weight *
                        (ship_speed ** 2 * mass_ice * self.ship_mass * (mass_ice + 2 * self.ship_mass)) /
                        (2 * (self.ship_mass + mass_ice) ** 2)  # kg m^2 / s^2
                )

                # compute the cost and update the costmap
                self.populate_costmap(self.cost_map,
                                      centre=centre,  # in costmap grid units
                                      radius=radius,  # in costmap grid units
                                      pixels=(rr, cc),
                                      constant_term=constant_term,
                                      max_cost=MAX_COST)

                # make sure there are no pixels with 0 cost
                # can disable check with -O in command line
                # assert np.all(self.cost_map[rr, cc] >= 0)

            if self.obs_cost_buffer_factor:
                # need to update the binary occupancy map with the original vertices
                rr2, cc2 = draw.polygon(ob_vert[:, 1], ob_vert[:, 0], shape=self.cost_map.shape)
            else:
                rr2, cc2 = rr, cc
            self.bin_occ_map[rr2, cc2] = 1

        if self.ice_resistance_weight:
            # compute sea ice concentration
            sic = compute_sic(
                self.bin_occ_map,
                kernel=self.sic_kernel,
                stride=1,
                # show_plot=True
            )

            self.cost_map *= (
                # see ice resistance force model
                # "Autonomous Passage Planning for a Polar Vessel" https://arxiv.org/pdf/2209.02389.pdf
                sic ** self.ice_resistance_weight
            )

        # apply a cost to the boundaries of the channel
        self.boundary_cost()

    def plot(self, ship_pos=None, ship_vertices=None, prim=None, goal=None):
        f, ax = plt.subplots(figsize=(6, 10))
        # plot the costmap
        cost_map = self.cost_map.copy()

        if cost_map.sum() == 0:
            cost_map[:] = 1
        else:
            if self.margin:
                cost_map[cost_map == np.max(cost_map)] = np.nan  # set the max to nan
            f2, ax2 = plt.subplots(figsize=(6, 10))
            ax2.imshow(self.bin_occ_map, cmap='Greys', origin='lower')
            ax2.set_title('Binary Occupancy Map')

        im = ax.imshow(cost_map, origin='lower', cmap='viridis')

        # first plot all the obstacles
        for obs in self.all_obstacles:
            ax.add_patch(
                patches.Polygon(obs, True, fill=False)
            )

        # plot the polygons
        for obs in self.obstacles:
            if 'centre' in obs:
                # plot the centre of each polygon
                x, y = obs['centre']
                ax.plot(x, y, 'rx', markersize=2)

                # plot circle around polygon with computed radius
                p = np.arange(0, 2 * np.pi, 0.01)
                ax.plot(x + obs['radius'] * np.cos(p),
                        y + obs['radius'] * np.sin(p), 'c', linewidth=1)

        ax.set_title('Costmap unit: {}x{} m'.format(1 / self.scale, 1 / self.scale))
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        # plot ship if necessary
        if ship_pos is not None:
            assert ship_vertices is not None
            theta = ship_pos[2]
            R = Rxy(theta)
            ax.plot(ship_pos[0], ship_pos[1], 'ro', markersize=2)
            ax.add_patch(patches.Polygon(ship_vertices @ R.T + [ship_pos[0], ship_pos[1]], True, fill=True))

            # plot the motion primitives if necessary
            if prim is not None:
                ax.set_title('Costmap unit: {}x{} m \n Lattice unit: {} m, Turning radius: {} l.u.'
                             .format(1 / self.scale, 1 / self.scale,
                                     prim.scale / self.scale,
                                     prim.turning_radius / prim.scale))
                origin = (0, 0, 0)
                edge_set = prim.edge_set_dict[origin]
                R2 = Rxy_3d(theta)
                for edge in edge_set:
                    path = prim.paths[(origin, tuple(edge))]
                    x, y, _ = R2 @ path
                    ax.plot([i + ship_pos[0] for i in x],
                            [j + ship_pos[1] for j in y], 'r', linewidth=0.5)

        if goal is not None:
            ax.axhline(goal, color='r', linestyle='--', label='goal')

        Plot.scale_axis_labels(ax, self.scale)
        ax.legend()

        if self.obstacles is not None and len(self.obstacles) > 0:
            f.colorbar(im, ax=ax)

            ob_radii = [ob['radius'] / self.scale for ob in self.obstacles]
            print('Radius: average = {:.4f} m, max = {:.4f} m, min = {:.4f} m'
                  .format(np.mean(ob_radii),
                          np.max(ob_radii),
                          np.min(ob_radii)))
            ob_areas = [poly_area(ob['vertices'] / self.scale) for ob in self.obstacles]
            print('Area: average = {:.4f} m^2, max = {:.4f} m^2, min = {:.4f} m^2'
                  .format(np.mean(ob_areas),
                          np.max(ob_areas),
                          np.min(ob_areas)))
            ob_masses = [ob['mass'] for ob in self.obstacles]
            print('Mass: average = {:.4f} kg, max = {:.4f} kg, min = {:.4f} kg'
                  .format(np.mean(ob_masses),
                          np.max(ob_masses),
                          np.min(ob_masses)))

        plt.show()


def demo():
    from ship_ice_planner.utils.sim_utils import generate_obstacles
    from ship_ice_planner.utils.sim_utils import ICE_DENSITY, ICE_THICKNESS, SHIP_MASS
    from ship_ice_planner.ship import FULL_SCALE_PSV_VERTICES
    from ship_ice_planner.ship import Ship

    # seed for deterministic results
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    # channel dimensions
    width = 50  # m
    height = 300  # m
    scale = 1  # pixels per meter

    # ship
    ship = Ship(scale=scale,
                vertices=FULL_SCALE_PSV_VERTICES,
                mass=SHIP_MASS)
    start = (width / 2 * scale,
             0 * scale,
             np.pi / 2)
    speed = 1  # m/s

    # goal
    goal = 200 * scale

    # generate obstacles
    obs_dict, obstacles = generate_obstacles(
        num_obs=100,
        # this is all in meters
        min_r=4,
        max_r=10,
        min_x=0,
        max_x=width,
        min_y=40,
        max_y=height,
        seed=seed
    )
    # initialize costmap
    costmap = CostMap(
        scale=scale,
        length=height, width=width,
        ship_mass=ship.mass,  # kg
        collision_cost_weight=1,
        obs_cost_buffer_factor=0.1,
        ice_thickness=ICE_THICKNESS,
        ice_density=ICE_DENSITY,
        ice_resistance_weight=1,
        sic_kernel=(int(width * scale // 2), int(width * scale // 2)),
    )

    # update obstacles with costmap
    costmap.update(obstacles,
                   ship_pos_y=start[1] - ship.length / 2,
                   ship_speed=speed,
                   goal=goal)

    # plot costmap
    costmap.plot(ship_pos=start,
                 ship_vertices=ship.vertices,
                 goal=goal)

    print('Mean cost: {:.4f}\nMax cost: {:.4f}'.format(np.mean(costmap.cost_map), np.max(costmap.cost_map)))


if __name__ == '__main__':
    # for testing costmap generation
    demo()
