""" Bare bones ship-ice simulation with pymunk """
import numpy as np
import pymunk
from matplotlib import pyplot as plt

from ship_ice_planner.src.cost_map import CostMap
from ship_ice_planner.src.ship import Ship
from ship_ice_planner.src.utils.plot import Plot
from ship_ice_planner.src.utils.sim_utils import create_polygon

if __name__ == '__main__':
    print("simulation start")

    # size of map/environment
    m, n = 100, 40

    # ship params
    ship_pose = (20, 20, np.pi/2)  # ship starting position and heading
    ship_vel = 10  # ship speed
    ship_vertices = np.asarray([[1., -0.],
                                [0.9, 0.10],
                                [0.5, 0.25],
                                [-1., 0.25],
                                [-1., -0.25],
                                [0.5, -0.25],
                                [0.9, -0.10]]) * 4  # can scale ship size here

    # obstacle params
    obs_params = dict(
        num_obs=20,  # number of random ice obstacles
        min_r=4,  # max and min radii of obstacles
        max_r=8,
        min_x=0,  # max and min x position of obs
        max_x=40,
        min_y=25,  # max and min y position of obs
        max_y=100,
    )

    # sim params
    t_max = np.inf  # max simulation time

    # generate obstacles
    # obs_dicts is a list of dicts, each dict is of the form {'radius': f, 'vertices', [[]], 'centre', (,)}
    # obstacles is a list of only the obstacle vertices
    obs_dicts, obstacles = CostMap.generate_obstacles(**obs_params)

    # make a new pymunk env with ship and obstacle
    space = pymunk.Space()
    static_body = space.static_body  # create a static body for friction constraints

    # init ship sim objects
    ship_body, ship_shape = Ship.sim(ship_vertices, ship_pose, body_type=pymunk.Body.KINEMATIC,
                                     velocity=(np.cos(ship_pose[2]) * ship_vel, np.sin(ship_pose[2]) * ship_vel))
    space.add(ship_body, ship_shape)
    ship_shape.collision_type = 1

    # init polygon objects
    polygons = []
    for ob in obs_dicts:
        poly = create_polygon(
                space, (ob['vertices'] - np.array(ob['centre'])).tolist(),
                *ob['centre'], density=10
            )
        poly.collision_type = 2
        polygons.append(poly)

    # flag which keeps sim running into first collision
    running = True

    def collide(arbiter, space, data):  # collision handler
        global running
        print('collision!')
        # running = False  # kill simulation after the end of collision
        return True

    # add callback for when collision occurs
    handler = space.add_collision_handler(1, 2)
    # from pymunk docs
    # separate: two shapes have just stopped touching for the first time
    # begin: two shapes just started touching for the first time
    handler.separate = collide

    # initialize plotting/animation
    plot = Plot(
        np.zeros((m, n)), obs_dicts,
        ship_pos=ship_pose, ship_vertices=np.asarray(ship_vertices),
        map_figsize=None, y_axis_limit=m,
    )
    plt.show(block=False)

    steps = 10
    while running:
        for _ in range(steps):
            space.step(0.02 / steps)

        # get updated obstacle info
        obstacles = CostMap.get_obs_from_poly(polygons)

        # update animation
        plt.pause(0.01)  # makes sure matplotlib has time to update figure
        # update ship patch based on updated ship pose
        plot.update_ship(ship_body, ship_shape)
        plot.update_obstacles(obstacles=obstacles)

    print('final velocity obstacle')
    for poly in polygons:
        print('\t', tuple(poly.body.velocity))
    print('final velocity ship', tuple(ship_body.velocity))
    plt.show()

    print('Done')
    plt.close('all')
