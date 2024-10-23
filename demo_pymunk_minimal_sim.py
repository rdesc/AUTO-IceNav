"""
Bare bones ship-ice simulation with pymunk
Does not include the full-scale vessel dynamics
"""
import math
import random

import numpy as np
import pymunk
import pymunk.batch

from ship_ice_planner.geometry.utils import get_global_obs_coords
from ship_ice_planner.ship import FULL_SCALE_PSV_VERTICES
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.sim_utils import *

SEED = 1

np.random.seed(SEED)
random.seed(SEED)


if __name__ == '__main__':
    # can use -O flag to disable assertions or -OO to disable docstrings
    print('Launching pymunk minimal ship-ice sim demo...')

    # size of map/environment
    length, width = 200, 80  # m

    # ship params
    ship_pose = (width / 2, -40, np.pi / 2)  # ship starting position and heading
    ship_vel = 2                             # ship speed m/s
    ship_vertices = np.asarray(FULL_SCALE_PSV_VERTICES)

    # obstacle params
    obs_params = dict(
        num_obs=200,  # number of random ice obstacles
        min_r=4,  # max and min radii of obstacles
        max_r=20,
        min_x=0,  # max and min x position of obs
        max_x=width,
        min_y=25,  # max and min y position of obs
        max_y=length,
    )

    # sim params
    t_max = np.inf   # max simulation time
    dt = 0.02        # move simulation forward by dt at each main loop iteration
    steps = 10       # divide dt into N steps for more accurate simulation

    # plot params
    sim_figsize = (7, 7)    # larger size slows down animation
    save_animation = False  # option to save animation to file instead of showing plot

    # generate obstacles
    # obs_dicts is a list of dicts, each dict is of the form {'radius': f, 'vertices', [[]], 'centre', (,)}
    # obstacles is a list of only the obstacle vertices
    obs_dicts, obstacles = generate_obstacles(**obs_params, seed=SEED)
    sqrt_floe_areas = [math.sqrt(ob['area']) for ob in obs_dicts]
    print('Num obstacles', len(obs_dicts))

    # make a new pymunk env with ship and obstacle
    space = init_pymunk_space()

    # init pymunk physics objects
    ship_shape = create_sim_ship(space,  # polygon for ship
                                 ship_vertices.tolist(),
                                 ship_pose,
                                 body_type=pymunk.Body.KINEMATIC,
                                 velocity=(np.cos(ship_pose[2]) * ship_vel, np.sin(ship_pose[2]) * ship_vel))
    ship_shape.collision_type = 1  # for collision callbacks
    ship_body = ship_shape.body

    # polygons for ice floes
    polygons = generate_sim_obs(space, obs_dicts)
    poly_vertices = []  # generate these once
    for p in polygons:
        p.collision_type = 2  # for collision callbacks
        vertices = list_vec2d_to_numpy(p.get_vertices())
        poly_vertices.append(
            np.concatenate([vertices, [vertices[0]]])  # close the polygon
        )

    # make hard boundaries for the channel
    # create_channel_borders(space, length, width)

    # flag which keeps sim running into first collision
    running = True

    def collide(arbiter, space, data):  # collision handler
        global running
        print('collision!')
        # running = False  # kill simulation after the end of collision
        return True

    def pre_solve_handler(arbiter, space, data):
        return True

    def post_solve_handler(arbiter, space, data):
        pass

    # add callback for when collision occurs
    handler = space.add_collision_handler(1, 2)
    # from pymunk docs
    # separate: two shapes have just stopped touching for the first time
    # begin: two shapes just started touching for the first time
    # post_solve: two shapes are touching and collision response processed
    handler.pre_solve = pre_solve_handler
    handler.post_solve = post_solve_handler
    handler.separate = collide

    # initialize plotting/animation
    plot = Plot(
        obstacles=obs_dicts,
        ship_pos=ship_pose, ship_vertices=ship_vertices, save_animation=save_animation,
        map_figsize=None, sim_figsize=sim_figsize, y_axis_limit=length, track_fps=True,
    )

    def on_close(event):
        global running
        if event.key == 'escape':
            running = False
    plot.sim_fig.canvas.mpl_connect('key_press_event', on_close)

    buffer = pymunk.batch.Buffer()  # new batch API https://www.pymunk.org/en/latest/pymunk.batch.html
    batch_fields = (pymunk.batch.BodyFields.POSITION
                    | pymunk.batch.BodyFields.ANGLE
                    | pymunk.batch.BodyFields.VELOCITY
                    | pymunk.batch.BodyFields.ANGULAR_VELOCITY
                    )
    batch_dim = 6

    sim_time = 0

    while running:
        for _ in range(steps):
            space.step(dt / steps)

        sim_time += dt

        # get batched body data
        buffer.clear()
        pymunk.batch.get_space_bodies(space, batch_fields, buffer)
        batched_data = np.asarray(
            list(memoryview(buffer.float_buf()).cast('d'))[batch_dim:]  # ignore ship body
        ).reshape(-1, batch_dim)

        # apply forces on ice floes
        apply_drag_from_water(polygons,
                              sqrt_floe_areas,
                              batched_data[:, 3:5],  # updated velocities
                              batched_data[:, 5],    # updated angular velocities
                              dt)
        # apply_random_disturbance(polygons)  # gives floes some small initial velocity before collisions with ship

        # get updated obstacles
        obstacles = get_global_obs_coords(poly_vertices,
                                          batched_data[:, :2],  # updated positions
                                          batched_data[:, 2],   # updated angles
                                          )

        # update animation
        plot.update_ship(ship_vertices, *ship_body.position, ship_body.angle)
        plot.update_obstacles(obstacles=obstacles)

        fps = plot.update_fps()
        plot.title_text.set_text(f'FPS: {fps:.0f}, Real time speed: {fps * dt:.1f}x, Time: {sim_time:.1f} s')

        plot.animate_sim()

    plot.close()
    print('Done')
