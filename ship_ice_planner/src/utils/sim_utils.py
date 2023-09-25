from typing import List

import numpy as np
import pymunk

from ship_ice_planner.src.ship import Ship


def create_polygon(space, vertices, x, y, density):
    body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
    body.position = (x, y)
    dummy_shape = pymunk.Poly(None, vertices)
    centre_of_g = dummy_shape.center_of_gravity
    vs = [(x - centre_of_g[0], y - centre_of_g[1]) for x, y in vertices]

    shape = pymunk.Poly(body, vs, radius=0.02)
    shape.density = density
    shape.elasticity = 0.01
    shape.friction = 1.0
    space.add(body, shape)
    return shape


def generate_sim_obs(space, obstacles: List[dict], density):
    return [
        create_polygon(
            space, (obs['vertices'] - np.array(obs['centre'])).tolist(),
            *obs['centre'], density=density
        )
        for obs in obstacles
    ]


def simulate_ship_ice_collision(path, ship_vertices, obs_dicts, ship_vel=1, dt=0.25, steps=10):
    """
    Simulate collision between ship and ice as ship travels along path
    Assumes only 1 collision between ship and obstacle can occur
    """
    assert path.shape[1] == 3

    # make a new pymunk env with ship and obstacle
    space = pymunk.Space()
    static_body = space.static_body  # create a static body for friction constraints

    # init ship sim objects
    # ship dynamics will not change in response to collision if body type is KINEMATIC
    start_pose = path[0]
    ship_body, ship_shape = Ship.sim(ship_vertices, start_pose, body_type=pymunk.Body.KINEMATIC,
                                     velocity=(np.cos(start_pose[2]) * ship_vel, np.sin(start_pose[2]) * ship_vel))
    space.add(ship_body, ship_shape)
    ship_shape.collision_type = 1

    # init polygon objects
    for ob in obs_dicts:
        poly = create_polygon(
            space, (ob['vertices'] - np.array(ob['centre'])).tolist(),
            *ob['centre'], density=10
        )
        poly.collision_type = 2  # this will identify the obstacle in the collision shape pair object (i.e. arbiter)

    # flag to keep sim running until collision
    collision_ob = None
    initial_ob_pos = None

    # collision handler
    def collide(arbiter, space, data):
        nonlocal collision_ob, initial_ob_pos
        # print('collide!')
        if collision_ob is None:
            collision_ob = arbiter.shapes[1]  # keep a reference of obstacle so we can get velocity
            initial_ob_pos = collision_ob.body.position
            assert arbiter.shapes[1].collision_type == poly.collision_type
        return True

    handler = space.add_collision_handler(1, 2)
    # from pymunk docs
    # separate: two shapes have just stopped touching for the first time
    # begin: two shapes just started touching for the first time
    handler.begin = collide

    # assert abs(dt * ship_vel - (path_length(path[:, :2] / len(path)))) < 0.05 * (dt * ship_vel)
    for p in path:  #  FIXME: resolution of path, velocity, dt, and steps are all related
        # the amount that sim moves ship forward should be equal to resolution of path
        for _ in range(steps):  # this slows down function quite a bit but increases sim accuracy
            space.step(dt / steps)
        # update velocity and angle such that ship follows path exactly
        ship_body.velocity = (np.cos(p[2]) * ship_vel, np.sin(p[2]) * ship_vel)
        ship_body.angle = -p[2]

    if collision_ob is None:
        return None  # this means there were no collision
    return (
        tuple(collision_ob.body.velocity),
        tuple(collision_ob.body.position - initial_ob_pos),
        tuple(ship_body.velocity)
    )
