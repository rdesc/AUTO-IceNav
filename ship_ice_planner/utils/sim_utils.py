import math
from typing import List, Tuple

import numpy as np
import pymunk
from pymunk import Vec2d, Body, Poly
from skimage import draw

from ship_ice_planner.geometry.polygon import poly_area, generate_polygon
from ship_ice_planner.geometry.utils import Rxy

__all__ = [
    'generate_obstacles',
    'create_sim_ship',
    'create_polygon',
    'create_channel_borders',
    'generate_sim_obs',
    'apply_random_disturbance',
    'apply_current',
    'apply_drag_from_water',
    'clamp_velocity',
    'list_vec2d_to_numpy',
    'init_pymunk_space',
    'batched_apply_drag_from_water',
    'get_floe_masses',
    'compute_collision_force_on_ship',
]

# simulation parameters -- parameters are from:
# ref1: "Numerical estimation of ship resistance in broken ice and investigation on the effect of floe geometry"
#        https://www.sciencedirect.com/science/article/pii/S095183392030160X
# ref2: "Ship resistance when operating in floating ice floes: A combined CFD&DEM approach"
#        https://www.sciencedirect.com/science/article/pii/S0951833920301118
# ref3: "GPU-Event-Mechanics Evaluation of Ice Impact Load Statistics"
#        https://www.researchgate.net/publication/266670398_GPU-Event-Mechanics_Evaluation_of_Ice_Impact_Load_Statistics
# ref4: "Numerical simulation of ice impacts on ship hulls in broken ice fields"
#        https://www.sciencedirect.com/science/article/pii/S0029801819301787
# ref5: "Planar multi-body model of iceberg free drift and towing in broken ice"
#        https://www.sciencedirect.com/science/article/pii/S0165232X15001810
# ref6: "An overview of the Oden Arctic Technology Research Cruise 2015 (OATRC2015)"
#        https://www.sciencedirect.com/science/article/pii/S0165232X17305736
# ref7: "Numerical Study of a Moored Structure in Moving Broken Ice Driven by Current and Wave"
#        https://asmedigitalcollection.asme.org/offshoremechanics/article-abstract/141/3/031501/454645/Numerical-Study-of-a-Moored-Structure-in-Moving
# ref8: "Simulator for Arctic Marine Structures (SAMS)"
#        https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2600326
ICE_DENSITY = 900.           # ref6 -- kg/m^3
WATER_DENSITY = 1025         # ref6 -- kg/m^3
ICE_THICKNESS = 1.2          # ref3 -- m
CD_DRAG_COEFFICIENT = 1.0    # ref7  (0.5 from ref6)
ICE_SHIP_RESTITUTION = 0.1   # ref5
ICE_ICE_RESTITUTION = 0.1    # ref5
ICE_SHIP_FRICTION = 0.05     # ref2
ICE_ICE_FRICTION = 0.35      # ref2
ANGULAR_VELOCITY_DECAY = 0.03  # should be in the range (0, 1), proper drag for angular velocity is not implemented
SHIP_MASS = 6000.0e3  # kg  https://github.com/cybergalactic/PythonVehicleSimulator/blob/master/src/python_vehicle_simulator/vehicles/supply.py
ICE_SUBMERGED_PERCENTAGE = ICE_DENSITY / WATER_DENSITY
DRAG_FORCE_CONSTANT = 0.5 * WATER_DENSITY * CD_DRAG_COEFFICIENT * ICE_THICKNESS * ICE_SUBMERGED_PERCENTAGE

# pymunk parameters
SIMPLE_DAMPING = .99  # between 0 and 1 where 1 is no damping
SOLVER_ITERATIONS = 50  # number of iterations for the solver, controls the accuracy of the simulation
POLY_BUFFER_RADIUS = 0.01  # m

# misc parameters
INIT_VELOCITY = 1e-2  # m/s
ICE_SMALL_DISTURBANCE = 0.01  # should be some small number, this multiplies the mass of the object
LARGE_SPEED_THRESHOLD = 10.  # m/s
MAX_COLLISION_FORCE_VECTOR_ON_SHIP = [1.5e6, 1.5e6, 1e8]  # ref8 -- N, N, Nm


def init_pymunk_space():
    # setup pymunk environment
    space = pymunk.Space()

    # damping to apply on ice floes, this is in addition to custom drag force
    space.damping = SIMPLE_DAMPING

    space.iterations = SOLVER_ITERATIONS

    # space = pymunk.Space(threaded=True)
    # space.threads = 2

    return space


def generate_obstacles(
        num_obs: int,
        min_r: int,
        max_r: int,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        allow_overlap=False,
        seed=None,
        **kwargs
) -> Tuple[List[dict], List[np.ndarray]]:
    # list to keep track of generated obs
    obstacles = []

    # generate a set of random circles
    _, circles = draw.random_shapes(image_shape=(int(max_y - min_y), int(max_x - min_x)),
                                    max_shapes=num_obs, min_shapes=num_obs,
                                    max_size=max_r * 2, min_size=min_r * 2,
                                    shape='circle', allow_overlap=allow_overlap,
                                    rng=seed)  # num_trials=100 by defaults

    # iterate over each circle and generate a polygon
    for circ in circles:
        # get bounding box coordinates
        r, c = circ[1]  # top right and bottom left coords
        radius = abs(r[0] - r[1]) / 2
        centre = (c[0] + radius + min_x, r[0] + radius + min_y)

        # now generate polygon given centre and radius
        vertices = generate_polygon(diameter=radius * 2, origin=centre)

        if vertices is not None:
            obstacles.append({
                'vertices': vertices,
                'centre': centre,
                'radius': radius,
                'area': poly_area(vertices)
            })

    return obstacles, [ob['vertices'] for ob in obstacles]


def create_sim_ship(space, vertices: List, start_pos: Tuple[float, float, float], body_type=None, velocity=(0, 0)) -> Poly:
    x, y, psi = start_pos
    # setup for pymunk
    if body_type is None:
        body_type = Body.KINEMATIC  # Body.DYNAMIC
    body = Body(body_type=body_type)  # mass and moment ignored when kinematic body type
    body.position = (x, y)
    body.velocity = Vec2d(*velocity)
    body.angle = psi  # Rotation of the body in radians
    shape = Poly(body, vertices, radius=POLY_BUFFER_RADIUS)

    # from pymunk: default calculation multiplies the elasticity/friction of the two shapes together.
    shape.elasticity = ICE_SHIP_RESTITUTION / math.sqrt(ICE_ICE_RESTITUTION)
    shape.friction = ICE_SHIP_FRICTION / math.sqrt(ICE_ICE_FRICTION)
    shape.density = SHIP_MASS / poly_area(np.asarray(vertices))  # pymunk will compute mass as density * area

    space.add(body, shape)

    return shape


def create_polygon(space, vertices, x, y, initial_velocity=None) -> Poly:
    body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
    body.position = (x, y)

    shape = pymunk.Poly(body, vertices, radius=POLY_BUFFER_RADIUS)
    shape.density = ICE_DENSITY * ICE_THICKNESS  # density for 2D ice, pymunk will compute mass as density * area

    # from pymunk: default calculation multiplies the elasticity/friction of the two shapes together.
    shape.elasticity = math.sqrt(ICE_ICE_RESTITUTION)
    shape.friction = math.sqrt(ICE_ICE_FRICTION)

    if initial_velocity is not None:
        body.velocity = Vec2d(*initial_velocity)
    else:
        # initialize to small value for numerical stability
        body.velocity = Vec2d(0, INIT_VELOCITY)

    space.add(body, shape)

    return shape


def create_channel_borders(space, map_length, map_width):
    # setup boundaries of ice field
    boundary_shape1 = pymunk.Segment(space.static_body,
                                     a=(0, -map_length),    # ensures no floes can escape
                                     b=(0, map_length * 2),
                                     radius=0.0)
    boundary_shape2 = pymunk.Segment(space.static_body,
                                     a=(map_width, -map_length),
                                     b=(map_width, map_length * 2),
                                     radius=0.0)

    boundary_shape2.friction = math.sqrt(ICE_ICE_FRICTION)
    boundary_shape1.friction = math.sqrt(ICE_ICE_FRICTION)
    boundary_shape1.elasticity = 0.0
    boundary_shape2.elasticity = 0.0

    space.add(boundary_shape1, boundary_shape2)


def get_floe_masses(obs_dicts):
    # make a dummy space to compute the mass of the ice floes
    dummy_space = pymunk.Space()
    polygons = generate_sim_obs(dummy_space, obs_dicts)
    # the masses may slightly differ from the mass in the obstacle dict due to the way pymunk computes mass
    return np.array([poly.mass for poly in polygons])


def list_vec2d_to_numpy(vectors: List[Vec2d]) -> np.ndarray:
    return np.array([[v.x, v.y] for v in vectors])


def generate_sim_obs(space, obstacles: List[dict]) -> List[Poly]:
    return [
        create_polygon(
            space, (obs['vertices'] - obs['centre']).tolist(), *obs['centre']
        )
        for obs in obstacles
    ]


def clamp_velocity(shape, max_speed):
    if shape.body.velocity.length > max_speed:
        shape.body.velocity = shape.body.velocity.normalized() * max_speed


def compute_collision_force_on_ship(impulse: List,
                                    contact_pts: List,
                                    surge: float,
                                    sway: float,
                                    psi: float,
                                    dt: float) -> List:
    """
    Aggregates all the collision impulses and contact points and computes
    the net force and torque on the ship in the body frame of reference.

    :param impulse: list of impulses applied on the ship
    :param contact_pts: list of contact points where the impulses were applied
    :param surge: surge velocity of the ship
    :param sway: sway velocity of the ship
    :param psi: yaw angle of the ship
    :param dt: time step
    """
    # initialize variables for force in surge, force in sway and torque in yaw
    force_X = 0.
    force_Y = 0.
    torque_N = 0.

    # rotation matrix to rotate the impulse vector from world frame to body frame, so need the inverse of rotation matrix
    R = Rxy(psi).T

    speed = np.sqrt(surge ** 2 + sway ** 2)
    if not len(impulse) or speed == 0 or surge < 0:
        return [0, 0, 0]  # don't apply a force if the ship is not moving or moving backwards

    # iterate over each impulse and compute the force and torque
    for i in range(len(impulse)):
        if np.linalg.norm(impulse[i]) > 0:
            curr_impulse = R @ impulse[i]
            force_X += curr_impulse[0] / dt  # divide impulse by dt to get force. Note, this is not very accurate!
            force_Y += curr_impulse[1] / dt
            torque_N += np.cross(contact_pts[i], curr_impulse / dt)

    return [
        -min(abs(force_X), MAX_COLLISION_FORCE_VECTOR_ON_SHIP[0]),
        np.sign(force_Y) * min(abs(force_Y), MAX_COLLISION_FORCE_VECTOR_ON_SHIP[1]),
        np.sign(torque_N) * min(abs(torque_N), MAX_COLLISION_FORCE_VECTOR_ON_SHIP[2])
    ]


def apply_random_disturbance(polygons):
    """
    Super dumb way of simulating random disturbances in water
    by applying random impulse to each polygon

    This can be batched for speedup
    """
    [
        poly.body.apply_impulse_at_local_point(
            impulse=Vec2d(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) * poly.mass * ICE_SMALL_DISTURBANCE,
            point=(0, 0)
        )
        for poly in polygons
    ]


def apply_current(polygons, current_vector):
    """
    Apply ocean current to ice floes

    This can be batched for speedup
    """
    [
        poly.body.apply_impulse_at_local_point(
            impulse=Vec2d(*current_vector) * poly.mass,
            point=(0, 0)
        )
        for poly in polygons
    ]


def apply_drag_from_water(polygons, sqrt_areas, velocities, angular_velocities, dt):
    """
    Simplified version of drag force from water applied on ice floes

    The variable 'A' should be the cross-sectional area or the projection
    of the floe area onto the direction of the relative velocity of the floe
    We can simplify this by simply setting A = sqrt(floe_area) * ice_thickness
    TODO: compute the proper cross-sectional area (it should not be too difficult) as illustrated in figure 3 of ref1

    We can also simplify the drag moment by adding damping to the angular velocity

    Useful discussions:
    - https://stackoverflow.com/a/44530585
    - https://physics.stackexchange.com/a/651758

    Buoyancy and fluid drag example
    https://github.com/slembcke/Chipmunk2D/blob/master/demo/Buoyancy.c
    """
    for poly, sqrt_area, v, w in zip(polygons, sqrt_areas, velocities, angular_velocities):
        poly.body.angular_velocity = w * (1 - ANGULAR_VELOCITY_DECAY)

        speed = math.sqrt(v[0] ** 2 + v[1] ** 2)

        poly.body.velocity = Vec2d(*v) + dt * (
                    DRAG_FORCE_CONSTANT
                    * sqrt_area  # effective width of the ice floe
                    * speed
                    * -v
            ) / poly.body.mass

        # sanity checks -- can disable with -O flag which can provide a speedup
        assert speed < LARGE_SPEED_THRESHOLD
        assert speed > poly.body.velocity.length


def batched_apply_drag_from_water(sqrt_areas, masses, velocities, angular_velocities, ship_velocity, dt, update_buffer):
    """
    Much faster version of `apply_drag_from_water`
    """
    speeds = np.linalg.norm(velocities, axis=1)

    new_velocities = velocities + dt * (
        DRAG_FORCE_CONSTANT
        * sqrt_areas[:, None]
        * speeds[:, None]
        * -velocities
    ) / masses[:, None]

    new_angular_velocities = angular_velocities * (1 - ANGULAR_VELOCITY_DECAY)

    v_arr = np.ravel(np.r_[[ship_velocity],  # leave ship velocity unchanged
                           np.c_[new_velocities, new_angular_velocities]])

    update_buffer.set_float_buf(v_arr.tobytes())
