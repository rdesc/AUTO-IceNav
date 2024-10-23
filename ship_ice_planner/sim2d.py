""" Main script for running simulation experiments with autonomous ship navigation in ice """
import math
import os
import random
import time
from multiprocessing import Process, Pipe, Queue
from queue import Empty, Full

import numpy as np
import pymunk
import pymunk.constraints
from matplotlib import pyplot as plt
from pymunk import Vec2d
import pymunk.batch

from ship_ice_planner.evaluation.evaluate_run_sim import floe_mass_hist_plot, ke_impulse_vs_time_plot, \
    control_vs_time_plot, state_vs_time_plot, impact_locs_impulse_plot
from ship_ice_planner.launch import launch
from ship_ice_planner.controller.sim_dynamics import SimShipDynamics
from ship_ice_planner.geometry.polygon import poly_area
from ship_ice_planner.geometry.utils import get_global_obs_coords
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import DotDict
from ship_ice_planner.utils.sim_utils import *


# global vars for dir/file names
PLOT_DIR = 'sim_plots'
TRIAL_SIM_PLOT = 'sim.pdf'
FLOE_MASS_HIST_PLOT = 'floe_mass_hist.pdf'
KE_IMPULSE_VS_TIME_PLOT = 'ke_impulse_vs_time.pdf'
IMPACT_LOCS_IMPULSE_PLOT = 'impact_locations_impulse.pdf'
STATE_VS_TIME_PLOT = 'state_vs_time.pdf'
CONTROL_VS_TIME_PLOT = 'control_vs_time.pdf'


def sim(
        cfg_file: str = None,
        cfg: DotDict = None,
        debug: bool = False,
        logging: bool = False,
        log_level: int = 10,
        init_queue: dict = None
):
    if cfg_file:
        # load config
        cfg = DotDict.load_from_file(cfg_file)
        cfg.cfg_file = cfg_file

    if cfg.output_dir:
        os.makedirs(cfg.output_dir)

    # multiprocessing setup
    queue = Queue(maxsize=1)  # LIFO queue to send state information to planner
    conn_recv, conn_send = Pipe(duplex=False)  # pipe to send new path to controller and for plotting
    planner = Process(target=launch,
                      kwargs=dict(cfg=cfg,
                                  debug=debug,
                                  logging=logging,
                                  log_level=log_level,
                                  pipe=conn_send,
                                  queue=queue))
    planner.start()

    # get sim params from config
    t_max = cfg.sim.t_max if cfg.sim.t_max else np.inf  # max number of main loop iterations
    dt = cfg.sim_dynamics.dt
    steps = cfg.sim.steps

    # seed to make sim deterministic
    seed = cfg.get('seed', 0)
    np.random.seed(seed)
    random.seed(seed)

    # timeout for polling of path updates
    # if timeout > 0, then the simulation loop will block until
    # a new path is received after a replan has been triggered
    planner_timeout = cfg.sim.get('planner_timeout', 0.0)

    # get goal, start, and obstacles
    if init_queue:
        goal = init_queue['goal']
        start = init_queue['ship_state']
        obs_dicts = init_queue['obstacles']

    else:
        goal, start = cfg.ship.goal_pos, cfg.ship.start_pos
        # generate obstacles and an associated polygon object
        obs_dicts, _ = generate_obstacles(**cfg.obstacle, seed=seed)

    obstacles = [ob['vertices'] for ob in obs_dicts]
    sqrt_floe_areas = np.array([math.sqrt(poly_area(ob['vertices'])) for ob in obs_dicts])  # for drag calculation

    # setup controller and simulated ship dynamics
    sim_dynamics = SimShipDynamics(
        eta=start, nu=[0, 0, 0],
        output_dir=cfg.output_dir,
        **cfg.sim_dynamics
    )
    state = sim_dynamics.state

    # setup pymunk environment
    space = init_pymunk_space()

    # setup boundaries of ice field
    # create_channel_borders(space, *cfg.map_shape)  # disabling for now, causes weird behavior

    # initialize collision metrics
    system_ke_loss = []   # https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.total_ke
                          # source code in Chimpunk2D cpArbiterTotalKE
    delta_ke_ice = []
    total_impulse = []    # https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.total_impulse
                          # source code in Chimpunk2D cpArbiterTotalImpulse
    collided_ob_idx = []  # idx for the collided obstacle, assumes id corresponds to the index in obs_dicts
    contact_pts = []      # collision contact points in the local coordinates of the ship

    # setup pymunk collision callbacks
    def pre_solve_handler(arbiter, space, data):
        ice_body = arbiter.shapes[1].body
        ice_body.pre_collision_KE = ice_body.kinetic_energy  # hacky, adding a field to pymunk body object
        return True

    def post_solve_handler(arbiter, space, data):
        nonlocal system_ke_loss, delta_ke_ice, total_impulse, collided_ob_idx, contact_pts
        ship_shape, ice_shape = arbiter.shapes

        # update metrics
        system_ke_loss.append(arbiter.total_ke)
        delta_ke_ice.append(ice_shape.body.kinetic_energy - ice_shape.body.pre_collision_KE)
        total_impulse.append(arbiter.total_impulse)
        collided_ob_idx.append(ice_shape.idx)

        # find the impact locations in the local coordinates of the ship
        # see https://stackoverflow.com/a/78017626 and run demo below to understand contact points
        # https://github.com/viblo/pymunk/blob/master/pymunk/examples/collisions.py
        if len(arbiter.contact_point_set.points) == 2:  # max 2 contact points
            # take the average
            c1, c2 = arbiter.contact_point_set.points
            contact_pts.append(list(ship_shape.body.world_to_local((c1.point_b + c2.point_b) / 2)))
        else:
            c1 = arbiter.contact_point_set.points[0]
            contact_pts.append(list(ship_shape.body.world_to_local(c1.point_b)))

    handler = space.add_collision_handler(1, 2)
    handler.pre_solve = pre_solve_handler
    handler.post_solve = post_solve_handler

    # init pymunk physics objects
    ship_shape = create_sim_ship(space,  # polygon for ship
                                 cfg.ship.vertices,
                                 state.eta,
                                 body_type=pymunk.Body.KINEMATIC,
                                 velocity=state.get_global_velocity())
    ship_shape.collision_type = 1  # for collision callbacks
    ship_body = ship_shape.body

    # polygons for ice floes
    polygons = generate_sim_obs(space, obs_dicts)
    poly_vertices = []  # generate these once
    for idx, p in enumerate(polygons):
        p.collision_type = 2  # for collision callbacks
        vertices = list_vec2d_to_numpy(p.get_vertices())
        poly_vertices.append(
            np.concatenate([vertices, [vertices[0]]])  # close the polygon (speeds up plotting somehow...)
        )
        p.idx = idx  # hacky but makes it easier to keep track which polygon collided
    floe_masses = np.array([poly.mass for poly in polygons])  # may differ slightly from the mass in the exp config

    # uses new batch module https://www.pymunk.org/en/latest/pymunk.batch.html
    buffer_get_body = pymunk.batch.Buffer()
    batch_fields_get_body = (pymunk.batch.BodyFields.POSITION
                             | pymunk.batch.BodyFields.ANGLE
                             | pymunk.batch.BodyFields.VELOCITY
                             | pymunk.batch.BodyFields.ANGULAR_VELOCITY
                             )
    batch_dim_get_body = 6

    buffer_set_body = pymunk.batch.Buffer()
    batch_fields_set_body = (pymunk.batch.BodyFields.VELOCITY
                             | pymunk.batch.BodyFields.ANGULAR_VELOCITY
                             )

    # send first message
    queue.put(dict(
        goal=[goal[0],
              goal[1] + cfg.get('goal_offset', 0)],  # option to offset goal
        ship_state=start,
        obstacles=obstacles,
        masses=floe_masses,
        # metadata=dict(  # can optionally send metadata
        # )
    ))

    # get path
    path = conn_recv.recv()
    path = np.asarray(path)

    # setup trajectory tracking
    sim_dynamics.init_trajectory_tracking(path)

    # initialize plotting / animation
    plot = None
    running = True
    save_fig_dir = os.path.join(cfg.output_dir, PLOT_DIR) if cfg.output_dir else None
    if cfg.anim.show or cfg.anim.save:
        plot = Plot(obstacles=obstacles, path=path.T, legend=False, track_fps=True, y_axis_limit=cfg.plot.y_axis_limit,
                    ship_vertices=cfg.ship.vertices, target=sim_dynamics.setpoint[:2], inf_stream=cfg.anim.inf_stream,
                    ship_pos=state.eta, map_figsize=None, sim_figsize=(5, 5), remove_sim_ticks=True, goal=goal[1],
                    save_fig_dir=save_fig_dir, map_shape=cfg.map_shape,
                    save_animation=cfg.anim.save, anim_fps=cfg.anim.fps
                    )
        def on_close(event):
            nonlocal running
            if event.key == 'escape':
                running = False

        plot.sim_fig.canvas.mpl_connect('key_press_event', on_close)

    # for plotting ship path
    ship_actual_path = ([], [])  # list for x and y

    iteration = 0    # to keep track of simulation steps
    t = time.time()  # to keep track of time taken for each sim iteration

    try:
        # main simulation loop
        while True:
            # check termination conditions
            if iteration >= t_max:
                print('\nReached max number of iterations:', t_max)
                break

            if running is False:
                print('\nReceived stop signal, exiting...')
                break

            if state.y >= goal[1]:
                print('\nReached goal:', goal)
                break

            time_per_step = time.time() - t
            t = time.time()

            # get batched body data
            buffer_get_body.clear()
            pymunk.batch.get_space_bodies(space, batch_fields_get_body, buffer_get_body)
            batched_data = np.asarray(
                list(memoryview(buffer_get_body.float_buf()).cast('d'))[batch_dim_get_body:]  # ignore ship body
            ).reshape(-1, batch_dim_get_body)

            # get updated ship pose
            state.eta = (ship_body.position.x, ship_body.position.y, ship_body.angle)

            if iteration % int(1 / dt) == 0:
                print(
                    f'Simulation step {iteration} / {t_max}, '
                    f'ship eta ({state.x:.2f}m, {state.y:.2f}m, {state.psi:.2f}rad), '
                    f'sim time {sim_dynamics.sim_time:.2f} s, '
                    f'time per step {time_per_step:.4f} s ',
                    end='\r',
                )

            if planner.is_alive():
                if sim_dynamics.check_trigger_replan():
                    # empty queue to ensure latest state data is pushed
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass

                    try:
                        queue.put(dict(
                            ship_state=state.eta,
                            obstacles=(poly_vertices,  # let planner parse the obstacles
                                       batched_data[:, :2],  # updated positions
                                       batched_data[:, 2]),  # updated angles
                            masses=floe_masses
                        ), timeout=1)

                        if planner_timeout:
                            conn_recv.poll(planner_timeout)  # wait for new path, block until timeout

                    except Full:
                        # we should only be reaching this point in rare cases!
                        print('Something went wrong with sending message to planner...')
                        pass

                # check for path
                if conn_recv.poll():
                    msg = conn_recv.recv()
                    # confirm we have path
                    if len(msg) == 0:
                        print('Error, received empty path!')
                    else:
                        new_path = np.asarray(msg)  # n x 3
                        # confirm path is a minimum of 2 points
                        if len(new_path) > 1:
                            path = new_path
                            sim_dynamics.setpoint_generator.replan_update(
                                state.get_vessel_speed(), (state.x, state.y), path
                            )

            # update controller
            sim_dynamics.control()

            # compute collision forces on ship
            tau_env = compute_collision_force_on_ship(total_impulse,
                                                      contact_pts,
                                                      state.u, state.v, state.psi,
                                                      dt / steps)
            sim_dynamics.vessel_model.set_environment_force(tau_env)

            # store simulation data
            sim_dynamics.log_step(
                system_ke_loss=system_ke_loss,
                delta_ke_ice=delta_ke_ice,
                total_impulse=total_impulse,
                collided_ob_idx=collided_ob_idx,
                contact_pts=contact_pts,
                tau_env=tau_env,
                real_time=t
            )

            if iteration % cfg.anim.plot_steps == 0 and plot is not None:
                # rendering is slow!! better option would be to use a proper library...
                ship_actual_path[0].append(state.x)
                ship_actual_path[1].append(state.y)

                plot.update_path(path.T, target=sim_dynamics.setpoint[:2])  # , ship_state=ship_actual_path)
                plot.update_ship(cfg.ship.vertices, *state.eta,
                                 move_yaxis_threshold=cfg.anim.move_yaxis_threshold)
                plot.update_obstacles(obstacles=get_global_obs_coords(  # updating obs is very slow...
                    poly_vertices,
                    batched_data[:, :2],  # updated positions
                    batched_data[:, 2])   # updated angles
                )

                fps = plot.update_fps()

                plot.title_text.set_text(
                    f'FPS: {fps:.0f}, '
                    f'Real time speed: {fps * dt * cfg.anim.plot_steps:.1f}x, '
                    f'Time: {sim_dynamics.sim_time:.1f} s\n'
                    f'surge {state.u:.2f} (m/s), '
                    f'sway {-state.v:.2f} (m/s), '                    # in body frame, positive sway is to the right
                    f'yaw rate {-state.r * 180 / np.pi:.2f} (deg/s)'  # in body frame, positive yaw is clockwise
                    # in the sim, these are reversed! so for plotting purposes flip the sign
                )

                plot.animate_sim()

            # simulate ship dynamics
            sim_dynamics.sim_step()

            # apply velocity commands to ship body
            ship_body.velocity = Vec2d(*state.get_global_velocity())
            ship_body.angular_velocity = state.r

            # apply forces on ice floes
            batched_apply_drag_from_water(sqrt_floe_areas,
                                          floe_masses,
                                          batched_data[:, 3:5],  # updated velocities
                                          batched_data[:, 5],    # updated angular velocities
                                          [*ship_body.velocity, ship_body.angular_velocity],
                                          dt,
                                          buffer_set_body,
                                          )
            pymunk.batch.set_space_bodies(
                space, batch_fields_set_body, buffer_set_body
            )
            # apply_random_disturbance(polygons)  # these are not optimized using batch module
            # apply_current(polygons, cfg.sim.ocean_current)  # (not fully tested! would need to also update sim_dynamics.py)

            # clear collision metrics
            system_ke_loss.clear(), delta_ke_ice.clear(), total_impulse.clear(), collided_ob_idx.clear(), contact_pts.clear()

            # move simulation forward
            for _ in range(steps):
                space.step(dt / steps)

            iteration += 1  # increment simulation step

    except KeyboardInterrupt:
        print('\nReceived keyboard interrupt, exiting...')

    finally:
        print('\nDone simulation... Processing simulation data...')

        conn_recv.close()
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put(None)
        queue.close()

        if plot is not None:
            plot.close()

        # get logged simulation data
        sim_data = sim_dynamics.get_state_history()

        # make a plot showing the final state of the simulation
        plot = Plot(
            obstacles=get_global_obs_coords(poly_vertices, batched_data[:, :2], batched_data[:, 2]),
            path=path.T, goal=goal[1], map_figsize=None, remove_sim_ticks=False, show=False,
            ship_pos=sim_data[['x', 'y']].to_numpy().T, map_shape=cfg.map_shape, legend=False
        )
        plot.add_ship_patch(plot.sim_ax, cfg.ship.vertices, *state.eta)
        plot.save(save_fig_dir, TRIAL_SIM_PLOT)

        # plots
        collided_ob_idx = sim_data['collided_ob_idx']
        if not np.sum(collided_ob_idx):
            print('No collisions occurred in simulation, skipping collision stats...')
        else:
            floe_mass_hist_plot(
                sim_data, floe_masses,
                save_fig=os.path.join(save_fig_dir, FLOE_MASS_HIST_PLOT) if save_fig_dir else None)
            ke_impulse_vs_time_plot(
                sim_data,
                save_fig=os.path.join(save_fig_dir, KE_IMPULSE_VS_TIME_PLOT) if save_fig_dir else None)
            impact_locs_impulse_plot(
                sim_data, cfg.ship.vertices,
                save_fig=os.path.join(save_fig_dir, IMPACT_LOCS_IMPULSE_PLOT) if save_fig_dir else None)

        state_vs_time_plot(
            sim_data,
            save_fig=os.path.join(save_fig_dir, STATE_VS_TIME_PLOT) if save_fig_dir else None
        )
        control_vs_time_plot(
            sim_data,
            dim_U=len(sim_dynamics.state.u_control),
            control_labels=sim_dynamics.vessel_model.controls,
            save_fig=os.path.join(save_fig_dir, CONTROL_VS_TIME_PLOT) if save_fig_dir else None
        )

        if plt.get_backend() != 'agg':
            plt.show()
        plt.close('all')

        print('Clean exiting planner process...')
        planner.terminate()
        if planner.is_alive():
            planner.join(timeout=2)

        print('Done')
