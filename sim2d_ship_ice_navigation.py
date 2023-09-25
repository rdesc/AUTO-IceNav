""" Main script for running simulation experiments with autonomous ship navigation in ice """
import os
import pickle
import random
from multiprocessing import Process, Pipe, Queue
from operator import le, ge
from queue import Empty

import numpy as np
import pymunk
import pymunk.constraints
from matplotlib import pyplot as plt, patches
from pymunk import Vec2d

from ship_ice_planner.launch import launch
from ship_ice_planner.src.controller.dp import DP
from ship_ice_planner.src.cost_map import CostMap
from ship_ice_planner.src.evaluation.metrics import tracking_error, total_work_done
from ship_ice_planner.src.geometry.polygon import poly_area
from ship_ice_planner.src.ship import Ship
from ship_ice_planner.src.utils.plot import Plot
from ship_ice_planner.src.utils.sim_utils import generate_sim_obs
from ship_ice_planner.src.utils.utils import DotDict


def sim(cfg_file=None, cfg=None, debug=False, logging=False, log_level=10, init_queue=None):
    if cfg_file:
        # load config
        cfg = DotDict.load_from_file(cfg_file)
        cfg.cfg_file = cfg_file

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

    # get important params
    steps = cfg.sim.steps
    t_max = cfg.sim.t_max if cfg.sim.t_max else np.inf
    horizon = cfg.a_star.horizon
    replan = cfg.a_star.replan
    inf_stream = cfg.anim.inf_stream
    new_obs_count = 1 if inf_stream else None
    seed = cfg.get('seed', None)
    dt = cfg.controller.dt

    np.random.seed(seed)  # seed to make sims deterministic
    random.seed(seed)

    # setup pymunk environment
    space = pymunk.Space()  # threaded=True causes some issues
    space.iterations = cfg.sim.iterations
    space.gravity = cfg.sim.gravity
    space.damping = cfg.sim.damping

    # keep track of running total of total kinetic energy / total impulse
    # computed using pymunk api call, source code here
    # https://github.com/slembcke/Chipmunk2D/blob/edf83e5603c5a0a104996bd816fca6d3facedd6a/src/cpArbiter.c#L158-L172
    total_ke = [0, []]  # keep track of both running total and ke at each collision
    total_impulse = [0, []]
    # keep track of running total of work
    total_work = [0, []]

    # keep track of all the obstacles that collide with ship
    clln_obs = set()

    # keep track of contact points
    contact_pts = []

    # setup a collision callback to keep track of total ke
    # def pre_solve_handler(arbiter, space, data):
    #     nonlocal ship_ke
    #     ship_ke = arbiter.shapes[0].body.kinetic_energy
    #     print('ship_ke', ship_ke, 'mass', arbiter.shapes[0].body.mass, 'velocity', arbiter.shapes[0].body.velocity)
    #     return True
    # # http://www.pymunk.org/en/latest/pymunk.html#pymunk.Body.each_arbiter

    def post_solve_handler(arbiter, space, data):
        nonlocal total_ke, total_impulse, clln_obs
        total_ke[0] += arbiter.total_ke
        total_ke[1].append(arbiter.total_ke)

        total_impulse[0] += arbiter.total_impulse.length
        total_impulse[1].append(list(arbiter.total_impulse))

        if arbiter.is_first_contact:
            clln_obs.add(arbiter.shapes[1])

        # max of two sets of points, easy to see with a picture with two overlapping convex shapes
        # find the impact locations in the local coordinates of the ship
        for i in arbiter.contact_point_set.points:
            contact_pts.append(list(arbiter.shapes[0].body.world_to_local((i.point_b + i.point_a) / 2)))

    # handler = space.add_default_collision_handler()
    handler = space.add_collision_handler(1, 2)
    # from pymunk docs
    # post_solve: two shapes are touching and collision response processed
    handler.post_solve = post_solve_handler

    if init_queue:
        goal, start, obs_dicts = init_queue.values()

    else:
        goal, start = cfg.ship.goal_pos, cfg.ship.start_pos
        # generate obstacles and an associated polygon object
        obs_dicts, _ = CostMap.generate_obstacles(**cfg.obstacle, seed=seed)

    # filter out obstacles that have zero area
    obs_dicts[:] = [ob for ob in obs_dicts if poly_area(ob['vertices']) != 0]
    obstacles = [ob['vertices'] for ob in obs_dicts]

    # send first message
    queue.put(dict(
        goal=goal,
        ship_state=start,
        obstacles=obstacles,
        metadata={'velocity': 0.0}
    ))

    polygons = generate_sim_obs(space, obs_dicts, cfg.sim.obstacle_density)
    for p in polygons:
        p.collision_type = 2

    # initialize ship sim objects
    ship_body, ship_shape = Ship.sim(cfg.ship.vertices, start)
    ship_shape.collision_type = 1
    space.add(ship_body, ship_shape)
    # run initial simulation steps to let environment settle
    for _ in range(1000):
        space.step(dt / steps)
    prev_obs = CostMap.get_obs_from_poly(polygons)

    # get path
    path = conn_recv.recv()
    path = np.asarray(path)
    path = path[path[:, 1] < horizon + path[0, 1]]

    # setup dp controller
    cx = path.T[0]
    cy = path.T[1]
    ch = path.T[2]
    dp = DP(x=start[0], y=start[1], yaw=start[2],
            cx=cx, cy=cy, ch=ch, output_dir=cfg.output_dir,
            **cfg.controller)
    dp.log_step()
    state = dp.state

    # initialize plotting/animation
    if cfg.anim.show:
        plot = Plot(
            np.zeros((cfg.costmap.m, cfg.costmap.n)), obs_dicts, path.T,
            ship_pos=start, ship_vertices=np.asarray(ship_shape.get_vertices()),
            horizon=horizon, map_figsize=None, y_axis_limit=cfg.plot.y_axis_limit,
            target=tuple(dp.setpoint[:2]), inf_stream=True, goal=goal[1]
        )
        plt.ion()  # turn on interactive mode
        plt.show()

    ship_state = ([], [])  # keep track of ship path
    past_path = ([], [])  # keep track of planned path behind ship
    t = 0  # start time tick
    goal_op = ge if not cfg.get('reverse_dir') else le

    try:
        # main simulation loop
        while t < t_max:
            t += 1
            if t >= t_max:
                print('Reached max time: ', t_max)
                break

            if goal_op(ship_body.position.y, goal[1]):
                print('Reached goal: ', goal)
                break

            if t % cfg.anim.plan_steps == 0:
                print('Simulation time {} / {}, ship position x={} y={}'
                      .format(t, t_max, ship_body.position.x, ship_body.position.y), end='\r')

                if inf_stream:
                    # add/remove polygons as necessary based on ship progression
                    new_obs_count = update_polygons(
                        polygons, space, ship_body.position.y, cfg.plot.y_axis_limit,
                        cfg.sim.obstacle_density, cfg.obstacle, new_obs_count, cfg.anim.new_obs_dist,
                        cfg.anim.move_yaxis_threshold, seed
                    )

                # get updated obstacles
                obstacles = CostMap.get_obs_from_poly(polygons)

                # update work metric
                work = total_work_done(prev_obs, obstacles)
                total_work[0] += work
                total_work[1].append(work)
                prev_obs = obstacles

                if replan:
                    # empty queue to ensure latest state data is pushed
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass

                    if not queue.full():
                        queue.put(dict(
                            ship_state=(state.x, state.y, state.yaw),
                            obstacles=obstacles,
                            metadata={'velocity': ship_body.velocity.length}
                        ), block=False)

                    # check for path
                    if conn_recv.poll():
                        msg = conn_recv.recv()
                        # confirm we have path
                        if len(msg) == 0:
                            print('Error, received empty path!')
                        else:
                            new_path = np.asarray(msg)  # n x 3
                            # cut out the part of path that sticks out past the horizon
                            new_path = new_path[new_path[:, 1] < horizon + new_path[0, 1]]
                            # confirm path is a minimum of 2 points
                            if len(new_path) > 1:
                                path = new_path
                                cx = path.T[0]
                                cy = path.T[1]
                                ch = path.T[2]
                                dp.target_course.update(cx, cy, ch)

            if cfg.get('ice_current', None):
                # apply impulse to bodies
                for p in polygons:
                    # note, ice may go outside of ice field boundaries
                    p.body.apply_impulse_at_local_point((cfg.ice_current[0] * p.mass,
                                                         cfg.ice_current[1] * p.mass))

            # update DP controller
            dp(ship_body.position.x,
               ship_body.position.y,
               ship_body.angle)

            # apply velocity commands to ship body
            ship_body.angular_velocity = state.r * np.pi / 180
            x_vel, y_vel = state.get_global_velocity()  # get velocities in global frame
            ship_body.velocity = Vec2d(x_vel, y_vel)

            # move simulation forward
            for _ in range(steps):
                space.step(dt / steps)

            # update ship pose
            state.update_pose(ship_body.position.x,
                              ship_body.position.y,
                              ship_body.angle)

            ship_state[0].append(state.x)
            ship_state[1].append(state.y)

            # log updates including tracking error
            (e_x, e_y, e_yaw), track_idx = tracking_error([state.x, state.y, state.yaw], path, get_idx=True)
            past_path[0].append(path[track_idx][0])
            past_path[1].append(path[track_idx][1])

            # update setpoint
            x_s, y_s, h_s = dp.get_setpoint()
            dp.setpoint = np.asarray([x_s, y_s, np.unwrap([state.yaw, h_s])[1]])

            dp.log_step(e_x=e_x, e_y=e_y, e_yaw=e_yaw,
                        setpoint=dp.setpoint.tolist(),
                        total_ke=total_ke[0],
                        total_work=total_work[0],
                        total_impulse=total_impulse[0])

            if t % cfg.anim.plot_steps == 0 and cfg.anim.show:
                plt.pause(0.001)

                # update animation
                plot.update_path(path[track_idx:].T, target=(x_s, y_s), ship_state=ship_state,
                                 past_path=past_path, start_y=path[0, 1])
                plot.update_ship(ship_body, ship_shape, move_yaxis_threshold=cfg.anim.move_yaxis_threshold)
                plot.update_obstacles(obstacles=CostMap.get_obs_from_poly(polygons))
                # get updated obstacles
                plot.animate_sim(save_fig_dir=os.path.join(cfg.output_dir, 'sim_plots')
                                 if (cfg.anim.save and cfg.output_dir) else None, suffix=t)
                plot.sim_fig.suptitle('velocity ({:.2f}, {:.2f}, {:.2f}) [m/s, m/s, rad/s]'
                                      '\nsim iteration {:d}'
                                      .format(ship_body.velocity.x, ship_body.velocity.y, ship_body.angular_velocity, t))

    finally:
        print('Done simulation\nCleaning up...')
        if planner.is_alive():
            # flush queue
            try:
                queue.get_nowait()
            except Empty:
                pass

            # flush pipe
            if not conn_recv.closed and conn_recv.poll():
                conn_recv.recv()

            conn_recv.close(), conn_send.close()

            # flush queue
            queue.put(None)
            queue.close()
            queue.join_thread()
            planner.terminate()
            planner.join()

        del queue, conn_send, conn_recv, planner

        print('Total KE', total_ke[0])
        print('Total impulse', total_impulse[0])
        print('Total work {}'.format(total_work[0]))
        plt.ioff()
        data = dp.get_state_history()
        if len(contact_pts) > 0:
            collision_stats(total_ke[1], data['total_ke'],
                            total_impulse[1], data['total_impulse'],
                            total_work[1], data['total_work'],
                            CostMap.get_obs_from_poly(list(clln_obs)), obstacles,
                            contact_pts, ship_shape.get_vertices(),
                            os.path.join(cfg.output_dir, 'collision') if cfg.output_dir else None)

        if cfg.output_dir or cfg.anim.show:
            pose_data = dp.get_state_history()[['x', 'y']].to_numpy().T
            Plot(
                np.zeros((cfg.costmap.m, cfg.costmap.n)), [{'vertices': ob} for ob in obstacles], goal=goal[1],
                ship_pos=pose_data, map_figsize=None, y_axis_limit=cfg.costmap.m,
            ).save(cfg.output_dir, 'sim', fig='sim')  # TODO: highlight the collided obstacles
        if cfg.anim.show:
            plt.show()
        plt.close('all')


def update_polygons(polygons,
                    space,
                    ship_pos_y,
                    y_axis_limit,
                    obstacle_density,
                    obs_params,     # params for generating obstacles
                    new_obs_count,  # number of times new obstacles have been regenerated
                    new_obs_dist,   # distance traveled before new obstacles are added
                    move_yaxis_threshold,
                    seed
                    ):
    num_obs, max_r = obs_params.get('num_obs'), obs_params.get('max_r')
    # remove polygons that are out of view and behind ship
    for poly in polygons:
        if poly.body.position.y < (ship_pos_y - move_yaxis_threshold - max_r):
            polygons.remove(poly)

    # check if need to generate new polygons
    if ship_pos_y > new_obs_count * new_obs_dist:
        # compute number of new obs to add
        new_obs_num = num_obs - len(polygons)

        if new_obs_num:
            start_pos_y = new_obs_count * new_obs_dist + y_axis_limit
            obs_params = {k: v for k, v in obs_params.items() if k not in ['min_y', 'max_y', 'num_obs']}
            polys_to_add, _ = CostMap.generate_obstacles(
                min_y=start_pos_y, max_y=start_pos_y + new_obs_dist, num_obs=new_obs_num, **obs_params, seed=seed
            )  # this does not guarantee returning the number of requested obstacles

            if polys_to_add:
                polygons.extend(
                    generate_sim_obs(space, polys_to_add, obstacle_density)
                )

        # update count
        new_obs_count += 1

    return new_obs_count


def collision_stats(ke, cum_ke,
                    imp, cum_imp,
                    work, cum_work,
                    clln_obs, all_obs,
                    contact_pts, ship_vertices,
                    save_fig_dir=None):
    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(ke)
    ax[0, 0].set_ylabel('kinetic energy')
    ax[0, 0].set_title('Per iteration')
    ax[0, 1].plot(cum_ke)
    ax[0, 1].set_title('Running total')

    ax[1, 0].plot([(x ** 2 + y ** 2) ** 0.5 for (x, y) in imp])
    ax[1, 0].set_ylabel('impulse (magnitude)')
    ax[1, 1].plot(cum_imp)

    ax[2, 0].plot(work)
    ax[2, 0].set_ylabel('work')
    ax[2, 1].plot(cum_work)

    # compare the obstacle distributions
    clln_obs_mass = [poly_area(obs) for obs in clln_obs]
    all_obs_mass = [poly_area(obs) for obs in all_obs]

    fig2, ax = plt.subplots()
    ax.hist([clln_obs_mass, all_obs_mass], label=['impact', 'all'])
    ax.set_title('Number of floes {}\nNumber of collisions {}'.format(len(all_obs_mass), len(clln_obs_mass)))
    plt.legend(loc='upper right')

    fig3, ax = plt.subplots()
    contact_pts = np.asarray(contact_pts)
    ax.plot(contact_pts.T[0], contact_pts.T[1], 'b.', alpha=0.1)
    ax.set_aspect('equal')
    ax.add_patch(patches.Polygon(ship_vertices, True, fill=False))

    fig4, ax = plt.subplots()
    # compute the lateral distance from centerline
    ax.hist(contact_pts.T[1])

    if save_fig_dir:
        os.makedirs(save_fig_dir)
        fig.savefig(os.path.join(save_fig_dir, 'metrics.png'))
        fig2.savefig(os.path.join(save_fig_dir, 'mass_hist.png'))
        fig3.savefig(os.path.join(save_fig_dir, 'impact_locs.png'))
        fig4.savefig(os.path.join(save_fig_dir, 'impact_hist.png'))
        pickle.dump({'ke': ke, 'cum_ke': cum_ke,
                     'imp': imp, 'cum_imp': cum_imp,
                     'work': work, 'cum_work': cum_work,
                     'clln_obs': clln_obs, 'all_obs': all_obs,
                     'contact_pts': contact_pts, 'ship_vertices': ship_vertices},
                    open(os.path.join(save_fig_dir, 'raw.pk'), 'wb'))


def demo():
    # demo to run a single simulation, for consecutive simulations see experiments/sim_exp.py
    print('2D simulation start')
    ddict = pickle.load(open('experiments/experiments_02-05_100.pk', 'rb'))
    exp_dict = ddict['exp']
    exp = exp_dict[0.5][0]  # 0.5 is the concentration, 0 is the trial number

    sim('configs/sim2d_config.yaml',
        # bottom two flags are for planner
        debug=False,   # enable debugging mode
        logging=False,  # enable logging
        log_level=10,  # log level for planner https://docs.python.org/3/library/logging.html#levels
        init_queue={
            **exp
        })


if __name__ == '__main__':
    demo()
