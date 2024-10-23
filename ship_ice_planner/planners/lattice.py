"""
Main planner script
"""
import logging
import os
import pickle
import time
import traceback

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from ship_ice_planner import PLANNER_PLOT_DIR, PATH_DIR, METRICS_FILE
from ship_ice_planner.a_star_search import AStar
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.opti.optimize_path import PathOptimizer
from ship_ice_planner.primitives import Primitives
from ship_ice_planner.ship import Ship
from ship_ice_planner.swath import generate_swath, view_all_swaths
from ship_ice_planner.utils.message_dispatcher import get_communication_interface
from ship_ice_planner.utils.storage import Storage
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import resample_path
from ship_ice_planner.utils.path_compare import Path


def lattice_planner(cfg, debug=False, **kwargs):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('Starting up lattice planner...')

    try:
        # setup message dispatcher
        md = get_communication_interface(**kwargs)
        md.start()

        # instantiate main objects
        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        prim = Primitives(**cfg.prim)
        swath_dict, swath_shape = generate_swath(ship, prim)
        costmap = CostMap(length=cfg.map_shape[0],
                          width=cfg.map_shape[1],
                          ship_mass=ship.mass,
                          padding=swath_shape[0] // 2,
                          **cfg.costmap)
        a_star = AStar(full_costmap=costmap,
                       prim=prim,
                       ship=ship,
                       swath_dict=swath_dict,
                       swath_shape=swath_shape,
                       **cfg.a_star)
        path_compare = Path(threshold_path_progress=cfg.get('threshold_path_progress', 0),
                            threshold_dist_to_path=cfg.get('threshold_dist_to_path', 0) * costmap.scale)
        metrics = Storage(output_dir=cfg.output_dir, file=METRICS_FILE)

        optim_step = cfg.get('optim', False)
        if optim_step:
            path_optimizer = PathOptimizer()
            path_optimizer.set_parameters(ship_length=ship.length,
                                          position_bounds=(costmap.shape[1] - 1,
                                                           costmap.shape[0] - 1),
                                          control_bounds=1 / prim.turning_radius,
                                          ship_vertices=ship.vertices,
                                          **cfg.optim
                                          )

        if debug:
            logger.debug('Showing debug plots for ship footprint, primitives, costmap, and swaths...')
            ship.plot(prim.turning_radius,
                      title='Turning radius {} m\nShip dimensions {} x {} m'.format(
                          prim.turning_radius / costmap.scale,
                          ship.length / costmap.scale, ship.width / costmap.scale)
                      )
            prim.plot()
            costmap.plot(ship_pos=[(costmap.shape[1] - 1) / 2, 0, np.pi / 2],  # dummy value
                         ship_vertices=ship.vertices,
                         prim=prim)
            view_all_swaths(swath_dict)
            # output an upper bound for the total number of nodes for a lattice of this size
            # this calculation does not consider the reduced number of headings on the edges of the lattice
            logger.info('Max number of lattice nodes for a {}x{} grid and prim scale {}: {}'.format(
                *costmap.shape, prim.scale,
                int(((costmap.shape[0] - 1) / prim.scale)) * int(
                    ((costmap.shape[1] - 1) / prim.scale)) * prim.num_headings))

        # directory to store plots
        plot_dir = os.path.join(cfg.output_dir, PLANNER_PLOT_DIR) if cfg.output_dir else None
        if plot_dir:
            os.makedirs(plot_dir)
        # directory to store generated path at each iteration
        path_dir = os.path.join(cfg.output_dir, PATH_DIR) if cfg.output_dir else None
        if path_dir:
            os.makedirs(path_dir)
        if not cfg.plot.show and plot_dir:
            matplotlib.use('Agg')

        # keep track of the planning count
        replan_count = 0
        # keep track of planner rate
        compute_time = []
        prev_goal_y = np.inf
        # keep track of ship xy position in a list for plotting
        ship_actual_path = ([], [])
        horizon = cfg.get('horizon', 0) * costmap.scale
        plot = None

        max_replan = cfg.get('max_replan')
        max_replan = max_replan if max_replan is not None else np.infty

        # start main planner loop
        while replan_count < max_replan:
            logger.info('Re-planning count: {}'.format(replan_count))

            # get new state data
            md.receive_message()

            # start timer
            t0 = time.time()

            # check if the shutdown flag is set
            if md.shutdown:
                logger.info('\033[93mReceived shutdown signal!\033[0m')
                break

            ship_pos = md.ship_state
            goal = md.goal
            obs = md.obstacles
            obs_masses = md.masses

            # scale by the scaling factor
            ship_pos[:2] *= costmap.scale

            # compute the current goal accounting for horizon
            if goal is not None:
                prev_goal_y = goal[1] * costmap.scale
            if horizon:
                sub_goal = ship_pos[1] + horizon
                goal_y = min(prev_goal_y, sub_goal)
            else:
                goal_y = prev_goal_y

            # stop planner when ship is within 1 ship length of the goal
            if ship_pos[1] >= goal_y - ship.length:
                logger.info('\033[92mAt final goal!\033[0m')
                break

            # check if there is new obstacle information
            if obs is not None:
                # update costmap
                t = time.time()
                costmap.update(obs_vertices=obs,
                               obs_masses=obs_masses,
                               # only process obstacles that are past the ship's position with ship length as a buffer distance
                               ship_pos_y=ship_pos[1] - ship.length,
                               ship_speed=cfg.sim_dynamics.target_speed,
                               goal=goal_y)
                logger.info('Costmap update time: {}'.format(time.time() - t))

            if debug:
                logger.debug('Showing debug plot for costmap...')
                costmap.plot(ship_pos=ship_pos,
                             ship_vertices=ship.vertices,
                             prim=prim,
                             goal=goal_y)

            # compute path to goal
            t = time.time()
            search_result = a_star.search(
                start=tuple(ship_pos),
                goal_y=goal_y,
            )
            logger.info('A* search time: {}'.format(time.time() - t))

            if not search_result:
                logger.error('\033[93mPlanner failed to find a path!\033[0m')
                if path_dir:
                    with open(os.path.join(path_dir, str(replan_count) + '_failed.pkl'), 'wb') as handle:
                        pickle.dump(
                            {**a_star.diagnostics,
                             'replan_count': replan_count,
                             'stamp': t0,
                             'goal': goal_y,
                             'ship_state': ship_pos,
                             'obstacles': costmap.obstacles,
                             'costmap': costmap.cost_map,
                             'raw_message': md.raw_message,
                             'processed_message': md.processed_message
                             },
                            handle, protocol=pickle.HIGHEST_PROTOCOL
                        )
                replan_count += 1
                continue

            # unpack result
            new_path, new_swath, node_path, nodes_expanded, g_score, swath_cost, path_length, edge_seq = search_result

            # compare new path to prev path
            path_compare.update(new_path, new_swath, ship_pos)

            if optim_step:
                t1 = time.time()

                optim_metrics = {}
                initial_path_warm_start = None

                if ship_pos[1] < goal_y - ship.length:
                    path_optimizer.set_initial_solution(initial_path=path_compare.path.copy(),
                                                        goal=min(goal_y,
                                                                 ship_pos[1] + cfg.optim.horizon * costmap.scale),
                                                        ship_pose=ship_pos)
                    # this helps with stability of the solver when path is near channel boundaries
                    costmap.clip_boundary_cost()
                    path_optimizer.costmap_to_lut(costmap.cost_map, subset_margin=ship.length)

                    sol = path_optimizer(plot=cfg.optim.plot,
                                         # kwargs for plotting
                                         obstacles=costmap.obstacles,
                                         ship_vertices=ship.vertices,
                                         anim=cfg.optim.anim,
                                         debug=cfg.optim.debug)

                    solved, optimized_path, optim_metrics, initial_path_warm_start = sol

                    if not solved:
                        optimized_path = path_compare.path  # send the path from A* if optimization fails
                        logger.warning('\033[93mOptimization step failed!\033[0m')

                path_real_world_scale = np.c_[(optimized_path[:2] / costmap.scale).T, optimized_path[2]]
                logger.info('Optimization step time: {}'.format(time.time() - t1))

            else:
                # optimization stage not enabled, so just use the path from A*!
                path_real_world_scale = np.c_[(path_compare.path[:2] / costmap.scale).T, path_compare.path[2]]

            compute_time.append((time.time() - t0))
            logger.info('Step time: {}'.format(compute_time[-1]))
            logger.info('Average planner rate: {} Hz\n'.format(1 / np.mean(compute_time)))

            # send path, return path in real world coordinates
            # shape will be n x 3
            md.send_message(resample_path(path_real_world_scale, cfg.path_step_size))

            t1 = time.time()  # second timer to check how long logging and plotting take

            # log metrics
            metrics.put_scalars(
                iteration=replan_count,
                compute_time=compute_time[-1],
                node_cnt=node_path.shape[1],
                expanded_cnt=len(nodes_expanded),
                g_score=g_score,
                swath_cost=swath_cost,
                path_length=path_length
            )
            if optim_step:
                metrics.put_scalars(**optim_metrics)

            logger.info(metrics.data)
            metrics.step()

            if path_dir and cfg.get('save_paths'):
                with open(os.path.join(path_dir, str(replan_count) + '.pkl'), 'wb') as handle:
                    data = {'replan_count': replan_count,
                            'stamp': t0,
                            # planning problem setup
                            # these are all in costmap grid units
                            'goal': goal_y,
                            'ship_state': ship_pos,
                            'obstacles': costmap.obstacles,
                            'costmap': costmap.cost_map,
                            # path computed by planner during this iteration and related information
                            # this is also in costmap grid units
                            'new_path': new_path,
                            'node_path': node_path,
                            'edge_seq': edge_seq,
                            'expanded': nodes_expanded,
                            # message data
                            'raw_message': md.raw_message,
                            'processed_message': md.processed_message
                            # we should also be saving the transmitted message...
                            }
                    if optim_step:
                        data = {**data,
                                'initial_path_warm_start': initial_path_warm_start,
                                'global_path': path_compare.path,
                                'path': optimized_path,
                                }
                    else:
                        data = {**data,
                                'path': path_compare.path
                                }
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if cfg.plot.show or plot_dir:
                ship_actual_path[0].append(ship_pos[0])
                ship_actual_path[1].append(ship_pos[1])

                if plot is None:
                    plot_args = dict(
                        costmap=costmap.cost_map,
                        obstacles=costmap.all_obstacles,
                        ship_vertices=ship.vertices,
                        ship_pos=ship_pos,
                        nodes_expanded=nodes_expanded,
                        sim_figsize=None,
                        scale=costmap.scale,
                        y_axis_limit=cfg.map_shape[0] * costmap.scale,
                        save_fig_dir=plot_dir,
                        show=cfg.plot.show,
                        swath=path_compare.swath,
                    )
                    if optim_step:
                        plot = Plot(
                            **plot_args,
                            path=optimized_path,
                            horizon=cfg.optim.horizon * costmap.scale,
                            global_path=path_compare.path,
                        )

                    else:
                        plot = Plot(
                            **plot_args,
                            path=path_compare.path,
                            horizon=horizon,
                            path_nodes=(node_path[0], node_path[1]),
                        )
                    # for even more debugging
                    # plot.show_prims_from_nodes_edges(plot.map_ax, prim, node_path.T, edge_seq)

                else:
                    if cfg.plot.show:
                        plt.pause(0.1)
                    else:
                        plt.draw()

                    # update plots
                    if optim_step:
                        plot.update_path(optimized_path,
                                         path_compare.swath,
                                         global_path=path_compare.path,
                                         nodes_expanded=nodes_expanded,
                                         ship_state=ship_actual_path,
                                         )
                    else:
                        plot.update_path(path_compare.path,
                                         path_compare.swath,
                                         path_nodes=(node_path[0], node_path[1]),
                                         nodes_expanded=nodes_expanded,
                                         ship_state=ship_actual_path
                                         )
                    plot.update_obstacles(costmap.all_obstacles)
                    plot.update_ship(ship.vertices, *ship_pos)
                    plot.update_map(costmap.cost_map)
                    plot.animate_map(suffix=replan_count)  # save plot to disk

                if debug:
                    plt.show()

            replan_count += 1
            logger.info('Logging and plotting time: {}'.format(time.time() - t1))

    except Exception as e:
        logger.error('\033[91mException raised: {}\n{}\033[0m'.format(e, traceback.format_exc()))

    finally:
        if plot is not None:
            plot.close()
        md.close()
        return metrics.get_history()  # returns a list of scalars/metrics for each planning step
