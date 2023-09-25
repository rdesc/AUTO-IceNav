import logging
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt

from ship_ice_planner.src.a_star_search import AStar
from ship_ice_planner.src.cost_map import CostMap
from ship_ice_planner.src.primitives import Primitives
from ship_ice_planner.src.ship import Ship
from ship_ice_planner.src.swath import generate_swath, view_all_swaths
from ship_ice_planner.src.utils.message_dispatcher import MessageDispatcher
from ship_ice_planner.src.utils.storage import Storage
from ship_ice_planner.src.utils.plot import Plot
from ship_ice_planner.src.utils.utils import Path

# global vars for dir/file names
PLOT_DIR = 'plots'
PATH_DIR = 'paths'
METRICS_FILE = 'metrics.txt'


# @profile
def lattice_planner(cfg, debug=False, **kwargs):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('Starting up lattice planner...')

    # setup message dispatcher
    md = MessageDispatcher(**kwargs)

    # instantiate main objects
    costmap = CostMap(horizon=cfg.a_star.horizon,
                      ship_mass=cfg.ship.mass, **cfg.costmap)
    ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
    prim = Primitives(**cfg.prim)
    swath_dict = generate_swath(ship, prim)
    # ship.plot(prim.turning_radius)
    # prim.plot()
    # view_all_swaths(swath_dict); exit()
    a_star = AStar(cmap=costmap,
                   prim=prim,
                   ship=ship,
                   swath_dict=swath_dict,
                   **cfg.a_star)
    path_obj = Path()
    metrics = Storage(output_dir=cfg.output_dir, file=METRICS_FILE)

    if debug:
        ship.plot(prim.turning_radius)
        prim.plot()
        m, n = costmap.shape
        # output an upper bound for the total number of nodes for a lattice of this size
        # this calculation does not consider the reduced number of headings on the edges of the lattice
        logger.info('Total number of nodes for a {}x{} grid (horizon) and prim scale {}: {}'.format(
            costmap.horizon, n, int(prim.scale),
            (costmap.horizon / prim.scale + 1) * (n / prim.scale + 1) * prim.num_headings))

    # directory to store plots
    plot_dir = os.path.join(cfg.output_dir, PLOT_DIR) if cfg.output_dir else None
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    # directory to store generated path at each iteration
    path_dir = os.path.join(cfg.output_dir, PATH_DIR) if cfg.output_dir else None
    if path_dir:
        os.makedirs(path_dir, exist_ok=True)

    # keep track of the planning count
    replan_count = 0
    # keep track of planner rate
    compute_time = []
    last_goal_y = np.inf

    # start main planner loop
    while replan_count < cfg.get('max_replan', np.infty):
        logger.info('Re-planning count: {}'.format(replan_count))

        # start timer
        t0 = time.time()

        # get new state data
        md.receive_message()
        ship_pos = md.ship_state
        goal = md.goal
        obs = md.obstacles

        # compute next goal
        if goal is not None:
            goal_y = min(goal[1], (ship_pos[1] + cfg.a_star.horizon)) * cfg.costmap.scale
            last_goal_y = goal[1]
        else:
            goal_y = min(last_goal_y, (ship_pos[1] + cfg.a_star.horizon)) * cfg.costmap.scale

        # scale x and y by the scaling factor
        ship_pos[:2] *= cfg.costmap.scale

        # check if ship has made it past the goal line segment
        if ship_pos[1] >= goal_y or md.shutdown:
            # plt.close(plot.map_fig)
            logger.info('At final goal!')
            break

        # check if there is new obstacle information
        if obs is not None:
            # update costmap
            costmap.update(obs, ship_pos[1] - ship.max_ship_length / 2,
                           vs=(md.metadata.get('velocity', 0.3) * cfg.costmap.scale + 1e-8))

        if debug:
            costmap.plot(obs, ship_pos, ship.vertices, prim)

        # compute path to goal
        search_result = a_star.search(
            start=(ship_pos[0], ship_pos[1], ship_pos[2]),
            goal_y=goal_y,
        )

        if not search_result:
            logger.error('Planner failed to find a path!')
            if path_dir:
                with open(os.path.join(path_dir, str(replan_count) + '_failed.pkl'), 'wb') as handle:
                    pickle.dump(
                        {**a_star.diagnostics,
                         'replan_count': replan_count,
                         'stamp': t0,
                         'ship_state': ship_pos,
                         'obstacles': costmap.obstacles,
                         'raw_message': md.raw_message,
                         'processed_message': md.processed_message
                         },
                        handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
            replan_count += 1
            continue

        # unpack result
        (full_path, full_swath), \
        (node_path, node_path_length), \
        (node_path_smth, new_nodes), \
        (nodes_expanded, g_score, swath_cost, length, edge_seq) = search_result
        x1, y1, _ = node_path  # this is the original node path prior to smoothing
        x2, y2 = new_nodes if len(new_nodes) != 0 else (0, 0)  # these are the nodes added from smoothing

        # compare new path to prev path
        send_new_path = path_obj.update(full_path, full_swath, costmap.cost_map, ship_pos[1],
                                        threshold_dist=cfg.get('threshold_dist', 0) * length,
                                        threshold_cost=cfg.get('threshold_cost'))

        # send path, return path in original scale
        # shape will be n x 3
        path_true_scale = np.c_[(path_obj.path[:2] / cfg.costmap.scale).T, path_obj.path[2]]  # TODO: confirm heading is ok

        if send_new_path:
            md.send_message(path_true_scale)

        compute_time.append((time.time() - t0))
        logger.info('Step time: {}'.format(compute_time[-1]))
        logger.info('Average planner rate: {} Hz\n'.format(1 / np.mean(compute_time)))

        t1 = time.time()  # second timer to check how long logging and plotting take

        # log metrics
        metrics.put_scalars(
            iteration=replan_count,
            compute_time=compute_time[-1],
            node_cnt=node_path.shape[1],
            smth_node_cnt=node_path_smth.shape[1]
            if len(node_path_smth) != 0 else 0,
            expanded_cnt=len(nodes_expanded),
            g_score=g_score,
            swath_cost=swath_cost,
            path_length=length / costmap.scale,
            send_new_path=send_new_path
        )
        logger.info(metrics.data)
        metrics.step()

        if path_dir:
            with open(os.path.join(path_dir, str(replan_count) + '.pkl'), 'wb') as handle:
                pickle.dump(
                    {'replan_count': replan_count,
                     'stamp': t0,
                     # information related to new generated path
                     'new_path': full_path,
                     'new_swath': full_swath,
                     'node_path': node_path,
                     'node_path_smth': node_path_smth,
                     'edge_seq': edge_seq,
                     'expanded': nodes_expanded,
                     # path chosen between new and prev
                     'path': path_obj.path,
                     'swath': path_obj.swath,
                     'send_new_path': send_new_path,
                     # planning problem setup
                     'goal': goal_y,
                     'ship_state': ship_pos,
                     'obstacles': costmap.obstacles,
                     'costmap': costmap.cost_map,
                     'raw_message': md.raw_message,
                     'processed_message': md.processed_message},
                    handle, protocol=pickle.HIGHEST_PROTOCOL
                )

        if cfg.plot.show:
            if replan_count == 0:
                # NOTE: plotting slows down planner quite a bit
                # instantiate plotting instance on first planning step
                plot = Plot(
                    costmap.cost_map, costmap.obstacles, full_path,
                    ship_vertices=ship.vertices, ship_pos=ship_pos,
                    path_nodes=(x1, y1), nodes_expanded=nodes_expanded, smoothing_nodes=(x2, y2),
                    swath=full_swath, horizon=costmap.horizon, sim_figsize=None,
                    scale=costmap.scale, y_axis_limit=min(cfg.plot.y_axis_limit, cfg.costmap.m) * costmap.scale,
                )
                # TODO: fix issue with horizon when goal less than horizon
                # TODO: reset costmap color scaling
                # plot.show_prims_from_nodes_edges(plot.map_ax, prim, node_path.T, edge_seq)  # for even more debugging
                if not debug:
                    plt.ion()  # turn on interactive mode
                plt.show()
                plot.save(save_fig_dir=plot_dir, suffix=str(replan_count))

            else:
                plt.pause(0.001)

                # update plots
                plot.update_path(path_obj.path,
                                 path_obj.swath,
                                 path_nodes=(x1, y1),
                                 smoothing_nodes=(x2, y2),
                                 nodes_expanded=nodes_expanded)
                plot.update_map(costmap.cost_map, costmap.obstacles, ship.vertices, ship_pos)
                plot.animate_map(save_fig_dir=plot_dir, suffix=replan_count)  # save plot to disk

        replan_count += 1
        logger.info('Logging and plotting time: {}'.format(time.time() - t1))

    return metrics.get_history()  # returns a list of scalars/metrics for each planning step
