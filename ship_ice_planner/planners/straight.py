""" Baseline planner that sends a straight path to the goal """
import logging
import os
import pickle
import time
import traceback

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from ship_ice_planner import PLANNER_PLOT_DIR, PATH_DIR
from ship_ice_planner.utils.message_dispatcher import get_communication_interface
from ship_ice_planner.utils.plot import Plot


def straight_planner(cfg, **kwargs):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('Starting up straight planner...')

    try:
        # setup message dispatcher
        md = get_communication_interface(**kwargs)
        md.start()

        vertices = np.asarray(cfg.ship.vertices)

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
        curr_goal = np.asarray([0, np.inf, 0])
        ship_actual_path = ([], [])

        # always plan straight path from the same initial x position
        start_x = None

        max_replan = cfg.get('max_replan')
        max_replan = max_replan if max_replan is not None else np.infty

        # start main planner loop
        while replan_count < max_replan:
            logger.info('Re-planning count: {}'.format(replan_count))

            # get new state data
            md.receive_message()

            # get timestamp
            t0 = time.time()

            # check if the shutdown flag is set
            if md.shutdown:
                logger.info('\033[93mReceived shutdown signal!\033[0m')
                break

            ship_pos = md.ship_state
            goal = md.goal
            obs = md.obstacles

            if start_x is None:
                start_x = ship_pos[0]

            if goal is not None:
                curr_goal = goal

            # check if ship has made it past the goal line segment
            if ship_pos[1] >= curr_goal[1]:
                logger.info('\033[92mAt final goal!\033[0m')
                break

            # only send a new path if new goal
            if goal is not None:
                # can load a precomputed path here, e.g. for open water baseline trials
                if cfg.get('load_path', False) and replan_count == 0:
                    path = pickle.load(open(cfg.load_path, 'rb'))['path']

                else:
                    # compute number of points to sample
                    y = np.arange(ship_pos[1], curr_goal[1], cfg.path_step_size)
                    path = np.asarray([[start_x] * len(y), y, [np.pi / 2] * len(y)])

                # only send path at the first planning step
                md.send_message(path.T)
            else:
                logger.warning('No goal received, not sending path')
                continue

            if path_dir and cfg.get('save_paths'):
                with open(os.path.join(path_dir, str(replan_count) + '.pkl'), 'wb') as handle:
                    pickle.dump(
                        {'replan_count': replan_count,
                         'stamp': t0,
                         'path': path,
                         'raw_message': md.raw_message,
                         'processed_message': md.processed_message
                         },
                        handle, protocol=pickle.HIGHEST_PROTOCOL
                    )

            if cfg.plot.show or plot_dir:
                ship_actual_path[0].append(ship_pos[0])
                ship_actual_path[1].append(ship_pos[1])

                if replan_count == 0:
                    # instantiate plotting instance on first planning step
                    plot = Plot(
                        obstacles=obs, path=path, sim_figsize=None,
                        ship_vertices=vertices, ship_pos=ship_pos,
                        save_fig_dir=plot_dir, show=cfg.plot.show
                    )

                else:
                    if cfg.plot.show:
                        plt.pause(0.1)
                    else:
                        plt.draw()

                    # update plots
                    plot.update_path(path, ship_state=ship_actual_path)
                    plot.update_obstacles(obs)
                    plot.update_ship(vertices, *ship_pos)
                    plot.animate_map(suffix=replan_count)  # save plot to disk

            replan_count += 1

    except Exception as e:
        logger.error('\033[91mException raised: {}\n{}\033[0m'.format(e, traceback.format_exc()))

    finally:
        md.close()
