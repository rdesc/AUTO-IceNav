""" Baseline planner that sends a straight path to the goal """
import logging
import os
import pickle
import time

import numpy as np

from ship_ice_planner.src.planners.lattice import PATH_DIR
from ship_ice_planner.src.utils.message_dispatcher import MessageDispatcher

STEP_SIZE = 0.0125  # step size when sampling path, should be similar to other planners for fair comparison


def straight_planner(cfg, **kwargs):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('Starting up straight planner...')

    # setup message dispatcher
    md = MessageDispatcher(**kwargs)

    # directory to store generated path at each iteration
    path_dir = os.path.join(cfg.output_dir, PATH_DIR) if cfg.output_dir else None
    if path_dir:
        os.makedirs(path_dir)

    # keep track of the planning count
    replan_count = 0
    curr_goal = np.asarray([0, np.inf, 0])

    while replan_count < cfg.get('max_replan', np.infty):
        # start timer
        t0 = time.time()

        # get new state data
        md.receive_message()
        ship_pos = md.ship_state
        goal = md.goal

        if goal is not None:
            curr_goal = goal

        if ship_pos[1] >= curr_goal[1] or md.shutdown:
            logger.info('At final goal!')
            break

        # only send a new path if new goal
        if goal is not None:
            # compute number of points to sample
            y = np.arange(ship_pos[1], curr_goal[1], STEP_SIZE)
            path = np.asarray([[ship_pos[0]] * len(y), y, [np.pi / 2] * len(y)])

            # send path
            md.send_message(path.T)
        else:
            logger.warning('No goal received, not sending path')
            continue

        if path_dir:
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

        replan_count += 1
        time.sleep(0.5)
