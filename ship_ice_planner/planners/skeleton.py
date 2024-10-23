"""
Morphological skeleton path planning approach described here https://ieeexplore.ieee.org/document/9389165

Smoothing is applied so need to consider how the curvature of the path
compares to the minimum turning radius constraint of the ship.
"""
import logging
import os
import pickle
import time
import traceback

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import networkx as nx
import numpy as np
import sknw
from scipy.signal import savgol_filter
from skimage import draw
from skimage.morphology import skeletonize

from ship_ice_planner.geometry.polygon import shrink_or_swell_polygon
from ship_ice_planner import PLANNER_PLOT_DIR, PATH_DIR, METRICS_FILE
from ship_ice_planner.utils.message_dispatcher import get_communication_interface
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.storage import Storage
from ship_ice_planner.utils.utils import resample_path

#############################################################################
# Skeleton planner parameters
#############################################################################
# parameters for full scale
SCALE = 1                       # scales the image by this factor when applying skeletonize
WD_LEN = 201                    # length of filter window for path smoothing
NUM_SMOOTHING_ITERS = 3         # number of times to apply smoothing filter
PATH_PADDING = WD_LEN // 2      # padding to add to path to fix edge artifacts from smoothing
GOAL_EXTEND = 200               # in metres, the amount to extend the goal y coordinate by, avoids having to pick an
                                # arbitrary goal point since we can just cut path at a fixed y position
REMOVE_EDGES_NEAR_BORDER = 9    # how many meters to remove from the edge of the channel to avoid ship hitting channel boundary
                                # set to half the width of the ship
SHRINK_FACTOR = 0.1             # percentage amount to shrink each obstacle to ensure path is always found
SHRINK_RETRIES = 2              # number of times to try shrinking obstacles before giving up

# parameters for model (NRC) scale
# SCALE = 32
# WD_LEN = 301
# NUM_SMOOTHING_ITERS = 3
# PATH_PADDING = WD_LEN
# GOAL_EXTEND = 5
# REMOVE_EDGES_NEAR_BORDER = 0.2
# SHRINK_FACTOR = 0.05
# SHRINK_RETRIES = 3


def skeleton_planner(cfg, debug=False, **kwargs):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('Starting up skeleton planner...')

    try:
        # setup message dispatcher
        md = get_communication_interface(**kwargs)
        md.start()

        # instantiate main objects
        metrics = Storage(output_dir=cfg.output_dir, file=METRICS_FILE)

        # set distance threshold for removing edges near channel border
        vertices = np.asarray(cfg.ship.vertices)
        ship_length = vertices[:, 0].max() - vertices[:, 0].min()

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
        ship_actual_path = ([], [])
        prev_goal = np.array([0, np.inf])
        horizon = cfg.get('horizon', 0)

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

            # compute the current goal accounting for horizon
            if goal is not None:
                prev_goal = goal
            if horizon:
                curr_goal = np.array([prev_goal[0],
                                      min(prev_goal[1], ship_pos[1] + horizon)])
            else:
                curr_goal = prev_goal

            # stop planner when ship is within 1 ship length of the goal
            if ship_pos[1] >= curr_goal[1] - ship_length:
                logger.info('\033[92mAt final goal!\033[0m')
                break

            path = morph_skeleton(map_shape=cfg.map_shape,
                                  state_data={'ship_state': ship_pos, 'goal': curr_goal, 'obstacles': obs},
                                  dist_thres=REMOVE_EDGES_NEAR_BORDER, debug=debug, shrink_factor=SHRINK_FACTOR)

            if path is False:
                curr_shrink = SHRINK_FACTOR * 2
                retry_num = 0

                for i in range(SHRINK_RETRIES):
                    curr_shrink += SHRINK_FACTOR
                    retry_num += 1
                    logger.warning('Planner failed to find a path! Trying again with obstacle shrinking'
                                   'with SHRINK = {} and retry number {}...'.format(curr_shrink, retry_num))

                    path = morph_skeleton(map_shape=cfg.map_shape,
                                          state_data={'ship_state': ship_pos, 'goal': curr_goal, 'obstacles': obs},
                                          dist_thres=REMOVE_EDGES_NEAR_BORDER, debug=debug, shrink_factor=curr_shrink)

                if path is False:
                    logger.error('Shrinking failed to help planner find a path!')

                    if path_dir:
                        with open(os.path.join(path_dir, str(replan_count) + '_failed.pkl'), 'wb') as handle:
                            pickle.dump(
                                {'replan_count': replan_count,
                                 'stamp': t0,
                                 'raw_message': md.raw_message,
                                 'processed_message': md.processed_message
                                 },
                                handle, protocol=pickle.HIGHEST_PROTOCOL
                            )
                    replan_count += 1
                    continue

            logger.info('Found path! start {} goal {}'.format(ship_pos, curr_goal))

            compute_time.append((time.time() - t0))
            logger.info('Step time: {}'.format(compute_time[-1]))
            logger.info('Average planner rate: {} Hz\n'.format(1 / np.mean(compute_time)))

            # send path
            md.send_message(resample_path(path.T, step_size=cfg.path_step_size))

            # log metrics
            metrics.put_scalars(
                iteration=replan_count,
                compute_time=compute_time[-1],
            )
            logger.info(metrics.data)
            metrics.step()

            if path_dir and cfg.get('save_paths'):
                # saving to disk takes about 1 ms
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
        return metrics.get_history()  # returns a list of scalars/metrics for each planning step


def morph_skeleton(map_shape, state_data, dist_thres=None, debug=False, shrink_factor=None):
    im = np.zeros(((int(max(state_data['goal'][1], map_shape[0]) + GOAL_EXTEND) * SCALE) + 1,
                   map_shape[1] * SCALE), dtype='uint8')
    obs = []

    # scale the obstacles
    for ob in state_data['obstacles']:
        ob = np.asarray(ob)

        if shrink_factor:
            # apply shrinking to each obstacle
            ob = shrink_or_swell_polygon(ob, factor=shrink_factor)

        obs.append((ob * SCALE).astype(np.int32))  # int32 is needed for cv2.fillPoly

    cv2.fillPoly(im, obs, (255, 255, 255))
    im = im.astype('bool')
    ske = skeletonize(~im).astype(np.uint16)

    # remove the pixels on skeleton that are too close to channel borders
    if dist_thres:
        margin = int(dist_thres * SCALE)
        if margin != 0:
            ske[:, :margin] = 0
            ske[:, -margin:] = 0

    # add a pixel for the start and final position
    goal = [state_data['goal'][0],
            state_data['goal'][1] + GOAL_EXTEND]  # extend goal y coordinate
    x, y, psi = state_data['ship_state']
    ship_state = (x, max(0, y), psi)  # minimum y coordinate for starting ship position is 0

    min_dlist = []  # for debugging

    # draw line segment to closest pixel in skeleton
    for p in [ship_state, goal]:
        p = np.asarray(p, dtype=int) * SCALE

        # get min distance to pixel on skeleton
        ske_c = ske.copy()
        ske_c = ske_c.astype(bool).astype('uint8')
        ske_c[ske_c == 0] = 255  # assumes unoccupied cells are 0
        ske_c[ske_c == 1] = 0  # everything else is considered occupied
        dist = cv2.distanceTransform(src=ske_c,
                                     distanceType=cv2.DIST_L2,
                                     maskSize=cv2.DIST_MASK_PRECISE)

        min_d = dist[p[1], p[0]]
        min_dlist.append(min_d)

        # check edge case when p is already a pixel on skeleton
        if ske[p[1], p[0]]:
            # find a neighbouring pixel that is not yet on skeleton
            # 8 nearest neighbours + centre
            for j in [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]:
                px, py = p[0] + j[0], p[1] + j[1]
                if 0 < px < im.shape[1] and 0 < py < im.shape[0] and not ske[py, px]:
                    ske[py, px] = 1
                    break

        else:
            # find a pixel on skeleton in circle of radius min_d + [-1, 0, 1, 2]
            for inc in [0, 1, -1, 2]:
                rr, cc = draw.disk((p[1], p[0]), min_d + inc, shape=ske.shape)
                if ske[rr, cc].sum() != 0:
                    break

            assert ske[rr, cc].sum() != 0
            # iterate over pixels inside of circle
            x_best, y_best, d_best = None, None, np.inf
            for r, c in zip(rr, cc):
                d = ((p[0] - c) ** 2 + (p[1] - r) ** 2) ** 0.5
                if d < d_best and ske[r, c]:
                    # make sure this is not the point p itself
                    if c == p[0] and r == p[1]:
                        continue
                    d_best = d
                    x_best, y_best = c, r

            if x_best is None or y_best is None:
                raise ValueError

            # draw the line
            lin = draw.line_nd([p[1], p[0]], [y_best, x_best])
            ske[lin] = 1

    # build graph from skeleton
    graph = sknw.build_sknw(ske)

    # find the key for the start and goal nodes in the graph
    s_key, g_key = (ship_state[0] * SCALE, ship_state[1] * SCALE), (goal[0] * SCALE, goal[1] * SCALE)
    s_node, g_node = {s_key: None, 'dist': np.inf}, {g_key: None, 'dist': np.inf}
    for i in graph.nodes:
        curr_node = graph.nodes[i]['o']
        for node, key in zip([s_node, g_node], [s_key, g_key]):
            if node['dist'] == 0:
                continue  # already found the node in the graph so skip
            d = ((curr_node[0] - key[1]) ** 2 + (curr_node[1] - key[0]) ** 2) ** 0.5
            if d < node['dist']:
                node[key] = i
                node['dist'] = d

    s_node, g_node = s_node[s_key], g_node[g_key]
    assert s_node is not None and g_node is not None

    # skip if start and goal nodes are the same
    if s_node == g_node:
        return False

    try:
        astar_path = nx.algorithms.astar_path(graph, s_node, g_node)  # can define heuristic function here
    except nx.NetworkXNoPath:
        return False

    # now build path
    skel_path = []
    for i, j in zip(astar_path[:-1], astar_path[1:]):
        # get edge
        edge = graph.edges[i, j]['pts']
        skel_path.extend(edge)

    # convert to 2d numpy array
    skel_path = np.asarray(skel_path).T[::-1]  # shape is 2 x n

    # pad the path to fix issues with savgol filter at the edge
    smooth_path = np.hstack(([[x - PATH_PADDING * np.cos(psi)], [y - PATH_PADDING * np.sin(psi)]], skel_path))

    # resample path
    smooth_path = resample_path(smooth_path.T, step_size=1, linear=True).T

    # apply smoothing
    for _ in range(NUM_SMOOTHING_ITERS):
        smooth_path = np.asarray([savgol_filter(smooth_path[0], WD_LEN, 3, mode='nearest'),
                                  savgol_filter(smooth_path[1], WD_LEN, 3, mode='nearest')])

    # transform path to original scaling
    smooth_path = np.c_[smooth_path.T[:-1] / SCALE,
                        # compute the heading along path
                        np.arctan2(np.diff(smooth_path[1]), np.diff(smooth_path[0]))].T

    # truncate path up until original goal and original start
    smooth_path = smooth_path[..., smooth_path[1] <= (goal[1] - GOAL_EXTEND + 1)]
    smooth_path = smooth_path[..., y <= smooth_path[1]]

    if debug:
        f, ax = plt.subplots(1, 4, sharex=True, sharey=True)
        ax[0].imshow(im, cmap='gray', origin='lower')
        for ob in state_data['obstacles']:
            ax[0].add_patch(patches.Polygon(ob * SCALE, True, fill=False, ec='m'))
        ax[0].set_title('Original image')
        ax[0].set_xlim(0, im.shape[1])
        ax[0].set_ylim(0, im.shape[0])

        ax[1].imshow(ske, cmap='gray', origin='lower')
        for p, min_d in zip([ship_state, goal], min_dlist):
            p = np.asarray(p) * SCALE
            ax[1].plot(np.cos(np.arange(0, 2 * np.pi, 0.1)) * min_d + p[0],
                       np.sin(np.arange(0, 2 * np.pi, 0.1)) * min_d + p[1])
        ax[1].set_title('Skeleton')

        ax[2].imshow(im, cmap='gray', origin='lower')
        # draw edges by pts
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            ax[2].plot(ps[:, 1], ps[:, 0], 'green')

        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax[2].plot(ps[:, 1], ps[:, 0], 'r.')
        ax[2].set_title('Graph')

        # show on plots start and goal points
        for a in ax:
            a.plot(ship_state[0] * SCALE, ship_state[1] * SCALE, 'gx')
            a.plot(goal[0] * SCALE, goal[1] * SCALE, 'gx')

        # draw path
        ax[3].imshow(im, cmap='gray', origin='lower')
        ax[3].plot(skel_path[0], skel_path[1], 'm--')
        ax[3].plot(smooth_path[0] * SCALE, smooth_path[1] * SCALE, 'c')
        ax[3].set_title('Path')

        f2, ax2 = plt.subplots(3, 1)
        ax2[0].plot(smooth_path[0])
        ax2[0].set_title('position x')
        ax2[1].plot(smooth_path[1])
        ax2[1].set_title('position y')
        ax2[2].plot(smooth_path[2])
        ax2[2].set_title('heading psi')

        plt.show()

    return smooth_path  # shape is 3 x n


def demo():
    # demo with full scale ice field
    from ship_ice_planner import FULL_SCALE_SIM_EXP_CONFIG
    exp_data = pickle.load(open(FULL_SCALE_SIM_EXP_CONFIG, 'rb'))['exp']
    ice_concentration = 0.5
    ice_field_idx = 0
    obstacles = exp_data[ice_concentration][ice_field_idx]['obstacles']
    obstacles = [ob['vertices'] for ob in obstacles]

    data = {
        'goal': (100., 1100.),
        'ship_state': (100., 0., np.pi / 2),
        'obstacles': obstacles
    }
    morph_skeleton(map_shape=(1100, 200),
                   state_data=data,
                   dist_thres=REMOVE_EDGES_NEAR_BORDER,  # removes edges near channel border, set to half the width of ship
                   shrink_factor=SHRINK_FACTOR,          # shrinks obstacles by 10%
                   debug=True
                   )


if __name__ == '__main__':
    # run with `python -m ship_ice_planner.planners.skeleton`
    demo()
