""" Morphological skeleton path planning approach described here https://ieeexplore.ieee.org/document/9389165 """
import logging
import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.signal import savgol_filter
import sknw
from skimage import draw
from skimage.morphology import skeletonize

from ship_ice_planner.src.cost_map import CostMap
from ship_ice_planner.src.evaluation.metrics import path_length
from ship_ice_planner.src.geometry.polygon import poly_centroid
from ship_ice_planner.src.planners.lattice import METRICS_FILE, PLOT_DIR, PATH_DIR
from ship_ice_planner.src.ship import Ship
from ship_ice_planner.src.swath import compute_swath_cost
from ship_ice_planner.src.utils.message_dispatcher import MessageDispatcher
from ship_ice_planner.src.utils.plot import Plot
from ship_ice_planner.src.utils.storage import Storage

SCALE = 4         # scales the image by this factor when applying skeletonize
WD_LEN = 51       # length of filter window for path smoothing
GOAL_EXTEND = 0   # in metres, the amount to extend the goal y coordinate by,
                  # avoids having to pick an arbitrary goal point since we
                  # can just cut path at a fixed y position
SHRINK = 0.2      # amount to take off each obstacle vertex to ensure path is always found


def skeleton_planner(cfg, debug=False, **kwargs):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('Starting up skeleton planner...')

    # setup message dispatcher
    md = MessageDispatcher(**kwargs)

    # instantiate main objects
    # the costmap and ship objects are only used for the purposes
    # of plotting and computing metrics, the skeleton planner
    # does not interface with either of these components
    costmap = CostMap(horizon=cfg.a_star.horizon, **cfg.costmap)
    ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
    metrics = Storage(output_dir=cfg.output_dir, file=METRICS_FILE)

    # set distance threshold for removing edges near channel border to 2 boat widths
    vertices = np.asarray(cfg.ship.vertices)
    dist_thres = min(vertices[:, 0].max() - vertices[:, 0].min(),
                     vertices[:, 1].max() - vertices[:, 1].min()) * 2

    # directory to store plots
    plot_dir = os.path.join(cfg.output_dir, PLOT_DIR) if cfg.output_dir else None
    if plot_dir:
        os.makedirs(plot_dir)
    # directory to store generated path at each iteration
    path_dir = os.path.join(cfg.output_dir, PATH_DIR) if cfg.output_dir else None
    if path_dir:
        os.makedirs(path_dir)

    # keep track of the planning count
    replan_count = 0
    # keep track of planner rate
    compute_time = []
    curr_goal = np.asarray([0, np.inf, 0])

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

        if goal is not None:
            curr_goal = goal

        # check if ship has made it past the goal line segment
        if ship_pos[1] >= curr_goal[1] or md.shutdown:
            logger.info('At final goal!')
            break

        # check if there is new obstacle information
        if obs is not None:
            # update costmap
            costmap.update(obs, ship_pos[1] - ship.max_ship_length)

        path = morph_skeleton(map_shape=(cfg.costmap.m, cfg.costmap.n),
                              state_data={'ship_state': ship_pos, 'goal': curr_goal, 'obstacles': obs},
                              dist_thres=dist_thres, debug=debug)

        if path is False:
            logger.warning('Planner failed to find a path! Trying again with obstacle shrinking...')

            path = morph_skeleton(map_shape=(cfg.costmap.m, cfg.costmap.n),
                                  state_data={'ship_state': ship_pos, 'goal': curr_goal, 'obstacles': obs},
                                  dist_thres=dist_thres, debug=debug, shrink=SHRINK)

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

        logger.info('Found path! node {} goal {}'.format(ship_pos, curr_goal))
        # send path
        md.send_message(path.T)

        compute_time.append((time.time() - t0))
        logger.info('Step time: {}'.format(compute_time[-1]))
        logger.info('Average planner rate: {} Hz\n'.format(1 / np.mean(compute_time)))

        # scale path to same scaling as costmap
        path[:2] *= costmap.scale

        # compute and log metrics
        # one issue with swath cost is it depends on the number
        # of sampled points along the path, i.e. if we downsample
        # the path the swath cost will likely decrease
        swath, swath_cost = compute_swath_cost(costmap.cost_map, path.T, ship.vertices)
        length = path_length(path[:2].T)

        # log metrics
        metrics.put_scalars(
            iteration=replan_count,
            compute_time=compute_time[-1],
            swath_cost=swath_cost,
            path_length=length
        )
        logger.info(metrics.data)
        metrics.step()

        if path_dir:
            with open(os.path.join(path_dir, str(replan_count) + '.pkl'), 'wb') as handle:
                pickle.dump(
                    {'replan_count': replan_count,
                     'stamp': t0,
                     'path': path,
                     'obstacles': costmap.obstacles,
                     'raw_message': md.raw_message,
                     'processed_message': md.processed_message
                     },
                    handle, protocol=pickle.HIGHEST_PROTOCOL
                )

        if cfg.plot.show:
            if replan_count == 0:
                # instantiate plotting instance on first planning step
                plot = Plot(
                    costmap.cost_map, costmap.obstacles, path,
                    swath=swath, horizon=None, sim_figsize=None, scale=costmap.scale,
                    y_axis_limit=min(cfg.plot.y_axis_limit, cfg.costmap.m) * costmap.scale,
                )
                plt.ion()  # turn on interactive mode
                plt.show()

            else:
                plt.pause(0.001)

                # update plots
                plot.update_path(path, swath)
                plot.update_map(costmap.cost_map, costmap.obstacles)
                plot.animate_map(save_fig_dir=plot_dir, suffix=replan_count)  # save plot to disk

        replan_count += 1

    return metrics.get_history()  # returns a list of scalars/metrics for each planning step


def morph_skeleton(map_shape, state_data, dist_thres=None, debug=False, shrink=None):
    im = np.zeros(((map_shape[0] + GOAL_EXTEND) * SCALE, map_shape[1] * SCALE), dtype='uint8')
    obs = []
    # scale the obstacles
    for x in state_data['obstacles']:
        x = np.asarray(x)

        if shrink:
            # apply shrinking to each obstacle
            centre = np.abs(poly_centroid(x))
            x = np.asarray(
                [[np.sign(a) * (abs(a) - shrink), np.sign(b) * (abs(b) - shrink)] for a, b in x - centre]
            ) + centre

        x = (x * SCALE).astype(np.int32)
        obs.append(x)

    cv2.fillPoly(im, obs, (255, 255, 255))
    im = im.astype(np.bool)
    ske = skeletonize(~im).astype(np.uint16)

    # remove the pixels on skeleton that are too close to channel borders
    if dist_thres:
        ske[:, :int(dist_thres * SCALE)] = 0
        ske[:, -int(dist_thres * SCALE):] = 0
    
    # add a pixel for the start and final position
    goal = [state_data['goal'][0],
            state_data['goal'][1] + GOAL_EXTEND]  # extend goal y coordinate
    ship_state = state_data['ship_state']

    min_dlist = []  # for debugging
    # 8 nearest neighbours + centre
    nn = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    # draw line segment to closest pixel in skeleton
    for p in [ship_state, goal]:
        p = np.asarray(p) * SCALE

        # get min distance to pixel on skeleton
        ske_c = ske.copy()
        ske_c = ske_c.astype(bool).astype('uint8')
        ske_c[ske_c == 0] = 255  # assumes unoccupied cells are 0
        ske_c[ske_c == 1] = 0  # everything else is considered occupied
        dist = cv2.distanceTransform(src=ske_c,
                                     distanceType=cv2.DIST_L2,
                                     maskSize=cv2.DIST_MASK_PRECISE)

        if p[1] > ske.shape[0] or p[0] > ske.shape[1]:
            p = (ske.shape[1] - 10, ske.shape[0] - 10)
        
        min_d = dist[int(round(p[1])), int(round(p[0]))]
        min_dlist.append(min_d)

        # check edge case when p is already a pixel on skeleton
        if ske[int(round(p[1])), int(round(p[0]))]:
            # find a neighbouring pixel that is not yet on skeleton
            for j in nn:
                x, y = p[0] + j[0], p[1] + j[1]
                if 0 < round(x) < im.shape[1] and 0 < round(y) < im.shape[0] and not ske[int(round(y)), int(round(x))]:
                    ske[int(round(y)), int(round(x))] = 1
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
            assert ske[lin].sum() != len(lin[0])
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

    # define heuristic function
    h = lambda a, b: ((graph.nodes[a]['o'][0] - graph.nodes[b]['o'][0]) ** 2 + (
            graph.nodes[a]['o'][1] - graph.nodes[b]['o'][1]) ** 2) ** 0.5
    try:
        path = nx.algorithms.astar_path(graph, s_node, g_node, heuristic=h)
    except nx.NetworkXNoPath:
        return False

    # now build path
    full_path = []
    for i, j in zip(path[:-1], path[1:]):
        # get edge
        edge = graph.edges[i, j]['pts']

        # reverse order of pts if necessary
        if tuple(edge[0]) not in {tuple(item) for item in graph.nodes[i]['pts']}:
            edge = edge[::-1]

        full_path.extend(edge)

    # convert to 2d numpy array
    full_path = np.asarray(full_path).T[::-1]  # shape is 2 x n

    # apply quadratic smoothing
    smooth_path = np.asarray([savgol_filter(full_path[0], WD_LEN, 2, mode='nearest'),
                              savgol_filter(full_path[1], WD_LEN, 2, mode='nearest')])

    # truncate path up until original goal
    smooth_path = smooth_path[..., smooth_path[1] <= (goal[1] - GOAL_EXTEND) * SCALE]

    if debug:
        f, ax = plt.subplots(1, 4)
        ax[0].imshow(im, cmap='gray', origin='lower')
        ax[0].set_title('Original image')

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
        ax[3].plot(full_path[0], full_path[1], 'm--')
        ax[3].plot(smooth_path[0], smooth_path[1], 'c')
        ax[3].set_title('Path')

        plt.show()

    # compute the heading along path
    theta = [
        np.arctan2(j[1] - i[1], j[0] - i[0])
        for i, j in zip(smooth_path.T[:-1], smooth_path.T[1:])
    ]

    # transform path to original scaling and then return
    return np.c_[smooth_path.T[:-1] / SCALE, theta].T  # shape is 3 x n


def demo():
    data = {
        'goal': (6, 70),
        'ship_state': (5, 6, np.pi / 2),
        'obstacles': pickle.load(open('data/demo_ice_data.pk', 'rb'))
    }
    shape = (76, 12)
    morph_skeleton(shape, data, dist_thres=0.5, shrink=0.2, debug=True)


if __name__ == '__main__':
    demo()
