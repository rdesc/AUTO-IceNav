""" Generates randomized experiment configurations """
import pickle
import random
from typing import List

import numpy as np
import packcircles as pc
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import draw
from tqdm import tqdm

from ship_ice_planner.src.geometry.polygon import generate_polygon, poly_area


FILE_NAME = 'experiment_configs.pk'         # file name to save experiment configurations
SHOW_PLOT = True                            # show plots of generated experiments
N = 10                                      # number of experiments per ice field concentration
CONCENTRATIONS = [0.3, 0.4, 0.5, 0.6, 0.7]  # ice field concentrations
MAP_SHAPE = (70, 12)                        # size of rectangular ice field
OBSTACLE = {
    'min_r': 0.2,                           # min and max radius of obstacles
    'max_r': 2,
    'min_y': 2,                             # boundaries of where obstacles can be placed
    'max_y': 70,
    'circular': False,                      # if True, obstacles are more circular, otherwise they are convex polygons
    'exp_dist': True                        # if True, obstacle radii are sampled from an exponential distribution
}
SHIP_STATE = {
    # range for x, y, theta for generating a random ship starting position
    'range_x': [3, 9],  # if set to None, then the ship starts at the x position with lowest ice concentration
    'range_y': [1, 1],
    'range_theta': [np.pi / 2, np.pi / 2]
}
GOAL = 70               # goal is a line segment with y position at GOAL
TOL = 0.01              # tolerance of deviation of actual concentration from desired concentration
SCALE = 8               # scales map by this factor for occupancy grid resolution
IM_SHAPE = (MAP_SHAPE[0] * SCALE, MAP_SHAPE[1] * SCALE)
VERTICES = np.asarray([[1., -0.],  # ship vertices (for plotting)
                       [0.9, 0.10],
                       [0.5, 0.25],
                       [-1., 0.25],
                       [-1., -0.25],
                       [0.5, -0.25],
                       [0.9, -0.10]])
PADDING = 0.0
VERTICES = np.asarray(
    [[np.sign(a) * (abs(a) + PADDING), np.sign(b) * (abs(b) + PADDING)] for a, b in VERTICES]
)
SEED = 99

np.random.seed(SEED)
random.seed(SEED)


def compute_circle_ob_concentration(r: np.ndarray):
    circ_area = ((r ** 2) * np.pi).sum()
    return circ_area / (MAP_SHAPE[0] * MAP_SHAPE[1])


def compute_poly_ob_concentration(polys):
    im = np.zeros((MAP_SHAPE[0] * SCALE, MAP_SHAPE[1] * SCALE))
    area = 0
    for p in polys:
        area += p['area']
        rr, cc = p['pixels']
        im[rr, cc] = 1

    return area / (MAP_SHAPE[1] * (OBSTACLE['max_y'] - OBSTACLE['min_y'])), im


def increase_concentration(obstacles, desired_concentration):
    actual_concentration, im = compute_poly_ob_concentration(obstacles)

    max_r = OBSTACLE['max_r']
    num_added = 0
    trials = 0

    while actual_concentration < desired_concentration - TOL and max_r - OBSTACLE['min_r'] > 0.05:
        if trials % 10 == 0:
            print('num trials', trials, 'current concentration', actual_concentration, 'max_r', max_r, 'added obs', num_added)
        trials += 1
        r = np.random.uniform(OBSTACLE['min_r'], max_r)
        slice_shape = int(max(1 / SCALE, r * 2) * SCALE)  # in map units

        # find all the slices that would a new obstacle using a sliding window approach
        new_obs_centres = []
        rand_offset_x = np.random.choice(np.arange(slice_shape))
        rand_offset_y = np.random.choice(np.arange(slice_shape))
        for i in range(rand_offset_y, im.shape[0] - slice_shape + 1, slice_shape):
            for j in range(rand_offset_x, im.shape[1] - slice_shape + 1, slice_shape):
                # skip if slice is beyond ice edge
                if OBSTACLE['min_y'] * SCALE <= i and i + slice_shape <= OBSTACLE['max_y'] * SCALE:
                    if im[i: i + slice_shape, j: j + slice_shape].sum() == 0:
                        new_obs_centres.append([(j + slice_shape / 2) / SCALE, (i + slice_shape / 2) / SCALE])

        if len(new_obs_centres) == 0:
            # decrease upper bound on r if no slices were found
            max_r = r
        else:
            # randomly choose one of these slices and generate an obstacle
            indices = np.random.choice(len(new_obs_centres), size=int(len(new_obs_centres) * 0.5), replace=False)
            for ind in indices:
                slice_choice = new_obs_centres[ind]
                x, y = slice_choice
                r = slice_shape / SCALE / 2
                vertices = generate_polygon(diameter=r * 2, origin=(x, y), circular=OBSTACLE['circular'])

                if vertices is not None:
                    # add ob to obstacles list
                    rr, cc = draw.polygon(vertices[:, 1] * SCALE, vertices[:, 0] * SCALE, shape=IM_SHAPE)
                    if len(rr) == 0 and im[rr, cc].sum() > 0:
                        continue
                    obstacles.append({
                        'vertices': vertices,
                        'centre': (x, y),
                        'radius': r,
                        'pixels': (rr, cc),
                        'area': poly_area(vertices)
                    })

                    num_added += 1

                    # compute new poly concentration
                    actual_concentration, im = compute_poly_ob_concentration(obstacles)

    print('added {} obstacles over {} trails!\ndesired concentration {}, actual concentration {}'
          .format(num_added, trials, desired_concentration, actual_concentration))

    return obstacles


def decrease_concentration(obstacles: List[dict], desired_concentration: float):
    actual_concentration, _ = compute_poly_ob_concentration(obstacles)
    num_deleted = 0
    while actual_concentration > desired_concentration + TOL:
        ind = np.random.choice(np.arange(len(obstacles)))
        obstacles = np.delete(obstacles, ind, axis=0)
        np.random.shuffle(obstacles)
        actual_concentration, _ = compute_poly_ob_concentration(obstacles)
        num_deleted += 1

    print('deleted {} obstacles!\ndesired concentration {}, actual concentration {}'
          .format(num_deleted, desired_concentration, actual_concentration))
    return obstacles


def find_best_start_x(obstacles, slice_shape=(10, 3)):
    # generate corresponding occupancy grid given obstacles
    im = np.zeros((MAP_SHAPE[0] * SCALE, MAP_SHAPE[1] * SCALE))
    for ob in obstacles:
        rr, cc = draw.polygon(ob['vertices'][:, 1] * SCALE, ob['vertices'][:, 0] * SCALE, shape=im.shape)
        im[rr, cc] = 1

    # find the slice that has the lowest concentration in obstacles using a sliding window approach
    # slice_shape[1] needs to be an odd number and be less than MAP_SHAPE[1]
    # we skip the first and last slice to avoid the ship starting too close to channel boundaries
    c = []
    for i in range((im.shape[1] - slice_shape[1] * SCALE) // SCALE):
        if i == 0:
            c.append(np.inf)
        sub_im = im[: slice_shape[0] * SCALE, i * SCALE: (i + slice_shape[1]) * SCALE]
        c.append(sub_im.sum() / np.multiply(*sub_im.shape))

    # get the indices for all the minimums
    min_idx = np.where(np.asarray(c) == np.min(c))[0]

    # return index closet to the middle if there are multiple mins
    if len(min_idx) != 0:
        best_idx = min_idx[np.argmin(np.abs((min_idx + (min_idx + slice_shape[1])) // 2 - MAP_SHAPE[1] / 2))].item()

    else:
        best_idx = min_idx[0]

    return (best_idx + (best_idx + slice_shape[1])) // 2


def plot(circs, polys, pose, concentration):
    fig, ax = plt.subplots(1, 3)
    # plot circles
    for (x, y, r) in circs:
        patch = plt.Circle((x, y), r, fc='b', ec='k', alpha=0.65)
        ax[0].add_patch(patch)
    ax[0].set_aspect('equal')

    # compute actual concentration
    ax[0].set_title('Circle Packing Result')

    # plot polygons
    for p in polys:
        patch = patches.Polygon(p['vertices'], True, fill=True, fc='b', ec='k', alpha=0.65)
        ax[1].add_patch(patch)
    ax[1].set_aspect('equal')

    # compute actual concentration
    poly_conc, im = compute_poly_ob_concentration(polys)
    ax[1].set_title('Desired Concentration {:.2f}\nActual Concentration {:.2f}'.format(concentration, poly_conc))

    ax[2].set_title('Occupancy Grid')
    ax[2].imshow(im, origin='lower', cmap='gray', extent=[0, MAP_SHAPE[1], 0, MAP_SHAPE[0]])

    # show ship footprint
    R = np.asarray([
        [np.cos(pose[2]), -np.sin(pose[2])],
        [np.sin(pose[2]), np.cos(pose[2])]
    ])
    # show pose on plots
    for a in ax:
        a.add_patch(patches.Polygon(VERTICES @ R.T + [pose[0], pose[1]], True, fill=True, fc='g'))
        a.plot(pose[0], pose[1], 'rx')

    # show goal
    for a in ax:
        a.plot([0, MAP_SHAPE[1]], [GOAL, GOAL], 'g-')

    plt.show()

    fig, ax = plt.subplots()
    masses = [p['area'] for p in polys]
    ax.hist(masses, bins=20)
    ax.set_title('Mass Distribution')
    ax.set_xlabel('Mass')
    plt.show()


def generate_rand_exp(show_plot=False):
    # dict to store the experiments
    exp_dict = {
        'meta_data': {
            'N': N,
            'concentrations': CONCENTRATIONS,
            'map_shape': MAP_SHAPE,
            'obstacle_config': OBSTACLE,
            'ship_state_config': SHIP_STATE,
            'goal': GOAL,
            'scale': SCALE,
            'seed': SEED
        },
        'exp': {c: {i: {'goal': None, 'ship_state': None, 'obstacles': None} for i in range(N)} for c in CONCENTRATIONS}
    }

    # approximate how many circles we need to pack environment assuming average radius
    if OBSTACLE['exp_dist']:
        avg_r = OBSTACLE['min_r'] * 1.5
    else:
        avg_r = (OBSTACLE['min_r'] + OBSTACLE['max_r']) / 2
    num_circ = (np.pi * (((MAP_SHAPE[0] ** 2 + MAP_SHAPE[1] ** 2) ** 0.5) / 2) ** 2) / (np.pi * avg_r ** 2)
    # approach is to first pack the environment with circles then convert circles
    # to polygons and then do rejection sampling to attain the desired concentration
    for conc in tqdm(CONCENTRATIONS):
        for i in tqdm(range(N)):
            # sample random radii
            if OBSTACLE['exp_dist']:
                radii = np.maximum(OBSTACLE['min_r'], np.minimum(
                    OBSTACLE['max_r'], np.random.exponential(scale=avg_r, size=int(num_circ))
                ))
            else:
                radii = np.random.uniform(OBSTACLE['min_r'], OBSTACLE['max_r'], size=int(num_circ))
            gen = pc.pack(radii)  # this is deterministic! Randomness comes from radii
            circles = np.asarray([(x, y, r) for (x, y, r) in gen])
            circles[:, 1] += (-circles[:, 1].min())
            circles[:, 0] += MAP_SHAPE[1]

            # remove the circles outside of environment boundaries
            circles = circles[np.logical_and(circles[:, 0] >= 0,
                                             circles[:, 0] <= MAP_SHAPE[1])]
            circles = circles[np.logical_and(circles[:, 1] >= 0,
                                             circles[:, 1] <= MAP_SHAPE[0])]
            # apply constraints specified in obstacle parameters
            circles = circles[np.logical_and(circles[:, 1] >= OBSTACLE.get('min_y', 0),
                                             circles[:, 1] <= OBSTACLE.get('max_y', MAP_SHAPE[0]))]

            np.random.shuffle(circles)

            # now generate polygons for each circle
            obstacles = []

            for (x, y, radius) in circles:
                vertices = generate_polygon(diameter=radius * 2, origin=(x, y), circular=OBSTACLE['circular'])

                if vertices is not None:
                    # take intersection of vertices and environment boundaries
                    vertices[:, 0][vertices[:, 0] < 0] = 0
                    vertices[:, 0][vertices[:, 0] >= MAP_SHAPE[1]] = MAP_SHAPE[1]

                    min_y = OBSTACLE.get('min_y', False) or 0
                    max_y = OBSTACLE.get('max_y', False) or MAP_SHAPE[0]
                    vertices[:, 1][vertices[:, 1] < min_y] = min_y
                    vertices[:, 1][vertices[:, 1] > max_y] = max_y

                    rr, cc = draw.polygon(vertices[:, 1] * SCALE, vertices[:, 0] * SCALE, shape=IM_SHAPE)
                    obstacles.append({
                        'vertices': vertices,
                        'centre': (x, y),
                        'radius': radius,
                        'pixels': (rr, cc),
                        'area': poly_area(vertices)
                    })

            # get concentration of ice field with polygon obstacles
            poly_concentration, _ = compute_poly_ob_concentration(obstacles)
            if abs(conc - poly_concentration) > TOL:
                print('\ndesired concentration {}, actual concentration {}'.format(conc, poly_concentration))
                if conc > poly_concentration:
                    # randomly add obstacles:
                    obstacles = increase_concentration(obstacles, conc)
                else:
                    obstacles = decrease_concentration(obstacles, conc)

            # add obstacles to dict
            exp_dict['exp'][conc][i]['obstacles'] = obstacles

            # add goal to dict
            exp_dict['exp'][conc][i]['goal'] = GOAL

            # generate ship starting state
            if SHIP_STATE['range_x'] is None:
                x = find_best_start_x(obstacles)
            else:
                x = np.random.uniform(low=SHIP_STATE['range_x'][0], high=SHIP_STATE['range_x'][1])
            y = np.random.uniform(low=SHIP_STATE['range_y'][0], high=SHIP_STATE['range_y'][1])
            theta = np.random.uniform(low=SHIP_STATE['range_theta'][0], high=SHIP_STATE['range_theta'][1])
            ship_state = (x, y, theta)

            # add to ship state to dict
            exp_dict['exp'][conc][i]['ship_state'] = ship_state

            if show_plot:
                plot(circles, obstacles, ship_state, conc)

    # save to disk
    if FILE_NAME:
        with open(FILE_NAME, 'wb') as f:
            pickle.dump(exp_dict, f)


if __name__ == '__main__':
    generate_rand_exp(show_plot=SHOW_PLOT)
