""" Generates randomized experiment configurations """
import os.path
import pickle
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import packcircles as pc
from matplotlib import patches
from scipy.stats.distributions import lognorm
from skimage import draw
from tqdm import tqdm

from ship_ice_planner.geometry.polygon import *
from ship_ice_planner.geometry.utils import Rxy
from ship_ice_planner.ship import FULL_SCALE_PSV_VERTICES, OEB_PSV_VERTICES
from ship_ice_planner.utils.plot import ICE_PATCH_COLOR, SHIP_PATCH_COLOR, OPEN_WATER_COLOR
from ship_ice_planner.utils.sim_utils import ICE_DENSITY, ICE_THICKNESS

FILE_NAME = 'data/experiment_configs.pkl'         # file name to save experiment configurations
SHOW_PLOT = False                                 # show plots of generated experiments
N = 100                                           # number of experiments per ice field concentration
CONCENTRATIONS = [0.2, 0.3, 0.4, 0.5]             # ice field concentrations
START_ICE_FIELD_DIST = 100.0                      # distance from ship starting position to start of ice field
MAP_SHAPE = (1000 + START_ICE_FIELD_DIST,
             200)                                 # size of rectangular ice field -- length x width in metres
SHIP_STATE = {
    # starting ship x, y position and psi heading
    'x': MAP_SHAPE[1] / 2,
    'y': 0,  # ship starts at bottom of map
    'psi': np.pi / 2
}
GOAL = (MAP_SHAPE[1] / 2, MAP_SHAPE[0])
OBSTACLE = {
    # 2 - 100 m width floe range from paper
    # "Observed changes in sea-ice floe size distribution during early summer in the western Weddell Sea"
    'min_r': 2,                                   # min and max radius of obstacles
    'max_r': 50,
    'min_area': 16,                               # min area of obstacles
    'min_y': START_ICE_FIELD_DIST,                # boundaries of where obstacles can be placed
    'max_y': MAP_SHAPE[0],
    'circular': False,                            # if True, obstacles are more circular
    'num_vertices_range': (20, 30),               # range for number of vertices for random convex polygons
    'final_vertices_range': (5, 20)               # downsamples the polygon vertices to a number in this range
}
TOL = 0.01                                        # tolerance of deviation of actual concentration from desired
SCALE = 1                                         # scales map by this factor for occupancy grid resolution
IM_SHAPE = (int(MAP_SHAPE[0] * SCALE),
            int(MAP_SHAPE[1] * SCALE))
SHIP_VERTICES = FULL_SCALE_PSV_VERTICES           # ship vertices for plotting

# ice parameters for ice floe generation
ICE_INIT_NUMBER_CIRCLES = 11000                   # number of initial ice floes to generate
                                                  # should be large enough so that circles pack the map
ICE_AREA_REMOVE_MAX = 400  # m^2

# ice floe generation is deterministic
SEED = 99

np.random.seed(SEED)
random.seed(SEED)
RNG = np.random.RandomState()  # for later use
RNG.set_state(np.random.get_state())


def compute_poly_ob_concentration(polys):
    im = np.zeros(IM_SHAPE)
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
    iters = 0

    while actual_concentration < desired_concentration - TOL and max_r - OBSTACLE['min_r'] > 0.05:
        if iters % 10 == 0:
            print('num iters', iters, 'current concentration', actual_concentration, 'max_r', max_r, 'added obs', num_added)
        iters += 1
        r = np.random.uniform(OBSTACLE['min_r'], max_r)
        slice_shape = int(max(1 / SCALE, r * 2) * SCALE)  # in map units

        # find all the slices that would fit a new obstacle using a sliding window approach
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
                vertices = generate_polygon(diameter=r * 2,
                                            origin=(x, y),
                                            circular=OBSTACLE['circular'],
                                            num_vertices_range=OBSTACLE['num_vertices_range'],
                                            final_vertices_range=OBSTACLE['final_vertices_range'])

                if vertices is not None:
                    # add ob to obstacles list
                    area = poly_area(vertices)
                    rr, cc = draw.polygon(vertices[:, 1] * SCALE, vertices[:, 0] * SCALE, shape=IM_SHAPE)

                    if len(rr) == 0 or im[rr, cc].sum() > 0 or area < OBSTACLE['min_area']:
                        continue  # skip this polygon!

                    obstacles.append({
                        'vertices': vertices,
                        'centre': (x, y),
                        'radius': poly_radius(vertices, centre_pos=(x, y)),
                        'pixels': (rr, cc),
                        'area': area,
                        'mass': area_to_mass(area)
                    })
                    num_added += 1

                    # compute new poly concentration
                    actual_concentration, im = compute_poly_ob_concentration(obstacles)

    print('added {} obstacles over {} iterations!\ndesired concentration {}, actual concentration {}'
          .format(num_added, iters, desired_concentration, actual_concentration))

    return obstacles


def decrease_concentration(obstacles: List[dict], desired_concentration: float):
    # randomize the order of obstacles
    np.random.shuffle(obstacles)
    actual_concentration, _ = compute_poly_ob_concentration(obstacles)
    num_deleted = 0
    while actual_concentration > desired_concentration + TOL:
        ind = np.random.choice(np.arange(len(obstacles)))
        if ICE_AREA_REMOVE_MAX and obstacles[ind]['area'] > ICE_AREA_REMOVE_MAX:
            # hack to remove ugly artifacts where the algorithm removes large obstacles
            # leaving a big open-water circle
            continue
        obstacles = np.delete(obstacles, ind, axis=0)
        actual_concentration, _ = compute_poly_ob_concentration(obstacles)
        num_deleted += 1

    print('deleted {} obstacles!\ndesired concentration {}, actual concentration {}'
          .format(num_deleted, desired_concentration, actual_concentration))
    return obstacles


def pack_circles_and_ice_field_plot(circs, polys, pose, concentration):
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
    ax[1].set_title('Desired Concentration {:.2f}\nActual Concentration {:.2f}\nObstacle count {}'
                    .format(concentration, poly_conc, len(polys)))

    ax[2].set_title('Occupancy Grid')
    ax[2].imshow(im, origin='lower', cmap='gray', extent=[0, MAP_SHAPE[1], 0, MAP_SHAPE[0]])

    # show ship footprint
    R = np.asarray([
        [np.cos(pose[2]), -np.sin(pose[2])],
        [np.sin(pose[2]), np.cos(pose[2])]
    ])
    # show pose on plots
    for a in ax:
        a.add_patch(patches.Polygon(SHIP_VERTICES @ R.T + [pose[0], pose[1]], True, fill=True, fc='w', ec='k'))
        a.plot(pose[0], pose[1], 'rx')

    # show goal
    for a in ax:
        a.plot([0, MAP_SHAPE[1]], [GOAL[1], GOAL[1]], 'g-')

    plt.show()

    fig, ax = plt.subplots()
    masses = [p['area'] for p in polys]
    ax.hist(masses, bins=20)
    ax.set_title('Mass Distribution')
    ax.set_xlabel('Mass')
    plt.show()

    floe_areas = [p['area'] for p in polys]
    floe_widths = np.sqrt(floe_areas)
    compute_fractional_area_distribution(floe_widths, floe_areas, MAP_SHAPE[0] * MAP_SHAPE[1])

    print_ob_stats(polys)


def generate_rand_exp():
    # make sure experiment config does not exist yet
    if FILE_NAME is not None:
        assert not os.path.exists(FILE_NAME), 'Experiment configuration file "{}" already exists!'.format(FILE_NAME)

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

    # approach is to first pack the environment with circles then convert circles
    # to polygons and then do rejection sampling to attain the desired concentration
    for conc in tqdm(CONCENTRATIONS):
        for i in tqdm(range(N)):
            # sample random radii from a proper ice mass distribution
            radii = area_to_radii(mass_to_area(sample_ice_mass_from_lognorm(n=ICE_INIT_NUMBER_CIRCLES, plot=False)))
            # other ways to sample radii
            # radii = sample_ice_radii_from_uniform(OBSTACLE['min_r'], OBSTACLE['max_r'], MAP_SHAPE)
            # radii = sample_ice_radii_from_exp(OBSTACLE['min_r'], OBSTACLE['max_r'], MAP_SHAPE)

            # clamp the radii values
            radii = np.clip(radii, OBSTACLE['min_r'], OBSTACLE['max_r'])

            gen = pc.pack(radii)  # this is deterministic! Randomness comes from radii

            circles = np.asarray([(x, y, r) for (x, y, r) in gen])
            circles[:, 1] += MAP_SHAPE[0] / 2
            circles[:, 0] += MAP_SHAPE[1] / 2
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
            im = np.zeros(IM_SHAPE)

            # order the circles from largest to smallest
            # this step ensure we place the large obstacles first
            circles = circles[np.argsort(circles[:, 2])[::-1]]

            for (x, y, radius) in circles:
                vertices = generate_polygon(diameter=radius * 2,
                                            origin=(x, y),
                                            circular=OBSTACLE['circular'],
                                            num_vertices_range=OBSTACLE['num_vertices_range'],
                                            final_vertices_range=OBSTACLE['final_vertices_range'])

                if vertices is not None:
                    # take intersection of vertices and environment boundaries
                    vertices[:, 0][vertices[:, 0] < 0] = 0
                    vertices[:, 0][vertices[:, 0] >= MAP_SHAPE[1]] = MAP_SHAPE[1]

                    min_y = OBSTACLE.get('min_y', False) or 0
                    max_y = OBSTACLE.get('max_y', False) or MAP_SHAPE[0]
                    vertices[:, 1][vertices[:, 1] < min_y] = min_y
                    vertices[:, 1][vertices[:, 1] > max_y] = max_y

                    area = poly_area(vertices)
                    rr, cc = draw.polygon(vertices[:, 1] * SCALE, vertices[:, 0] * SCALE, shape=IM_SHAPE)

                    if len(rr) == 0 or im[rr, cc].sum() > 0 or area < OBSTACLE['min_area']:
                        # skip this polygon if:
                        #  - it doesn't fit in the environment
                        #  - it overlaps with another obstacle
                        #  - it's area is too small
                        continue

                    obstacles.append({
                        'vertices': vertices,
                        'centre': (x, y),
                        'radius': poly_radius(vertices, centre_pos=(x, y)),  # m
                        'pixels': (rr, cc),
                        'area': area,  # area in m^2
                        'mass': area_to_mass(area)  # mass in kg
                    })

                    # compute new poly concentration
                    actual_concentration, im = compute_poly_ob_concentration(obstacles)

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

            # add ship start to dict
            ship_state = (SHIP_STATE['x'], SHIP_STATE['y'], SHIP_STATE['psi'])
            exp_dict['exp'][conc][i]['ship_state'] = ship_state

            if SHOW_PLOT:
                pack_circles_and_ice_field_plot(circles, obstacles, ship_state, conc)

    # save to disk
    if FILE_NAME:
        with open(FILE_NAME, 'wb') as f:
            pickle.dump(exp_dict, f)
            print('Saved experiment configuration file to', os.path.abspath(FILE_NAME))
    print('Done!')


def build_obs_dicts(obstacles: List):
    obs_dicts = []
    for p in obstacles:
        p = np.asarray(p)
        centre = poly_centroid(p)
        area = poly_area(p)
        obs_dicts.append({
            'vertices': p,
            'centre': centre,
            'radius': poly_radius(p, centre),
            'area': area,
            'mass': area_to_mass(area)
        })
    return obs_dicts


def sample_ice_radii_from_uniform(min_radius, max_radius, map_shape):
    avg_r = (min_radius + max_radius) / 2
    num_circ = (np.pi * (((map_shape[0] ** 2 + map_shape[1] ** 2) ** 0.5) / 2) ** 2) / (np.pi * avg_r ** 2)

    # sample random radii
    radii = np.random.uniform(min_radius, max_radius, size=int(num_circ))

    return radii


def sample_ice_radii_from_exp(min_radius, max_radius, map_shape):
    avg_r = min_radius * 1.5
    num_circ = (np.pi * (((map_shape[0] ** 2 + map_shape[1] ** 2) ** 0.5) / 2) ** 2) / (np.pi * avg_r ** 2)

    # sample random radii
    radii = np.maximum(min_radius, np.minimum(
        max_radius, np.random.exponential(scale=avg_r, size=int(num_circ))
    ))

    return radii


def sample_ice_mass_from_lognorm(n, plot=False) -> np.ndarray:
    """
    log-normal parameters 'loc', 'scale' are from paper "GPU-Event-Mechanics Evaluation of Ice Impact Load"
    From paper: " The floe characteristic dimensions (defined as the square root of the area)
                  ranged from 2m to 20 m, with a mean of 6.9m and a standard deviation of 3.9m. "

    Note, log-normal can look similar to powerlaw. Some papers give powerlaw fit to ice floe size distribution such as
    "Observed changes in sea-ice floe size distribution during early summer in the western Weddell Sea"
    """
    loc = 10.21      # note: these don't perfectly reproduce the mean and std given in the paper
    scale = 0.9324
    s = 0.54         # note: shape parameter was not given in paper!! I just picked a value that gave reasonable results

    # sample from lognormal distribution
    r = lognorm.rvs(s, loc=loc, scale=scale, size=n, random_state=RNG)

    if plot:
        count, bins, ignored = plt.hist(r, 100, density=True, align='mid')
        x = np.linspace(min(bins), max(bins), 10000)
        pdf = lognorm.pdf(x, s, loc, scale)
        plt.plot(x, pdf, linewidth=2, color='r')
        plt.show()
        plt.hist(np.exp(r), 100, density=True, align='mid')
        plt.show()

        # print some stats about the distribution
        print('median of the distribution', lognorm.median(s, loc, scale))
        print('mean of the distribution', lognorm.mean(s, loc, scale))     # equal to `np.exp(np.log(scale) + s**2 / 2) + loc`
        print('variance of the distribution', lognorm.var(s, loc, scale))  # equal to `np.exp(2 * np.log(scale) + s**2) * (np.exp(s**2) - 1)`

        # print some stats
        print('mean area', mass_to_area(np.exp(r)).mean(), 'standard deviation', mass_to_area(np.exp(r)).std())
        print('mean size', np.sqrt(mass_to_area(np.exp(r))).mean(), 'standard deviation', np.sqrt(mass_to_area(np.exp(r))).std())
        print('min size', np.sqrt(mass_to_area(np.exp(r))).min(), 'max size', np.sqrt(mass_to_area(np.exp(r))).max())

        radii = area_to_radii(mass_to_area(np.exp(r)))
        radii = np.clip(radii, OBSTACLE['min_r'], OBSTACLE['max_r'])
        gen = pc.pack(radii)
        circles = np.asarray([(x, y, r) for (x, y, r) in gen])

        # plot the circles
        fig, ax = plt.subplots()
        ax.plot(1, 1)
        for (x, y, r) in circles:
            patch = plt.Circle((x, y), r, fc='b', ec='k', alpha=0.65)
            ax.add_patch(patch)
        ax.set_aspect('equal')
        plt.show()

    return np.exp(r)  # return array of mass values in kg


def compute_fractional_area_distribution(widths, areas, total_area, bin_spacing=100):
    widths = np.asarray(widths)
    areas = np.asarray(areas)

    min_width = np.min(widths)
    max_width = np.max(widths)
    bins = np.linspace(min_width, max_width, bin_spacing)

    frac_area = []
    for p in bins:
        frac_area.append(np.sum(areas[widths > p]) / total_area)

    plt.plot(bins, frac_area)
    plt.xlabel('Effective floe width')  # aka characteristic Length
    plt.ylabel('Fractional area')
    # make x axis log scale
    plt.xscale('log')
    plt.show()


def plot_cdf(masses, bin_spacing=100):
    min_mass = np.min(masses)
    max_mass = np.max(masses)
    bins = np.linspace(min_mass, max_mass, bin_spacing)

    cdf = []
    for p in bins:
        cdf.append(sum(masses <= p) / len(masses))

    plt.plot(bins, cdf)
    plt.xlabel('Floe mass (kg)')
    plt.ylabel('Cumulative probability')
    plt.xscale('log')
    plt.show()


def print_ob_stats(obs):
    print('number of obstacles', len(obs))
    print('mean area', np.mean([p['area'] for p in obs]))
    print('std area', np.std([p['area'] for p in obs]))
    print('min area', np.min([p['area'] for p in obs]))
    print('max area', np.max([p['area'] for p in obs]))
    print('mean characteristic length', np.mean([np.sqrt(p['area']) for p in obs]))
    print('std characteristic length', np.std([np.sqrt(p['area']) for p in obs]))
    print('min characteristic length', np.min([np.sqrt(p['area']) for p in obs]))
    print('max characteristic length', np.max([np.sqrt(p['area']) for p in obs]))


def view_experiments(pickle_file, num_trials_per_plot=10, num_trials_skip=10):
    with open(pickle_file, 'rb') as f:
        exp_dict = pickle.load(f)

    curr_plot_count = 0

    for c in exp_dict['exp']:
        print('\nConcentration:', c)
        for i in exp_dict['exp'][c]:
            if num_trials_skip is not None and i % num_trials_skip != 0:
                continue

            obs = exp_dict['exp'][c][i]['obstacles']
            ship_state = exp_dict['exp'][c][i]['ship_state']
            map_shape = exp_dict['meta_data']['map_shape']

            ice_field_plot(c, i, ship_state, obs, map_shape)
            curr_plot_count += 1

            print_ob_stats(obs)

            if curr_plot_count == num_trials_per_plot:
                curr_plot_count = 0
                plt.show()

    # print ob stats for entire dataset
    all_obs = []
    for c in exp_dict['exp']:
        for i in exp_dict['exp'][c]:
            all_obs.extend(exp_dict['exp'][c][i]['obstacles'])
    print('\nall obs')
    print_ob_stats(all_obs)


def ice_field_plot(concentration, ice_field_idx, ship_state, obs, map_shape=None, ship_vertices=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    for p in obs:
        patch = patches.Polygon(p['vertices'],  True, fill=True, fc=ICE_PATCH_COLOR, ec='k', linewidth=0.5)
        ax.add_patch(patch)

    ax.set_aspect('equal')
    ax.plot(ship_state[0], ship_state[1], 'rx')
    ax.set_title('Concentration: {:.2f} Ice Field Index: {}\nObs count: {}'.format(concentration, ice_field_idx, len(obs)))

    if map_shape is not None:
        ax.set_xlim(0, map_shape[1])

    if ship_vertices is not None:
        ax.add_patch(
            patches.Polygon(
                ship_vertices @ Rxy(ship_state[2]).T + [ship_state[0], ship_state[1]],
                True, fill=True, fc=SHIP_PATCH_COLOR, ec='k'
            ))
    ax.set_facecolor(OPEN_WATER_COLOR)


def mass_to_area(ice_mass):
    return ice_mass / (ICE_DENSITY * ICE_THICKNESS)  # m^2


def area_to_mass(ice_area):
    return ice_area * ICE_DENSITY * ICE_THICKNESS  # kg


def area_to_radii(ice_area):
    return np.sqrt(ice_area / np.pi)


if __name__ == '__main__':
    # sample_ice_mass_from_lognorm(ICE_INIT_NUMBER_CIRCLES, plot=True)
    # view_experiments(FILE_NAME)
    generate_rand_exp()
