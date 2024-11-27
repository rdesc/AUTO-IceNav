"""
Ice tracking from the segmentations using OpenCV Kalman filter

https://docs.opencv.org/4.7.0/de/d70/samples_2cpp_2kalman_8cpp-example.html#_a7
https://en.wikipedia.org/wiki/Kalman_filter#Details
"""
import glob
import linecache
import os.path
import pickle
from os.path import join

import cv2
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.ndimage
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from scipy.signal import savgol_filter
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import seaborn as sns
from tqdm import tqdm

from ship_ice_planner.NRC_wrapper.daq_columns import *
from ship_ice_planner.evaluation.evaluate_trial_difficulty import get_trial_difficulty, MAX_DIFFICULTY_LEVEL, \
    MIN_DIFFICULTY_LEVEL, DIFFICULTY_LEVELS
from ship_ice_planner.evaluation.process_daq_data import get_start_end_timesteps, VERTICES
from ship_ice_planner.evaluation.timing_file_parser import get_polygons_from_timing_file, \
    get_timing_step_for_timestamp
from ship_ice_planner.geometry.polygon import separate_polygons
from ship_ice_planner.geometry.utils import Rxy

TRACKING_FIGURE_FILENAME = 'ice_tracking.png'
TRACKING_DATA_FILENAME = 'ice_tracking.pkl'
RESULTS_FILENAME = 'performance_vs_difficulty.pkl'

FRAME_RATE = 29.97
NUM_STEPS_PER_UPDATE = 1
DT = 1 / FRAME_RATE * NUM_STEPS_PER_UPDATE
MIN_OB_AREA = 0.1 ** 2
OB_THICKNESS = 0.012  # m
OB_DENSITY = 917  # kg/m^3 for ice
IOU_MATCH_THRESHOLD = 0.2
NO_DROP_MEASUREMENT_TIME = 0          # time in seconds after which enable drop measurement logic
MAX_NUM_SKIPPED_KF_UPDATES = int(1 / DT * 5)
INITIALIZATION_TIME_LOOK_BACK = 2     # time in seconds before start step to consider for initialization step
SHRINK_SWELL_OBSTACLE_FACTOR = 0.1    # factor to shrink or swell the obstacle by to split polygons that are too close

MIN_OB_SPEED_TRUNCATE_TRIAL = 0.03    # minimum speed to consider when computing track length
MIN_TIME_TRUNCATE_TRIAL = 2           # minimum time in seconds above minimum speed to consider when computing track length
MEDIAN_FILTER_SIZE = int(1 / DT) * 2  # size of median filter to apply to speed

# Kalman filter params
F_MATRIX = np.asarray([  # state-transition model
    # the true state at time k is evolved from the state at (k-1) according to
    # x_k = Fx_[k-1] + w_k where w_k ~ N(0, Q)
    [1, 0, DT, 0, 0],
    [0, 1, 0, DT, 0],
    [0, 0, 1,  0, 0],
    [0, 0, 0,  1, 0],
    [0, 0, 0,  0, 1]
], np.float32)

H_MATRIX = np.asarray([  # observation model
    # at time k an observation z_k of the true state x_k is made according to
    # z_k = Hx_k + v_k where v_k ~ N(0, R)
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1],
], np.float32)

Q_MATRIX = np.asarray([  # covariance of the process noise
    # assume a constant acceleration model that is normally distributed with mean 0 and standard deviation sigma_a
    [1/4*DT**4,         0, 1/2*DT**3,         0, 0],
    [        0, 1/4*DT**4,         0, 1/2*DT**3, 0],
    [1/2*DT**3,         0,     DT**2,         0, 0],
    [        0, 1/2*DT**3,         0,     DT**2, 0],
    [        0,         0,         0,         0, 0.000000001],
], np.float32) * (0.4 ** 2)  # sigma_a^2

R_MATRIX = np.asarray([  # covariance of the observation noise
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], np.float32) ** 2


class TrackedObject:

    def __init__(self, object_id):
        self.id = object_id

        self.estimates = []     # estimated states at each timestep (x, y, vx, vy, area)
                                # (x, y) is the position of the polygon centroid
                                # (vx, vy) is the velocity of the polygon centroid
                                # 'area' is the area of the polygon
                                # shape is (n, 5)
        self.measurements = []  # measurements at each timestep (x, y, area)
                                # shape is (n, 3)
        self.polygon = []       # polygon (i.e. list of vertices) at each timestep
        self.residual = []      # residual at each timestep (both the pre-fit and post-fit residual)

        self.KF = None          # kalman filter
        self.num_skipped_KF_updates = 0  # number of consecutive skips

        self.done_tracking = False

    def log_step(self, measurement, polygon, skip_KF_update=False):
        self.measurements.append(measurement.flatten())
        self.polygon.append(polygon)
        if skip_KF_update:
            self.estimates.append(self.KF.statePre.flatten())   # predicted (a priori) state estimate
            self.num_skipped_KF_updates += 1
        else:
            self.estimates.append(self.KF.statePost.flatten())  # updated (a posteriori) state estimate
            self.num_skipped_KF_updates = 0
        self.residual.append([measurement.flatten() - H_MATRIX @ self.KF.statePre.flatten(),
                              measurement.flatten() - H_MATRIX @ self.KF.statePost.flatten()])

    def init_KF(self, x, y, area, vx=0, vy=0):
        self.KF = cv2.KalmanFilter(5, 3)
        self.KF.measurementMatrix = H_MATRIX
        self.KF.transitionMatrix = F_MATRIX
        self.KF.processNoiseCov = Q_MATRIX
        self.KF.measurementNoiseCov = R_MATRIX

        self.KF.statePost = np.asarray([[x],    # initial positions
                                        [y],
                                        [vx],   # initial velocities
                                        [vy],
                                        [area]], np.float32)
        self.KF.errorCovPost = np.asarray([
            [1,   0,   0,   0,   0],
            [0,   1,   0,   0,   0],
            [0,   0, 0.2,   0,   0],
            [0,   0,   0, 0.2,   0],
            [0,   0,   0,   0,   1],
        ], np.float32) ** 2

        self.estimates = []
        self.measurements = []
        self.polygon = []
        self.residual = []


def get_ship_footprint_at_pose(pose, padding=None):
    if padding:
        vertices = np.asarray(
            [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in VERTICES]
            # add padding to remove polygons picked up on ship
        )
    else:
        vertices = VERTICES
    return vertices @ Rxy(np.deg2rad(pose[2])).T + np.asarray(pose[:2])


def get_radius(circle_area):
    return np.sqrt(circle_area / np.pi)


def compute_iou(centre, radius, poly):
    circle = Point(*centre).buffer(radius)
    poly = Polygon(poly)
    if not poly.is_valid:
        poly = Polygon(poly).convex_hull
    return poly.intersection(circle).area / unary_union([circle, poly]).area


def post_process_polys(polys):
    sep_polys = [separate_polygons(p,
                                   factor=SHRINK_SWELL_OBSTACLE_FACTOR,
                                   smoothen_corners=True,
                                   filter_out_small_polygons=True,
                                   min_ob_area=MIN_OB_AREA)
                 for p in polys]
    return [poly for poly_list in sep_polys for poly in poly_list]


def smooth_path(path):
    smoothed_path = scipy.signal.savgol_filter(path, window_length=int(len(path) / 3), polyorder=3, axis=0, mode='interp')

    # plt.plot(path[:, 0], path[:, 1], '-', label='path')
    # plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], '--', label='smoothed path')
    # plt.legend()
    # plt.gca().set_aspect('equal')
    # plt.show()

    return smoothed_path


def compute_work_metrics(estimates, time, median_size, speed_threshold,
                         time_threshold=None, object_id=-1, plot=False):
    """
    Compute the various different work metrics we consider for the ice tracking data

    We have 3 different ways of approximating work:
    1. Change in kinetic energy of the object to get net work
    2. Integral of the dot product of net force and velocity over time to get net work
    3. Compute the length of the tracked object's path and multiply by the mass
    """
    speed = np.linalg.norm(estimates[:, 2:4], axis=1)
    median_speed = scipy.ndimage.median_filter(speed, size=median_size)
    indices = np.where(median_speed > speed_threshold)[0]

    if len(indices) == 0:
        return None  # tracked ice never exceeded speed threshold

    start_idx = np.min(indices)
    end_idx = np.max(indices)

    if time_threshold:
        # minimum time (seconds) above speed threshold
        # find all the consecutive indices where the speed is above the threshold
        grouped_time = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        time_total = sum([time[group[-1]] - time[group[0]] for group in grouped_time])

        if time_total < time_threshold:
            return None

    # work approx #1
    # compute the net work from the net forces to get object to max speed
    # first find the max speed
    max_speed = np.max(speed)
    max_speed_idx = np.argmax(speed)
    mean_initial_speed = np.mean(speed[:start_idx])
    area = estimates[max_speed_idx, 4]
    mass = area * OB_DENSITY * OB_THICKNESS
    delta_KE_metric = 1 / 2 * mass * (max_speed ** 2 - mean_initial_speed ** 2)

    # work approx #2
    # approximate the work done by the ship on the object by integrating
    # the acceleration dot velocity over time
    v = estimates[:, 2:4]
    a = np.gradient(v, time, axis=0)
    dt = np.gradient(time)
    # compute work as the integral of dot product of acceleration and velocity over time
    # for ice that starts with initial speed ~= 0 and terminates with speed ~= 0
    # the integral should evaluate close to 0
    work_net_force = mass * dt * (a[:, 0] * v[:, 0] + a[:, 1] * v[:, 1])
    indices = np.where(median_speed < mean_initial_speed)[0]
    start2_idx = indices[np.argmin(abs(start_idx - indices))]
    # only consider the work when we think ob is influenced by ship
    work_net_force_metric = work_net_force[start2_idx:end_idx]
    work_net_force_metric = work_net_force_metric[work_net_force_metric > 0].sum()  # only sum up the positive work

    # work approx #3
    # generate a smooth path
    smoothed_path = smooth_path(path=estimates[start_idx:end_idx, :2])
    smoothed_path_length = np.sum(np.linalg.norm(np.diff(smoothed_path, axis=0), axis=1))
    path_length_x_mass_metric = smoothed_path_length * mass

    if plot:
        f, ax = plt.subplots(1, 3)
        ax[0].plot(time[:len(speed)], speed, label='speed, max - min = {:.2f} m/s'.format(max_speed - mean_initial_speed))
        ax[0].plot(time[:len(speed)], median_speed, '--', label='median')
        ax[0].plot([time[0], time[len(speed) - 1]], [speed_threshold, speed_threshold], '-', label='speed threshold')
        ax[0].plot([time[0], time[start_idx]], [mean_initial_speed, mean_initial_speed], '-', label='mean initial speed')
        ax[0].plot(time[start_idx], median_speed[start_idx], 'rx')
        ax[0].plot(time[end_idx], median_speed[end_idx], 'rx')
        ax[0].set_title('speed vs time\n' +
                        r'$W = \Delta KE = $' + '{:.4f} J'
                        .format(delta_KE_metric))
        ax[0].set_xlabel('time (s)')
        ax[0].set_ylabel('speed (m/s)')
        ax[0].legend()
        ax[1].plot(time, work_net_force, label='work')
        ax[1].plot([time[start2_idx], time[end_idx]], [0, 0], 'r-', label='considered for metric')
        ax[1].legend()
        ax[1].set_xlabel('time (s)')
        ax[1].set_ylabel('work (J)')
        ax[1].set_title('work vs time\n' +
                        r'$W = m\int a \cdot v \ dt = $' + '{:.4f} J'.format(work_net_force_metric))
        ax[1].get_shared_x_axes().join(ax[0], ax[1])
        ax[2].plot(estimates[start_idx:end_idx, 0], estimates[start_idx:end_idx, 1], 'b-', label='path')
        ax[2].plot(*smoothed_path.T, 'r--', label='smoothed path')
        ax[2].set_xlabel('x (m)')
        ax[2].set_ylabel('y (m)')
        ax[2].set_aspect('equal')
        ax[2].legend()
        f.suptitle('Object id {}, mass = {:.2f} kg'.format(object_id, mass))
        plt.show()

    return (
        (start_idx, end_idx),
        smoothed_path,
        mass,
        {
            # these first two metrics should be similar (work from net force typically x1 or x2 larger)
            # they are very similar if the object just had 1 big collision with the ship
            'work_net_force': work_net_force_metric,
            'delta_KE': delta_KE_metric,
            'path_length_x_mass': path_length_x_mass_metric,
        }
    )


def compute_poly_centroid(polygons, min_area=0., ship_footprint=None):
    """
    :param polygons: (m, 2)
    :param min_area: minimum area m^2 of polygon to consider
    :param ship_footprint:
    """
    filtered_polygons = []
    centroids = []
    if ship_footprint is not None:
        ship_footprint = Polygon(ship_footprint)
    for poly in polygons:
        if not Polygon(poly).is_valid:
            shapely_poly = Polygon(poly).convex_hull
        else:
            shapely_poly = Polygon(poly)
        if shapely_poly.area < min_area:
            continue
        if ship_footprint is not None and ship_footprint.contains(shapely_poly):
            continue
        if min_area > 0 or ship_footprint is not None:
            filtered_polygons.append(poly)
        centroids.append(shapely_poly.centroid.coords[0])

    if min_area > 0 or ship_footprint is not None:
        return np.asarray(centroids), filtered_polygons

    return np.asarray(centroids)


def track_ice(trial_dir, plot=False):
    # load in the daq file and get the start and end timesteps
    csv_file = glob.glob(join(trial_dir, '*50Hz.csv'))
    if len(csv_file):
        daq_df = load_daq_as_df(csv_file[0])
    else:
        print('skipping trial {}, missing daq file!'.format(trial_dir))
        return

    start, end = get_start_end_timesteps(daq_df, csv_file[0])
    start_timing_step = get_timing_step_for_timestamp(trial_dir, daq_df[TIME_COL].iloc[start])
    end_timing_step = get_timing_step_for_timestamp(trial_dir, daq_df[TIME_COL].iloc[end])

    print('start step {}, end step {}'.format(start_timing_step, end_timing_step))

    # look back the last N seconds and find the frame which has the higher number of obs and use that for
    polys = get_polygons_from_timing_file(trial_dir, timestamp=daq_df[TIME_COL].iloc[start])

    if len(polys) == 0:
        print('skipping trial {}, no polygons!'.format(trial_dir))
        return

    polys = post_process_polys(polys)

    print('len polys', len(polys))

    largest_size_polys = polys
    timing_step = start_timing_step - int(INITIALIZATION_TIME_LOOK_BACK * FRAME_RATE)
    for i in range(0, int(INITIALIZATION_TIME_LOOK_BACK * FRAME_RATE), NUM_STEPS_PER_UPDATE):
        polys = get_polygons_from_timing_file(trial_dir, step=timing_step + i)
        polys = post_process_polys(polys)
        if len(largest_size_polys) < len(polys):
            largest_size_polys = polys
            start_timing_step = timing_step + i

    print('start step {}, len polys {}'.format(start_timing_step, len(largest_size_polys)))

    polys, timestamp, _  = get_polygons_from_timing_file(trial_dir, step=start_timing_step, return_timestamp=True)
    polys = post_process_polys(polys)
    time_data = [timestamp]

    daq_index = np.argmin(np.abs(daq_df[TIME_COL] - timestamp))
    ship_pose = [daq_df[POSE_COLS].iloc[daq_index].to_list()]

    # filter obstacles if necessary
    centroids, filtered_polys = compute_poly_centroid(polys,
                                                      min_area=MIN_OB_AREA,
                                                      ship_footprint=get_ship_footprint_at_pose(ship_pose[0],
                                                                                                padding=0.1))
    # initialize a tracked object for each poly
    tracked_objects = []
    for idx, item in enumerate(zip(centroids, filtered_polys)):
        curr_tracked = TrackedObject(idx)
        curr_centroid, curr_poly = item
        curr_tracked.init_KF(*curr_centroid, Polygon(curr_poly).area)
        curr_tracked.log_step(np.asarray([*curr_centroid, Polygon(curr_poly).area]), curr_poly)
        tracked_objects.append(curr_tracked)

    print('tracking {} objects'.format(len(tracked_objects)))

    for count, step in tqdm(enumerate(range(start_timing_step + 1, end_timing_step, NUM_STEPS_PER_UPDATE))):
        polys, timestamp, _ = get_polygons_from_timing_file(trial_dir, step=step, return_timestamp=True)
        polys = post_process_polys(polys)
        time_data.append(timestamp)
        daq_index = np.argmin(np.abs(daq_df[TIME_COL] - timestamp))
        ship_pose.append(daq_df[POSE_COLS].iloc[daq_index].to_list())
        centroids = compute_poly_centroid(polys)

        for tracked in tracked_objects:
            if tracked.done_tracking:
                # skip since we're done tracking this object
                continue

            centroid_dists = np.linalg.norm(np.asarray(tracked.estimates[-1][:2]).T - centroids, axis=1)
            best_centroid_idx = np.argmin(centroid_dists)
            candidate_matched_poly = polys[best_centroid_idx]
            candidate_matched_centroid = centroids[best_centroid_idx]

            # use a circle to approximate the tracked object
            iou = compute_iou(tracked.estimates[-1][:2],             # estimated centroid
                              get_radius(tracked.estimates[-1][4]),  # estimated area
                              candidate_matched_poly)

            tracked.KF.predict()
            measurement = np.asarray([[candidate_matched_centroid[0]],
                                      [candidate_matched_centroid[1]],
                                      [Polygon(candidate_matched_poly).area]], np.float32)

            if (
                    iou < IOU_MATCH_THRESHOLD and
                    (step - start_timing_step) / FRAME_RATE > NO_DROP_MEASUREMENT_TIME  # skip if we just started tracking
            ):
                tracked.log_step(measurement,
                                 candidate_matched_poly,
                                 skip_KF_update=True)

            else:
                tracked.KF.correct(measurement)  # KF update
                tracked.log_step(measurement,
                                 candidate_matched_poly)

            if tracked.num_skipped_KF_updates > MAX_NUM_SKIPPED_KF_UPDATES:
                # lost the tracked object
                print('lost tracked object with id', tracked.id)
                tracked.done_tracking = True

        if plot and count % 10 == 0:
            plt.figure(1)
            plt.cla()

            for p in polys:
                plt.gca().add_patch(plt.Polygon(p, True, fill=True, color='k', alpha=0.2))
            ship_footprint = get_ship_footprint_at_pose(ship_pose[-1])
            plt.gca().add_patch(plt.Polygon(ship_footprint, True, fill=True, color='b', alpha=0.5))

            for tracked in tracked_objects:
                plt.gca().add_patch(plt.Polygon(tracked.polygon[-1], True, fill=True, color='k', alpha=0.5))
                plt.plot(tracked.measurements[-1][0], tracked.measurements[-1][1], 'cx', label='measurement')
                plt.plot(tracked.estimates[-1][0], tracked.estimates[-1][1], 'mx', label='prediction')
                plt.plot(*np.asarray(tracked.measurements).T[:2], 'b--', label='measurements')
                plt.plot(*np.asarray(tracked.estimates).T[:2], 'r-.', label='estimates')
                plt.plot([tracked.estimates[-1][0], tracked.estimates[-1][0] + tracked.estimates[-1][2] * 10],  # velocity vector 1 second
                         [tracked.estimates[-1][1], tracked.estimates[-1][1] + tracked.estimates[-1][3] * 10], 'g-', label='velocity')
                # add a circle representing the estimated area
                plt.gca().add_patch(plt.Circle((tracked.estimates[-1][0], tracked.estimates[-1][1]),
                                               get_radius(tracked.estimates[-1][4]), color='m', fill=False))

            plt.gca().set_aspect('equal', adjustable='box')
            plt.pause(0.1)

    # store to disk
    pickle.dump({
        'time': {
            'steps_range': (start_timing_step, end_timing_step, NUM_STEPS_PER_UPDATE),
            'timestamps': np.asarray(time_data)
        },
        'tracked': {item.id: {'estimates': np.asarray(item.estimates),
                              'measurements': np.asarray(item.measurements),
                              'residual': np.asarray(item.residual),
                              'polygons': item.polygon,  # this makes the file considerably larger
                              'done_tracking': item.done_tracking}
                    for item in tracked_objects},
        'ship_pose': np.asarray(ship_pose),
    }, open(join(trial_dir, TRACKING_DATA_FILENAME), 'wb'))

    print('Done')


def evaluate_trial_from_tracking_data(trial_dir,
                                      plot=True,
                                      anim=False):
    # get ice tracking data
    if not os.path.isfile(join(trial_dir, TRACKING_DATA_FILENAME)):
        track_ice(trial_dir, plot=False)

    if not os.path.isfile(join(trial_dir, TRACKING_DATA_FILENAME)):
        # above might have failed
        print('Error!')
        return

    csv_file = glob.glob(join(trial_dir, '*50Hz.csv'))
    if len(csv_file):
        daq_df = load_daq_as_df(csv_file[0])
    else:
        print('skipping trial {}, missing daq file!'.format(trial_dir))
        return
    data = pickle.load(open(join(trial_dir, TRACKING_DATA_FILENAME), 'rb'))
    start, end = get_start_end_timesteps(daq_df, csv_file[0])
    start_timing_step = get_timing_step_for_timestamp(trial_dir, daq_df[TIME_COL].iloc[start])
    end_timing_step = get_timing_step_for_timestamp(trial_dir, daq_df[TIME_COL].iloc[end])

    metrics = []
    for tracked_id in range(len(data['tracked'])):
        tracked = data['tracked'][tracked_id]
        estimates = tracked['estimates']

        curr_work_metric = compute_work_metrics(estimates, data['time']['timestamps'][:len(estimates)],
                                                median_size=MEDIAN_FILTER_SIZE,
                                                speed_threshold=MIN_OB_SPEED_TRUNCATE_TRIAL,
                                                time_threshold=MIN_TIME_TRUNCATE_TRIAL,
                                                object_id=tracked_id,
                                                plot=False  # for debugging and visualization
                                                )

        if curr_work_metric is not None:
            indices, smoothed_path, mass, _ = curr_work_metric
            curr_work_metric = curr_work_metric[-1]  # work metrics are a dict
            data['tracked'][tracked_id]['smoothed_path'] = smoothed_path
            data['tracked'][tracked_id]['start_end_indices'] = indices

            # check if the object was ever in contact with the ship
            collided_with_ship = False
            for idx, pose in enumerate(data['ship_pose']):
                if idx > len(estimates) - 1:
                    break
                ship_polygon = get_ship_footprint_at_pose(pose, padding=0.1)
                circle = Point(estimates[idx][:2]).buffer(get_radius(estimates[-1][4]))
                if Polygon(ship_polygon).intersects(circle):
                    collided_with_ship = True
                    break

            metrics.append({
                'tracked_id': tracked_id,
                'estimated_mass': mass,
                'collided_with_ship': collided_with_ship,
                **curr_work_metric
            })
    # make dataframe
    metrics_df = pd.DataFrame(metrics)

    if anim:
        fig, ax = plt.subplots()
        x_lim, y_lim = None, None  # for allowing interactive zoom in the figure

        for idx, step in enumerate(tqdm(range(*data['time']['steps_range']))):
            if step < start_timing_step or step > end_timing_step:
                continue

            if idx % 10 != 0:
                continue

            ax.cla()
            if x_lim is not None:
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)

            polys = get_polygons_from_timing_file(trial_dir, step=step)
            for p in polys:
                ax.add_patch(plt.Polygon(p, True, fill=True, color='k', alpha=0.2))

            ship_footprint = get_ship_footprint_at_pose(data['ship_pose'][idx])
            ax.add_patch(plt.Polygon(ship_footprint, True, fill=True, color='b', alpha=0.5))
            ax.plot(*data['ship_pose'][idx][:2], 'bx', label='ship pose')

            num_tracked = 0
            for tracked_id in range(len(data['tracked'])):
                tracked = data['tracked'][tracked_id]

                if (
                        'smoothed_path' not in tracked or
                        idx < tracked['start_end_indices'][0] or
                        idx > tracked['start_end_indices'][1]
                ):
                    continue

                estimates = tracked['estimates']
                measurements = tracked['measurements']

                if idx >= len(tracked['estimates']):
                    continue

                num_tracked += 1
                ax.plot(measurements[idx][0], measurements[idx][1], 'cx', label='measurement')
                ax.plot(estimates[idx][0], estimates[idx][1], 'mx', label='prediction')
                ax.text(estimates[idx][0], estimates[idx][1], str(tracked_id), fontsize=6)
                ax.plot([estimates[idx][0], estimates[idx][0] + estimates[idx][2] * 10],
                        # velocity vector 10 seconds
                        [estimates[idx][1], estimates[idx][1] + estimates[idx][3] * 10], 'g-', label='velocity')
                # add a circle representing the estimated area
                ax.add_patch(plt.Circle((estimates[idx][0], estimates[idx][1]),
                                        get_radius(estimates[idx][4]), color='m', fill=False))

                ax.plot(*estimates.T[:2, :idx + 1], 'k-')
                ax.plot(*tracked['smoothed_path'][:idx - tracked['start_end_indices'][0] + 1].T, 'r--', label='smoothed path')

            ax.set_title('step {}, tracked object count {}, trial {}'.format(step, num_tracked, os.path.basename(trial_dir)))
            ax.set_aspect('equal', adjustable='box')

            plt.pause(0.1)
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()

    if plot:
        fig1, ax1 = plt.subplots(figsize=(15, 15))
        step = end_timing_step
        idx = -1
        polys = get_polygons_from_timing_file(trial_dir, step=step)
        for p in polys:
            ax1.add_patch(plt.Polygon(p, True, fill=True, color='k', alpha=0.2))

        ship_footprint = get_ship_footprint_at_pose(data['ship_pose'][idx])
        ax1.add_patch(plt.Polygon(ship_footprint, True, fill=True, color='b', alpha=0.5))
        ax1.plot(*np.asarray(data['ship_pose']).T[:2], 'b-', label='ship pose')

        num_tracked = 0
        num_smoothed = 0
        for tracked_id in range(len(data['tracked'])):
            tracked = data['tracked'][tracked_id]
            estimates = tracked['estimates']

            num_tracked += 1
            ax1.plot(estimates[idx][0], estimates[idx][1], 'mx', label='prediction')
            # add a circle representing the estimated area
            ax1.add_patch(plt.Circle((estimates[idx][0], estimates[idx][1]),
                                     get_radius(estimates[idx][4]), color='m', fill=False))

            ax1.plot(*estimates.T[:2], 'k-')
            if 'smoothed_path' in tracked:
                num_smoothed += 1
                ax1.plot(*tracked['smoothed_path'].T, 'r--', label='smoothed path')

        ax1.set_title('step {}, tracked object count {}, smoothed count {}\n'
                      'params: min speed threshold={} m/s, min time threshold={} s'
                      .format(step, num_tracked, num_smoothed, MIN_OB_SPEED_TRUNCATE_TRIAL, MIN_TIME_TRUNCATE_TRIAL))
        ax1.set_aspect('equal', adjustable='box')
        fig1.tight_layout()
        fig1.savefig(join(trial_dir, TRACKING_FIGURE_FILENAME), dpi=300)

    return {
        # get the mean mass of the tracked ice objects that collided with the ship
        'mean_mass': metrics_df[metrics_df['collided_with_ship']]['estimated_mass'].mean(),
        'num_collided_ice_floes': sum(metrics_df['collided_with_ship']),
        'num_tracked': len(metrics_df),
        **metrics_df[['work_net_force', 'delta_KE', 'path_length_x_mass']].sum().to_dict(),
    }


def performance_vs_difficulty_plot(root_dir, metric, reverse_difficulty=True, data=None, put_in_bins=False):
    """
    Plot the performance metrics vs difficulty for all the trials
    """
    if data is None:
        trials_difficulty = get_trial_difficulty(root_dir, put_in_bins=put_in_bins)

        # collect the ice tracking metrics for each trial
        data = {}

        for curr_trial in tqdm(trials_difficulty):
            planner = curr_trial.split('_')[0]

            root_planner_dir = join(root_dir, planner)
            if 'lattice' in planner:
                root_planner_dir = root_planner_dir[:-1]

            if not os.path.isfile(join(root_planner_dir, curr_trial, TRACKING_DATA_FILENAME)):
                continue

            metrics = evaluate_trial_from_tracking_data(join(root_planner_dir, curr_trial), plot=False, anim=False)
            difficulty = (MAX_DIFFICULTY_LEVEL + MIN_DIFFICULTY_LEVEL) - trials_difficulty[curr_trial] \
                if (reverse_difficulty and not put_in_bins) else trials_difficulty[curr_trial]

            if planner not in data:
                data[planner] = [
                    {'trial': curr_trial,
                     'difficulty': difficulty,
                     **metrics},
                ]

            else:
                data[planner].append(
                    {'trial': curr_trial,
                     'difficulty': difficulty,
                     **metrics},
                )

        # save to disk
        pickle.dump(data, open(join(root_dir, RESULTS_FILENAME), 'wb'))

    with plt.style.context('science'):

        if put_in_bins:
            # make a grouped bar chart
            plot_data = []
            for planner in data:
                df = pd.DataFrame(data[planner])
                df = df[~df['trial'].str.contains('bonus')]  # ignore bonus trials

                if planner == 'lattice2':
                    df = df[df['trial'].str.contains('lattice2_adv')]

                if planner == 'lattice1':
                    df = df[~df['trial'].str.contains('lattice1_adv')]

                print('\nplanner', planner, 'num trials', len(df))
                for difficulty in DIFFICULTY_LEVELS:
                    print('\t',
                          'difficulty', difficulty,
                          'num trials', len(df[df['difficulty'] == difficulty]),
                          'mean metric', metric, df[df['difficulty'] == difficulty][metric].mean(),
                          'mean num ice pushed', df[df['difficulty'] == difficulty]['num_tracked'].mean(),
                          'mean collided ice mass', df[df['difficulty'] == difficulty]['mean_mass'].mean(),
                          )

                for index, row in df.iterrows():
                    plot_data.append({
                        'planner': planner,
                        'difficulty': row['difficulty'],
                        metric: row[metric]
                    })

            plot_df = pd.DataFrame(plot_data)
            f, ax = plt.subplots(figsize=(5, 4), dpi=300)
            ax = sns.boxplot(data=plot_df,
                             x='difficulty',
                             y=metric,
                             hue='planner',
                             hue_order=['straight', 'skeleton', 'lattice1'],
                             order=DIFFICULTY_LEVELS,
                             palette='Set2')
            sns.stripplot(data=plot_df,
                          x='difficulty',
                          y=metric,
                          hue='planner',
                          hue_order=['straight', 'skeleton', 'lattice1'],
                          order=DIFFICULTY_LEVELS,
                          palette='Set2',
                          dodge=True,
                          jitter=False,
                          alpha=0.6,
                          zorder=2,
                          linewidth=1,
                          ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=[(handles[0], handles[3]), (handles[1], handles[4]), (handles[2], handles[5])],
                      labels=['Straight', 'Skeleton', 'AUTO-IceNav'],
                      loc='upper left', handlelength=6,
                      handler_map={tuple: HandlerTuple(ndivide=None)})
            ax.set_ylabel('Work (J)')
            ax.set_xlabel('Difficulty')
            f.tight_layout()
            plt.show()

        else:
            f, ax = plt.subplots(figsize=(5, 5))
            markers = ['^', 'o', 's', 'X']
            for planner in data:
                df = pd.DataFrame(data[planner])
                ax.scatter(df['difficulty'], df[metric], label=planner, marker=markers.pop(0), s=100, alpha=0.7)
            ax.set_xlabel('difficulty (very easy → very difficult)')
            ax.set_ylabel('work (J)')
            ax.legend()
            f.tight_layout()
            plt.show()


if __name__ == '__main__':
    from ship_ice_planner.evaluation import NRC_OEB_2023_EXP_ROOT
    import scienceplots
    root_dir = NRC_OEB_2023_EXP_ROOT

    for trial in tqdm(os.listdir(root_dir)):
        if not os.path.isdir(os.path.join(root_dir, trial)):
            continue
        if 'bad_trials' in trial:
            continue

        print('trial: {}'.format(trial))
        curr_metrics = evaluate_trial_from_tracking_data(trial_dir=os.path.join(root_dir, trial),
                                                         plot=False,
                                                         anim=True)
        print(curr_metrics)
        linecache.clearcache()

    performance_vs_difficulty_plot(root_dir,
                                   metric='work_net_force',  # options are: path_length_x_mass, delta_KE, work_net_force
                                   put_in_bins=True,
                                   # data=pickle.load(open(os.path.join(root_dir, RESULTS_FILENAME), 'rb'))
    )
