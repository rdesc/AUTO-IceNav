"""
Processes data recorded from the GDAC Data Acquisition System
https://ieeexplore.ieee.org/document/9389165
"""
import json
import os
import pickle
from datetime import datetime
from os.path import *

import numpy as np
import pandas as pd
import scipy.ndimage
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm

from ship_ice_planner import PATH_DIR, METRICS_FILE
from ship_ice_planner.NRC_wrapper.daq_columns import *
from ship_ice_planner.NRC_wrapper.nrc_config import TOWER
from ship_ice_planner.evaluation import PLANNER_DATA_TRANSFORMED
from ship_ice_planner.evaluation.evaluate_trial_difficulty import get_trial_difficulty
from ship_ice_planner.evaluation.timing_file_parser import get_polygons_from_timing_file
from ship_ice_planner.ship import OEB_PSV_VERTICES
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import compute_path_length

GDAC_PLOTS_DIR = 'gdac_plots'                    # directory to store the gdac plots
GDAC_SUMMARY = 'gdac_summary.txt'                # stores the metrics as a JSON
GDAC_SUMMARY_PLOT = 'gdac_summary.png'           # overview of the daq data showing the final step of the trial
START_END_TIMESTEPS = 'start_end_timesteps.txt'  # stores the start and end time steps and time stamps for the trial

DT = 1 / 50  # time in seconds between each sample logged by GDAC

# for computing force = alpha * (rps)^2
FStbd    = 0.00284371401352545    # tunnel thruster 1 (m3) starboard
FPort    = -0.00284371401352545   # tunnel thruster 1 (m3) port
AStbd    = 0.000996364588886264   # tunnel thruster 2 (m4) starboard
APort    = -0.000996364588886264  # tunnel thruster 2 (m4) port
PAhead   = 0.0868329285660451     # main propeller 1 (m1) forward
PAstern  = -0.0583454222495970    # main propeller 1 (m1) backward
SAhead   = 0.0860615521092497     # main propeller 2 (m2) forward
SAstern  = -0.0583454222495970    # main propeller 2 (m2) backward

# ship vertices for visualization purposes
VERTICES = OEB_PSV_VERTICES


def compute_force(dfi: pd.DataFrame):
    """
    Computes the force for the main prop and tunnel
    """
    f1, f2 = np.zeros(len(dfi)), np.zeros(len(dfi))

    # main prop
    # make binary mask for when ship is going forward/backward
    c1_idx = dfi[BODY_VEL_COLS[0]] > 0

    # forward
    f1[c1_idx] = PAhead * dfi[c1_idx][RPS_COLS[0]] ** 2 + SAhead * dfi[c1_idx][RPS_COLS[1]] ** 2
    # backward
    f1[~c1_idx] = PAstern * dfi[~c1_idx][RPS_COLS[0]] ** 2 + SAstern * dfi[~c1_idx][RPS_COLS[1]] ** 2

    # make binary mask for when ship is going starboard/port
    c2_idx = dfi[BODY_VEL_COLS[1]] > 0

    # tunnel
    # starboard
    f2[c2_idx] = FStbd * dfi[c2_idx][RPS_COLS[2]] ** 2 + AStbd * dfi[c2_idx][RPS_COLS[3]] ** 2
    # port
    f2[~c2_idx] = FPort * dfi[~c2_idx][RPS_COLS[2]] ** 2 + APort * dfi[~c2_idx][RPS_COLS[3]] ** 2

    return f1, f2


def compute_energy_use(force, velocity, dt):
    """
    Computes the energy use of the ship
    """
    return abs(force) * abs(velocity) * dt


def update_daq_with_planner_iteration(file_path, time_delta=60 * 60 * 1.5):
    """Adds column to daq for planning iteration, this should have been logged with gdac during experiments!"""
    df = load_daq_as_df(file_path)
    # get all timestamps from planner data
    planner_timestamps = {}
    files_sorted = sorted(os.listdir(join(dirname(file_path), PATH_DIR)), key=lambda x: int(x.split('.')[0].split('_')[0]))
    for fp in files_sorted:
        if 'failed' in fp:
            continue
        if fp.endswith('.pkl'):
            key = pickle.load(
                open(join(dirname(file_path), PATH_DIR, fp), 'rb'))['processed_message']['metadata']['timestamp']
            key = key[0] + eval('0.' + str(key[1]))  # convert to float
            planner_timestamps[key] = int(fp.split('.')[0])

    daq_timestamps = [
        datetime.strptime(date[1:-1], '%Y-%m-%d %H:%M:%S.%f').timestamp() for date in df[DATE_COL].to_list()
    ]

    # add new column to df
    new_col = np.zeros(len(df), dtype=int) - 1

    for k, v in planner_timestamps.items():
        # find closest daq timestamp
        daq_index = np.argmin(np.abs(np.asarray(daq_timestamps) - k - time_delta))
        new_col[daq_index:] = v

    # update the daq file
    df[PLANNER_COL] = new_col
    df.to_csv(file_path, index=False)

    return df


def plot_gdac(file_path, step=int(1 / DT), start_idx=None):
    df = load_daq_as_df(file_path)
    plot_dir = join(dirname(file_path), GDAC_PLOTS_DIR)

    if not exists(plot_dir):
        os.makedirs(plot_dir)

    if start_idx is None and PF_MODE_COL in df.columns:
        start_idx, end_idx = get_start_end_timesteps(df, file_path)

        if PLANNER_COL not in df.columns:
            df = update_daq_with_planner_iteration(file_path)

        for idx in tqdm(range(1, len(df[start_idx:end_idx][::step]))):
            dfc = df.iloc[start_idx:idx * step + start_idx]
            planner_idx = dfc[PLANNER_COL].iloc[-1]
            if planner_idx == -1:
                planner_data = None
            else:
                planner_data = pickle.load(open(join(dirname(file_path), PLANNER_DATA_TRANSFORMED), 'rb'))[planner_idx]

            make_plot(join(plot_dir, str(idx) + '.png'), df, dfc, idx * step + start_idx,
                      planner_data=planner_data,    # plot planner data
                      timing_data=True,             # plot obstacles from timing file
                      root_dir=dirname(file_path)
                      )

    else:
        for idx in tqdm(range(1, len(df[start_idx:][::step]))):
            dfc = df.iloc[start_idx:idx * step + start_idx]
            make_plot(join(plot_dir, str(idx) + '.png'), df, dfc, idx * step + start_idx, root_dir=dirname(file_path))


def make_plot(plot_fp, df, dfc, idx, planner_data=None, timing_data=True, root_dir=None):
    f, ax = plt.subplot_mosaic([['A', 'B'],
                                ['A', 'C'],
                                ['A', 'D'],
                                ['A', 'E'],
                                ['A', 'F'],
                                ['A', 'G'],
                                ['A', 'H'],
                                ['A', 'I']],
                               figsize=(20, 15))
    x = dfc[POSITION_COLS[0]].to_numpy()
    y = dfc[POSITION_COLS[1]].to_numpy()
    yaw = dfc[ROTATION_COLS[2]].to_numpy()
    # rotating tank by 90 degrees then followed by a reflection so its nicer to visualize (and matches footage)
    Plot.add_ship_patch(ax['A'], VERTICES, y[-1], x[-1], np.pi - (yaw[-1] * np.pi / 180 + np.pi / 2))
    ax['A'].plot(y, x, 'b--', label='path')
    for tower in TOWER:
        ax['A'].plot(tower[1], tower[0], 'rx', label='tower')
    sx, sy, syaw = dfc.iloc[-1][SETPOINT_COLS]
    syaw = np.pi - (syaw * np.pi / 180 + np.pi / 2)
    ax['A'].plot([sy, sy + np.cos(syaw)], [sx, sx + np.sin(syaw)], 'm', label='setpoint')
    ax['A'].set_aspect('equal')
    ax['A'].set_title('Step %i, time %.2f (s)\nship pose %.2f, %.2f, %.2f'
                      % (idx, df.iloc[idx][TIME_COL], x[-1], y[-1], yaw[-1] * np.pi / 180))
    ax['A'].set_xlabel('y (m)')
    ax['A'].set_ylabel('x (m)')

    vel_x, vel_y = dfc[BODY_VEL_COLS[0]], dfc[BODY_VEL_COLS[1]]
    ax['B'].plot(dfc[TIME_COL], vel_x, label='vel x')
    ax['B'].set_xlim(dfc[TIME_COL].iloc[0], dfc[TIME_COL].iloc[-1])
    ax['B'].set_title('Velocity body u (m/s)')

    ax['C'].plot(dfc[TIME_COL], dfc[VEL_COLS[1]], label='vel y')
    ax['C'].set_xlim(dfc[TIME_COL].iloc[0], dfc[TIME_COL].iloc[-1])
    ax['C'].set_title('Velocity body v (m/s)')

    f1, f2 = compute_force(dfc)

    ax['D'].plot(dfc[TIME_COL], f1)
    ax['D'].set_xlim(dfc[TIME_COL].iloc[0], dfc[TIME_COL].iloc[-1])
    ax['D'].set_title('thrust main prop (N)')

    ax['E'].plot(dfc[TIME_COL], f2)
    ax['E'].set_xlim(dfc[TIME_COL].iloc[0], dfc[TIME_COL].iloc[-1])
    ax['E'].set_title('thrust tunnel (N)')

    ax['F'].plot(dfc[TIME_COL], dfc[ACCELER_COLS[0]])
    ax['F'].set_xlim(dfc[TIME_COL].iloc[0], dfc[TIME_COL].iloc[-1])
    ax['F'].set_title('acceleration x (m/s^2)')
    _, impact_idx = metrics_from_accelerometer_data(dfc)
    for im in impact_idx:
        ax['F'].plot(dfc[TIME_COL].iloc[im], dfc[ACCELER_COLS[0]].iloc[im], 'rx')

    ax['G'].plot(dfc[TIME_COL], dfc[ACCELER_COLS[1]])
    ax['G'].set_xlim(dfc[TIME_COL].iloc[0], dfc[TIME_COL].iloc[-1])
    ax['G'].set_title('acceleration y (m/s^2)')
    for im in impact_idx:
        ax['G'].plot(dfc[TIME_COL].iloc[im], dfc[ACCELER_COLS[1]].iloc[im], 'rx')

    ax['H'].plot(dfc[TIME_COL], dfc[ROTATION_COLS[1]])
    ax['H'].set_xlim(dfc[TIME_COL].iloc[0], dfc[TIME_COL].iloc[-1])
    ax['H'].set_title('pitch (deg)')

    ax['I'].plot(dfc[TIME_COL], dfc[ROTATION_COLS[0]])
    ax['I'].set_xlim(dfc[TIME_COL].iloc[0], dfc[TIME_COL].iloc[-1])
    ax['I'].set_title('roll (deg)')
    ax['I'].set_xlabel('time (s)')

    if planner_data is not None:
        # plot the planner path and obstacles
        path = planner_data['path']  # n x 3
        ax['A'].plot(path[:, 1], path[:, 0], 'r--', linewidth=2)

    if timing_data:
        if root_dir is None:
            root_dir = os.path.dirname(plot_fp)
        polys = get_polygons_from_timing_file(root_dir, df.iloc[idx][TIME_COL])
        for ob in polys:
            ax['A'].add_patch(Polygon(ob[:, ::-1], color='b', alpha=0.3))

    plt.tight_layout()
    plt.savefig(plot_fp, dpi=200)
    plt.close(f)


def get_start_end_timesteps(df, file_path):
    if isfile(join(dirname(file_path), START_END_TIMESTEPS)):
        start, end, *_ = open(join(dirname(file_path), START_END_TIMESTEPS), 'r').readlines()

    else:
        start = df[df[PF_MODE_COL] == 1].index[0]
        end = df[df[PF_MODE_COL] == 1].index[-1]
        start_time = df[DATE_COL].iloc[start]
        end_time = df[DATE_COL].iloc[end]

        # write to file
        with open(join(dirname(file_path), START_END_TIMESTEPS), 'w') as f:
            f.write('%s\n%s\n%s\n%s' % (start, end, start_time, end_time))

    return int(start), int(end)


def get_planner_metrics(root_dir):
    metrics_file = join(root_dir, METRICS_FILE)
    metrics = []

    if not isfile(metrics_file):
        return None

    with open(metrics_file, 'r') as f:
        for line in f.readlines():
            metrics.append(json.loads(line))

    metrics_df = pd.DataFrame(metrics)

    # hardcoded
    metric_cols = {
        'compute_time': 'Mean Planner Compute Time (s)',
        'expanded_cnt': 'Mean Planner Expanded Nodes',
    }

    metrics_to_keep = []
    for col in metric_cols:
        if col in metrics_df.columns:
            metrics_to_keep.append(col)

    metrics_df = metrics_df[metrics_to_keep]

    # rename columns
    metrics_df.rename(columns=metric_cols, inplace=True)

    return metrics_df


def metrics_from_accelerometer_data(dfc,
                                    threshold_acc=-0.005,
                                    median_filter_size=1,
                                    average_filter_size=25,
                                    ship_mass=70,  # kg
                                    impact_max_step_length=10
                                    ):
    """
    Detects collision impacts from accelerometer data by comparing the signal
    to the averaged signal and setting a threshold for the variance.

    The resultant force angle can also be found... see equation 3.1 here
    https://www.dynamicpublishers.com/Neural/NPSC2007/20-NPSC-2007-319-334.pdf

    Other good reference
    https://stubber.math-inf.uni-greifswald.de/~ebner/resources/uniG/collisionDetectionIRC.pdf
    """
    # if average_filter_size > len(dfc):
    #     print('average filter size too large, setting to half the length of the data')
    #     average_filter_size = len(dfc) // 2
    assert average_filter_size <= len(dfc)
    idx_xy = []
    impact_acc_xy = []

    for acc_col in (ACCELER_COLS[0], ACCELER_COLS[1]):
        acc = dfc[acc_col].to_numpy()

        # median filter
        if median_filter_size > 1:
            smoothed_acc = scipy.ndimage.median_filter(acc, size=median_filter_size)
        else:
            smoothed_acc = acc

        average_acc = np.convolve(smoothed_acc, np.ones(average_filter_size) / average_filter_size, mode='valid')
        # remove the front and end of averaged signal
        smoothed_acc = smoothed_acc[average_filter_size // 2:-average_filter_size // 2 + 1]

        impact_acc = smoothed_acc - average_acc
        variance_acc = (smoothed_acc - average_acc) ** 2

        # indices where the acceleration is past the threshold
        idx = np.where(variance_acc > threshold_acc ** 2)[0]

        idx_xy.append(idx)
        impact_acc_xy.append(impact_acc)

    # merge indices from acc x and acc y
    idx = np.unique([*idx_xy[0], *idx_xy[1]])
    forces = []
    for i in idx:
        forces.append(
            ship_mass * np.sqrt(impact_acc_xy[0][i] ** 2 + impact_acc_xy[1][i] ** 2)
        )
    idx += average_filter_size // 2

    # combine if necessary
    merged_idx = []
    merged_forces = []
    current_group_idx = []
    current_group_forces = []

    for num, f in zip(idx, forces):
        if not current_group_idx or num - current_group_idx[-1] <= impact_max_step_length:
            current_group_idx.append(num)
            current_group_forces.append(f)
        else:
            merged_idx.append(current_group_idx)
            merged_forces.append(current_group_forces)
            current_group_idx = [num]
            current_group_forces = [f]

    if current_group_idx:
        merged_idx.append(current_group_idx)
        merged_forces.append(current_group_forces)

    idx = [g1[np.argmax(g2)] for g1, g2 in zip(merged_idx, merged_forces)]
    forces = [np.mean(group) for group in merged_forces]

    return forces, idx


def evaluate_trial(exp_root, trial_name, daq_file_path):
    trial_dir = dirname(daq_file_path)
    df = load_daq_as_df(daq_file_path)

    if PLANNER_COL not in df.columns:
        df = update_daq_with_planner_iteration(daq_file_path)

    # get start and end timesteps
    start, end = get_start_end_timesteps(df, daq_file_path)
    dfc = df.iloc[start:end]

    metrics = {}

    metrics['Difficulty'] = get_trial_difficulty(exp_root, trial_name, put_in_bins=False)
    metrics['Difficulty Bin'] = get_trial_difficulty(exp_root, trial_name, put_in_bins=True)
    metrics['Transit Time (s)'] = dfc[TIME_COL].iloc[-1] - dfc[TIME_COL].iloc[0]
    metrics['Start Time (s)'] = dfc[TIME_COL].iloc[0]
    metrics['End Time (s)'] = dfc[TIME_COL].iloc[-1]
    metrics['Number of Polygons'] = len(get_polygons_from_timing_file(trial_dir, step=0))
    metrics['Number of Planning Iterations'] = float(df[PLANNER_COL].max() + 1)
    metrics['Actual Path Length (m)'] = compute_path_length(np.asarray([dfc[POSITION_COLS[0]], dfc[POSITION_COLS[1]]]).T)
    metrics['Mean Speed (m/s)'] = metrics['Actual Path Length (m)'] / metrics['Transit Time (s)']
    metrics['Mean Main Prop RPS (rps)'] = dfc[RPS_COLS[:2]].abs().mean().mean()
    metrics['Mean Tunnel RPS (rps)'] = dfc[RPS_COLS[2:]].abs().mean().mean()

    f1, f2 = compute_force(dfc)
    e1 = compute_energy_use(f1, dfc[BODY_VEL_COLS[0]].to_numpy(), DT)
    e2 = compute_energy_use(f2, dfc[BODY_VEL_COLS[1]].to_numpy(), DT)
    metrics['Mean Thrust Main Prop (N)'] = f1.mean()
    metrics['Mean Thrust Tunnel (N)'] = f2.mean()
    metrics['Mean Thrust (N)'] = (np.sqrt(f1 ** 2 + f2 ** 2)).mean()
    metrics['Total Energy Main Prop (J)'] = e1.sum()
    metrics['Total Energy Tunnel (J)'] = e2.sum()
    metrics['Total Energy Controls (J)'] = e1.sum() + e2.sum()

    impact_acc, impact_idx = metrics_from_accelerometer_data(dfc)
    metrics['Number of Collisions (IMU)'] = len(impact_idx)
    metrics['Mean Impact Force (N) (IMU)'] = np.mean(impact_acc)
    metrics['Max Impact Force (N) (IMU)'] = np.max(impact_acc)

    metrics['Start x (m)'] = dfc[POSITION_COLS[0]].iloc[0]
    metrics['Start y (m)'] = dfc[POSITION_COLS[1]].iloc[0]
    metrics['End x (m)'] = dfc[POSITION_COLS[0]].iloc[-1]
    metrics['End y (m)'] = dfc[POSITION_COLS[1]].iloc[-1]
    metrics['Displacement x (m)'] = abs(dfc[POSITION_COLS[0]].iloc[-1] - dfc[POSITION_COLS[0]].iloc[0])

    planner_metrics = get_planner_metrics(trial_dir)
    if planner_metrics is not None:
        for col in planner_metrics.columns:
            metrics[col] = float(planner_metrics[col].mean())

    with open(join(trial_dir, GDAC_SUMMARY), 'w') as f:
        json.dump(metrics, f, indent=4)

    # plot generated such that the orientation matches the overhead footage
    make_plot(join(trial_dir, GDAC_SUMMARY_PLOT), df, dfc, end)

    return metrics
