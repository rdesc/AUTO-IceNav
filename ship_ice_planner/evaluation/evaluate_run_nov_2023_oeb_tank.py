import glob
import linecache
import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

from ship_ice_planner.evaluation import NRC_OEB_2023_EXP_ROOT, PLANNER_DATA_TRANSFORMED
from ship_ice_planner.evaluation.process_daq_data import plot_gdac, evaluate_trial
from ship_ice_planner.utils.utils import resample_path
from ship_ice_planner.NRC_wrapper.socket_communication import PlannerNRCSocketServer
from ship_ice_planner.NRC_wrapper.nrc_config import transform_fn
from ship_ice_planner.NRC_wrapper.daq_columns import generate_csv_from_daq
from ship_ice_planner.image_process.process_overhead_video import process_overhead_video

PLANNER_RESULTS_CSV_FILE_NAME = 'results.csv'
ALL_TRIAL_RESULTS_RAW_CSV_FILE_NAME = 'raw_results.csv'
OVERHEAD_VIDEO_FILENAME = 'overlay_video_x{}_speed.mp4'
OVERHEAD_NO_OVERLAY_VIDEO_FILENAME = 'no_overlay_video_x{}_speed.mp4'
OVERHEAD_VIDEO_SPEED_UP = 4


def move_data_to_trial_folder(trial_dir, timing_dir, daq_dir):
    trial = os.path.basename(trial_dir)
    # check for daq or csv file
    daq_file = glob.glob(join(trial_dir, '*.daq'))
    csv_file = glob.glob(join(trial_dir, '*50Hz.csv'))
    if len(daq_file) == 0 and len(csv_file) == 0:
        daq_file = glob.glob(join(daq_dir, trial + '*.daq'))
        if len(daq_file) == 0:
            print('no .daq (or .csv) file found for trial {}'.format(trial))
        else:
            # move to daq folder
            os.rename(daq_file[0], join(trial_dir, os.path.basename(daq_file[0])))

    # check for timing file
    timing_file = glob.glob(join(trial_dir, '*.timing'))
    if len(timing_file) == 0:
        timing_file = glob.glob(join(timing_dir, trial + '*.timing'))
        mkv_file = glob.glob(join(timing_dir, trial + '*.mkv'))
        if len(timing_file) == 0:
            print('no .timing file found for trial {}'.format(trial))
        else:
            # move to timing folder
            os.rename(timing_file[0], join(trial_dir, os.path.basename(timing_file[0])))
        if len(mkv_file) == 0:
            print('no .mkv file found for trial {}'.format(trial))
        else:
            # move to timing folder
            os.rename(mkv_file[0], join(trial_dir, os.path.basename(mkv_file[0])))


def make_planner_data_pkl(root_dir, planner):
    """
    Did not log the planner messages sent to controller in the experiments
    so creating a dict that stores the planned paths in the world frame.

    Should have logged the sent planner messages!
    """
    # check file already exists
    if os.path.exists(join(root_dir, PLANNER_DATA_TRANSFORMED)):
        print('Transformed planner data already exists for {}'.format(root_dir))
        return

    data = {'metadata': {}}
    files_sorted = sorted(os.listdir(join(root_dir, 'paths')), key=lambda x: int(x.split('.')[0].split('_')[0]))
    for fp in files_sorted:
        if 'failed' in fp:
            index = fp.split('_')[0]
            data[index] = None
            continue
        index = int(fp.split('.')[0])
        curr_data = pickle.load(open(join(root_dir, 'paths', fp), 'rb'))
        message = PlannerNRCSocketServer.parse_raw_message(curr_data['raw_message'])
        if 'transform_matrix_planner_to_world' not in data['metadata']:
            m1 = PlannerNRCSocketServer.get_transform_planner_to_world(message)
            m2 = PlannerNRCSocketServer.get_transform_world_to_planner(message)
            data['metadata']['transform_matrix_planner_to_world'] = m1
            data['metadata']['transform_matrix_world_to_planner'] = m2
        path = curr_data['path']

        if 'lattice' in planner:
            scale = curr_data['ship_state'][0] / curr_data['processed_message']['ship_state'][0]
            path = resample_path(np.c_[(path[:2] / scale).T, path[2]], 0.01)

            if 'global_path' in curr_data:
                global_path = curr_data['global_path']
                global_path = np.c_[(global_path[:2] / scale).T, global_path[2]]

        else:
            path = path.T

        data[index] = {
            'path': transform_fn(m1, path, 3),
            'goal': PlannerNRCSocketServer.parse_goal(message),
        }

        if 'global_path' in curr_data:
            data[index]['global_path'] = transform_fn(m1, global_path, 3)

    # save to disk
    pickle.dump(data, open(join(root_dir, PLANNER_DATA_TRANSFORMED), 'wb'))


def main(experiment_root):
    all_dfs = []

    for planner in os.listdir(experiment_root):
        planner_dir = os.path.join(experiment_root, planner)
        if not os.path.isdir(planner_dir):
            continue

        planner_results = []
        for trial in tqdm(os.listdir(planner_dir)):

            trial_dir = os.path.join(planner_dir, trial)

            if not os.path.isdir(trial_dir):
                continue

            # get the daq file
            if len(glob.glob(os.path.join(trial_dir, '*50Hz.csv'))):
                DAQ_FILE = glob.glob(os.path.join(trial_dir, '*50Hz.csv'))[0]
            elif len(glob.glob(os.path.join(trial_dir, '*.daq'))):
                DAQ_FILE = glob.glob(os.path.join(trial_dir, '*.daq'))[0]
                print('converting daq file {} to csv...'.format(DAQ_FILE))  # for convenience
                generate_csv_from_daq(DAQ_FILE)
                DAQ_FILE = glob.glob(os.path.join(trial_dir, '*50Hz.csv'))[0]
            else:
                print('skipping trial {}, missing daq file!'.format(trial_dir))
                continue

            planner = trial.split('_')[0]
            if 'open_water' in trial:
                planner = 'open_water'

            if 'skeleton_centre_17' not in trial: continue

            print('\nDAQ_FILE: {}, planner: {}, trial: {}'.format(DAQ_FILE, planner, trial))

            # make a pkl file containing all the planned paths in the world frame
            make_planner_data_pkl(trial_dir, planner)

            # compute metrics
            metrics = evaluate_trial(experiment_root, trial, DAQ_FILE)

            metrics['Planner'] = planner
            metrics['Trial'] = trial
            metrics['Trial Number'] = trial.split('_')[-1]

            planner_results.append(metrics)

            # generate plots using daq data
            # plot_gdac(DAQ_FILE)

            # process overhead video
            video_file_name = glob.glob(os.path.join(trial_dir, '*cam0.mkv'))
            if len(video_file_name):
                video_file_name = video_file_name[0]
                print('video file name: {}'.format(video_file_name))
                # make overlay video
                process_overhead_video(video_file_name,
                                       show_video=False,
                                       show_costmap=False,
                                       overlay_obs=False,
                                       frames_to_read=OVERHEAD_VIDEO_SPEED_UP,
                                       save_overlay_video_filename=OVERHEAD_VIDEO_FILENAME.format(OVERHEAD_VIDEO_SPEED_UP))

                # make no overlay video
                # process_overhead_video(video_file_name,
                #                        show_video=False,
                #                        show_costmap=False,
                #                        overlay_ship_pos=False,
                #                        overlay_path=False,
                #                        overlay_obs=False,
                #                        overlay_ship_path=False,
                #                        frames_to_read=OVERHEAD_VIDEO_SPEED_UP,
                #                        save_overlay_video_filename=OVERHEAD_NO_OVERLAY_VIDEO_FILENAME.format(OVERHEAD_VIDEO_SPEED_UP),
                #                        save_first_frame=True)

            # NOTE!! clear cache, necessary for timing_file_parser.py
            linecache.clearcache()

        if not planner_results:
            continue

        # save results for planner
    #     df = pd.DataFrame(planner_results)
    #     df = df.set_index('Trial Number')
    #     df = df.sort_index()
    #     df = df.set_index('Trial')
    #     df.to_csv(os.path.join(planner_dir, PLANNER_RESULTS_CSV_FILE_NAME))
    #
    #     all_dfs.append(df)
    #
    # df = pd.concat(all_dfs)
    # df.to_csv(os.path.join(experiment_root, ALL_TRIAL_RESULTS_RAW_CSV_FILE_NAME))
    print('Done!')


if __name__ == '__main__':
    main(NRC_OEB_2023_EXP_ROOT)
