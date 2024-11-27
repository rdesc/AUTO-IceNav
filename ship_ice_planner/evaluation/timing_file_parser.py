"""
Read video timing files and parse them.
Vision server is started up at the same time as GDAC
"""
import os
import json
import linecache
import pickle
from ast import literal_eval

import numpy as np
from setuptools import glob

SKIP_FIRST_K_LINES = 5  # timing file contains k lines of metadata
TIMESTAMP_FILE = 'cam_timestamps.pkl'  # stores a list of timestamps for each frame starting at frame 1


def get_polygons_from_timing_file(root_dir, timestamp=None, step=None, return_timestamp=False):
    """
    NOTE: this method can cause memory issues if the cache is not freed from time to time

    clear cache via: linecache.clearcache()

    :param root_dir: directory containing the timing file
    :param timestamp: timestamp of the frame to return
    :param step: the index or step of the frame to return
    """
    assert (timestamp is not None) or (step is not None), 'Must provide either timestamp or step'

    # find the file inside the root_dir with .timing
    timing_fp = glob.glob(os.path.join(root_dir, '*.timing'))
    if len(timing_fp) == 0:
        return []
    else:
        timing_fp = timing_fp[0]
    timestamps_fp = glob.glob(os.path.join(root_dir, TIMESTAMP_FILE))

    if len(timestamps_fp) == 0:
        timestamps_data = create_timestamp_list_file(timing_fp)
    else:
        timestamps_data = pickle.load(open(timestamps_fp[0], 'rb'))

    if timestamp is not None:
        # find the index closest to timestep
        step = np.argmin(np.abs(np.asarray(timestamps_data) - timestamp))

    if return_timestamp:
        return parse_polygon_data(
            linecache.getline(timing_fp, int(step) + SKIP_FIRST_K_LINES)
        ), timestamps_data[step], step

    # get the polygons at idx
    return parse_polygon_data(
        linecache.getline(timing_fp, int(step) + SKIP_FIRST_K_LINES)
    )


def get_timing_step_for_timestamp(root_dir, timestamp):
    # find the file inside the root_dir with .timing
    timestamps_fp = glob.glob(os.path.join(root_dir, TIMESTAMP_FILE))
    timestamps_data = pickle.load(open(timestamps_fp[0], 'rb'))

    # find the index closest to timestep
    return np.argmin(np.abs(np.asarray(timestamps_data) - timestamp))


def create_timestamp_list_file(timing_file):
    # read the timing file
    with open(timing_file, 'r') as f:
        count = 0
        data = []

        while True:
            count += 1

            line = f.readline()

            if count < SKIP_FIRST_K_LINES:
                continue

            if not line:
                break

            # get the index and time
            metadata = line.split(',{')[0]
            timestamp = float(metadata.split(',')[1])
            data.append(timestamp)

    # store as pkl
    with open(os.path.join(os.path.dirname(timing_file), TIMESTAMP_FILE), 'wb') as f:
        pickle.dump(data, f)

    return data


def parse_polygon_data(line):
    # want list of arrays
    data = line.split(',{')
    if len(data) <= 1:
        return []
    data = '{' + data[1]
    data = json.loads(data)

    polys = []
    for poly in data.values():
        # convert list of strings to floats
        poly = literal_eval('[' + poly + ']')
        poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        polys.append(poly)

    return polys


def get_frame_rate_from_timing_file(root_dir):
    # find the file inside the root_dir with .timing
    timing_fp = glob.glob(os.path.join(root_dir, '*.timing'))
    if len(timing_fp) == 0:
        return None
    else:
        timing_fp = timing_fp[0]

    with open(timing_fp, 'r') as f:
        f.readline()
        line = f.readline()  # second line contains the frame rate
        frame_rate = float(line.split('Frame Rate: ')[-1])

    return frame_rate
