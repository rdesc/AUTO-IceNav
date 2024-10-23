import glob
import os
import random
from os.path import join, dirname, basename

import pandas as pd


DATASET_DIR = 'difficulty_evaluation_trial_image_dataset'  # stores both the images and the csv file containing scores
DATASET_FILE_NAME = 'dataset.csv'    # csv file containing the difficulty scores and mapping from trial to image
SEED = 0
MAX_DIFFICULTY_LEVEL = 1  # hardest, NOTE: in the paper this is reversed
MIN_DIFFICULTY_LEVEL = 5  # easiest
EASY = 'easy'
MEDIUM = 'medium'
HARD = 'hard'
DIFFICULTY_LEVELS = [EASY, MEDIUM, HARD]
EASY_BIN = (3.5, 5.0)    # inclusive
MEDIUM_BIN = (2.5, 3.5)  # exclusive
HARD_BIN = (1.0, 2.5)    # inclusive


def generate_image_dataset(experiment_root):
    """
    Generates a dataset of images from the first frame of each trial.
    The order of the images is randomized and planner information is removed.
    """
    random.seed(SEED)

    data = {}

    # get all the trial first frames
    trials = glob.glob(join(experiment_root, '**/first_frame.png'), recursive=True)

    # randomize the list
    random.shuffle(trials)

    # remove bad trials
    for trial in trials:
        if 'bad_trials' in trial:
            trials.remove(trial)

    for i, trial in enumerate(trials):
        data[i] = basename(dirname(trial))

    print('number of trials: {}'.format(len(trials)))

    # make the dataset folder
    os.system('mkdir -p {}'.format(join(experiment_root, DATASET_DIR)))

    # save each image to the dataset
    for i, trial in enumerate(trials):
        # copy the image to the dataset folder
        os.system('cp {} {}.png'.format(trial, join(experiment_root, DATASET_DIR, str(i))))

    # save dataset as csv
    df = pd.DataFrame.from_dict(data, orient='index', columns=['trial'])
    df.to_csv(join(experiment_root, DATASET_DIR, DATASET_FILE_NAME))


def get_trial_difficulty(experiment_root, trial=None, put_in_bins=False):
    dataset = pd.read_csv(join(experiment_root, DATASET_DIR, DATASET_FILE_NAME), index_col=0)
    dataset.set_index('trial', inplace=True)

    if trial is None:
        # return a dict where key is trial and value is average difficulty
        if not put_in_bins:
            return {
                trial: dataset.loc[trial].mean()
                for trial in dataset.index
            }
        else:
            return {
                trial: get_bin(dataset.loc[trial].mean())
                for trial in dataset.index
            }
    if trial not in dataset.index:
        return None  # trial not found

    # return the average difficulty of the user-specified trial
    if not put_in_bins:
        return dataset.loc[trial].mean()
    else:
        return get_bin(dataset.loc[trial].mean())


def get_bin(difficulty):
    if EASY_BIN[0] <= difficulty <= EASY_BIN[1]:
        return EASY
    elif MEDIUM_BIN[0] < difficulty < MEDIUM_BIN[1]:
        return MEDIUM
    elif HARD_BIN[0] <= difficulty <= HARD_BIN[1]:
        return HARD
    else:
        raise ValueError('Difficulty {} not in any bin'.format(difficulty))


if __name__ == '__main__':
    from ship_ice_planner.evaluation import NRC_OEB_2023_EXP_ROOT
    trial_diff = get_trial_difficulty(experiment_root=NRC_OEB_2023_EXP_ROOT,
                                      put_in_bins=True)
    data = []
    for trial, diff in trial_diff.items():
        data.append({
            'planner': trial.split('_')[0],
            'difficulty': diff
        })
    df = pd.DataFrame(data)
    print(df.groupby('planner')['difficulty'].value_counts())
