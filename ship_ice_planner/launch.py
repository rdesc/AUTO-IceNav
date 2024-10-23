import os

import numpy as np
import yaml

import ship_ice_planner
from ship_ice_planner import EXPERIMENT_ROOT_DIR, DEFAULT_CONFIG_FILE
from ship_ice_planner.planners import *
from ship_ice_planner.primitives import CACHED_PRIM_PATHS
from ship_ice_planner.swath import CACHED_SWATH
from ship_ice_planner.utils.utils import DotDict, setup_logger


def launch(cfg_file=None, cfg=None, debug=False, logging=True, log_level=10, **kwargs):
    try:
        os.remove(CACHED_PRIM_PATHS)
        os.remove(CACHED_SWATH)
    except FileNotFoundError:
        pass

    if cfg_file:
        cfg = DotDict.load_from_file(cfg_file)
        cfg.cfg_file = cfg_file

    # seed if necessary (only used in testing)
    if cfg.get('seed', None):
        np.random.seed(cfg.seed)

    # make directory to store planner outputs
    if cfg.output_dir:

        # socket_comm=True means this is a live tank test!
        # directory structure is root_dir/exp_name/planner/trial_id
        # trial_id is either the number of trials in the planner directory or set by output_dir in the config
        if kwargs.get('socket_comm'):
            if cfg.get('exp_name'):
                cfg.output_dir = os.path.join(EXPERIMENT_ROOT_DIR, cfg.exp_name, cfg.planner, cfg.output_dir)
            else:
                if EXPERIMENT_ROOT_DIR not in cfg.output_dir:
                    cfg.output_dir = os.path.join(EXPERIMENT_ROOT_DIR, cfg.output_dir)
                cfg.output_dir = os.path.join(cfg.output_dir, cfg.planner)
                if not os.path.isdir(cfg.output_dir):
                    cfg.output_dir = os.path.join(cfg.output_dir, '0')

                else:
                    cfg.output_dir = os.path.join(cfg.output_dir, str(len(os.listdir(cfg.output_dir))))

        # NOTE: do not change the output directory here if planner is being run with the sim!
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, DEFAULT_CONFIG_FILE), 'w') as outfile:
            yaml.dump(DotDict.to_dict(cfg), outfile, default_flow_style=False)

    # setup logger
    # can disable logging to console here
    logger = setup_logger(output=cfg.output_dir, console=bool(logging),
                          name=ship_ice_planner.__name__, level=log_level)

    # run planner
    if cfg.planner == 'skeleton':
        res = skeleton_planner(cfg, debug, **kwargs)

    elif cfg.planner == 'straight':
        res = straight_planner(cfg, **kwargs)

    elif cfg.planner == 'lattice':
        res = lattice_planner(cfg, debug, **kwargs)

    else:
        raise ValueError(f'Planner "{cfg.planner}" not recognized')

    logger.info('Done run with output directory')
    logger.info(cfg.output_dir)

    # cleanup run
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    handlers.clear()

    return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launcher for Ship Ice Planner')
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    parser.add_argument('-l', '--logging', action='store_true', help='Logging mode')
    parser.add_argument('-ll', '--log_level', type=int, default=10, help='Logging level')

    args = parser.parse_args()
    assert args.config is not None, 'Must provide a config file! See --help for usage'
    launch(cfg_file=args.config, debug=args.debug, logging=args.logging, log_level=args.log_level, socket_comm=True)
