import os

import numpy as np
import yaml

from ship_ice_planner import src
from ship_ice_planner.src.planners import *
from ship_ice_planner.src.utils.utils import DotDict, setup_logger


def launch(cfg_file=None, cfg=None, debug=False, logging=True, log_level=10, **kwargs):
    try:
        os.remove('.prim_paths.pkl')
        os.remove('.swath.pkl')
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
        # directory structure is experiment_name/planner/trial_num
        cfg.output_dir = os.path.join(cfg.output_dir, cfg.planner)
        if not os.path.isdir(cfg.output_dir):
            cfg.output_dir = os.path.join(cfg.output_dir, '0')

        else:
            cfg.output_dir = os.path.join(cfg.output_dir, str(len(os.listdir(cfg.output_dir))))

        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, 'config.yaml'), 'w') as outfile:
            yaml.dump(DotDict.to_dict(cfg), outfile, default_flow_style=False)

    # setup logger
    # can disable logging to console here
    logger = setup_logger(output=cfg.output_dir, console=bool(logging), name=src.__name__, level=log_level, prefix=str(cfg.output_dir) + ' ')

    # run planner
    if cfg.planner == 'skeleton':
        res = skeleton_planner(cfg, debug, **kwargs)

    elif cfg.planner == 'straight':
        res = straight_planner(cfg, **kwargs)

    else:
        res = lattice_planner(cfg, debug, **kwargs)

    logger.info('Done run with output directory')
    logger.info(cfg.output_dir)

    # cleanup run
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    handlers.clear()

    return res


def demo():
    from queue import Queue
    import pickle

    config_file = 'configs/no_physics.yaml'  # demo config
    cfg = DotDict.load_from_file(config_file)
    cfg.output_dir = ''
    cfg.max_replan = 1
    cfg.plot.show = True
    queue = Queue()
    queue.put(dict(
        goal=(0, 70),
        ship_state=(6, 10, np.pi / 2),
        obstacles=pickle.load(open('data/demo_ice_data.pk', 'rb'))
    ))
    launch(cfg=cfg, queue=queue, debug=True, logging=True)  # python -O to run python with optimization


if __name__ == '__main__':
    demo()
