"""
Tests weighted A* and confirms solution path is no worse than (1 + ε) times the optimal solution path
https://en.wikipedia.org/wiki/A*_search_algorithm#Bounded_relaxation
"""
import os
import pickle
import shutil
import unittest
from copy import deepcopy
from queue import Queue

import numpy as np

from ship_ice_planner.launch import launch
from ship_ice_planner.utils.utils import DotDict


class TestWeightedAStar(unittest.TestCase):
    def setUp(self):
        cfg_file = 'test_config.yaml'
        self.cfg = DotDict.load_from_file(cfg_file)
        self.cfg.plot.show = False
        self.cfg.max_replan = 1
        self.cfg.output_dir = 'tmp'
        self.cfg.sim_dynamics = DotDict()
        self.cfg.sim_dynamics.target_speed = 1

    def tearDown(self):
        shutil.rmtree('tmp')
        os.remove('.prim_paths.pkl')
        os.remove('.swath.pkl')

    def test_bounded_relaxation(self):
        results = []
        weights = np.arange(1, 5, 0.3)
        for w in weights:
            print('Static weight:', w)
            self.cfg.a_star.weight = w
            self.cfg.output_dir = os.path.join('tmp', 'weight' + str(w))

            queue = Queue()
            queue.put(dict(
                goal=(None, 40),
                ship_state=(6, 2, np.pi / 2),
                obstacles=[ob['vertices'] for ob in pickle.load(open('../data/demo_NRC_ice_tank_data.pkl', 'rb'))]
            ))
            res = launch(cfg=deepcopy(self.cfg), debug=False, logging=True, queue=queue)
            results.append(*res)

        a = results[0]['g_score']
        for i, w in zip(results[1:], weights[1:]):
            b = i['g_score']
            self.assertTrue(a * w >= b)  # confirm the solution path is no worse than w times optimal solution path


if __name__ == '__main__':
    unittest.main()
