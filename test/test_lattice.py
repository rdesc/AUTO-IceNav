import os
import pickle
import shutil
import unittest
from copy import copy
from queue import Queue

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from ship_ice_planner.launch import launch
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import DotDict


class TestLattice(unittest.TestCase):
    def setUp(self):
        cfg_file = 'test_config.yaml'
        self.cfg = DotDict.load_from_file(cfg_file)

    def tearDown(self):
        shutil.rmtree('tmp')
        os.remove('.prim_paths.pkl')
        os.remove('.swath.pkl')

    def test_fill_lattice(self):
        """
        Expands all possible nodes in the state lattice given map dimensions, a set of primitives
        and a start node. Note, the planner will fail to find a path since an unreachable goal is set.
        This test can also be used to measure the performance of the A* algorithm. With numba
        enabled we get a runtime of 0.7 s for ~6000 expanded nodes.
        """
        cfg = copy(self.cfg)
        cfg.map_shape = (30, 12)
        cfg.a_star.weight = 0
        cfg.max_replan = 1
        cfg.planner = 'lattice'
        cfg.plot.show = False   # set to True to see the expanded nodes which generate a nice looking lattice
        cfg.save_paths = True
        cfg.optim = False

        iteration = 0
        for prim_name in ['PRIM_8H_1', 'PRIM_16H_1']:
            cfg.prim.prim_name = prim_name

            for idx, theta in enumerate([np.pi / 2,
                                         np.pi / 2 + 0.01,
                                         np.pi / 3,
                                         np.pi / 2 + 1 / 3,
                                         2 * np.pi / 3.001]):
                iteration += 1

                # init queue, send ship state and goal then pass to planner
                queue = Queue()
                start = (6, 0, theta)
                queue.put(dict(
                    goal=(None, 1000),
                    ship_state=start
                ))
                cfg.output_dir = 'tmp/' + str(iteration)
                launch(cfg=cfg, debug=False, logging=True, queue=queue)

                with open(os.path.join(cfg.output_dir, 'paths', '0_failed.pkl'), 'rb') as f:
                    data: dict = pickle.load(f)

                expanded = {
                    k: (v[0][0], v[0][1] + data['limits'][0], v[0][2])
                    for k, v in data['expanded'].to_dict().items()
                }

                print('Number of expanded nodes', len(expanded))

                if cfg.plot.show:
                    f, ax = plt.subplots()
                    c, data = Plot.aggregate_nodes(expanded)
                    sc = ax.scatter(data[:, 0], data[:, 1], s=2, c=c, cmap='viridis')
                    cbar = plt.colorbar(sc)
                    cbar.locator = MaxNLocator(integer=True)
                    cbar.update_ticks()
                    cbar.set_label('Number of Headings')
                    ax.set_title('Number of expanded nodes: {}\n Start: {:.2f}, {:.2f}, {:.2f}'.format(
                        len(expanded), start[0], start[1], start[2]))
                    ax.set_aspect('equal')
                    plt.show()


if __name__ == '__main__':
    unittest.main()
