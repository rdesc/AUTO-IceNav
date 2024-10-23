import time
import unittest

import dubins
import numpy as np
from matplotlib import pyplot as plt

# C version is >10x faster than pure
# python and >1.5x faster than numba
from ship_ice_planner.dubins_helpers.heuristic import dubins_heuristic


class TestHeuristic(unittest.TestCase):
    def setUp(self):
        self.x0 = 10
        self.y0 = 10
        self.r_min = 3
        self.plot = False

    def test_edge_case(self):
        goal = 11
        theta = np.arange(0, np.pi * 2, 0.1)
        for t in theta:
            p = (self.x0, self.y0, t)
            m = 1 if p[2] <= np.pi / 2 or p[2] >= 3 * np.pi / 2 else - 1
            omega_y = p[1] + m * self.r_min * np.cos(p[2])

            if omega_y >= goal:
                h, p_final = dubins_heuristic(p, goal, self.r_min)
                path_length, segment_lengths = dubins_pkg(p, p_final, self.r_min, debug=self.plot)

                self.assertTrue(segment_lengths[1] < 1e-2)  # confirm there is no straight segment
                self.assertTrue(abs(h - path_length) < 1e-2)

    def test_general_case(self):
        goal = 20
        theta = np.arange(0, np.pi * 2, 0.1)
        for t in [*theta, np.pi / 2]:
            p = (self.x0, self.y0, t)
            h, p_final = dubins_heuristic(p, goal, self.r_min)
            path_length, _ = dubins_pkg(p, p_final, self.r_min, debug=self.plot)
            self.assertTrue(abs(h - path_length) < 1e-2)

    def test_past_goal(self):
        goal = 5
        p = (self.x0, self.y0, 0)
        self.assertEqual(dubins_heuristic(p, goal, self.r_min)[0], 0)

    def test_boundary(self):
        boundary = (9, 11)
        for goal in [12, 20]:
            for t in [0, np.pi]:
                p = (self.x0, self.y0, t)
                h, p_final = dubins_heuristic(p, goal, self.r_min, boundary=boundary)
                self.assertTrue(p_final[0] > boundary[1] or p_final[0] < boundary[0])
                self.assertEqual(h, np.inf)

                if self.plot:
                    plt.vlines(boundary[0], self.y0 - self.r_min, self.y0 + self.r_min)
                    plt.vlines(boundary[1], self.y0 - self.r_min, self.y0 + self.r_min)
                    path_length, _ = dubins_pkg(p, p_final, self.r_min, debug=self.plot)

    def test_execution_time(self):
        t = time.time()
        N = 10_000_000
        goal = 20

        for i in range(0, N // 2):
            p = (10, 18, i * 2 * np.pi / N)
            dubins_heuristic(p, goal, self.r_min)

        for i in range(0, N // 2):
            p = (10, 0, i * 2 * np.pi / N)
            dubins_heuristic(p, goal, self.r_min)

        print('time', time.time() - t)


def dubins_pkg(p0, p1, r_min, step_size=0.001, eps=1e-4, debug=True):
    dubins_path = dubins.shortest_path(p0, p1, r_min - eps)
    path = np.asarray(dubins_path.sample_many(step_size)[0])

    if debug:
        plt.plot(p0[0], p0[1], 'bx')
        plt.plot([p0[0], p0[0] + np.cos(p0[2])],
                 [p0[1], p0[1] + np.sin(p0[2])], 'b--')
        plt.plot(path[:, 0], path[:, 1])
        plt.plot(p1[0], p1[1], 'mx')
        plt.plot([p1[0], p1[0] + np.cos(p1[2])],
                 [p1[1], p1[1] + np.sin(p1[2])], 'm--')
        plt.gca().set_aspect('equal')
        plt.show()

    return (
        dubins_path.path_length(),
        # segment types https://github.com/AndrewWalker/Dubins-Curves/blob/master/src/dubins.c#L38
        (dubins_path.segment_length(0), dubins_path.segment_length(1), dubins_path.segment_length(2))
    )


if __name__ == '__main__':
    unittest.main()
