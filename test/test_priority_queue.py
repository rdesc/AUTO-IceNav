import unittest

import numpy as np

from ship_ice_planner.utils.priority_queue import PriorityQueue


class TestPriorityQueue(unittest.TestCase):
    def setUp(self):
        self.queue = PriorityQueue(item=(2, True))  # initialize queue with a dummy priority and data

    def test_update(self):
        self.queue.update(orig_item=(2, True), new_item=(1, True))
        self.assertEqual(self.queue.invalidate, {(2, True)})
        self.assertTrue((2, True) in self.queue.queue)
        self.assertTrue((1, True) in self.queue.queue)

    def test_get_item(self):
        for _ in range(100):
            item = np.random.rand()
            self.queue.put((item, True))
            self.queue.update((item, True), (item-1, True))

        while self.queue.empty():
            min_item = min(self.queue.queue)
            self.assertEqual(self.queue.get_item(), min_item)

        self.assertEqual(len(self.queue.invalidate), 100)


if __name__ == '__main__':
    unittest.main()
