""" Script for visualizing and testing dubins heuristic"""
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from ship_ice_planner.dubins_helpers.heuristic import dubins_heuristic


def tangent_circles(p: Tuple[float, float, float], r_min):
    """
    Generates the two tangent circles at p
    Returns the two tuples where each tuple is the
    circle centre and angle on the tangent circle
    """
    m = 1 if p[2] <= np.pi / 2 or p[2] >= 3 * np.pi / 2 else - 1
    theta = (3 * np.pi / 2 + p[2]) % (2 * np.pi)

    # get centres
    c1 = p[0] + m * r_min * np.cos(theta), p[1] + m * r_min * np.sin(theta)
    c2 = p[0] - m * r_min * np.cos(theta), p[1] - m * r_min * np.sin(theta)

    # get angle on circle
    a1 = get_angle(p, c1)
    a2 = get_angle(p, c2)

    return (c1, a1), (c2, a2)


def get_angle(p, c):
    return np.unwrap([2 * np.pi, np.arctan2(p[1] - c[1], p[0] - c[0])])[1] % (2 * np.pi)


def get_circle_exit(p, c, r_min, m):
    """
    Get the configuration where the path exits the tangent circle
    """
    return m * r_min * (1 - np.sin(p[2])) + p[0], c[1], np.pi / 2


def goal_on_circle(c, goal, r_min, m):
    """
    Finds the point on the circle that intersects with goal
    """
    x = (r_min ** 2 - (c[1] - goal) ** 2) ** 0.5
    temp_p = (c[0] + m * x, goal, 0)
    theta = get_angle(temp_p, c)
    return temp_p[0], temp_p[1], theta


def main():
    r_min = 2
    goal = 12
    boundary = (6, 12)

    # test tangent circle
    for i in range(1, 8):
        for d in [-0.1, 0, 0.1]:
            p = (10, 10, i * np.pi / 4 + d)
            circles = tangent_circles(p, r_min)
            m = 1 if p[2] <= np.pi / 2 or p[2] >= 3 * np.pi / 2 else - 1

            plt.hlines(goal, p[0] + r_min * 2, p[0] - r_min * 2, label='goal line')
            plt.vlines(boundary[0], p[1] - r_min * 2, goal, colors='r', label='boundary')
            plt.vlines(boundary[1], p[1] - r_min * 2, goal, colors='r')
            plt.plot(p[0], p[1], 'bx')
            plt.plot([p[0], p[0] + np.cos(p[2])], [p[1], p[1] + np.sin(p[2])], 'b--', label='start config', zorder=10)

            for item in circles:
                centre, angle = item
                x = np.arange(0, 2 * np.pi, 0.1)
                plt.plot(*centre, 'cx')
                plt.plot(np.cos(x) * r_min + centre[0], np.sin(x) * r_min + centre[1], 'c')
                plt.plot([centre[0], centre[0] + np.cos(angle)],
                         [centre[1], centre[1] + np.sin(angle)], 'c--')

                if centre[1] >= goal:
                    goal_p = goal_on_circle(centre, goal, r_min, m)
                    plt.plot(goal_p[0], goal_p[1], 'ro', label='C exit')
                    plt.plot([centre[0], centre[0] + np.cos(goal_p[2])],
                             [centre[1], centre[1] + np.sin(goal_p[2])], 'c--')

            if not any(i[0][1] >= goal for i in circles):
                centre = circles[0][0] if circles[0][0][1] >= goal else circles[1][0]
                p_exit = get_circle_exit(p, centre, r_min, m)
                plt.plot(p_exit[0], p_exit[1], 'bo', label='CS exit')

            h, final_p = dubins_heuristic(p, goal, r_min, boundary=boundary)
            print('start configuration', p, 'heuristic (length of Dubins path)', h, 'final configuration', final_p)

            plt.plot([final_p[0], final_p[0] + np.cos(final_p[2])],
                     [final_p[1], final_p[1] + np.sin(final_p[2])], 'm--', label='goal config')
            plt.title('Heuristic %.2f' % h)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.gca().set_aspect('equal')
            plt.show()
            plt.pause(0.01)


if __name__ == '__main__':
    main()  # generates plots for debugging
