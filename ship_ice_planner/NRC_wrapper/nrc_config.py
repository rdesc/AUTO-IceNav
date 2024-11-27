"""
    The NRC tank coordinate system:

    x         --------------
    ^         |            |
    |         |            |
    |--> y    |            |
              |   tank     |
              |            |
              |            |
              |            |
              --------------

    The NRC tank uses the north-east-down coordinate system.
    Where yaw = 0 is north and yaw = pi/2 is east. The origin
    is at the top left corner of the tank.

    The Planner code coordinate system:

              --------------
              |            |
              |            |
              |            |
              |   tank     |
    y         |            |
    ^         |            |
    |         |            |
    |--> x    --------------

    The planner coordinate system has y pointing up and x pointing right where
    the origin is at the bottom left corner of the tank.
    Yaw = 0 is the positive x direction and yaw = pi/2 is the positive y direction.
"""
import numpy as np

# plastic ice properties
ICE_DENSITY = 991  # kg/m^3
ICE_THICKNESS = 0.012  # m

# 3d homography matrix i.e. the perspective transformation matrix
H_MATRIX_CAM_TO_WORLD = np.asarray(
    [[ 1.16096759e-02,  5.48038652e-03, -3.19808759e+01],
     [-1.67479199e-04,  8.43524568e-03,  7.88072594e+00],
     [-1.35419683e-05, -2.21296116e-04,  1.00000000e+00]],
)

H_MATRIX_WORLD_TO_CAM = np.asarray(
    [[8.95490812e+01, 1.40479417e+01,  2.75315007e+03],
     [5.34508750e-01, 9.83232076e+01, -7.57764195e+02],
     [1.33095553e-03, 2.19487807e-02,  8.69592798e-01]]
)

# tower coordinates from OEB 2023 experiments
TOWER = [[-34.39, 7.945],   # BOTTOM LEFT
         [-14.39, 7.945],   # TOP LEFT
         [-34.39, 17.945],  # BOTTOM RIGHT
         [-14.39, 17.945]]  # TOP RIGHT

# tower coordinates from OEB 2022 experiments
# TOWER = [[-38.5, 8.4],
#          [-18.1, 8.4],
#          [-38.5, 15.4],
#          [-18.1, 15.4]]

BOTTOM_LEFT, TOP_LEFT, BOTTOM_RIGHT, TOP_RIGHT = TOWER

# artificially shrink the tank width by 4 metre
BOTTOM_LEFT[1] += 4
TOP_LEFT[1] += 4

# for open water trials
# BOTTOM_RIGHT[1] -= 4
# TOP_RIGHT[1] -= 4

TANK_LENGTH = abs(BOTTOM_LEFT[0] - TOP_LEFT[0])
TANK_WIDTH = abs(TOP_RIGHT[1] - TOP_LEFT[1])
MID_TANK = ((BOTTOM_LEFT[0] + TOP_LEFT[0]) / 2,
            (BOTTOM_LEFT[1] + BOTTOM_RIGHT[1]) / 2)

# hardcoded the goal
BUFFER_GOAL = 1  # metres

GOAL_UP = (TOP_RIGHT[0] - BUFFER_GOAL, MID_TANK[1])
GOAL_DOWN = (BOTTOM_RIGHT[0] + BUFFER_GOAL, MID_TANK[1])

# # plot the tower coordinates for debugging
# import matplotlib.pyplot as plt
# for item, label in zip(TOWER, ['BL', 'TL', 'BR', 'TR']):
#     plt.plot(item[1], item[0], 'x', label=label)
# plt.legend()
# plt.gca().set_aspect('equal')
# plt.show()

transform_matrix_world_to_planner1 = np.array([
    [0, 1,  0, -BOTTOM_LEFT[1]],
    [1, 0,  0, -BOTTOM_LEFT[0]],
    [0, 0, -1,       np.pi / 2],
    [0, 0,  0,               1]
])

# convenient functions for doing the transformations
# x is the input and n is the size of a state, e.g. n=2 for (x,y) and n=3 for (x,y,yaw)
transform_world_to_planner1_fn = lambda x, n: x @ transform_matrix_world_to_planner1[:n, :n].T + transform_matrix_world_to_planner1[:n, -1]

transform_matrix_planner_to_world1 = np.array([
    [0, 1,  0, BOTTOM_LEFT[0]],
    [1, 0,  0, BOTTOM_LEFT[1]],
    [0, 0, -1,      np.pi / 2],
    [0, 0,  0,              1]
])

transform_planner_to_world1_fn = lambda x, n: x @ transform_matrix_planner_to_world1[:n, :n].T + transform_matrix_planner_to_world1[:n, -1]

transform_matrix_world_to_planner2 = np.array([
    [ 0, -1,  0,  TOP_RIGHT[1]],
    [-1,  0,  0,  TOP_RIGHT[0]],
    [ 0,  0, -1, 3 * np.pi / 2],
    [ 0,  0,  0,             1]
])

transform_world_to_planner2_fn = lambda x, n: x @ transform_matrix_world_to_planner2[:n, :n].T + transform_matrix_world_to_planner2[:n, -1]

transform_matrix_planner_to_world2 = np.array([
    [ 0, -1,  0,  TOP_RIGHT[0]],
    [-1,  0,  0,  TOP_RIGHT[1]],
    [ 0,  0, -1, 3 * np.pi / 2],
    [ 0,  0,  0,             1]
])

transform_planner_to_world2_fn = lambda x, n: x @ transform_matrix_planner_to_world2[:n, :n].T + transform_matrix_planner_to_world2[:n, -1]

transform_fn = lambda m, x, n: x @ m[:n, :n].T + m[:n, -1]