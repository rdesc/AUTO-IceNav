""" Path optimization using CasADi """
import logging
import time
from typing import Tuple

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from casadi import MX, Opti, Function, horzcat, vertcat, norm_2, sum1, sum2
from matplotlib import patches, colors

from ship_ice_planner.dubins_helpers.control import compute_turn_rate_control
from ship_ice_planner.geometry.polygon import generate_body_points_polygon
from ship_ice_planner.geometry.utils import Rxy
from ship_ice_planner.swath import compute_swath_cost
from ship_ice_planner.utils.utils import resample_path, compute_path_length


class SolverCallback:
    # tiny class for conveniently storing solution from solver
    def __init__(self):
        self.X_list = []  # for each iteration save out the current solution
        self.U_list = []
        self.L_list = []
        self.obj_list = []

    def __call__(self, path=None, control=None, path_length=None, obj=None):
        self.X_list.append(path)
        self.U_list.append(control)
        self.L_list.append(path_length)
        self.obj_list.append(obj)


class PathOptimizer:

    def __init__(self):
        self.lut = None                  # smooth function of the form lut((y, x)) to look up costs at position (x, y)
        self.costmap = None              # the costmap used to generate the look-up table, used for plotting
        self.costmap_offsets = None      # only a subset of the costmap is used given the ship position and goal

        # parameters
        self.pos_bounds = None           # bounds on position
        self.control_bounds = None       # bound on control input
        self.goal_margin = None          # margin around goal position
        self.initial_ds = None           # initial distance between waypoints/decision variables
        self.body_points = None          # points inside ship footprint where costs are computed
        self.body_points_spacing = None  # spacing between body points
        self.body_weights = None         # weights for each body point
        self.smooth_control_weight = None  # weight for penalizing large changes in control input
        # casadi specific params
        self.p_opts = {}
        self.s_opts = {}
        # these parameters are set from initial solution
        self.N = None                    # number of control intervals
        self.goal = None                 # goal position
        self.ship_pose = None            # initial ship pose
        self.initial_path_length = None  # path length of initial path
        self.warm_start_path = None      # initial path used to warm start the solver
        self.global_path = None          # the global path to constrain the solution to (optional)

        # constant to make sure the scale of objective is the same as the swath cost
        self.objective_bound_constant = None

        # casadi objects
        self.opti = None

        self.plot_dir = None
        self.logger = logging.getLogger(__name__)

    def set_parameters(self,
                       ship_length: float,
                       position_bounds: Tuple,
                       control_bounds: float,
                       wp_spacing: float,
                       ship_vertices: np.ndarray,
                       goal_margin: float = 0.,
                       body_point_params: dict = None,
                       p_opts: dict = None,
                       s_opts: dict = None,
                       plot_dir: str = None,
                       smooth_control_weight: float = 0,
                       **kwargs
                       ):
        self.plot_dir = plot_dir
        self.pos_bounds = position_bounds
        self.control_bounds = control_bounds
        self.goal_margin = goal_margin
        self.p_opts = p_opts
        self.s_opts = s_opts
        self.smooth_control_weight = smooth_control_weight
        self.initial_ds = wp_spacing
        self._set_body_points(vertices=ship_vertices, **body_point_params)
        self._validate_params_and_compute_objective_bound_constant(ship_length)

        self.logger.info('Position bounds {}, {}. Control bounds {}'.format(self.pos_bounds[0],
                                                                            self.pos_bounds[1],
                                                                            self.control_bounds))

    def _validate_params_and_compute_objective_bound_constant(self, ship_length):
        # these should all be in costmap grid units
        self.max_ds = ship_length / self.body_points_spacing ** 2
        assert self.initial_ds <= self.max_ds, 'spacing between waypoints (ds) must be less than {}'.format(self.max_ds)
        self.objective_bound_constant = self.initial_ds / self.max_ds
        self.logger.info('objective bound constant: {:.4f}, max allowable delta s {:.4f}'
                         .format(self.objective_bound_constant, self.max_ds))

    def _set_body_points(self, vertices, spacing=1., margin=0., weights=None, plot=False):
        self.body_points_spacing = spacing

        body_points, body_weights = generate_body_points_polygon(vertices,
                                                                 spacing,
                                                                 margin,
                                                                 weights=weights,
                                                                 only_inside_polygon=False,  # body points make a rectangle that enclose ship body
                                                                 plot=plot)

        self.logger.info('body weights {}'.format(list(body_weights)))
        self.body_points = body_points
        self.body_weights = MX(body_weights)

    def set_initial_solution(self,
                             initial_path: np.ndarray,
                             goal: float,
                             ship_pose: np.ndarray,
                             global_path: np.ndarray = None):
        self.ship_pose = ship_pose

        # cut the path to the goal region
        idx = np.argmin(np.abs(initial_path[1, :] - goal))
        initial_path_clipped = initial_path[:, :idx + 1]

        # cut the path to the ship pose
        dist = np.linalg.norm(initial_path_clipped[:2] - ship_pose[:2, None], axis=0)
        trunc_ind = np.argmin(dist)
        initial_path_clipped = initial_path_clipped[:, trunc_ind:]

        # resample
        self.warm_start_path = resample_path(initial_path_clipped.T, self.initial_ds, linear=False, plot=False).T
        self.initial_path_length = compute_path_length(self.warm_start_path[:2].T)

        self.goal = self.warm_start_path[1, -1]  # should be close to the goal passed in the args

        # the number of waypoints determines number of control intervals, should be close to self.initial_path_length / self.initial_ds
        self.N = self.warm_start_path.shape[1] - 1

        if global_path is not None:
            idx = np.argmin(np.abs(global_path[1, :] - self.goal))
            self.global_path = global_path[:, :idx + 1]

        self.logger.info('Number of control intervals {:.0f}, Start position ({:.2f}, {:.2f}, {:.2f}), Goal {:.2f}'.
                         format(self.N, *ship_pose, self.goal))

    def costmap_to_lut(self, costmap, use_costmap_subset=True, subset_margin=0):
        """
        Generates a smooth function for the costmap using a 2D casadi look-up table
        https://web.casadi.org/docs/#using-lookup-tables

        Need to ensure that there is a high cost along the side boundaries of the costmap
        """
        t1 = time.time()
        self.costmap = costmap

        if use_costmap_subset:
            lower = max(0, int(self.warm_start_path[1, 0] - subset_margin))
            upper = min(costmap.shape[0], int(self.goal + self.goal_margin + subset_margin))

            self.costmap_offsets = (lower, upper)
            costmap_subset = costmap[self.costmap_offsets[0]: self.costmap_offsets[1]]
        else:
            self.costmap_offsets = (0, costmap.shape[0])
            costmap_subset = costmap

        xgrid = np.arange(0, costmap_subset.shape[0])
        ygrid = np.arange(0, costmap_subset.shape[1])

        # generate look-up table function for costmap
        # order of args should be lut((y, x)) which is consistent with costmap[row, col]
        self.lut = ca.interpolant('LUT', 'bspline', [xgrid, ygrid], costmap_subset.ravel(order='F'),
                                  {'algorithm': 'smooth_linear',  # options are 'smooth_linear' and 'not_a_knot'
                                   'smooth_linear_frac': 0.49,    # value should be inside (0, 0.5)
                                   # 'verbose': True,
                                   # 'jit': True
                                   })
        self.logger.info('lut generation time {:.2f} s'.format(time.time() - t1))

        # for debugging purposes, can convert the lut back to a discrete costmap
        # f, ax = plt.subplots(1, 2)
        # ax[0].imshow(costmap_subset); ax[1].imshow(self._lut_to_costmap(self.lut, costmap_subset.shape)); plt.show(); exit()

        return self.lut

    @staticmethod
    def _lut_to_costmap(lut, shape, scale=10):
        """
        For debugging purposes, convert the casadi look-up table back to a discrete costmap
        """
        costmap = np.zeros((shape[0] * scale, shape[1] * scale))
        for i in range(shape[0] * scale):
            for j in range(shape[1] * scale):
                costmap[i, j] = lut((i / scale, j / scale))
        return costmap

    def plot_solution_evolution(self,
                                obj_list,
                                X_list,
                                U_list,
                                path_length,
                                obstacles,
                                ship_vertices,
                                debug=True,
                                anim=False,
                                ):
        """
        Plot the evolution of the solution over the optimization iterations
        Very useful for visualizing and debugging optimization stage

        :param obj_list: objective values at each iteration
        :param X_list: list of state trajectories at each iteration
        :param U_list: list of control trajectories at each iteration
        :param path_length: path length for the optimized path
        :param obstacles: list of obstacle vertices
        :param ship_vertices: ship vertices
        :param debug: whether to compute and plot swath cost at each iteration (can be slow)
        :param anim: whether to animate the plot
        """
        assert len(obj_list) == len(X_list) == len(U_list)
        num_iters = len(obj_list)
        if debug:
            swath_cost_list = [
                compute_swath_cost(self.costmap, X_list[i].T, ship_vertices, resample=0.1)[1]
                for i in range(num_iters)
            ]

        f1, ax = plt.subplots(1, 3, figsize=(25, 10))
        obj_ax = ax[0]
        swath_ax = obj_ax
        path1_ax = ax[1]
        path2_ax = ax[2]

        for i in range(num_iters + 1):
            for a in ax.ravel():
                a.cla()
            swath_ax.cla()

            if not anim:
                i = num_iters

            if debug:
                swath_ax.plot(swath_cost_list[:i], 'g--', label='swath cost')

            if i == num_iters:
                obj_ax.plot(obj_list, 'b', label='objective')

                if debug:
                    obj_ax.set_title('collision objective improvement {:.1f} %\nswath cost improvement {:.1f} %'
                                     .format((100 * (obj_list[-1] - obj_list[0]) / obj_list[0]),
                                             100 * (swath_cost_list[-1] - swath_cost_list[0]) / swath_cost_list[0]))
                else:
                    obj_ax.set_title('collision objective improvement {:.1f} %\n'
                                     .format((100 * (obj_list[-1] - obj_list[0]) / obj_list[0])))
            else:
                if debug:
                    obj_ax.plot(obj_list[:i], 'b', label='objective')
                    obj_ax.set_title('collision objective improvement {:.1f} %\nswath cost improvement {:.1f} %'
                                     .format((100 * (obj_list[i] - obj_list[0]) / obj_list[0]),
                                             100 * (swath_cost_list[i] - swath_cost_list[0]) / swath_cost_list[0]))
                else:
                    obj_ax.plot(obj_list[:i], 'b', label='objective')
                    obj_ax.set_title('collision objective improvement {:.1f} %'
                                     .format((100 * (obj_list[i] - obj_list[0]) / obj_list[0])))

            obj_ax.set_xlabel('iteration')
            obj_ax.set_ylabel('objective')
            obj_ax.legend()

            path1_ax.imshow(self.costmap, origin='lower')

            if i == num_iters:
                path1_ax.plot(X_list[-1][0, :], X_list[-1][1, :], 'r.', label='optimized')
                path1_ax.set_title('initial cost {:.1f}\noptimized cost {:.1f}'
                                   .format(float(obj_list[0]), float(obj_list[-1])))
                # plot 'x' at each body point
                for x_, y_, theta_ in X_list[-1].T[:1]:
                    pts_x = np.cos(theta_) * self.body_points[:, 0] - np.sin(theta_) * self.body_points[:, 1] + x_
                    pts_y = np.sin(theta_) * self.body_points[:, 0] + np.cos(theta_) * self.body_points[:, 1] + y_
                    path1_ax.plot(pts_x, pts_y, 'wx', markersize=1)

                # show the swath
                swath = compute_swath_cost(self.costmap, X_list[-1].T, ship_vertices, resample=0.01)[0]
                swath_im = np.zeros(swath.shape + (4,))  # init RGBA array
                swath_im[:] = colors.to_rgba('w')
                swath_im[:, :, 3] = swath
                path1_ax.imshow(swath_im, origin='lower', alpha=0.2)

            else:
                path1_ax.plot(X_list[i][0, :], X_list[i][1, :], 'r', label='optimized')
                path1_ax.set_title('initial cost {:.1f}\noptimized cost {:.1f}'
                                   .format(float(obj_list[0]), float(obj_list[i])))
            # first solution is the initial solution i.e. warm start to solver
            path1_ax.plot(X_list[0][0, :], X_list[0][1, :], 'k--', label='initial')
            path1_ax.set_aspect('equal')
            path1_ax.legend()

            if i == num_iters:
                path2_ax.plot(X_list[-1][0, :], X_list[-1][1, :], 'r', label='optimized')
                path2_ax.set_title('initial cost {:.1f}\noptimized cost {:.1f}'
                                   .format(float(obj_list[0]), float(obj_list[-1])))

                # show ship footprint along path
                for x_, y_, theta_ in X_list[-1].T[::2]:
                    pts = Rxy(theta_) @ ship_vertices.T + np.array([[x_], [y_]])
                    path2_ax.add_patch(patches.Polygon(pts.T, color='k', fill=False, linewidth=0.3))

                # add text with an offset showing the path index
                # for i, (x_, y_, theta_) in enumerate(X_list[best_iter].T[::10]):
                #     path2_ax.text(x_ + 5, y_, str(i * 10), c='m', fontsize=10)

                # show ship position and heading
                if self.ship_pose is not None:
                    path2_ax.plot(self.ship_pose[0], self.ship_pose[1], 'g.', markersize=10)
                    path2_ax.arrow(self.ship_pose[0], self.ship_pose[1],
                                   10 * np.cos(self.ship_pose[2]), 10 * np.sin(self.ship_pose[2]),
                                   head_width=5, head_length=5, fc='g', ec='g')

            else:
                path2_ax.plot(X_list[i][0, :], X_list[i][1, :], 'r', label='optimized')
                path2_ax.set_title('initial cost {:.1f}\noptimized cost {:.1f}'
                                   .format(float(obj_list[0]), float(obj_list[i])))

            for obs in obstacles:
                path2_ax.add_patch(
                    patches.Polygon(obs['vertices'], True, fill=True, lw=1, ec='k', fc='b', alpha=0.3)
                )
            path2_ax.plot(X_list[0][0, :], X_list[0][1, :], 'k--', label='initial')
            path2_ax.legend()
            path2_ax.set_aspect('equal')

            if anim:
                plt.pause(0.001)
            else:
                break

        # zoom in on plot based on path
        margin = 30
        for ax in [path1_ax, path2_ax]:
            ax.set_xlim(max(0, np.min(X_list[-1][0, :]) - margin),
                        min(self.pos_bounds[0], np.max(X_list[-1][0, :]) + margin))
            ax.set_ylim(max(0, np.min(X_list[-1][1, :]) - margin),
                        min(self.pos_bounds[1], np.max(X_list[-1][1, :]) + margin))
            plt.draw()

        # plot the controls and states
        f2, ax = plt.subplots(4, 1, figsize=(5, 10), sharex=True)
        arc_length = np.linspace(0, path_length, self.N + 1)
        ax[0].plot(arc_length, X_list[-1][0, :])
        ax[0].set_ylabel('x')
        ax[1].plot(arc_length, X_list[-1][1, :])
        ax[1].set_ylabel('y')
        ax[2].plot(arc_length, X_list[-1][2, :])
        ax[2].set_ylabel('yaw')
        ax[3].plot(arc_length[:-1], U_list[-1])
        ax[3].set_ylabel('turn rate')
        ax[3].set_xlabel('arc length (s)')
        f2.suptitle('state and controls')
        plt.tight_layout()

        f3, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
        swath1, swath_cumsum1 = compute_swath_cost(self.costmap, X_list[0].T, ship_vertices, compute_cumsum=True)
        ax[0].plot(swath_cumsum1, label='initial')
        ax[1].plot(np.diff([0, *swath_cumsum1]), label='initial')
        swath2, swath_cumsum2 = compute_swath_cost(self.costmap, X_list[-1].T, ship_vertices, compute_cumsum=True)
        ax[0].plot(swath_cumsum2, label='optimized')
        ax[1].plot(np.diff([0, *swath_cumsum2]), label='optimized')
        ax[0].set_title('swath cost cumulative sum')
        ax[1].set_title('swath cost')
        ax[0].set_xlabel('path waypoint index')
        ax[1].set_xlabel('path waypoint index')
        ax[0].set_ylabel('swath cost cumsum')
        ax[1].set_ylabel('swath cost')
        ax[1].legend()
        plt.tight_layout()

        plt.show()

    def _F(self, x, u):
        """
        Unicycle kinematic model for the dubins car
        dx/ds = F(x,u)
        """
        return ca.vertcat(ca.cos(x[2]), ca.sin(x[2]), u)

    def __call__(self, plot=False, **kwargs):
        """
        Build and run optimization problem with CasADi

        There are many ways to improve speed here

        Potentially useful links:
        - https://groups.google.com/g/casadi-users/c/bAwynN7A1JA

        """
        self.opti = Opti()  # Optimization problem

        # ---- decision variables ---------
        X = self.opti.variable(3, self.N + 1)
        pos = X[:2, :]                                 # x, y positions, shape is 2 x N+1
        heading = X[2, :]                              # heading psi, shape is 1 x N+1
        U = self.opti.variable(1, self.N)        # control input
        L = self.opti.variable(1)                      # path length
        ds = L / self.N

        # ---- objective          ---------
        px = MX.sym('px')
        py = MX.sym('py')
        psi = MX.sym('psi')
        kappa = MX.sym('kappa')  # curvature which is also the control input

        # function that computes the cost at each body point given the position, heading, and curvature
        f = Function('f', [px, py, psi, kappa], [sum1(
            self.lut(
                (vertcat(
                    horzcat(ca.cos(psi), -ca.sin(psi), px),
                    horzcat(ca.sin(psi), ca.cos(psi), py)
                ) @ horzcat(self.body_points, MX.ones(len(self.body_points))).T)[::-1, :]
            ).T
            * self.body_weights
            * ca.hypot(ca.cos(psi)  # dx / ds
                       - ca.sin(psi) * self.body_points[:, 0] * kappa - ca.cos(psi) * self.body_points[:, 1] * kappa,

                       ca.sin(psi)  # dy / ds
                       + ca.cos(psi) * self.body_points[:, 0] * kappa - ca.sin(psi) * self.body_points[:, 1] * kappa)
        )])

        # this speeds up construction and compilation time but not evaluation
        # can do 'serial' or 'thread' for parallel evaluation
        F = f.map(U.shape[1], 'serial')
        total_collision_cost = ds / self.max_ds * sum2(F(pos[0, :-1],
                                                         pos[1, :-1] - self.costmap_offsets[0],
                                                         heading[:-1],
                                                         U))
        obj = L + total_collision_cost

        # ---- position constraints -------
        if self.ship_pose is not None:
            self.opti.subject_to(X[:, 0] == self.ship_pose)  # start at initial ship pose
        else:
            self.opti.subject_to(X[:, 0] == self.warm_start_path[:, 0])

        if self.global_path is not None:
            # path must terminate within some distance of the global path
            self.opti.subject_to(norm_2(pos[:, -1] - self.global_path[:2, -1] + 1e-4) <= self.goal_margin)
        else:
            # path must terminate at the goal
            self.opti.subject_to(norm_2(pos[1, -1] - self.goal + 1e-4) <= self.goal_margin)

        # cannot go outside map boundaries
        # note, we also ensure ship body does not go outside the costmap boundaries using the costmap
        self.opti.subject_to(self.opti.bounded(0, pos, self.pos_bounds))

        # ---- control constraints --------
        self.opti.subject_to(self.opti.bounded(-self.control_bounds, U, self.control_bounds))

        # ---- constraints for the kinematic model --------
        for k in range(self.N):  # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = self._F(X[:, k], U[k])
            k2 = self._F(X[:, k] + ds / 2 * k1, U[k])
            k3 = self._F(X[:, k] + ds / 2 * k2, U[k])
            k4 = self._F(X[:, k] + ds * k3, U[k])
            x_next = X[:, k] + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

            if self.smooth_control_weight > 0 and 0 <= k < self.N - 1:
                # penalize large changes in turn rate
                obj += self.smooth_control_weight * ((U[k + 1] - U[k]) / ds) ** 2

        # ---- set objective --------
        self.opti.minimize(obj)

        # ---- initial values for solver ---
        self.opti.set_initial(U, np.clip(
            compute_turn_rate_control(self.warm_start_path),
            -self.control_bounds,
            self.control_bounds
        ))
        self.opti.set_initial(X, self.warm_start_path)
        self.opti.set_initial(L, self.initial_path_length)

        # ---- solve NLP              ------
        self.opti.solver('ipopt',  # set numerical backend
                         self.p_opts,    # p opts
                         self.s_opts)    # https://coin-or.github.io/Ipopt/OPTIONS.html

        callback = SolverCallback()
        self.opti.callback(lambda i: callback(self.opti.debug.value(X),
                                              self.opti.debug.value(U),
                                              self.opti.debug.value(L),
                                              self.opti.debug.value(total_collision_cost)  # this slows down solver a little bit
                                              ))

        solved = False
        try:
            sol = self.opti.solve()
            solved = True

        finally:
            obj_list = self.opti.debug.stats()['iterations']['obj']
            sol_path = callback.X_list[-1]

            metrics = {
                'initial_obj': {'path length': callback.L_list[0], 'total collision cost': callback.obj_list[0], 'total objective': obj_list[0]},
                'final_obj': {'path length': callback.L_list[-1], 'total collision cost': callback.obj_list[-1], 'total objective': obj_list[-1]},
                'improvement': (callback.L_list[-1] + callback.obj_list[-1]) / (callback.L_list[0] + callback.obj_list[0]) - 1,
                'solved': solved,
            }
            self.logger.info(metrics)

            if plot:
                self.plot_solution_evolution(
                    obj_list=callback.obj_list,  # this is just the total collision cost
                    X_list=callback.X_list,
                    U_list=callback.U_list,
                    path_length=callback.L_list[-1],
                    **kwargs)

            return (
                solved,               # whether the optimization problem was solved
                sol_path,             # the optimized path (if solved, otherwise the last path produced by the solver)
                                      # ideally the path is resampled using the unicycle kinematic equations
                metrics,              # some useful metrics to log
                self.warm_start_path  # the initial path used to warm start the solver
                )
