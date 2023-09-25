"""Dynamic Positioning (DP) demo with no physics simulation"""
import pickle
from multiprocessing import Process, Pipe, Queue
from queue import Empty

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from ship_ice_planner.launch import launch
from ship_ice_planner.src.controller.dp import DP
from ship_ice_planner.src.evaluation.metrics import tracking_error
from ship_ice_planner.src.utils.utils import DotDict

# Simulation params
SHOW_ANIMATION = True
UPDATE_FREQUENCY_HZ = 2  # frequency at which to request a new path and update sim plot
T = np.inf  # max simulation steps
START = (6, 5, np.pi / 2)
GOAL = [6, 72]
OBSTACLES = pickle.load(open('data/demo_ice_data.pk', 'rb'))

# Planner params
CFG = DotDict.load_from_file('configs/no_physics.yaml')
CFG.planner = 'lattice'  # 'lattice', 'skeleton', or 'straight'
DEBUG = False
LOGGING = False


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc='r', ec='k', **kwargs):
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, **kwargs)
        plt.plot(x, y, **kwargs)


def main():
    # multiprocessing setup
    queue = Queue(maxsize=1)  # LIFO queue to send state information to A*
    conn_recv, conn_send = Pipe(duplex=False)  # pipe to send new path to controller and for plotting

    planner = Process(target=launch,
                      # args=(,),
                      kwargs=dict(cfg=CFG, debug=DEBUG, logging=LOGGING,
                                  pipe=conn_send, queue=queue))
    planner.start()

    # send first message
    queue.put(dict(
        goal=GOAL,
        ship_state=START,
        obstacles=OBSTACLES
    ))

    # get path
    path = conn_recv.recv()
    path = np.asarray(path)
    cx = path.T[0]
    cy = path.T[1]
    ch = path.T[2]

    steps = 0
    # setup controller
    dp = DP(x=START[0], y=START[1], yaw=START[2],
            cx=cx, cy=cy, ch=ch, **CFG.controller)
    dp.log_step()
    state = dp.state

    try:
        while T >= steps and state.y < GOAL[1]:
            steps += 1
            if steps % (50 / UPDATE_FREQUENCY_HZ) == 0:
                # empty queue to ensure latest state data is pushed
                try:
                    queue.get_nowait()
                except Empty:
                    pass
                queue.put(dict(
                    ship_state=(state.x, state.y, state.yaw),
                    obstacles=OBSTACLES
                ))

                # check for path
                if conn_recv.poll():
                    path = conn_recv.recv()
                    path = np.asarray(path)
                    cx = path.T[0]
                    cy = path.T[1]
                    ch = path.T[2]
                    dp.target_course.update(cx, cy, ch)

            # update DP controller
            dp(state.x, state.y, state.yaw)

            # integrate based on updated control signals
            state.update_pose(*state.integrate())

            # log updates
            dp.log_step()

            # update setpoint
            x_s, y_s, h_s = dp.get_setpoint()
            dp.setpoint = np.asarray([x_s, y_s, np.unwrap([state.yaw, h_s])[1]])

            if SHOW_ANIMATION and steps % (50 / UPDATE_FREQUENCY_HZ) == 0:  # pragma: no cover
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                plot_arrow(state.x, state.y, state.yaw, zorder=3)
                plt.plot(cx, cy, '-r', label='course', zorder=1)
                states = dp.get_state_history()
                plt.plot(states['x'], states['y'], '-b', label='trajectory', zorder=2)
                plt.plot(x_s, y_s, 'xg', label='target', zorder=3)
                plot_arrow(x_s, y_s, h_s, fc='g', ec='g', zorder=3)
                ax = plt.gca()
                for ob in OBSTACLES:
                    ax.add_patch(patches.Polygon(ob, True, fill=False))
                ax.set_aspect('equal')
                plt.grid(True)
                plt.title('yaw rate {:.4f} [deg/s], surge {:.4f} [m/s], sway {:.4f} [m/s]'
                          .format(state.r, state.u, state.v))
                plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
                plt.pause(0.001)

    finally:
        print('Done DP demo! steps', steps)

        plt.cla()
        states = np.asarray(dp.get_state_history().to_numpy()).T
        t, x, y, yaw, r, u, v, F_r, F_u, F_v, *_ = states
        plt.plot(cx, cy, '.r', label='course')
        plt.plot(x, y, '-b', label='trajectory')
        plt.axis('equal')
        plt.legend()
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.grid(True)

        f, ax = plt.subplots(9, 1, sharex='all', figsize=(5, 12))
        ax[0].plot(t, x)
        ax[0].set_title('x')
        ax[1].plot(t, y)
        ax[1].set_title('y')
        ax[2].plot(t, yaw)
        ax[2].set_title('yaw')
        ax[3].plot(t, r)
        ax[3].set_title('yaw rate')
        ax[4].plot(t, u)
        ax[4].set_title('surge velocity')
        ax[5].plot(t, v)
        ax[5].set_title('sway velocity')
        ax[6].plot(t, F_r)
        ax[6].set_title('yaw rate force')
        ax[7].plot(t, F_u)
        ax[7].set_title('surge force')
        ax[8].plot(t, F_v)
        ax[8].set_title('sway force')

        f.tight_layout()

        track_error = np.asarray([tracking_error(pose, path) for pose in np.asarray([x, y, yaw]).T[::10]])
        print('Average tracking error {:.4f} m'.format(np.hypot(track_error[:, 0], track_error[:, 1]).mean()))

        plt.show()

        queue.put(None)  # sends signal to shutdown process
        planner.join()


if __name__ == '__main__':
    print('Dynamic positioning path tracking simulation start')
    main()
