"""Dynamic Positioning (DP) demo with no physics simulation"""
import pickle
from multiprocessing import Process, Pipe, Queue
from queue import Empty

import matplotlib.pyplot as plt
import numpy as np

from ship_ice_planner import *
from ship_ice_planner.evaluation.evaluate_run_sim import control_vs_time_plot, state_vs_time_plot, tracking_error_plot
from ship_ice_planner.launch import launch
from ship_ice_planner.controller.sim_dynamics import SimShipDynamics
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import DotDict

# Simulation params
SHOW_ANIMATION = True
INF_STREAM = True          # simulation view moves with ship along ice field
SHOW_ICE = True
UPDATE_FREQUENCY_HZ = 2    # frequency at which to update sim plot
                           # increasing means more frequent plot updates between sim updates
T = np.inf                 # max simulation steps
FINAL_PLOT_DATA_STEP = 10  # plot every nth data point
DEBUG = False              # option to enable planner debugging
LOGGING = False            # option to enable planner logging

# #############################################################################
# full scale parameters
CFG = DotDict.load_from_file(FULL_SCALE_SIM_PARAM_CONFIG)
CFG.output_dir = None      # option to store trial data to disk
CFG.planner = 'lattice'    # 'lattice', 'skeleton', or 'straight'
CFG.plot.show = False      # option to enable planner plotting
# CFG.max_replan = 1         # comment out to have no limit on replans
START = (100, 0, np.pi / 2)  # (x, y, psi)
GOAL = [100, 1100]  # (x, y)
exp_data = pickle.load(open(FULL_SCALE_SIM_EXP_CONFIG, 'rb'))['exp']
ice_concentration = 0.5
ice_field_idx = 1
obstacles = exp_data[ice_concentration][ice_field_idx]['obstacles']
OBSTACLES = [ob['vertices'] for ob in obstacles]
FLOE_MASSES = [ob['mass'] for ob in obstacles]

# #############################################################################
# model (NRC) scale parameters
# CFG = DotDict.load_from_file(NRC_OEB_SIM_PARAM_CONFIG)
# CFG.output_dir = None      # option to store trial data to disk
# CFG.planner = 'lattice'    # 'lattice', 'skeleton', or 'straight'
# CFG.plot.show = False      # option to enable planner plotting
# CFG.optim = False
# CFG.max_replan = 1         # comment out to have no limit on replans
# CFG.horizon = 70
# START = (6, 2, np.pi / 2)
# GOAL = [6, 70]
# CFG.map_shape = (76, 12)
# CFG.plot.y_axis_limit = None
# obstacles = pickle.load(open(NRC_ICE_SIM_EXP_CONFIG, 'rb'))
# OBSTACLES = [ob['vertices'] for ob in obstacles]
# FLOE_MASSES = [ob['mass'] for ob in obstacles]

#############################################################################


def main():
    # multiprocessing setup
    queue = Queue(maxsize=1)  # LIFO queue to send state information to planner
    conn_recv, conn_send = Pipe(duplex=False)  # pipe to send new path to controller and for plotting

    planner = Process(target=launch,
                      # args=(,),
                      kwargs=dict(cfg=CFG, debug=DEBUG, logging=LOGGING,
                                  pipe=conn_send, queue=queue))
    planner.start()

    # send first message
    queue.put(dict(
        goal=[GOAL[0],
              GOAL[1] + CFG.get('goal_offset', 0)],  # option to offset goal
        ship_state=START,
        obstacles=OBSTACLES,
        masses=FLOE_MASSES
    ))

    # get path
    path = conn_recv.recv()
    path = np.asarray(path)

    # setup controller and simulated ship dynamics
    sim_dynamics = SimShipDynamics(
        eta=START, nu=[0, 0, 0],
        output_dir=CFG.output_dir,
        **CFG.sim_dynamics
    )
    sim_dynamics.init_trajectory_tracking(path)
    state = sim_dynamics.state

    # setup plotting for animation
    plot = None
    running = True
    if SHOW_ANIMATION:
        if SHOW_ICE:
            show_obs = OBSTACLES
        else:
            show_obs = []
        plot = Plot(obstacles=show_obs, path=path.T, legend=True, track_fps=True, y_axis_limit=CFG.plot.y_axis_limit,
                    ship_vertices=CFG.ship.vertices, target=sim_dynamics.setpoint[:2], inf_stream=INF_STREAM,
                    ship_pos=state.eta, map_figsize=None, sim_figsize=(10, 10), remove_sim_ticks=False, goal=GOAL[1],
                    map_shape=CFG.map_shape)

        def on_close(event):
            nonlocal running
            if event.key == 'escape':
                running = False

        plot.sim_fig.canvas.mpl_connect('key_press_event', on_close)

    # for plotting ship path
    ship_actual_path = ([], [])  # list for x and y
    steps = 0

    try:
        while T >= steps and state.y < GOAL[1] and running:
            steps += 1
            if planner.is_alive():
                if sim_dynamics.check_trigger_replan():
                    # empty queue to ensure latest state data is pushed
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass
                    queue.put(dict(
                        ship_state=(state.x, state.y, state.psi),
                        obstacles=OBSTACLES,
                        masses=FLOE_MASSES
                    ))

                # check for path
                if conn_recv.poll():
                    path = conn_recv.recv()
                    path = np.asarray(path)
                    sim_dynamics.setpoint_generator.replan_update(
                        state.get_vessel_speed(), (state.x, state.y), path
                    )

            # update controller
            sim_dynamics.control()

            # store simulation data
            sim_dynamics.log_step()

            # simulate ship dynamics
            sim_dynamics.sim_step()
            state.integrate()

            if SHOW_ANIMATION and steps % (1 / CFG.sim_dynamics.dt / UPDATE_FREQUENCY_HZ) == 0:
                ship_actual_path[0].append(state.x)
                ship_actual_path[1].append(state.y)

                plot.update_path(path.T, ship_state=ship_actual_path, target=sim_dynamics.setpoint[:2])
                plot.update_ship(CFG.ship.vertices, state.x, state.y, state.psi)

                fps = plot.update_fps()

                plot.title_text.set_text(
                    f'FPS: {fps:.0f}, '
                    f'Real time speed: {fps / UPDATE_FREQUENCY_HZ:.1f}x, '
                    f'Time: {sim_dynamics.sim_time:.1f} s\n'
                    f'surge {state.u:.2f} (m/s), '
                    f'sway {-state.v:.2f} (m/s), '                    # in body frame, positive sway is to the right
                    f'yaw rate {-state.r * 180 / np.pi:.2f} (deg/s)'  # in body frame, positive yaw is clockwise
                    # in the sim, these are reversed! so for plotting purposes flip the sign
                )

                plot.animate_sim()

    except KeyboardInterrupt:
        print('Received keyboard interrupt, exiting...')

    finally:
        print('Done DP demo! steps', steps)

        conn_recv.close()
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put(None)
        queue.close()

        if plot is not None:
            plot.close()

        sim_data = sim_dynamics.get_state_history().iloc[::FINAL_PLOT_DATA_STEP]

        # generate plots
        if CFG.get('max_replan') == 1:
            # only show tracking error figure if there was only one planning iteration
            tracking_error_plot(sim_data, path, CFG.map_shape)
        state_vs_time_plot(sim_data)
        control_vs_time_plot(sim_data,
                             dim_U=len(sim_dynamics.state.u_control),
                             control_labels=sim_dynamics.vessel_model.controls
                             )
        plt.show()

        print('Clean exiting planner process...')
        planner.terminate()
        if planner.is_alive():
            planner.join(timeout=2)

        print('Done')


if __name__ == '__main__':
    print('Launching dynamic positioning demo with ice physics turned off...')
    main()
