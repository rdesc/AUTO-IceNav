PLANNER_COL     = 'Planning Iteration'
PF_MODE_COL     = 'PF Mode (1)'  # planner flag where 1 = planner, 0 = manual
TIME_COL        = 'Time (s)'
DATE_COL        = 'DateTime'
VEL_COLS        = ['MoES Vel Estimate X (m/s)', 'MoES Vel Estimate Y (m/s)', 'MoES Vel Estimate Z (m/s)', 'MoES W (m/s)']
BODY_VEL_COLS   = ['MoES U (m/s)', 'MoES V (m/s)']
RPS_COLS        = ['M1 RPS (rps)', 'M2 RPS (rps)', 'M3 RPS (rps)', 'M4 RPS (rps)']
SETPOINT_COLS   = ['Setpoint X (m)', 'Setpoint Y (m)', 'Setpoint Yaw (deg)']
ACCELER_COLS    = ['Xsens X Acceleration (g)', 'Xsens Y Acceleration (g)', 'Xsens Z Acceleration (g)']
POSITION_COLS   = ['X (Body 1) (m)', 'Y (Body 1) (m)', 'Z (Body 1) (m)']
ROTATION_COLS   = ['Roll (Body 1) (deg)', 'Pitch (Body 1) (deg)', 'Yaw (Body 1) (deg)']
POSE_COLS = [POSITION_COLS[0], POSITION_COLS[1], ROTATION_COLS[2]]


def load_daq_as_df(daq_file_name):
    """
    Loads daq as a pandas DataFrame

    Timing data is not included by default when loading daq file directly.
    Easiest option is to simply load the corresponding csv file.
    """
    import numpy as np
    import pandas as pd

    if 'csv' in daq_file_name:
        return pd.read_csv(daq_file_name)
    else:
        # Needs the ocre package to be installed. This can be done with:
        # pip install ocre_python_library-1.37.1-py39-none-manylinux_2_17_x86_64.whl --no-deps
        #
        # Make sure to use the --no-deps flag to not mess with conda environment.
        from sweet.tools.dataset import DaqFile

        # Load the daq file
        daq_file = DaqFile(daq_file_name)

        # Returns a sweet.tools.channel.ChannelCollection object, which is a group of
        # sweet.tools.channel.Channel objects.
        channels = daq_file.load()
        daq_file.close()

        return pd.DataFrame(data=np.asarray([channel.y_data for channel in channels.values()]).T,
                            columns=list(channels.keys()))


def generate_csv_from_daq(daq_file_name):
    """
    Generates a csv file from a daq file.
    Handling daq file requires the ocre package from NRC
    """
    from os.path import join, dirname, basename
    from sweet.tools.dataset import DaqFile, CSVChannelFile

    channels = DaqFile(daq_file_name).load()
    rates = set([int(channels[name].get_sampling_rate()) for name in channels])
    for rate in sorted(rates):
        subset = channels.get_subset_by_rate(rate)
        output_file = join(dirname(daq_file_name), f"{basename(daq_file_name).split('.daq')[0]}_{rate}Hz.csv")
        csv_file = CSVChannelFile(output_file, "w")
        csv_file.save(subset, include_absolute=True)
        csv_file.close()
