# Notes about running NRC experiments

## Prerequisites
Run the following command from [NRC_wrapper](../ship_ice_planner/NRC_wrapper) to install the required package:
```shell
pip install ocre_python_library-1.37.1-py39-none-manylinux_2_17_x86_64.whl --no-deps
```

Test to make sure the socket server and client demo works:
```shell
python -m ship_ice_planner.NRC_wrapper.socket_communication
```

## Relevant scripts
**NRC_wrapper/**
- [socket_communication.py](../ship_ice_planner/NRC_wrapper/socket_communication.py):
  socket-based communication interface to send and receive data.
- [nrc_config.py](../ship_ice_planner/NRC_wrapper/nrc_config.py):
  configuration file which specifies the following:
  - coordinate transformations between the NRC setup and 
    the coordinate system used in the planner code
  - coordinates for the four corner points that define the boundary of the ice field
  - buffer distance for the goal region
  - homography matrix for the camera
- [daq_columns.py](../ship_ice_planner/NRC_wrapper/daq_columns.py): 
  convenient global variables for the relevant columns of the daq data and
  helpers to load daq as a pandas dataframe
- [planner_message.proto](../ship_ice_planner/NRC_wrapper/planner_message.proto):
  protocol buffer definition file that defines sent and received message structure
- [comm.py](../ship_ice_planner/NRC_wrapper/comm.py):
  NRC code for communication interface 

**evaluation/**
- [timing_file_parser.py](../ship_ice_planner/evaluation/timing_file_parser.py): 
  very convenient script to parse the timing files containing the ice segmentation data
- [process_daq_data.py](../ship_ice_planner/evaluation/process_daq_data.py):
  processes the data logged by GDAC 
- [evaluate_ice_data.py](../ship_ice_planner/evaluation/evaluate_ice_data.py):
  does tracking of the ice pieces to derive useful performance metrics such as
  net work done on the ice by the ship
- [evaluate_run_nov_2023_oeb_tank.py](../ship_ice_planner/evaluation/evaluate_run_nov_2023_oeb_tank.py)
  script to evaluate the data from the November 2023 NRC experiments.
  For each trial, the following steps are performed:
  - Compute relevant metrics
  - Generate time series plots of sensor and thruster data
  - Overlay ice segmentation and path data on overhead camera footage

**image_process/**
- [process_overhead_video.py](../ship_ice_planner/image_process/process_overhead_video.py):
  script which generates the pretty videos with the overhead camera footage, ice segmentation, and path data

## Important setup steps
- A sufficient number of ice pieces should be placed in the tank to ensure 
  the ice provides resistance to the ship. Alternatively, the ice pieces should have sufficient mass
- Calibration of the NRC setup which includes:
  - Camera calibration and homography matrix computation
  - Coordinate transformation between the NRC setup and the planner code
  - Calibration trials recording IMU readings during a ship-ice collision
    (e.g. collision with an ice piece at varying speeds and with varying head-on collision angles).
    This is crucial for computing meaningful collision metrics from acceleration measurements.
- Set up a trigger signal or message when the trial has started that is logged in the DAQ data
- Set up a naming scheme to identify the trials across the different data logging sources.
  One option is to create a shared [spreadsheet in google sheets](https://docs.google.com/spreadsheets/d/1lUalqfpz-LGA-F_wfFPRMCLfTSgBt3ryVDntHUWD858/edit?usp=sharing)
  where each row is a trial which contains the following columns: 
  trial ID, method, start position, trial number, comments.
- Timestamps need to be recorded in the planner code to easily match the DAQ data for post-processing
- An ideal number of trials per method is >= 20 trials
- Ensure the planner code saves the raw message data that is sent and received
- Ensure the settings are set properly in 
  [socket_communication.py](../ship_ice_planner/NRC_wrapper/socket_communication.py)
  ```python
  _HOST = '192.168.58.250'
  _PORT = 30002
  ```

## Running the NRC experiments
In the current setup, the planner code is run on a separate computer from the controller and 
vision system. Prior to launching the planner for a new trial, the following parameters should be
set in the configuration file:
```yaml
exp_name: OEB_2023               # directory structure for output files is output/EXP_NAME/PLANNER/TRIAL_ID
                                 # exp_name can be used to group trials together
output_dir: AUTO-IceNav-trial01  # trial_id is set by OUTPUT_DIR (set this for every trial!)
planner: 'lattice'               # path planner options are 'skeleton', 'straight', or 'lattice'
                                 # AUTO-IceNav uses 'lattice' planner
comments: ''                     # comments about trial being performed
```

Once the parameters are updated, the planner can be launched with the configuration file:
```bash
python -m ship_ice_planner.launch --config configs/NRC_OEB_config.yaml --logging
```
This will start the socket server which will listen for the initial message from the NRC setup.
The planner expects to receive the ship position, ship orientation, ice segmentation data, and relevant metadata.

## Testing in Simulation
In preparation for the NRC experiments, it is crucial to test the planner code in simulation!

To do so, update the simulation parameters in [sim_utils.py](../ship_ice_planner/utils/sim_utils.py) to match the NRC setup:
```python
ICE_DENSITY = 991.
WATER_DENSITY = 1000.
ICE_THICKNESS = 0.012
```

Next, launch the simulator with ice data from a previous NRC trial:
```bash
python demo_sim2d_ship_ice_navigation.py \
data/demo_NRC_OEB_data.pkl \
configs/NRC_OEB_config.yaml \
--output_dir output/oeb_test01 \
--start 3. 2. 1.57 \
--goal 3. 20.
```

## Data from previous NRC experiments
Contact [Rodrigue de Schaetzen](https://rdesc.dev/).