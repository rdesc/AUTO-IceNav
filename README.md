# Autonomous Ship in Ice Planner
Code for our ICRA 2023 [paper](https://arxiv.org/abs/2302.11601) "Real-Time Navigation for Autonomous Surface Vehicles In Ice-Covered Waters".
Links to [ICRA presentation](https://youtu.be/OAYU6cKKdXU?feature=shared) and [demo video](https://youtu.be/v626IKxXmhQ?feature=shared).

https://github.com/rdesc/Autonomous-Ship-In-Ice/assets/39059473/4b6d276f-bd09-4ad3-9f91-9433c05e381e

## Installation
1. clone the project
```bash
git clone https://github.com/rdesc/Autonomous-Ship-In-Ice
```

2. Create and activate new environment
```bash
conda create --name ship_ice_planner python=3.6
conda activate ship_ice_planner
```

3. Install requirements from "requirements.txt". Note, may need to install the
[dubins](https://github.com/AndrewWalker/pydubins) package manually
(see [here](https://github.com/AndrewWalker/pydubins/issues/16#issuecomment-1138899416) for instructions).
```bash
pip install -r requirements.txt
```

## Usage
The planner is intended to run in a separate python process from the simulator/controller where one process, e.g. 
the physics simulator, is responsible for sending the current goal, ship state, and ice field information to the planner and 
the planner process uses this information to generate a plan which gets sent back to the main process.

#### Configuration File
All the parameters are configured via a yaml file. Sample configuration files are provided in the `config` directory.

#### Physics Simulation
Run the following console command to start the physics simulation and planner in separate processes:
```shell
python sim2d_ship_ice_navigation.py
````
The script will launch the [demo](https://github.com/rdesc/Autonomous-Ship-In-Ice/blob/ea447c9b3489f2782adfd39a4d05c9b05e224c6a/sim2d_ship_ice_navigation.py#L437)
function which will initialize the physics simulation and planner processes with a starting ship state, goal, and ice environment.

To generate empirical data of planner performance in simulation, the following script runs a series of simulation trials in sequence as well
as baseline planners (e.g. straight planner and morphological skeleton planner).
```shell
python -m experiments.sim_exp.py
```

#### Standalone Planner
Alternatively, the planner can generate paths without navigating a ship (either real world or in simulation) via the following console command:
```shell
python -m ship_ice_planner.launch 
```
The script will launch the [demo](https://github.com/rdesc/Autonomous-Ship-In-Ice/blob/ea447c9b3489f2782adfd39a4d05c9b05e224c6a/ship_ice_planner/launch.py#L67) function.
Note, the config parameter `output_dir` needs to be set to a valid directory path for the planner to save the generated paths.

The planner can also be imported as a python module and launched, e.g.:
```python
from ship_ice_planner.launch import launch

config_file = 'configs/no_physics.yaml'  # demo config
queue = Queue()
queue.put(dict(
    goal=(0, 70),
    ship_state=(6, 10, np.pi / 2),
    obstacles=pickle.load(open('data/demo_ice_data.pk', 'rb'))
))

launch(cfg_file=config_file, queue=queue, debug=True, logging=True)
```
By default `debug` mode is set to False and `logging` to console is set to True.

#### Evaluation
To evaluate the results of the simulation trials, run the following script:
```shell
python -m ship_ice_planner.src.evaluation.evaluate_run_sim.py 
```

#### Generate ice field simulation data
To generate ice field simulation data, use the following script:
```shell
python -m experiments.generate_rand_exp.py
```
Example of ice field generated with 0.4 concentration.
<p align="center">
 <img src="https://github.com/rdesc/Autonomous-Ship-In-Ice/blob/ea447c9b3489f2782adfd39a4d05c9b05e224c6a/docs/images/ice_field_concentration_0.4.png" width="640" height="480"> 
</p>
