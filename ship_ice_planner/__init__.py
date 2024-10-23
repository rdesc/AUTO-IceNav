# global vars for dir/file names
DATA_ROOT_DIR = 'data'
EXPERIMENT_ROOT_DIR = 'output'
PLANNER_PLOT_DIR = 'planner_plots'
PATH_DIR = 'paths'
METRICS_FILE = 'metrics.txt'
LOG_FILE = 'log.txt'
DEFAULT_CONFIG_FILE = 'config.yaml'
FULL_SCALE_SIM_EXP_CONFIG = 'data/experiment_configs.pkl'
NRC_OEB_SIM_EXP_CONFIG = 'data/demo_NRC_OEB_data.pkl'
NRC_ICE_SIM_EXP_CONFIG = 'data/demo_NRC_ice_tank_data.pkl'
FULL_SCALE_SIM_PARAM_CONFIG = 'configs/sim2d_config.yaml'
NRC_OEB_SIM_PARAM_CONFIG = 'configs/NRC_OEB_config.yaml'

try:
    import numba as nb  # speeds up some computations
    NUMBA_IS_AVAILABLE = True
except ImportError:
    NUMBA_IS_AVAILABLE = False
