"""
Default configuration values for pymin.
"""

# initialization sentinel
UNSET = object()
# note: None is reserved for returns where no value is set

# paths
CONFIG_PATH = "config"
DEFAULT_CONFIG_PATH = "default_config.yml"

# histogram settings
MIN_EVENTS_DATA = 10
MIN_EVENTS_MC_DIAG = 1000
MIN_EVENTS_MC_OFFDIAG = 2000

HIST_MIN = 80.0
HIST_MAX = 100.0
HIST_BINS = 80
