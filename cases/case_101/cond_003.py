"""Case 100's spiral and S cargo condition fitted within 50 mm."""

import numpy as np

from cases.case_100 import cond_003 as _source
from .scaling_helpers import build_scaled_condition


PARAMS, SCALE_FACTOR = build_scaled_condition(_source)
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]
INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
PATH_POINTS = np.asarray([entry[1] for entry in TARGET_SCHEDULE])
SPIRAL_POINTS = _source.SPIRAL_POINTS * SCALE_FACTOR
S_POINTS = _source.S_POINTS * SCALE_FACTOR
NUM_ROBOTS = _source.NUM_ROBOTS
