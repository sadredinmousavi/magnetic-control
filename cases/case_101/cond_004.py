"""Case 100's elongated-cargo rotation and figure-eight fitted within 50 mm."""

import numpy as np

from cases.case_100 import cond_004 as _source
from .scaling_helpers import build_scaled_condition


PARAMS, SCALE_FACTOR = build_scaled_condition(_source)
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]
INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
PATH_POINTS = np.asarray([entry[1] for entry in TARGET_SCHEDULE])
CARGO_SIZE = _source.CARGO_SIZE * SCALE_FACTOR
CARGO_CENTER = _source.CARGO_CENTER * SCALE_FACTOR
NUM_ROBOTS = _source.NUM_ROBOTS
