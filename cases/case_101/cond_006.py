"""Case 100's Path-v4 condition and channel walls scaled together."""

import numpy as np

from cases.case_100 import cond_006 as _source
from .scaling_helpers import build_scaled_condition


PARAMS, SCALE_FACTOR = build_scaled_condition(_source)
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]
INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
PATH_POINTS = np.asarray([entry[1] for entry in TARGET_SCHEDULE])
WALL_SEGMENTS = PARAMS["WALL_SEGMENTS"]
PATH_V4_EDGE_POLYLINES = tuple(points * SCALE_FACTOR for points in _source.PATH_V4_EDGE_POLYLINES)
NUM_ROBOTS = _source.NUM_ROBOTS
