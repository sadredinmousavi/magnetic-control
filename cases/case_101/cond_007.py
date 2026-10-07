"""Case 100's square cargo path fitted within 50 mm."""

import numpy as np

from cases.case_100 import cond_007 as _source
from .scaling_helpers import build_scaled_condition


PARAMS, SCALE_FACTOR = build_scaled_condition(_source)
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]
INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
PATH_POINTS = _source.PATH_POINTS * SCALE_FACTOR
CARGO_CENTER = _source.CARGO_CENTER * SCALE_FACTOR
SQUARE_HALF_SIDE = _source.SQUARE_HALF_SIDE * SCALE_FACTOR
SQUARE_CORNERS = _source.SQUARE_CORNERS * SCALE_FACTOR
NUM_ROBOTS = _source.NUM_ROBOTS
