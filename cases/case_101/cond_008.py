"""Case 100's circle-star cargo path fitted within 50 mm."""

from cases.case_100 import cond_008 as _source
from .scaling_helpers import build_scaled_condition


PARAMS, SCALE_FACTOR = build_scaled_condition(_source)
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]
INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
PATH_POINTS = _source.PATH_POINTS * SCALE_FACTOR
CARGO_CENTER = _source.CARGO_CENTER * SCALE_FACTOR
PATH_RADIUS = _source.PATH_RADIUS * SCALE_FACTOR
UPPER_CIRCLE_POINTS = _source.UPPER_CIRCLE_POINTS * SCALE_FACTOR
LOWER_HALF_STAR_POINTS = _source.LOWER_HALF_STAR_POINTS * SCALE_FACTOR
NUM_ROBOTS = _source.NUM_ROBOTS
