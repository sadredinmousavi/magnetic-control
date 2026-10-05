"""Case 100 condition 008's circle-star cargo path, appended to Case 001."""

from copy import deepcopy

from cases.case_100 import cond_008 as _source


NUM_ROBOTS = _source.NUM_ROBOTS
PICKUP_DURATION = _source.PICKUP_DURATION
PATH_STEP_DURATION = _source.PATH_STEP_DURATION
ROBOT_START_RADIUS = _source.ROBOT_START_RADIUS
CARGO_CENTER = _source.CARGO_CENTER.copy()
PATH_RADIUS = _source.PATH_RADIUS
UPPER_POINT_COUNT = _source.UPPER_POINT_COUNT
UPPER_CIRCLE_POINTS = _source.UPPER_CIRCLE_POINTS.copy()
LOWER_HALF_STAR_POINTS = _source.LOWER_HALF_STAR_POINTS.copy()
PATH_POINTS = _source.PATH_POINTS.copy()

# Preserve the pickup and route settings; the loader supplies Case 001's
# base magnetic and microrobot parameters.
PARAMS = deepcopy(_source.PARAMS)
INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]
