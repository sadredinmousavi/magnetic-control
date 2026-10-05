"""Case 100 condition 007's square cargo path, appended to Case 001."""

from copy import deepcopy

from cases.case_100 import cond_007 as _source


NUM_ROBOTS = _source.NUM_ROBOTS
PICKUP_DURATION = _source.PICKUP_DURATION
PATH_STEP_DURATION = _source.PATH_STEP_DURATION
ROBOT_START_RADIUS = _source.ROBOT_START_RADIUS
CARGO_CENTER = _source.CARGO_CENTER.copy()
SQUARE_HALF_SIDE = _source.SQUARE_HALF_SIDE
POINTS_PER_SIDE = _source.POINTS_PER_SIDE
SQUARE_CORNERS = _source.SQUARE_CORNERS.copy()
PATH_POINTS = _source.PATH_POINTS.copy()

# Preserve the pickup and route settings; the loader supplies Case 001's
# base magnetic and microrobot parameters.
PARAMS = deepcopy(_source.PARAMS)
INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]
