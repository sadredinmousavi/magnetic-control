"""Case 002 condition 7's Path-v4 route, appended to Case 001."""

from cases.case_002 import cond_007 as _source


# Re-export the route and CAD geometry so this condition exposes the same
# path/point helpers as its source condition.
CAD_BOUNDING_BOX = _source.CAD_BOUNDING_BOX
CAD_EDGE_POLYLINES = _source.CAD_EDGE_POLYLINES
WALL_SEGMENTS = _source.WALL_SEGMENTS
CENTER_TO_PATH_POINTS = _source.CENTER_TO_PATH_POINTS
LOWER_CIRCLE_POINTS = _source.LOWER_CIRCLE_POINTS
RIGHT_TURN_POINTS = _source.RIGHT_TURN_POINTS
UPPER_RIGHT_PATH_POINTS = _source.UPPER_RIGHT_PATH_POINTS
LEFT_PATH_POINTS = _source.LEFT_PATH_POINTS
PATH_TO_EXIT_POINTS = _source.PATH_TO_EXIT_POINTS
PATH_POINTS = _source.PATH_POINTS
TARGET_SCHEDULE = _source.TARGET_SCHEDULE
INITIAL_ROBOT_POSITIONS = _source.INITIAL_ROBOT_POSITIONS
PATH_V4_EDGE_POLYLINES = _source.PATH_V4_EDGE_POLYLINES

# Use Case 001's base physical parameters while preserving all route-specific
# settings from Case 002 condition 7.
PARAMS = dict(_source.PARAMS)
