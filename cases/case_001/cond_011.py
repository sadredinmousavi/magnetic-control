"""Case 002 condition 5's Path-v3 route, appended to Case 001."""

from cases.case_002 import cond_005 as _source


# Re-export the route and CAD geometry so this condition exposes the same
# path/point helpers as its source condition.
CAD_BOUNDING_BOX = _source.CAD_BOUNDING_BOX
CAD_EDGE_POLYLINES = _source.CAD_EDGE_POLYLINES
WALL_SEGMENTS = _source.WALL_SEGMENTS
ENTRY_PATH_POINTS = _source.ENTRY_PATH_POINTS
UPPER_PATH_POINTS = _source.UPPER_PATH_POINTS
TRANSITION_PATH_POINTS = _source.TRANSITION_PATH_POINTS
LOWER_PATH_POINTS = _source.LOWER_PATH_POINTS
LOWER_PATH_WITH_MIDPOINTS = _source.LOWER_PATH_WITH_MIDPOINTS
PATH_POINTS = _source.PATH_POINTS
TARGET_SCHEDULE = _source.TARGET_SCHEDULE
INITIAL_ROBOT_POSITIONS = _source.INITIAL_ROBOT_POSITIONS
PATH_V3_EDGE_POLYLINES = _source.PATH_V3_EDGE_POLYLINES

# Use Case 001's base physical parameters while preserving all route-specific
# settings from Case 002 condition 5.
PARAMS = dict(_source.PARAMS)
