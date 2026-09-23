"""Run the Path-v4 rectangular-cargo route with Case 001 magnets/robots."""

from copy import deepcopy

import numpy as np

from cases.case_002 import cond_008 as _source


# Copy the complete Condition 008 geometry and route so this condition can be
# compared using Case 001's stronger/larger microrobot configuration.
NUM_ROBOTS = _source.NUM_ROBOTS
FORMATION_DURATION = _source.FORMATION_DURATION
PATH_STEP_DURATION = _source.PATH_STEP_DURATION
PATH_SCALE = _source.PATH_SCALE

CARGO_SIZE = _source.CARGO_SIZE.copy()
CARGO_SIZE[0] += 0.010
CARGO_CENTER = _source.CARGO_CENTER.copy()
CAD_BOUNDING_BOX = _source.CAD_BOUNDING_BOX.copy()
CAD_EDGE_POLYLINES = tuple(polyline.copy() for polyline in _source.CAD_EDGE_POLYLINES)
WALL_SEGMENTS = tuple(
    (start.copy(), end.copy()) for start, end in _source.WALL_SEGMENTS
)

# Move from the center toward the lower-left entrance along the midpoint
# between the two wall edges, then join the circular channel smoothly.
CENTER_TO_PATH_POINTS = np.array([
    [ 0.0000,  0.0000],
    [-0.0080, -0.0050],
    [-0.0160, -0.0080],
    [-0.0240, -0.0110],
    [-0.0320, -0.0140],
    [-0.0400, -0.0170],
])
LOWER_CIRCLE_POINTS = _source.LOWER_CIRCLE_POINTS.copy()
RIGHT_TURN_POINTS = _source.RIGHT_TURN_POINTS.copy()
UPPER_RIGHT_PATH_POINTS = _source.UPPER_RIGHT_PATH_POINTS.copy()
LEFT_PATH_POINTS = _source.LEFT_PATH_POINTS.copy()
PATH_TO_EXIT_POINTS = _source.PATH_TO_EXIT_POINTS.copy()
PATH_POINTS = np.vstack((
    PATH_SCALE * CENTER_TO_PATH_POINTS,
    _source.PATH_POINTS[len(_source.CENTER_TO_PATH_POINTS):],
))
INITIAL_ROBOT_POSITIONS = CARGO_CENTER + np.array([
    [-0.0100,  0.0080],
    [ 0.0000,  0.0080],
    [ 0.0100,  0.0080],
    [-0.0100, -0.0080],
    [ 0.0000, -0.0080],
    [ 0.0100, -0.0080],
    [-0.0155,  0.0000],
    [ 0.0155,  0.0000],
])

_segment_directions = np.diff(PATH_POINTS, axis=0)
_segment_directions /= np.linalg.norm(_segment_directions, axis=1, keepdims=True)
_path_tangents = np.vstack((
    _segment_directions[0],
    _segment_directions[:-1] + _segment_directions[1:],
    _segment_directions[-1],
))
_path_angles = np.arctan2(_path_tangents[:, 1], _path_tangents[:, 0])
_ratios = np.full(len(PATH_POINTS), 3.0)
_ratios[0] = 5.0
_stiffness_angles = _path_angles + np.pi / 2
EXIT_ALIGNMENT_HOLD_DURATION = 10.0
_exit_alignment_index = len(PATH_POINTS) - len(PATH_TO_EXIT_POINTS)
_exit_alignment_direction = (
    PATH_POINTS[_exit_alignment_index + 1]
    - PATH_POINTS[_exit_alignment_index]
)
_exit_alignment_stiffness_angle = (
    np.arctan2(_exit_alignment_direction[1], _exit_alignment_direction[0])
    + np.pi / 2.0
)

TARGET_SCHEDULE = [
    (0.0, PATH_POINTS[0], _ratios[0], _stiffness_angles[0]),
]
_schedule_time = 0.0
for index, point in enumerate(PATH_POINTS[1:], start=1):
    _schedule_time += FORMATION_DURATION if index == 1 else PATH_STEP_DURATION
    TARGET_SCHEDULE.append((
        _schedule_time,
        point,
        _ratios[index],
        _stiffness_angles[index],
    ))
    if index == _exit_alignment_index:
        # Hold at the first left-side approach point for one extra interval so
        # the weak/long formation axis aligns with the tangent toward center.
        _schedule_time += EXIT_ALIGNMENT_HOLD_DURATION
        TARGET_SCHEDULE.append((
            _schedule_time,
            point.copy(),
            _ratios[index],
            _exit_alignment_stiffness_angle,
        ))

PARAMS = deepcopy(_source.PARAMS)
PARAMS.update({
    "TARGET_SCHEDULE": TARGET_SCHEDULE,
    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,
    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + PATH_STEP_DURATION),
    "WALL_SEGMENTS": WALL_SEGMENTS,
    "PAYLOAD_SIZE": CARGO_SIZE,
    "PAYLOAD_INITIAL_POS": CARGO_CENTER,
    "ANIMATION_TITLE": (
        "Orientation Aware Swarm control for cargo manipulation"
    ),
})

PATH_V4_EDGE_POLYLINES = CAD_EDGE_POLYLINES
