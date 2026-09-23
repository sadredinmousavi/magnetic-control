"""Eight robots carry rectangular cargo from the center through Path-v4."""

import numpy as np

from .cond_007 import (
    CAD_BOUNDING_BOX as _CAD_BOUNDING_BOX,
    CAD_EDGE_POLYLINES as _CAD_EDGE_POLYLINES,
    FORMATION_DURATION,
    LEFT_PATH_POINTS,
    LOWER_CIRCLE_POINTS,
    PARAMS as COND_007_PARAMS,
    PATH_STEP_DURATION,
    RIGHT_TURN_POINTS,
    UPPER_RIGHT_PATH_POINTS,
    WALL_SEGMENTS as _WALL_SEGMENTS,
)


NUM_ROBOTS = 8
CARGO_SIZE = np.array([0.020, 0.015])
CARGO_CENTER = np.array([0.0, 0.0])
PATH_SCALE = 1.60

# Match Condition 006 by scaling the complete CAD geometry uniformly about
# the origin, keeping the walls and target route aligned.
CAD_BOUNDING_BOX = PATH_SCALE * _CAD_BOUNDING_BOX
CAD_EDGE_POLYLINES = tuple(
    PATH_SCALE * polyline for polyline in _CAD_EDGE_POLYLINES
)
WALL_SEGMENTS = tuple(
    (PATH_SCALE * start, PATH_SCALE * end)
    for start, end in _WALL_SEGMENTS
)

# Begin at the cargo in the center, enter the three-section route, then return
# through the center and leave through the open +X side.
CENTER_TO_PATH_POINTS = np.array([
    [0.0000,  0.0000],
    [-0.0100, 0.0000],
    [-0.0220, -0.0040],
    [-0.0340, -0.0100],
])

PATH_TO_EXIT_POINTS = np.array([
    [-0.0300, -0.0070],
    [-0.0150, -0.0030],
    [ 0.0000,  0.0000],
    [ 0.0200,  0.0000],
    [ 0.0400,  0.0000],
    [ 0.0700,  0.0000],
])

PATH_POINTS = PATH_SCALE * np.vstack((
    CENTER_TO_PATH_POINTS,
    LOWER_CIRCLE_POINTS,
    RIGHT_TURN_POINTS,
    UPPER_RIGHT_PATH_POINTS,
    LEFT_PATH_POINTS[1:],
    PATH_TO_EXIT_POINTS,
))

# Place eight robots around the initially horizontal rectangular cargo.
INITIAL_ROBOT_POSITIONS = CARGO_CENTER + np.array([
    [-0.0070,  0.0080],
    [ 0.0000,  0.0080],
    [ 0.0070,  0.0080],
    [-0.0070, -0.0080],
    [ 0.0000, -0.0080],
    [ 0.0070, -0.0080],
    [-0.0105,  0.0000],
    [ 0.0105,  0.0000],
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
# The supplied angle is the strong stiffness axis; the weak, elongated
# formation axis is therefore tangent to the route.
_stiffness_angles = _path_angles + np.pi / 2

TARGET_SCHEDULE = [
    (0.0, PATH_POINTS[0], _ratios[0], _stiffness_angles[0]),
    *[
        (
            FORMATION_DURATION + index * PATH_STEP_DURATION,
            point,
            _ratios[index + 1],
            _stiffness_angles[index + 1],
        )
        for index, point in enumerate(PATH_POINTS[1:])
    ],
]


PARAMS = {
    **COND_007_PARAMS,
    "TARGET_SCHEDULE": TARGET_SCHEDULE,
    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,
    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + PATH_STEP_DURATION),
    "ANIMATION_TITLE": "Eight Microrobots Carrying Rectangular Cargo Through Path-v4",
    "WALL_SEGMENTS": WALL_SEGMENTS,
    "USE_OVERDAMPED_DYNAMICS": False,
    "DYNAMICS_SPEEDUP": 1.0,
    "SOLVER_METHOD": "RK45",
    "SOLVER_RTOL": 1e-3,
    "SOLVER_ATOL": 1e-6,

    # Match the cargo interaction/dynamics settings from case_001.cond_009;
    # PAYLOAD_SIZE remains the geometry used for contact and rendering.
    "PAYLOAD_RADIUS": 0.015,
    "PAYLOAD_SIZE": CARGO_SIZE,
    "PAYLOAD_HEIGHT": 0.001,
    "PAYLOAD_DENSITY": 50.0,
    "PAYLOAD_DRAG_FACTOR": 50.0,
    "CONTACT_STIFFNESS": 2e-5,
    "CONTACT_DAMPING": 1e-5,
    "PAYLOAD_CAPILLARY_GAIN": 5e-7,
    "PAYLOAD_CAPILLARY_RANGE": 0.007,
    "PAYLOAD_ANGULAR_DRAG_FACTOR": 1.0,
    "PAYLOAD_INITIAL_POS": CARGO_CENTER,
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
    "PAYLOAD_INITIAL_ANGLE": 0.0,
    "PAYLOAD_INITIAL_ANGULAR_VEL": 0.0,
}

PATH_V4_EDGE_POLYLINES = CAD_EDGE_POLYLINES
