"""Seven robots leave the center through the Path-v4 CAD opening."""

import numpy as np

from .path_v4_geometry import CAD_BOUNDING_BOX, CAD_EDGE_POLYLINES, WALL_SEGMENTS


NUM_ROBOTS = 7
FORMATION_DURATION = 10.0
PATH_STEP_DURATION = 10.0

# Follow the three Path-v4 sections in their cyclic direction. The lower path
# is the large lower channel between the 36.819 mm and 61.819 mm arcs; it runs
# from left to right, matching the direction shown on the CAD path.
_lower_path_radius = 0.5 * (36.819 + 61.819) * 1e-3
_lower_angles = np.deg2rad(np.linspace(-158.0, -15.0, 17))
LOWER_CIRCLE_POINTS = _lower_path_radius * np.column_stack((
    np.cos(_lower_angles),
    np.sin(_lower_angles),
))

# Turn upward around the right arrowhead before entering the diagonal path.
RIGHT_TURN_POINTS = np.array([
    [0.0500, -0.0080],
    [0.0500,  0.0000],
    [0.0420,  0.0080],
    [0.0351,  0.0125],
])

UPPER_RIGHT_PATH_POINTS = np.linspace(
    [0.0351, 0.0125],
    [-0.0104, 0.0434],
    10,
)[1:]

LEFT_PATH_POINTS = np.array([
    [-0.0104,  0.0434],
    [-0.0210,  0.0410],
    [-0.0298,  0.0360],
    [-0.0320,  0.0290],
    [-0.0340,  0.0180],
    [-0.0370,  0.0060],
    [-0.0395, -0.0047],
    [-0.0430, -0.0100],
])

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

PATH_POINTS = np.vstack((
    CENTER_TO_PATH_POINTS,
    LOWER_CIRCLE_POINTS,
    RIGHT_TURN_POINTS,
    UPPER_RIGHT_PATH_POINTS,
    LEFT_PATH_POINTS[1:],
    PATH_TO_EXIT_POINTS,
))

_robot_angles = np.linspace(0.0, 2.0 * np.pi, NUM_ROBOTS, endpoint=False)
INITIAL_ROBOT_POSITIONS = PATH_POINTS[0] + np.column_stack((
    0.0015 * np.cos(_robot_angles),
    0.0015 * np.sin(_robot_angles),
))

_segment_directions = np.diff(PATH_POINTS, axis=0)
_segment_directions /= np.linalg.norm(_segment_directions, axis=1, keepdims=True)
_path_tangents = np.vstack((
    _segment_directions[0],
    _segment_directions[:-1] + _segment_directions[1:],
    _segment_directions[-1],
))
_path_angles = np.arctan2(_path_tangents[:, 1], _path_tangents[:, 0])

# Start with an elongated formation in the center, then tighten it along the
# three-section route.
_ratios = np.full(len(PATH_POINTS), 3.0)
_ratios[0] = 5.0
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
    "TARGET_SCHEDULE": TARGET_SCHEDULE,
    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,

    "GRID_MIN": -0.3,
    "GRID_MAX": 0.3,
    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + PATH_STEP_DURATION),
    "T_EVAL_POINTS": 600,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "SOLVER_MAX_STEP": 0.05,
    "ROBOT_INTERACTION_SCALE": 0.25,

    "ANIMATION_TITLE": "Seven Microrobots Traversing the Path-v4 Center Exit",
    "ANIMATION_DRAW_TRAJECTORIES": False,
    "ANIMATION_DRAW_TARGET_TRAJECTORY": True,
    "ANIMATION_DRAW_TARGET_POINTS": False,

    "WALL_SEGMENTS": WALL_SEGMENTS,
    "WALL_STIFFNESS": 5e-4,
    "WALL_DAMPING": 5e-6,
    "WALL_INTERACTION_RANGE": 0.0015,
    "WALL_RECOVERY_DEPTH": 0.004,

    # Payload remains disabled for this condition.
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),

    "PLOT_DRAW_TARGET_PATH": True,
}

# Public alias makes the imported source geometry discoverable from cond_007.
PATH_V4_EDGE_POLYLINES = CAD_EDGE_POLYLINES
