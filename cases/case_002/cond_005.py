"""Three robots traverse the supplied Path-v3 top-view CAD geometry."""

import numpy as np

from .path_v3_geometry import CAD_BOUNDING_BOX, CAD_EDGE_POLYLINES, WALL_SEGMENTS


NUM_ROBOTS = 3
FORMATION_DURATION = 10.0
PATH_STEP_DURATION = 10.0

# The JSON stores original CAD X/Z coordinates in millimetres. The geometry
# loader converts them to simulation metres without changing their orientation.
WORKSPACE_RADIUS = float(np.max(np.abs(CAD_BOUNDING_BOX)))

# The CAD export contains boundaries only, so this moving-target centerline is
# derived from the middle of the corresponding upper and lower corridors.
_upper_outer_radius = 53.157467466338716e-3
_upper_inner_radius = np.hypot(6.0, 40.717774107172545) * 1e-3
_upper_path_radius = 0.5 * (_upper_outer_radius + _upper_inner_radius)
_upper_start_angle = np.arctan2(2.2284225692326913, 41.097095531140226)
_upper_end_angle = np.arctan2(15.0, -38.32671559425268)
_upper_angles = np.linspace(_upper_start_angle, _upper_end_angle, 15)
UPPER_PATH_POINTS = _upper_path_radius * np.column_stack((
    np.cos(_upper_angles),
    np.sin(_upper_angles),
))[::-1]

# Midpoints between the paired CAD polylines forming the lower zigzag.
_lower_outer = np.array([
    [37.064889957586075, -9.771577430767287],
    [37.064889957586075, -49.01510136555551],
    [20.2519186454263, -32.20213005339571],
    [2.0311658818421137, -50.42288281697989],
    [-10.932451658150502, -37.45926527698727],
    [-25.8077547339946, -52.33456835283138],
    [-40.36055105329546, -37.781772033530515],
    [-52.89175457485157, -37.781772033530515],
])
_lower_inner = np.array([
    [25.06488995758608, -6.0],
    [25.06488995758608, -20.044538617078352],
    [20.251918645426308, -15.231567304918581],
    [2.0311658818421243, -33.45232006850276],
    [-10.932451658150502, -20.488702528510135],
    [-25.807754733994603, -35.36400560435423],
    [-35.38998830481832, -25.78177203353052],
    [-59.66825144757523, -25.78177203353052],
])
LOWER_PATH_POINTS = 0.5e-3 * (_lower_outer + _lower_inner)

# Pass through the open right-hand junction between the two CAD corridors.
TRANSITION_PATH_POINTS = np.array([
    [47.10e-3, -3.77e-3],
    [39.00e-3, -3.77e-3],
])

PATH_POINTS = np.vstack((
    UPPER_PATH_POINTS,
    TRANSITION_PATH_POINTS,
    LOWER_PATH_POINTS,
))

_robot_angles = np.linspace(0.0, 2.0 * np.pi, NUM_ROBOTS, endpoint=False)
INITIAL_ROBOT_POSITIONS = PATH_POINTS[0] + np.column_stack((
    0.0015 * np.cos(_robot_angles),
    0.0015 * np.sin(_robot_angles),
))

TARGET_SCHEDULE = [
    (0.0, PATH_POINTS[0], 1.0, 0.0),
    *[
        (
            FORMATION_DURATION + index * PATH_STEP_DURATION,
            point,
            1.0,
            0.0,
        )
        for index, point in enumerate(PATH_POINTS[1:])
    ],
]


PARAMS = {
    "TARGET_SCHEDULE": TARGET_SCHEDULE,
    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,

    "DISH_RADIUS": WORKSPACE_RADIUS,
    "GRID_MIN": -0.3,
    "GRID_MAX": 0.3,
    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + PATH_STEP_DURATION),
    "T_EVAL_POINTS": 600,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "SOLVER_MAX_STEP": 0.05,

    "ANIMATION_TITLE": "Swarm Control Through Path-v3 CAD Geometry",
    "ANIMATION_DRAW_TRAJECTORIES": False,
    "ANIMATION_DRAW_TARGET_TRAJECTORY": True,

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

# Public alias makes the imported source geometry discoverable from cond_005.
PATH_V3_EDGE_POLYLINES = CAD_EDGE_POLYLINES
