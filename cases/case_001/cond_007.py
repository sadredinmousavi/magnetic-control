"""Twenty microrobots carry a central payload along a spiral and capital S."""

import numpy as np


NUM_ROBOTS = 7
PAYLOAD_CENTER = np.array([0.0, 0.0])
PICKUP_DURATION = 24.0
PATH_STEP_DURATION = 10.0

# Place all 20 robots on a ring away from the circular payload so the pickup
# stage visibly gathers them toward the center.
_robot_angles = np.linspace(0.0, 2.0 * np.pi, NUM_ROBOTS, endpoint=False)
INITIAL_ROBOT_POSITIONS = np.column_stack((
    0.040 * np.cos(_robot_angles),
    0.040 * np.sin(_robot_angles),
))

# Two-turn, center-out spiral. The first point is omitted from the movement
# schedule because the pickup stage already holds the target at the center.
_spiral_angles = np.linspace(0.0, 4.0 * np.pi, 21)
_spiral_radii = np.linspace(0.0, 0.080, len(_spiral_angles))
SPIRAL_POINTS = np.column_stack((
    _spiral_radii * np.cos(_spiral_angles),
    _spiral_radii * np.sin(_spiral_angles),
))

# A block-capital English S, drawn continuously from upper-right to lower-left.
S_POINTS = np.array([
    [0.055, 0.065], [0.035, 0.075], [0.005, 0.078],
    [-0.025, 0.073], [-0.050, 0.058], [-0.060, 0.038],
    [-0.052, 0.018], [-0.030, 0.005], [0.000, 0.000],
    [0.030, -0.005], [0.052, -0.018], [0.060, -0.038],
    [0.050, -0.058], [0.025, -0.073], [-0.005, -0.078],
    [-0.035, -0.075], [-0.055, -0.065],
])

_movement_points = np.vstack((SPIRAL_POINTS[1:], S_POINTS, PAYLOAD_CENTER))
TARGET_SCHEDULE = [
    (0.0, PAYLOAD_CENTER, 1.0, np.deg2rad(0.0)),
    *[
        (
            PICKUP_DURATION + index * PATH_STEP_DURATION,
            point,
            1.0,
            np.deg2rad(0.0),
        )
        for index, point in enumerate(_movement_points)
    ],
]


PARAMS = {
    "TARGET_SCHEDULE": TARGET_SCHEDULE,
    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,

    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + PATH_STEP_DURATION),
    "T_EVAL_POINTS": 1000,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "USE_OVERDAMPED_DYNAMICS": True,
    "DYNAMICS_SPEEDUP": 1.0,
    "SOLVER_RTOL": 1e-4,
    "SOLVER_ATOL": 1e-7,

    "ANIMATION_TITLE": "Payload Manipulation",
    "ANIMATION_DRAW_TRAJECTORIES": False,
    "ANIMATION_DRAW_TARGET_TRAJECTORY": True,

    "PAYLOAD_RADIUS": 0.025,
    "PAYLOAD_HEIGHT": 0.001,
    "PAYLOAD_DENSITY": 50,
    "PAYLOAD_DRAG_FACTOR": 50,
    "CONTACT_STIFFNESS": 2e-5,
    "CONTACT_DAMPING": 5e-5,
    "PAYLOAD_CAPILLARY_GAIN": 5e-7,
    "PAYLOAD_CAPILLARY_RANGE": 0.007,
    "PAYLOAD_INITIAL_POS": PAYLOAD_CENTER,
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
