"""Seven microrobots follow an equilibrium point around a 5 cm circle."""

import numpy as np


NUM_ROBOTS = 7
CIRCLE_RADIUS = 0.05
CIRCLE_CENTER = np.array([0.0, 0.0])
NUM_CIRCLE_TARGETS = 8
PATH_STEP_DURATION = 5.0

# Include both 0 and 2*pi so the scheduled route completes the circle and
# returns exactly to its starting point.
_circle_angles = np.linspace(
    0.0,
    2.0 * np.pi,
    NUM_CIRCLE_TARGETS,
)
CIRCLE_POINTS = CIRCLE_CENTER + CIRCLE_RADIUS * np.column_stack((
    np.cos(_circle_angles),
    np.sin(_circle_angles),
))
PATH_POINTS = CIRCLE_POINTS

# Begin with a compact, evenly spaced group around the first target point.
_robot_angles = np.linspace(0.0, 2.0 * np.pi, NUM_ROBOTS, endpoint=False)
INITIAL_ROBOT_POSITIONS = PATH_POINTS[0] + np.column_stack((
    0.006 * np.cos(_robot_angles),
    0.006 * np.sin(_robot_angles),
))

TARGET_SCHEDULE = [
    (
        index * PATH_STEP_DURATION,
        point,
        1.0,
        np.deg2rad(0.0),
    )
    for index, point in enumerate(PATH_POINTS)
]


PARAMS = {
    "TARGET_SCHEDULE": TARGET_SCHEDULE,
    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,

    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + PATH_STEP_DURATION),
    "T_EVAL_POINTS": 900,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "USE_OVERDAMPED_DYNAMICS": True,
    "DYNAMICS_SPEEDUP": 1.0,
    "SOLVER_RTOL": 1e-5,
    "SOLVER_ATOL": 1e-8,

    "ANIMATION_TITLE": "Equilibrium Point Moving on a 5 cm Circle",
    "ANIMATION_DRAW_TRAJECTORIES": False,
    "ANIMATION_DRAW_TARGET_TRAJECTORY": True,
    "ANIMATION_DRAW_TARGET_POINTS": True,
    "PLOT_DRAW_TARGET_PATH": True,

    # Payload remains disabled, matching the copied Case 001 convention.
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
