"""Target points for rotating and translating an elongated cargo."""

import numpy as np

from .cond_003 import PARAMS as COND_003_PARAMS


CARGO_SIZE = np.array([0.060, 0.015])
CARGO_CENTER = np.array([0.0, 0.0])
ROBOT_START_CENTER = np.array([0.0, 0.040])
NUM_ROBOTS = 11
INITIAL_ROBOT_SPACING = 0.012
APPROACH_DELAY = 1.0
TARGET_STEP_DURATION = 8.0
FORMATION_RATIO = 3.0
INFINITY_PATH_TARGETS = 25
INFINITY_X_RADIUS = 0.060
INFINITY_Y_RADIUS = 0.025
INFINITY_INITIAL_WAIT = 4.0
POINT_WAIT_GROWTH = 1.10

# Two spaced rows prevent the strong initial robot-robot repulsion caused by
# the previous compact ellipse. The staggered 6+5 layout supports the odd
# robot count while keeping a minimum center distance of 12 mm.
_row_counts = ((NUM_ROBOTS + 1) // 2, NUM_ROBOTS // 2)
_y_offsets = INITIAL_ROBOT_SPACING * np.array([-0.5, 0.5])
INITIAL_ROBOT_POSITIONS = ROBOT_START_CENTER + np.array([
    [x_offset, y_offset]
    for row_count, y_offset in zip(_row_counts, _y_offsets)
    for x_offset in INITIAL_ROBOT_SPACING * (
        np.arange(row_count) - (row_count - 1) / 2.0
    )
])

# The target angle specifies the strong stiffness axis. Adding pi/2 makes the
# weaker, long formation axis follow the requested cargo direction.
def _target_entry(time, point, cargo_direction):
    return (
        time,
        np.asarray(point, dtype=float),
        FORMATION_RATIO,
        cargo_direction + np.pi / 2.0,
    )


TARGET_SCHEDULE = [
    _target_entry(0.0, ROBOT_START_CENTER, 0.0),
    _target_entry(APPROACH_DELAY, CARGO_CENTER, 0.0),
]

# Rotate the long formation direction through 360 degrees in 30-degree steps.
for step, cargo_direction in enumerate(
    np.linspace(0.0, 2.0 * np.pi, 13)[1:],
    start=1,
):
    TARGET_SCHEDULE.append(_target_entry(
        APPROACH_DELAY + step * TARGET_STEP_DURATION,
        CARGO_CENTER,
        cargo_direction,
    ))

# After the full turn, move to another location along the cargo's long axis.
_rotation_end_time = TARGET_SCHEDULE[-1][0]
for step, point in enumerate(([0.020, 0.0], [0.040, 0.0], [0.060, 0.0]), start=1):
    TARGET_SCHEDULE.append(_target_entry(
        _rotation_end_time + step * TARGET_STEP_DURATION,
        point,
        2.0 * np.pi,
    ))

# Trace a horizontal figure-eight from the rightmost point. The cargo's long
# axis follows the local path tangent. Each point remains active 10% longer
# than the previous one, and a final target returns to the center.
_path_time = TARGET_SCHEDULE[-1][0]
_point_wait = INFINITY_INITIAL_WAIT
_infinity_angles = np.linspace(0.0, 2.0 * np.pi, INFINITY_PATH_TARGETS)
_tangent_vectors = np.column_stack((
    -INFINITY_X_RADIUS * np.sin(_infinity_angles),
    2.0 * INFINITY_Y_RADIUS * np.cos(2.0 * _infinity_angles),
))
_tangent_angles = 0.5 * np.unwrap(
    2.0 * np.arctan2(_tangent_vectors[:, 1], _tangent_vectors[:, 0])
)
for angle, tangent_angle in zip(_infinity_angles, _tangent_angles):
    _path_time += _point_wait
    point = np.array([
        INFINITY_X_RADIUS * np.cos(angle),
        INFINITY_Y_RADIUS * np.sin(2.0 * angle),
    ])
    TARGET_SCHEDULE.append(_target_entry(
        _path_time,
        point,
        tangent_angle,
    ))
    _point_wait *= POINT_WAIT_GROWTH

_path_time += _point_wait
TARGET_SCHEDULE.append(_target_entry(
    _path_time,
    CARGO_CENTER,
    2.0 * np.pi,
))
FINAL_HOLD_DURATION = _point_wait * POINT_WAIT_GROWTH


PARAMS = {
    **COND_003_PARAMS,
    # Restored RK45: Radau was slower in this case.
    "SOLVER_METHOD": "RK45",
    # "SOLVER_METHOD": "Radau",
    # Previous inherited value; higher damping caused very small contact steps.
    # "CONTACT_DAMPING": 5e-5,
    "CONTACT_DAMPING": 1e-5,
    # Previous inherited tolerances; uncomment these and comment the next two
    # lines to restore the more accurate, slower solve.
    # "SOLVER_RTOL": 1e-4,
    # "SOLVER_ATOL": 1e-7,
    "SOLVER_RTOL": 1e-3,
    "SOLVER_ATOL": 1e-6,
    "TARGET_SCHEDULE": TARGET_SCHEDULE,
    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,
    "PAYLOAD_INITIAL_POS": CARGO_CENTER,
    "PAYLOAD_INITIAL_ANGLE": 0.0,
    "PAYLOAD_INITIAL_ANGULAR_VEL": 0.0,
    "PAYLOAD_ANGULAR_DRAG_FACTOR": 1.0,
    "PAYLOAD_SIZE": CARGO_SIZE,
    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + FINAL_HOLD_DURATION),
    "ANIMATION_TITLE": "Coordinated Translation and Rotation of an Elongated Object",
}
