"""Pick up a central cargo and transport it around a closed square."""

import numpy as np

from .cargo_path_helpers import (
    CARGO_CENTER,
    NUM_ROBOTS,
    PATH_STEP_DURATION,
    PICKUP_DURATION,
    ROBOT_START_RADIUS,
    build_cargo_path_params,
)


SQUARE_HALF_SIDE = 0.050
POINTS_PER_SIDE = 5
SQUARE_CORNERS = np.array([
    [-SQUARE_HALF_SIDE, -SQUARE_HALF_SIDE],
    [-SQUARE_HALF_SIDE, SQUARE_HALF_SIDE],
    [SQUARE_HALF_SIDE, SQUARE_HALF_SIDE],
    [SQUARE_HALF_SIDE, -SQUARE_HALF_SIDE],
    [-SQUARE_HALF_SIDE, -SQUARE_HALF_SIDE],
])

# Sample each side evenly and retain only one copy of shared corners.
PATH_POINTS = np.vstack([
    np.linspace(start, end, POINTS_PER_SIDE, endpoint=False)
    for start, end in zip(SQUARE_CORNERS[:-1], SQUARE_CORNERS[1:])
])
PATH_POINTS = np.vstack((PATH_POINTS, SQUARE_CORNERS[-1]))

PARAMS = build_cargo_path_params(
    PATH_POINTS,
    "Central Cargo Transport Around a Square",
)

# Public aliases make the pickup geometry easy to inspect and test.
INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]

