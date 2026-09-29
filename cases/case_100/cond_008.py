"""Carry central cargo on an upper semicircle and lower half-star path."""

import numpy as np

from .cargo_path_helpers import (
    CARGO_CENTER,
    NUM_ROBOTS,
    PATH_STEP_DURATION,
    PICKUP_DURATION,
    ROBOT_START_RADIUS,
    build_cargo_path_params,
)


PATH_RADIUS = 0.055
UPPER_POINT_COUNT = 13

# Travel from the left junction to the right junction over the upper circle.
_upper_angles = np.linspace(np.pi, 0.0, UPPER_POINT_COUNT)
UPPER_CIRCLE_POINTS = PATH_RADIUS * np.column_stack((
    np.cos(_upper_angles),
    np.sin(_upper_angles),
))

# This is the lower boundary of a five-point star. Its two end points are
# shifted onto y=0 and scaled so they join the semicircle without a gap.
_star_angles = np.deg2rad([18.0, -18.0, -54.0, -90.0, -126.0, -162.0, -198.0])
_star_radii = np.array([1.0, 0.382, 1.0, 0.382, 1.0, 0.382, 1.0])
_lower_star = _star_radii[:, None] * np.column_stack((
    np.cos(_star_angles),
    np.sin(_star_angles),
))
_lower_star[:, 1] -= _lower_star[0, 1]
_lower_star[:, 0] *= PATH_RADIUS / _lower_star[0, 0]
_lower_star[:, 1] *= PATH_RADIUS / abs(_lower_star[:, 1].min())
LOWER_HALF_STAR_POINTS = _lower_star

# The upper arc ends on the right; the star section returns to the left.
PATH_POINTS = np.vstack((
    UPPER_CIRCLE_POINTS,
    LOWER_HALF_STAR_POINTS[1:],
))

PARAMS = build_cargo_path_params(
    PATH_POINTS,
    "Central Cargo Transport on a Circle–Star Composite Path",
)

INITIAL_ROBOT_POSITIONS = PARAMS["INITIAL_ROBOT_POSITIONS"]
TARGET_SCHEDULE = PARAMS["TARGET_SCHEDULE"]

