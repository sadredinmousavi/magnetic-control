"""Shared calibration points and square sampling for compensator extraction."""

import numpy as np

from .grid_repeatability_helpers import build_scheduled_path


CALIBRATION_POINTS = np.array([
    [0.0, 0.0],
    [0.0, 0.01],
    [0.01, 0.0],
])
# Include both corners and the midpoint for symmetric radial error samples.
POINTS_PER_SIDE = 7


def centered_square_perimeter(side_length):
    """Sample a closed square, retaining shared corners only once."""
    half_side = side_length / 2.0
    corners = np.array([
        [-half_side, -half_side],
        [-half_side, half_side],
        [half_side, half_side],
        [half_side, -half_side],
        [-half_side, -half_side],
    ])
    edge_points = np.vstack([
        np.linspace(start, end, POINTS_PER_SIDE - 1, endpoint=False)
        for start, end in zip(corners[:-1], corners[1:])
    ])
    return np.vstack((edge_points, corners[-1]))


def build_extract_compensator_test(side_length):
    """Start with three axis-calibration targets, then trace one square."""
    square_points = centered_square_perimeter(side_length)
    path_points, target_schedule, params = build_scheduled_path(
        np.vstack((CALIBRATION_POINTS, square_points)),
        f"Extract Compensator: Axes and {side_length:.2f} m Centered Square",
    )
    return square_points, path_points, target_schedule, params
