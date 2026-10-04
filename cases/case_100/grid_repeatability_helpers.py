"""Helpers for square-grid positioning repeatability tests."""

import numpy as np


SQUARE_SIDE_LENGTH = 0.12
REPEAT_COUNT = 3
POINT_HOLD_DURATION = 3.0


def build_serpentine_grid(grid_size, side_length=SQUARE_SIDE_LENGTH):
    """Return every point in a centered square, ordered row by row."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")

    half_side = side_length / 2.0
    coordinates = np.linspace(-half_side, half_side, grid_size)
    rows = []
    for row_index, y_coordinate in enumerate(coordinates):
        x_coordinates = coordinates if row_index % 2 == 0 else coordinates[::-1]
        rows.extend((x_coordinate, y_coordinate) for x_coordinate in x_coordinates)
    return np.asarray(rows, dtype=float)


def build_scheduled_path(path_points, title):
    """Build condition parameters for an already ordered target path."""
    path_points = np.asarray(path_points, dtype=float)
    if path_points.ndim != 2 or path_points.shape[1] != 2 or len(path_points) < 2:
        raise ValueError("path_points must contain at least two 2D points.")
    target_schedule = [
        (
            index * POINT_HOLD_DURATION,
            point.copy(),
            1.0,
            np.deg2rad(0.0),
        )
        for index, point in enumerate(path_points)
    ]

    params = {
        "TARGET_SCHEDULE": target_schedule,
        "INITIAL_ROBOT_POSITIONS": path_points[:1].copy(),
        "SEQUENCE_WAIT": POINT_HOLD_DURATION,

        "T_SPAN": (0.0, len(path_points) * POINT_HOLD_DURATION),
        "T_EVAL_POINTS": max(300, len(path_points) * 15),
        "SOLVER_PROGRESS_INTERVAL": 0.5,
        "USE_OVERDAMPED_DYNAMICS": True,
        "DYNAMICS_SPEEDUP": 1.0,
        "SOLVER_RTOL": 1e-5,
        "SOLVER_ATOL": 1e-8,

        "ANIMATION_TITLE": title,
        "ANIMATION_DRAW_TRAJECTORIES": False,
        "ANIMATION_DRAW_TARGET_TRAJECTORY": True,
        "ANIMATION_DRAW_TARGET_POINTS": True,
        "PLOT_DRAW_TARGET_PATH": True,

        # No payload is used for positioning repeatability measurements.
        "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
        "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
    }
    return path_points, target_schedule, params


def build_path_test(grid_points, title):
    """Build a three-pass repeatability condition from ordered grid points."""
    grid_points = np.asarray(grid_points, dtype=float)
    if grid_points.ndim != 2 or grid_points.shape[1] != 2 or len(grid_points) < 2:
        raise ValueError("grid_points must contain at least two 2D points.")

    # Reverse direction after each pass. Consecutive passes share their end
    # point, avoiding a long repositioning move across the square. Every
    # nonzero move is therefore only one grid spacing.
    passes = [
        (grid_points if repeat_index % 2 == 0 else grid_points[::-1]).copy()
        for repeat_index in range(REPEAT_COUNT)
    ]
    path_points, target_schedule, params = build_scheduled_path(
        np.vstack(passes), title
    )
    return grid_points, path_points, target_schedule, params


def build_grid_test(grid_size, title):
    """Build a three-pass condition on a centered square grid."""
    return build_path_test(build_serpentine_grid(grid_size), title)
