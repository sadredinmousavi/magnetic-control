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


def build_grid_test(grid_size, title):
    """Build a three-pass condition for measuring point repeatability."""
    grid_points = build_serpentine_grid(grid_size)

    # Reverse direction after each pass. Consecutive passes share their end
    # point, avoiding a long repositioning move across the square. Every
    # nonzero move is therefore only one grid spacing.
    passes = [
        (grid_points if repeat_index % 2 == 0 else grid_points[::-1]).copy()
        for repeat_index in range(REPEAT_COUNT)
    ]
    path_points = np.vstack(passes)

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
    return grid_points, path_points, target_schedule, params
