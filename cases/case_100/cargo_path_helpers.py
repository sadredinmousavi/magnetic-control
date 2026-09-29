"""Shared central-cargo pickup setup for Case 100 path conditions."""

import numpy as np


CARGO_CENTER = np.array([0.0, 0.0])
NUM_ROBOTS = 7
ROBOT_START_RADIUS = 0.040
PICKUP_DURATION = 24.0
PATH_STEP_DURATION = 10.0


def initial_robot_ring(
    center=CARGO_CENTER,
    count=NUM_ROBOTS,
    radius=ROBOT_START_RADIUS,
):
    """Place a compact, evenly spaced swarm around the central cargo."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.asarray(center, dtype=float) + np.column_stack((
        radius * np.cos(angles),
        radius * np.sin(angles),
    ))


def build_cargo_path_params(path_points, title):
    """Build a condition that gathers at the origin, then transports cargo."""
    points = np.asarray(path_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Cargo path must contain at least two 2D points.")

    target_schedule = [
        (0.0, CARGO_CENTER.copy(), 1.0, np.deg2rad(0.0)),
        *[
            (
                PICKUP_DURATION + index * PATH_STEP_DURATION,
                point.copy(),
                1.0,
                np.deg2rad(0.0),
            )
            for index, point in enumerate(points)
        ],
    ]

    return {
        "TARGET_SCHEDULE": target_schedule,
        "INITIAL_ROBOT_POSITIONS": initial_robot_ring(),

        "T_SPAN": (0.0, target_schedule[-1][0] + PATH_STEP_DURATION),
        "T_EVAL_POINTS": max(800, len(target_schedule) * 30),
        "SOLVER_PROGRESS_INTERVAL": 0.5,
        "USE_OVERDAMPED_DYNAMICS": True,
        "DYNAMICS_SPEEDUP": 1.0,
        "SOLVER_RTOL": 1e-4,
        "SOLVER_ATOL": 1e-7,

        "ANIMATION_TITLE": title,
        "ANIMATION_DRAW_TRAJECTORIES": False,
        "ANIMATION_DRAW_TARGET_TRAJECTORY": True,
        "ANIMATION_DRAW_TARGET_POINTS": True,
        "PLOT_DRAW_TARGET_PATH": True,

        # Match Case 100 condition 003's circular cargo and contact model.
        "PAYLOAD_RADIUS": 0.015,
        "PAYLOAD_HEIGHT": 0.001,
        "PAYLOAD_DENSITY": 50,
        "PAYLOAD_DRAG_FACTOR": 50,
        "CONTACT_STIFFNESS": 2e-5,
        "CONTACT_DAMPING": 5e-5,
        "PAYLOAD_CAPILLARY_GAIN": 5e-7,
        "PAYLOAD_CAPILLARY_RANGE": 0.007,
        "PAYLOAD_INITIAL_POS": CARGO_CENTER.copy(),
        "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
    }

