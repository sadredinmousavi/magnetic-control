import numpy as np


def two_equilibrium_points(center, angle_rad, radius=0.05):
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    center = np.array(center)
    return center + radius * direction, center - radius * direction


SHAPE_DURATION = 30.0


PARAMS = {
    # --- Desired Parameters ---
    "TARGET_SCHEDULE": [
        # (0.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(80.0))),
        (0 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(0.0))),
        (1 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(45.0))),
        (2 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(90.0))),
        (3 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(135.0))),
        (4 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(180.0))),
        (5 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(225.0))),
        (6 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(270.0))),
        (7 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(315.0))),
        (8 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(360.0))),
        (9 * SHAPE_DURATION, *two_equilibrium_points([0.001, 0.00], np.deg2rad(80.0))),
    ],
    "TWO_EQUILIBRIUM_SOLVER": "stable",#plain
    "STABILITY_TRACE_MARGIN" : 1e-8,
    "STABILITY_DET_MARGIN" : 1e-14,

    "ROBOT_MAGNETIZATION": 3.0e4,  # Effective magnetization for PDMS + 20% SPION

    # --- Initial Robot Positions ---
    # Generate 30 robots in a grid pattern for simplicity
    "INITIAL_ROBOT_POSITIONS": np.array([
        [x, y] for x in np.linspace(-0.05, 0.05, 6)
                for y in np.linspace(-0.05, 0.05, 5)
    ]),  # 6x5=30

    # --- Payload Initial State ---
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),

    # --- Time / Solver Parameters ---
    "T_SPAN": (0, 13 * SHAPE_DURATION),
    "T_EVAL_POINTS": 300,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
}
