import numpy as np


SHAPE_DURATION = 15.0


PARAMS = {
    # --- Desired Parameters ---
    "TARGET_SCHEDULE": [
        (0 * SHAPE_DURATION, np.array([0.03, 0.03]), 1, np.deg2rad(80.0)),
        (1 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(0.0)),
        (2 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(45.0)),
        (3 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(90.0)),
        (4 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(135.0)),
        (5 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(180.0)),
        (6 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(225.0)),
        (7 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(270.0)),
        (8 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(315.0)),
        (9 * SHAPE_DURATION, np.array([0.03, 0.03]), 2, np.deg2rad(360.0)),
        (10 * SHAPE_DURATION, np.array([0.001, 0.00]), 1, np.deg2rad(80.0)),
    ],

    # --- Initial Robot Positions ---
    # Generate 30 robots in a grid pattern for simplicity
    "INITIAL_ROBOT_POSITIONS": np.array([
        [x, y] for x in np.linspace(-0.05, 0.05, 6)
                for y in np.linspace(-0.05, 0.05, 5)
    ]),  # 6x5=30
    # "INITIAL_ROBOT_POSITIONS": np.array([
    #     [10,10]
    # ]),  # 6x5=30

    # --- Time Parameters ---
    "T_SPAN": (0, 11 * SHAPE_DURATION),
    "T_EVAL_POINTS": 300,
    "SOLVER_PROGRESS_INTERVAL": 0.5,

    "ANIMATION_TITLE": (
        "Controlling the Ratio and Directions of Equilibrium-Point Eigenvalues"
    ),

    # --- Payload Initial State ---
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
