import numpy as np


PARAMS = {
    "TARGET_SCHEDULE": [
        (0.0,  np.array([ 0.00,  0.00]), 1, np.deg2rad(0.0)),
        (6.0,  np.array([ 0.06,  0.00]), 1, np.deg2rad(0.0)),
        (12.0, np.array([ 0.042,  0.042]), 1, np.deg2rad(45.0)),
        (18.0, np.array([ 0.00,  0.06]), 1, np.deg2rad(90.0)),
        (24.0, np.array([-0.042,  0.042]), 1, np.deg2rad(135.0)),
        (30.0, np.array([-0.06,  0.00]), 1, np.deg2rad(180.0)),
        (36.0, np.array([-0.042, -0.042]), 1, np.deg2rad(225.0)),
        (42.0, np.array([ 0.00, -0.06]), 1, np.deg2rad(270.0)),
        (48.0, np.array([ 0.042, -0.042]), 1, np.deg2rad(315.0)),
        (54.0, np.array([ 0.06,  0.00]), 1, np.deg2rad(360.0)),
        (60.0, np.array([ 0.00,  0.00]), 1, np.deg2rad(360.0)),
    ],

    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),

    # --- Time Parameters ---
    "T_SPAN": (0, 180.0),
    "T_EVAL_POINTS": 700,
    "SOLVER_PROGRESS_INTERVAL": 0.5,

    # --- Payload Initial State ---
    "PAYLOAD_INITIAL_POS": np.array([0.0, 0.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
