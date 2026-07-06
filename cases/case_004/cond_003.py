import numpy as np


PARAMS = {
    # Four-equilibrium entries use:
    # (start_time, target_pos_1, target_pos_2, target_pos_3, target_pos_4)
    "TARGET_SCHEDULE": [
        (
            0.0,
            np.array([0.0800, 0.0200]),
            np.array([-0.0800, 0.0200]),
            np.array([0.0800, 0.0300]),
            np.array([-0.0800, 0.0300]),
        ),
    ],

    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),

    # --- Time Parameters ---
    "T_SPAN": (0, 50.0),
    "T_EVAL_POINTS": 700,
    "SOLVER_PROGRESS_INTERVAL": 0.5,

    # --- Payload Initial State ---
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
