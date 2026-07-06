import numpy as np


PARAMS = {
    "TARGET_SCHEDULE": [
        (6.0,  np.array([0.00, 0.00]), 1, np.deg2rad(0.0)),
        (6.0,  np.array([0.0500,  0.0000]), 1, np.deg2rad(0.0)),
        (7.5,  np.array([0.0462,  0.0191]), 1, np.deg2rad(0.0)),
        (9.0,  np.array([0.0354,  0.0354]), 1, np.deg2rad(0.0)),
        (10.5, np.array([0.0191,  0.0462]), 1, np.deg2rad(0.0)),
        (12.0, np.array([0.0000,  0.0500]), 1, np.deg2rad(0.0)),
        (13.5, np.array([-0.0191, 0.0462]), 1, np.deg2rad(0.0)),
        (15.0, np.array([-0.0354, 0.0354]), 1, np.deg2rad(0.0)),
        (16.5, np.array([-0.0462, 0.0191]), 1, np.deg2rad(0.0)),
        (18.0, np.array([-0.0500, 0.0000]), 1, np.deg2rad(0.0)),
        (19.5, np.array([-0.0462, -0.0191]), 1, np.deg2rad(0.0)),
        (21.0, np.array([-0.0354, -0.0354]), 1, np.deg2rad(0.0)),
        (22.5, np.array([-0.0191, -0.0462]), 1, np.deg2rad(0.0)),
        (24.0, np.array([0.0000, -0.0500]), 1, np.deg2rad(0.0)),
        (25.5, np.array([0.0191, -0.0462]), 1, np.deg2rad(0.0)),
        (27.0, np.array([0.0354, -0.0354]), 1, np.deg2rad(0.0)),
        (28.5, np.array([0.0462, -0.0191]), 1, np.deg2rad(0.0)),
        (30.0, np.array([0.0500, 0.0000]), 1, np.deg2rad(0.0)),
        (31.5, np.array([0.0462, 0.0191]), 1, np.deg2rad(0.0)),
        (33.0, np.array([0.0354, 0.0354]), 1, np.deg2rad(0.0)),
        (34.5, np.array([0.0191, 0.0462]), 1, np.deg2rad(0.0)),
        (36.0, np.array([0.0000, 0.0500]), 1, np.deg2rad(0.0)),
        (38.0,  np.array([0.00, 0.00]), 1, np.deg2rad(0.0)),
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
