import numpy as np


PARAMS = {
    "TARGET_SCHEDULE": [
        (18.0, np.array([-0.035, -0.0125]), 1, np.deg2rad(0.0)),
        (24.0, np.array([0.00, 0.00]), 1, np.deg2rad(0.0)),
        (30.0, np.array([0.01, 0.00]), 1, np.deg2rad(0.0)),
        (36.0, np.array([0.01, 0.00]), 1, np.deg2rad(0.0)),
        (42.0, np.array([0.02, 0.005]), 1, np.deg2rad(0.0)),
        (48.0, np.array([0.03, 0.015]), 1, np.deg2rad(0.0)),
        (54.0, np.array([0.04, 0.03]), 1, np.deg2rad(0.0)),
        (60.0, np.array([0.035, 0.045]), 1, np.deg2rad(0.0)),
        (66.0, np.array([0.02, 0.055]), 1, np.deg2rad(0.0)),
        (72.0, np.array([0.00, 0.06]), 1, np.deg2rad(0.0)),
        (78.0, np.array([-0.02, 0.055]), 1, np.deg2rad(0.0)),
        (84.0, np.array([-0.035, 0.045]), 1, np.deg2rad(0.0)),
        (90.0, np.array([-0.045, 0.03]), 1, np.deg2rad(0.0)),
        (96.0, np.array([-0.04, 0.01]), 1, np.deg2rad(0.0)),
        (102.0, np.array([-0.025, -0.005]), 1, np.deg2rad(0.0)),
        (108.0, np.array([0.00, -0.015]), 1, np.deg2rad(0.0)),
        (114.0, np.array([0.025, -0.01]), 1, np.deg2rad(0.0)),
        (120.0, np.array([0.045, 0.00]), 1, np.deg2rad(0.0)),
        (126.0, np.array([0.06, 0.015]), 1, np.deg2rad(0.0)),
        (132.0, np.array([0.07, 0.035]), 1, np.deg2rad(0.0)),
        (138.0, np.array([0.065, 0.055]), 1, np.deg2rad(0.0)),
        (144.0, np.array([0.05, 0.07]), 1, np.deg2rad(0.0)),
        (150.0, np.array([0.03, 0.08]), 1, np.deg2rad(0.0)),
        (156.0, np.array([0.06, 0.06]), 1, np.deg2rad(0.0)),
        (162.0, np.array([-0.10, 0.00]), 1, np.deg2rad(0.0)),
    ],
    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),

    # --- Payload Initial State ---
    "PAYLOAD_INITIAL_POS": np.array([0.0, 0.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
