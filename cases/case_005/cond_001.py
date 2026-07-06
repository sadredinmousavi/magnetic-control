import numpy as np


PARAMS = {
    "TARGET_SCHEDULE": [
        (0.0,  np.array([ 0.0400,  0.0000]), np.array([-0.0400,  0.0000])),
        (6.0,  np.array([ 0.0283,  0.0283]), np.array([-0.0283,  0.0283])),
        (12.0, np.array([ 0.0000,  0.0400]), np.array([ 0.0000,  0.0400])),
        (18.0, np.array([-0.0283,  0.0283]), np.array([ 0.0283,  0.0283])),
        (24.0, np.array([-0.0400,  0.0000]), np.array([ 0.0400,  0.0000])),
        (30.0, np.array([-0.0283, -0.0283]), np.array([ 0.0283, -0.0283])),
        (36.0, np.array([ 0.0000, -0.0400]), np.array([ 0.0000, -0.0400])),
        (42.0, np.array([ 0.0283, -0.0283]), np.array([-0.0283, -0.0283])),
        (48.0, np.array([ 0.0400,  0.0000]), np.array([-0.0400,  0.0000])),
    ],

    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),

    # --- Payload Initial State ---
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
