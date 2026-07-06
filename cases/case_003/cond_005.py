import numpy as np

PARAMS = {
    "TARGET_SCHEDULE": [
        (6.0,  np.array([ 0.050,  0.000]), np.array([-0.050,  0.000])),
        (12.0, np.array([ 0.035,  0.035]), np.array([-0.035, -0.035])),
        (18.0, np.array([ 0.000,  0.050]), np.array([ 0.000, -0.050])),
        (24.0, np.array([-0.035,  0.035]), np.array([ 0.035, -0.035])),
        (30.0, np.array([-0.050,  0.000]), np.array([ 0.050,  0.000])),
        (36.0, np.array([-0.035, -0.035]), np.array([ 0.035,  0.035])),
        (42.0, np.array([ 0.000, -0.050]), np.array([ 0.000,  0.050])),
        (48.0, np.array([ 0.035, -0.035]), np.array([-0.035,  0.035])),
        (54.0, np.array([ 0.050,  0.000]), np.array([-0.050,  0.000])),
    ],
    # Two-equilibrium solver options: "stable", "plain", "center_repulsion".
    "TWO_EQUILIBRIUM_SOLVER": "stable",

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
