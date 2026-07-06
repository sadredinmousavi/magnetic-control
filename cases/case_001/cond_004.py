import numpy as np


PARAMS = {
    # --- Desired Parameters ---
    # The equilibrium center first gathers robots away from the payload, then
    # visits the payload at the origin, transports it, and finally retreats.
    "TARGET_SCHEDULE": [
        (6.0, np.array([-0.0, -0.00]), 1, np.deg2rad(0.0)),
        (12.0, np.array([-0.07, -0.05]), 1, np.deg2rad(0.0)),
        (18.0, np.array([+0.07, +0.05]), 1, np.deg2rad(0.0)),
        (24.0, np.array([0.00, 0.00]), 1, np.deg2rad(0.0)),
        (30.0, np.array([-0.04, 0.04]), 1, np.deg2rad(0.0)),
        (36.0, np.array([0.04, 0.04]), 1, np.deg2rad(0.0)),
        (42.0, np.array([0.04, -0.04]), 1, np.deg2rad(0.0)),
        (48.0, np.array([-0.04, 0.04]), 1, np.deg2rad(0.0)),
        
    ],

    "CAPILLARY_SIN_C": 0.01,

    # --- Initial Robot Positions ---
    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),

    # --- Time / Solver Parameters ---
    "T_SPAN": (0, 180.0),
    "T_EVAL_POINTS": 700,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "USE_OVERDAMPED_DYNAMICS": False,
    "DYNAMICS_SPEEDUP": 1.0,
    "SOLVER_RTOL": 1e-4,
    "SOLVER_ATOL": 1e-7,

    # --- Payload Parameters ---
    "PAYLOAD_RADIUS": 0.015,
    "PAYLOAD_HEIGHT": 0.001,
    "PAYLOAD_DENSITY": 50,
    "PAYLOAD_DRAG_FACTOR": 50,
    "CONTACT_STIFFNESS": 2e-5,
    "CONTACT_DAMPING": 5e-5,
    "PAYLOAD_CAPILLARY_GAIN": 5e-7,
    "PAYLOAD_CAPILLARY_RANGE": 0.007,
    "PAYLOAD_INITIAL_POS": np.array([0.0, 0.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
