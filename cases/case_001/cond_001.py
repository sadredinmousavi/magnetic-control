import numpy as np


PARAMS = {
    # --- Desired Parameters ---
    # TARGET_SCHEDULE entries:
    # Stable one-equilibrium mode:
    #   (start_time, target_pos, eig_ratio, eigvec_angle_rad)
    # Two-equilibrium mode, without eigenvalue/eigenvector constraints:
    #   (start_time, target_pos_1, target_pos_2)
    # Write the angle by hand in degrees and convert inline with np.deg2rad(...).
    "TARGET_SCHEDULE": [
        (0.0, np.array([0.00, 0.00]), 1, np.deg2rad(80.0)),
        (6.0, np.array([0.00, 0.00]), 2, np.deg2rad(0.0)),
        (12.0, np.array([0.00, 0.00]), 2, np.deg2rad(45.0)),
        (18.0, np.array([0.00, 0.00]), 2, np.deg2rad(90.0)),
        (24.0, np.array([0.00, 0.00]), 2, np.deg2rad(135.0)),
        (30.0, np.array([0.00, 0.00]), 2, np.deg2rad(180.0)),
        (36.0, np.array([0.00, 0.00]), 2, np.deg2rad(225.0)),
        (42.0, np.array([0.00, 0.00]), 2, np.deg2rad(270.0)),
        (48.0, np.array([0.00, 0.00]), 2, np.deg2rad(315.0)),
        (54.0, np.array([0.00, 0.00]), 2, np.deg2rad(360.0)),
        (60.0, np.array([0.001, 0.00]), 1, np.deg2rad(80.0)),
    ],

    # --- Initial Robot Positions ---
    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),
}
