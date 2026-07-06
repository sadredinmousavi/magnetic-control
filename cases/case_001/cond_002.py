import numpy as np


def two_equilibrium_points(center, angle_rad, radius=0.05):
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    center = np.array(center)
    return center + radius * direction, center - radius * direction


PARAMS = {
    # --- Desired Parameters ---
    # TARGET_SCHEDULE entries:
    # Stable one-equilibrium mode:
    #   (start_time, target_pos, eig_ratio, eigvec_angle_rad)
    # Two-equilibrium mode, without eigenvalue/eigenvector constraints:
    #   (start_time, target_pos_1, target_pos_2)
    # Write the angle by hand in degrees and convert inline with np.deg2rad(...).
    "TARGET_SCHEDULE": [
        (6.0,  np.array([ 0.060,  0.000]), 1, np.deg2rad(0.0)),
        (12.0, np.array([ 0.042,  0.042]), 1, np.deg2rad(45.0)),
        (18.0, np.array([ 0.000,  0.060]), 1, np.deg2rad(90.0)),
        (24.0, np.array([-0.042,  0.042]), 1, np.deg2rad(135.0)),
        (30.0, np.array([-0.060,  0.000]), 1, np.deg2rad(180.0)),
        (36.0, np.array([-0.042, -0.042]), 1, np.deg2rad(225.0)),
        (42.0, np.array([ 0.000, -0.060]), 1, np.deg2rad(270.0)),
        (48.0, np.array([ 0.042, -0.042]), 1, np.deg2rad(315.0)),
        (54.0, np.array([ 0.060,  0.000]), 1, np.deg2rad(360.0)),
    ],


    # --- Initial Robot Positions ---
    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),
}
