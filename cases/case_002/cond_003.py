import numpy as np


SHAPE_DURATION = 40.0


PARAMS = {
    # --- Desired Parameters ---
    # Move through a compact zig-zag channel with several knees. The path stays
    # near the workspace center, away from the PMs at radius 0.25 m.
    "TARGET_SCHEDULE": [
        (0 * SHAPE_DURATION, np.array([-0.115, -0.055]), 1, np.deg2rad(0.0)),
        (1 * SHAPE_DURATION, np.array([-0.075, -0.055]), 1, np.deg2rad(0.0)),
        (2 * SHAPE_DURATION, np.array([-0.075, 0.040]), 1, np.deg2rad(0.0)),
        (3 * SHAPE_DURATION, np.array([-0.020, 0.040]), 1, np.deg2rad(0.0)),
        (4 * SHAPE_DURATION, np.array([-0.020, -0.040]), 1, np.deg2rad(0.0)),
        (5 * SHAPE_DURATION, np.array([0.040, -0.040]), 1, np.deg2rad(0.0)),
        (6 * SHAPE_DURATION, np.array([0.040, 0.050]), 1, np.deg2rad(0.0)),
        (7 * SHAPE_DURATION, np.array([0.085, 0.050]), 1, np.deg2rad(0.0)),
        (8 * SHAPE_DURATION, np.array([0.115, 0.050]), 1, np.deg2rad(0.0)),
    ],

    "ROBOT_MAGNETIZATION": 3.0e4,  # Effective magnetization for PDMS + 20% SPION

    # --- Initial Robot Positions ---
    "INITIAL_ROBOT_POSITIONS": np.array([
        [x, y] for x in np.linspace(-0.125, -0.105, 6)
                for y in np.linspace(-0.065, -0.045, 5)
    ]),

    # --- Wall / Corridor Parameters ---
    # Soft wall segments form a compact corridor with open entrance/exit.
    # It is not a hard collision solver: walls repel robots when they approach.
    "WALL_SEGMENTS": [
        # Entrance horizontal section.
        (np.array([-0.115, -0.090]), np.array([-0.055, -0.090])),
        (np.array([-0.115, -0.020]), np.array([-0.055, -0.020])),
        # First vertical climb.
        (np.array([-0.110, -0.055]), np.array([-0.110, 0.040])),
        (np.array([-0.040, -0.055]), np.array([-0.040, 0.040])),
        # Upper horizontal section.
        (np.array([-0.075, 0.005]), np.array([-0.020, 0.005])),
        (np.array([-0.075, 0.075]), np.array([-0.020, 0.075])),
        # Middle vertical descent.
        (np.array([-0.055, 0.040]), np.array([-0.055, -0.040])),
        (np.array([0.015, 0.040]), np.array([0.015, -0.040])),
        # Lower horizontal section.
        (np.array([-0.020, -0.075]), np.array([0.040, -0.075])),
        (np.array([-0.020, -0.005]), np.array([0.040, -0.005])),
        # Final vertical climb.
        (np.array([0.005, -0.040]), np.array([0.005, 0.050])),
        (np.array([0.075, -0.040]), np.array([0.075, 0.050])),
        # Exit horizontal section.
        (np.array([0.040, 0.015]), np.array([0.115, 0.015])),
        (np.array([0.040, 0.085]), np.array([0.115, 0.085])),
    ],
    "WALL_STIFFNESS": 5e-6,
    "WALL_DAMPING": 5e-7,
    "WALL_INTERACTION_RANGE": 0.001,

    # --- Time / Solver Parameters ---
    "T_SPAN": (0, 9 * SHAPE_DURATION),
    "T_EVAL_POINTS": 400,
}
