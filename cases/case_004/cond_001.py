import numpy as np


CIRCLE_CENTER = np.array([0.05, 0.05])
CIRCLE_RADIUS = 0.04
STEP_TIME = 6.0


def two_equilibrium_points(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    return (
        CIRCLE_CENTER + CIRCLE_RADIUS * direction,
        CIRCLE_CENTER - CIRCLE_RADIUS * direction,
    )


PARAMS = {
    "TARGET_SCHEDULE": [
        (step * STEP_TIME, *two_equilibrium_points(angle_deg))
        for step, angle_deg in enumerate(
            [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
        )
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
