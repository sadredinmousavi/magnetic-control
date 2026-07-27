"""One microrobot tracing the lowercase word ``micro``."""

import numpy as np


# Letter strokes in a small normalized drawing coordinate system. Consecutive
# letters are connected because the simulation has one continuously moving
# robot (there is no pen-up state).
_WORD_POINTS = np.array([
    # m
    [0.00, 0.00], [0.00, 0.75], [0.00, 0.15],
    [0.18, 0.65], [0.36, 0.15], [0.54, 0.65], [0.72, 0.00],
    # i (including its dot)
    [0.92, 0.00], [0.92, 0.65], [0.92, 0.00],
    [0.92, 0.92], [0.92, 0.82], [0.92, 0.00],
    # c
    [1.72, 0.55], [1.58, 0.68], [1.38, 0.65], [1.25, 0.48],
    [1.22, 0.25], [1.34, 0.07], [1.55, 0.02], [1.72, 0.15],
    # r
    [1.92, 0.00], [1.92, 0.65], [1.92, 0.15],
    [2.10, 0.60], [2.30, 0.62],
    # o
    [2.78, 0.02], [2.58, 0.02], [2.43, 0.18], [2.40, 0.42],
    [2.52, 0.62], [2.72, 0.68], [2.90, 0.55], [2.96, 0.32],
    [2.90, 0.12], [2.78, 0.02],
])


def _scale_word(points):
    """Scale the word to exactly x=[-0.07, 0.07] and a 0.04 m height."""
    scaled = np.asarray(points, dtype=float).copy()
    scaled[:, 0] = -0.07 + 0.14 * (
        (scaled[:, 0] - scaled[:, 0].min())
        / (scaled[:, 0].max() - scaled[:, 0].min())
    )
    scaled[:, 1] = -0.02 + 0.04 * (
        (scaled[:, 1] - scaled[:, 1].min())
        / (scaled[:, 1].max() - scaled[:, 1].min())
    )
    return scaled


WORD_POINTS = _scale_word(_WORD_POINTS)
SECONDS_PER_POINT = 3.0


PARAMS = {
    "TARGET_SCHEDULE": [
        (index * SECONDS_PER_POINT, point, 1.0, np.deg2rad(0.0))
        for index, point in enumerate(WORD_POINTS)
    ],

    # Exactly one microrobot, initially located at the first point of the word.
    "INITIAL_ROBOT_POSITIONS": np.array([WORD_POINTS[0]]),

    "T_SPAN": (0.0, len(WORD_POINTS) * SECONDS_PER_POINT),
    "T_EVAL_POINTS": len(WORD_POINTS) * 15,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "USE_OVERDAMPED_DYNAMICS": True,
    "DYNAMICS_SPEEDUP": 1.0,
    "SOLVER_RTOL": 1e-5,
    "SOLVER_ATOL": 1e-8,

    # Make the written word visible in both the window and saved animation.
    "ANIMATION_DRAW_TRAJECTORIES": True,
    "ANIMATION_DRAW_CONTOUR": False,
    "ANIMATION_DRAW_STREAMLINES": False,
    "ANIMATION_DRAW_QUIVER": False,
    "VIDEO_DPI": 160,
    "VIDEO_FPS": 30,
    "VIDEO_CRF": 18,

    # Payload remains disabled, matching the base case convention.
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
