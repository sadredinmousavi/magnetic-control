"""Seven microrobots tracing the text ``MNLab`` together."""

import numpy as np


# Letter strokes in a small normalized drawing coordinate system. Consecutive
# letters are connected because the simulation has continuously moving
# robots (there is no pen-up state).
_TEXT_POINTS = np.array([
    # M
    [0.00, 0.00], [0.00, 1.00], [0.28, 0.45],
    [0.56, 1.00], [0.56, 0.00],
    # N
    [0.78, 0.00], [0.78, 1.00], [1.34, 0.00], [1.34, 1.00],
    # L
    [1.56, 1.00], [1.56, 0.00], [2.12, 0.00],
    # a
    [2.34, 0.05], [2.34, 0.52], [2.50, 0.70], [2.75, 0.70],
    [2.92, 0.52], [2.92, 0.05], [2.92, 0.52], [2.75, 0.70],
    [2.50, 0.70], [2.34, 0.52], [2.34, 0.25], [2.48, 0.05],
    [2.72, 0.05], [2.92, 0.22],
    # b
    [3.16, 0.00], [3.16, 1.00], [3.16, 0.55], [3.38, 0.72],
    [3.65, 0.67], [3.80, 0.48], [3.78, 0.25], [3.62, 0.08],
    [3.38, 0.05], [3.16, 0.25], [3.16, 0.00],
])


def _scale_text(points):
    """Scale the text to exactly x=[-0.08, 0.08] and a 0.048 m height."""
    scaled = np.asarray(points, dtype=float).copy()
    scaled[:, 0] = -0.08 + 0.16 * (
        (scaled[:, 0] - scaled[:, 0].min())
        / (scaled[:, 0].max() - scaled[:, 0].min())
    )
    scaled[:, 1] = -0.024 + 0.048 * (
        (scaled[:, 1] - scaled[:, 1].min())
        / (scaled[:, 1].max() - scaled[:, 1].min())
    )
    return scaled


TEXT_POINTS = _scale_text(_TEXT_POINTS)
SECONDS_PER_POINT = 3.0
NUM_ROBOTS = 7

# Start the robots on a small ring around the first writing point. Keeping the
# positions distinct avoids overlapping robots while preserving a compact group.
_robot_angles = np.linspace(0.0, 2.0 * np.pi, NUM_ROBOTS, endpoint=False)
INITIAL_ROBOT_POSITIONS = TEXT_POINTS[0] + np.column_stack((
    0.006 * np.cos(_robot_angles),
    0.006 * np.sin(_robot_angles),
))


PARAMS = {
    "TARGET_SCHEDULE": [
        (index * SECONDS_PER_POINT, point, 1.0, np.deg2rad(0.0))
        for index, point in enumerate(TEXT_POINTS)
    ],

    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,

    "T_SPAN": (0.0, len(TEXT_POINTS) * SECONDS_PER_POINT),
    "T_EVAL_POINTS": len(TEXT_POINTS) * 15,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "USE_OVERDAMPED_DYNAMICS": True,
    "DYNAMICS_SPEEDUP": 1.0,
    "SOLVER_RTOL": 1e-5,
    "SOLVER_ATOL": 1e-8,

    # Draw the equilibrium-point path without drawing individual robot paths.
    "ANIMATION_DRAW_TRAJECTORIES": False,
    "ANIMATION_DRAW_TARGET_TRAJECTORY": True,
    "ANIMATION_DRAW_CONTOUR": True,
    "ANIMATION_DRAW_STREAMLINES": False,
    "ANIMATION_DRAW_QUIVER": False,
    "ANIMATION_TITLE": "Equilibrium-Point Manipulation",
    "VIDEO_DPI": 160,
    "VIDEO_FPS": 30,
    "VIDEO_CRF": 18,

    # Payload remains disabled, matching the base case convention.
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
