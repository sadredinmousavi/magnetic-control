"""One-pass radially compensated test on the Case 100 5 x 5 grid."""

import numpy as np

from .grid_repeatability_helpers import build_scheduled_path, build_serpentine_grid


GRID_SIZE = 5

# Calibration table: desired radius and corresponding commanded radius.
# Values are stored in metres. The origin remains fixed.
CALIBRATION_DESIRED_RADII = 1e-3 * np.array([
    0.0, 10.0, 20.0, 30.0, 40.0,
    50.0, 60.0, 70.0, 80.0, 90.0,
])
CALIBRATION_COMMANDED_RADII = 1e-3 * np.array([
    0.0, 8.45, 15.41, 22.91, 32.41,
    44.85, 60.60, 79.51, 100.86, 123.41,
])


def compensate_points(points):
    """Interpolate commanded radii while preserving each point's angle."""
    desired_points = np.asarray(points, dtype=float)
    desired_radii = np.linalg.norm(desired_points, axis=1)
    commanded_radii = np.interp(
        desired_radii,
        CALIBRATION_DESIRED_RADII,
        CALIBRATION_COMMANDED_RADII,
    )
    radial_scale = np.divide(
        commanded_radii,
        desired_radii,
        out=np.ones_like(commanded_radii),
        where=desired_radii > 0.0,
    )
    return desired_points * radial_scale[:, None]


DESIRED_GRID_POINTS = build_serpentine_grid(GRID_SIZE)
COMPENSATED_GRID_POINTS = compensate_points(DESIRED_GRID_POINTS)
DESIRED_RADII = np.linalg.norm(DESIRED_GRID_POINTS, axis=1)
COMMANDED_RADII = np.linalg.norm(COMPENSATED_GRID_POINTS, axis=1)

GRID_POINTS = COMPENSATED_GRID_POINTS
PATH_POINTS, TARGET_SCHEDULE, PARAMS = build_scheduled_path(
    GRID_POINTS,
    "Radially Compensated 5 x 5 Square-Grid Repeatability Test",
)
