"""Cartesian position calibration on a 10 mm grid strictly inside r = 5 cm."""

import numpy as np

from .extract_compensator_helpers import CALIBRATION_POINTS
from .grid_repeatability_helpers import build_scheduled_path


RADIUS_LIMIT_MM = 50
GRID_SPACING_MM = 10
GRID_COORDINATES_MM = np.arange(-40, 41, GRID_SPACING_MM)

# Follow the supplied serpentine order: start at (-20, 40) mm, then
# alternate left-to-right and right-to-left across each descending row.
# Filter in integer millimetres to exclude r = 50 mm boundary points exactly.
_rows = []
for _row_index, _y in enumerate(GRID_COORDINATES_MM[::-1]):
    _x_coordinates = GRID_COORDINATES_MM[
        GRID_COORDINATES_MM ** 2 + _y ** 2 < RADIUS_LIMIT_MM ** 2
    ]
    if _row_index % 2:
        _x_coordinates = _x_coordinates[::-1]
    _rows.append(np.column_stack((_x_coordinates, np.full_like(_x_coordinates, _y))))

GRID_POINTS_MM = np.vstack(_rows)
GRID_POINTS = GRID_POINTS_MM * 1e-3

# Keep all grid points, including the later visits to the three references.
PATH_POINTS, TARGET_SCHEDULE, PARAMS = build_scheduled_path(
    np.vstack((CALIBRATION_POINTS, GRID_POINTS)),
    "Cartesian Position Calibration (r < 5 cm)",
)
