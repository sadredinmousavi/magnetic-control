"""Three-pass repeatability test on a centered 3 x 3, 0.12 m square grid."""

from .grid_repeatability_helpers import build_grid_test


GRID_SIZE = 3

GRID_POINTS, PATH_POINTS, TARGET_SCHEDULE, PARAMS = build_grid_test(
    GRID_SIZE,
    "3 x 3 Square-Grid Repeatability Test",
)
