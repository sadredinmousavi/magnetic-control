"""Three-pass repeatability test on a centered 5 x 5, 0.12 m square grid."""

from .grid_repeatability_helpers import build_grid_test


GRID_SIZE = 5

GRID_POINTS, PATH_POINTS, TARGET_SCHEDULE, PARAMS = build_grid_test(
    GRID_SIZE,
    "5 x 5 Square-Grid Repeatability Test",
)
