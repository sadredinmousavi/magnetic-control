"""One-pass normal test on a centered 5 x 5, 0.12 m square grid."""

from .grid_repeatability_helpers import build_scheduled_path, build_serpentine_grid


GRID_SIZE = 5

GRID_POINTS = build_serpentine_grid(GRID_SIZE)
PATH_POINTS, TARGET_SCHEDULE, PARAMS = build_scheduled_path(
    GRID_POINTS,
    "Normal 5 x 5 Square-Grid Repeatability Test",
)
