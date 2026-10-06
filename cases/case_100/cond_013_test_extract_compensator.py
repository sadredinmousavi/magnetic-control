"""Extract position-error samples on a centered 0.06 m square."""

from .extract_compensator_helpers import (
    CALIBRATION_POINTS,
    POINTS_PER_SIDE,
    build_extract_compensator_test,
)


SQUARE_SIDE_LENGTH = 0.06

SQUARE_POINTS, PATH_POINTS, TARGET_SCHEDULE, PARAMS = build_extract_compensator_test(
    SQUARE_SIDE_LENGTH,
)
