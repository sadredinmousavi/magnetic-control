import unittest

import numpy as np

from case_loader import case_output_name, load_case, validate_target_schedule
from cases.case_100 import cond_009_test, cond_010_test
from cases.case_100.grid_repeatability_helpers import REPEAT_COUNT, SQUARE_SIDE_LENGTH


class Case100GridRepeatabilityTests(unittest.TestCase):
    def _check_condition(self, condition, expected_size):
        self.assertEqual(condition.GRID_POINTS.shape, (expected_size**2, 2))
        self.assertEqual(
            condition.PATH_POINTS.shape,
            (REPEAT_COUNT * expected_size**2, 2),
        )
        np.testing.assert_allclose(
            condition.GRID_POINTS.min(axis=0),
            [-SQUARE_SIDE_LENGTH / 2.0] * 2,
        )
        np.testing.assert_allclose(
            condition.GRID_POINTS.max(axis=0),
            [SQUARE_SIDE_LENGTH / 2.0] * 2,
        )

        expected_points = {tuple(point) for point in condition.GRID_POINTS}
        for repeat_points in np.split(condition.PATH_POINTS, REPEAT_COUNT):
            self.assertEqual({tuple(point) for point in repeat_points}, expected_points)

        validate_target_schedule(condition.TARGET_SCHEDULE)
        self.assertEqual(condition.PARAMS["SEQUENCE_WAIT"], 3.0)

    def test_three_by_three_grid(self):
        self._check_condition(cond_009_test, 3)
        params = load_case("case_100.cond_009_test")
        self.assertEqual(len(params["TARGET_SCHEDULE"]), 27)

    def test_five_by_five_grid(self):
        self._check_condition(cond_010_test, 5)
        params = load_case("case_100.cond_010_test")
        self.assertEqual(len(params["TARGET_SCHEDULE"]), 75)

    def test_test_suffix_is_preserved_in_output_names(self):
        self.assertEqual(
            case_output_name("case_100.cond_009_test"),
            "case_100_cond_009_test",
        )
        self.assertEqual(
            case_output_name("case_100.cond_010_test"),
            "case_100_cond_010_test",
        )


if __name__ == "__main__":
    unittest.main()
