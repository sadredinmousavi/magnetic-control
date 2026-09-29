import unittest

import numpy as np

from case_loader import load_case, validate_target_schedule
from cases.case_100 import cond_007, cond_008


class Case100CargoPathTests(unittest.TestCase):
    def assert_central_pickup_condition(self, condition_name, condition):
        params = load_case(condition_name)
        validate_target_schedule(params["TARGET_SCHEDULE"])

        np.testing.assert_allclose(
            params["TARGET_SCHEDULE"][0][1], condition.CARGO_CENTER
        )
        self.assertEqual(
            params["TARGET_SCHEDULE"][1][0], condition.PICKUP_DURATION
        )
        np.testing.assert_allclose(
            params["PAYLOAD_INITIAL_POS"], condition.CARGO_CENTER
        )
        distances = np.linalg.norm(
            params["INITIAL_ROBOT_POSITIONS"] - condition.CARGO_CENTER,
            axis=1,
        )
        np.testing.assert_allclose(distances, condition.ROBOT_START_RADIUS)
        self.assertEqual(len(params["INITIAL_ROBOT_POSITIONS"]), 7)
        self.assertTrue(params["PLOT_DRAW_TARGET_PATH"])

    def test_condition_007_is_a_closed_square_after_pickup(self):
        self.assert_central_pickup_condition(
            "case_100.cond_007", cond_007
        )

        np.testing.assert_allclose(
            cond_007.PATH_POINTS[0], cond_007.PATH_POINTS[-1]
        )
        self.assertEqual(
            set(np.unique(np.abs(cond_007.SQUARE_CORNERS))),
            {cond_007.SQUARE_HALF_SIDE},
        )
        deltas = np.diff(cond_007.PATH_POINTS, axis=0)
        self.assertTrue(np.all(
            np.isclose(deltas[:, 0], 0.0)
            | np.isclose(deltas[:, 1], 0.0)
        ))

    def test_condition_008_joins_upper_circle_to_lower_half_star(self):
        self.assert_central_pickup_condition(
            "case_100.cond_008", cond_008
        )

        np.testing.assert_allclose(
            np.linalg.norm(cond_008.UPPER_CIRCLE_POINTS, axis=1),
            cond_008.PATH_RADIUS,
        )
        self.assertTrue(np.all(cond_008.UPPER_CIRCLE_POINTS[:, 1] >= -1e-12))
        self.assertTrue(np.all(cond_008.LOWER_HALF_STAR_POINTS[:, 1] <= 1e-12))
        np.testing.assert_allclose(
            cond_008.PATH_POINTS[0], cond_008.PATH_POINTS[-1], atol=1e-12
        )
        np.testing.assert_allclose(
            cond_008.UPPER_CIRCLE_POINTS[-1],
            cond_008.LOWER_HALF_STAR_POINTS[0],
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
