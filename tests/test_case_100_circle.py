import unittest

import numpy as np

from case_loader import load_case, validate_target_schedule
from cases.case_100 import cond_013


class Case100CircleTests(unittest.TestCase):
    def test_path_is_closed_five_centimetre_circle(self):
        offsets = cond_013.PATH_POINTS - cond_013.CIRCLE_CENTER

        np.testing.assert_allclose(
            np.linalg.norm(offsets, axis=1),
            cond_013.CIRCLE_RADIUS,
        )
        np.testing.assert_allclose(
            cond_013.PATH_POINTS[0],
            cond_013.PATH_POINTS[-1],
            atol=1e-12,
        )
        self.assertEqual(cond_013.CIRCLE_RADIUS, 0.05)
        self.assertEqual(len(cond_013.PATH_POINTS), 8)

    def test_condition_loads_with_valid_schedule(self):
        params = load_case("case_100.cond_013")

        validate_target_schedule(params["TARGET_SCHEDULE"])
        self.assertEqual(len(params["TARGET_SCHEDULE"]), 8)
        self.assertEqual(len(params["INITIAL_ROBOT_POSITIONS"]), 7)
        self.assertTrue(params["ANIMATION_DRAW_TARGET_TRAJECTORY"])
        self.assertTrue(params["ANIMATION_DRAW_TARGET_POINTS"])


if __name__ == "__main__":
    unittest.main()
