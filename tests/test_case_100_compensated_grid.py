import unittest

import numpy as np

from case_loader import load_case, validate_target_schedule
from cases.case_100 import cond_011_normal_test as normal_condition
from cases.case_100 import cond_012_compensated_test as condition


class Case100CompensatedGridTests(unittest.TestCase):
    def test_normal_condition_has_one_five_by_five_pass(self):
        self.assertEqual(normal_condition.GRID_POINTS.shape, (25, 2))
        self.assertEqual(normal_condition.PATH_POINTS.shape, (25, 2))
        np.testing.assert_allclose(
            normal_condition.PATH_POINTS, normal_condition.GRID_POINTS
        )

    def test_compensated_condition_has_one_five_by_five_pass(self):
        self.assertEqual(condition.DESIRED_GRID_POINTS.shape, (25, 2))
        self.assertEqual(condition.GRID_POINTS.shape, (25, 2))
        self.assertEqual(condition.PATH_POINTS.shape, (25, 2))
        np.testing.assert_allclose(condition.PATH_POINTS, condition.GRID_POINTS)

    def test_commanded_radii_are_interpolated_from_calibration_table(self):
        expected = np.interp(
            condition.DESIRED_RADII,
            condition.CALIBRATION_DESIRED_RADII,
            condition.CALIBRATION_COMMANDED_RADII,
        )
        np.testing.assert_allclose(condition.COMMANDED_RADII, expected, atol=1e-12)

        radius_30 = np.isclose(condition.DESIRED_RADII, 0.030)
        radius_60 = np.isclose(condition.DESIRED_RADII, 0.060)
        np.testing.assert_allclose(condition.COMMANDED_RADII[radius_30], 0.02291)
        np.testing.assert_allclose(condition.COMMANDED_RADII[radius_60], 0.06060)

    def test_compensation_preserves_each_points_radial_direction(self):
        cross_products = (
            condition.DESIRED_GRID_POINTS[:, 0] * condition.GRID_POINTS[:, 1]
            - condition.DESIRED_GRID_POINTS[:, 1] * condition.GRID_POINTS[:, 0]
        )
        np.testing.assert_allclose(cross_products, 0.0, atol=1e-14)
        np.testing.assert_allclose(condition.GRID_POINTS[12], [0.0, 0.0])

    def test_condition_loads_with_valid_schedule(self):
        normal_params = load_case("case_100.cond_011_normal_test")
        compensated_params = load_case("case_100.cond_012_compensated_test")
        validate_target_schedule(normal_params["TARGET_SCHEDULE"])
        validate_target_schedule(compensated_params["TARGET_SCHEDULE"])
        self.assertEqual(len(normal_params["TARGET_SCHEDULE"]), 25)
        self.assertEqual(len(compensated_params["TARGET_SCHEDULE"]), 25)


if __name__ == "__main__":
    unittest.main()
