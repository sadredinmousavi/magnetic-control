import importlib
import unittest

import numpy as np

from case_loader import load_case, validate_target_schedule


class PathV3GeometryTests(unittest.TestCase):
    def test_case_002_condition_005_uses_complete_cad_geometry(self):
        condition = importlib.import_module("cases.case_002.cond_005")

        self.assertEqual(len(condition.PATH_V3_EDGE_POLYLINES), 51)
        self.assertEqual(len(condition.WALL_SEGMENTS), 219)
        all_points = np.vstack(condition.PATH_V3_EDGE_POLYLINES)
        self.assertTrue(np.all(all_points >= -0.065))
        self.assertTrue(np.all(all_points <= 0.065))
        self.assertTrue(np.allclose(
            condition.CAD_BOUNDING_BOX,
            [[-0.065, -0.065], [0.065, 0.065]],
        ))

    def test_condition_configuration_and_schedule_are_valid(self):
        params = load_case("case_002.cond_005")

        validate_target_schedule(params["TARGET_SCHEDULE"])
        self.assertEqual(len(params["INITIAL_ROBOT_POSITIONS"]), 7)
        self.assertTrue(params["PLOT_DRAW_TARGET_PATH"])
        self.assertEqual(params["DISH_RADIUS"], 0.065)
        self.assertEqual(params["GRID_MIN"], -0.3)
        self.assertEqual(params["GRID_MAX"], 0.3)
        self.assertGreaterEqual(params["ROBOT_INTERACTION_SCALE"], 0.0)


if __name__ == "__main__":
    unittest.main()
