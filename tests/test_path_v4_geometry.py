import importlib
import unittest

import numpy as np

from case_loader import build_common_config, load_case, validate_target_schedule


class PathV4GeometryTests(unittest.TestCase):
    def test_condition_007_loads_native_cad_geometry(self):
        condition = importlib.import_module("cases.case_002.cond_007")

        self.assertEqual(len(condition.PATH_V4_EDGE_POLYLINES), 25)
        np.testing.assert_allclose(
            condition.CAD_BOUNDING_BOX,
            [[-0.065, -0.065], [0.065, 0.065]],
        )

    def test_condition_configuration_and_schedule_are_valid(self):
        params = load_case("case_002.cond_007")
        cfg = build_common_config(params)

        validate_target_schedule(params["TARGET_SCHEDULE"])
        self.assertEqual(len(params["INITIAL_ROBOT_POSITIONS"]), 7)
        self.assertTrue(params["PLOT_DRAW_TARGET_PATH"])
        self.assertEqual(cfg.NUM_ROBOTS, 7)

    def test_center_exit_path_has_wall_clearance(self):
        condition = importlib.import_module("cases.case_002.cond_007")
        cfg = build_common_config(load_case("case_002.cond_007"))
        walls = np.asarray(condition.WALL_SEGMENTS)
        starts, ends = walls[:, 0], walls[:, 1]
        directions = ends - starts
        length_squared = np.sum(directions**2, axis=1)

        samples = np.column_stack((
            np.linspace(condition.PATH_POINTS[0, 0], condition.PATH_POINTS[-1, 0], 301),
            np.zeros(301),
        ))
        projections = np.clip(
            np.sum((samples[:, None, :] - starts) * directions, axis=2)
            / length_squared,
            0.0,
            1.0,
        )
        closest = starts + projections[:, :, None] * directions
        clearance = np.min(np.linalg.norm(samples[:, None, :] - closest, axis=2))
        self.assertGreater(clearance, cfg.ROBOT_RADIUS + cfg.WALL_INTERACTION_RANGE)


if __name__ == "__main__":
    unittest.main()
