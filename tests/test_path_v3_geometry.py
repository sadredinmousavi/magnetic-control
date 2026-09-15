import importlib
import unittest

import numpy as np

from case_loader import load_case, validate_target_schedule
from cases.case_002.path_v3_geometry import CAD_EDGE_POLYLINES, WALL_OPENINGS


class PathV3GeometryTests(unittest.TestCase):
    def test_case_002_condition_005_uses_complete_cad_geometry(self):
        condition = importlib.import_module("cases.case_002.cond_005")

        self.assertEqual(len(condition.PATH_V3_EDGE_POLYLINES), 51)
        self.assertEqual(len(condition.WALL_SEGMENTS), 201)
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
        self.assertEqual(len(params["INITIAL_ROBOT_POSITIONS"]), 3)
        self.assertTrue(params["PLOT_DRAW_TARGET_PATH"])
        self.assertEqual(params["DISH_RADIUS"], 0.065)
        self.assertEqual(params["GRID_MIN"], -0.3)
        self.assertEqual(params["GRID_MAX"], 0.3)
        self.assertGreaterEqual(params["ROBOT_INTERACTION_SCALE"], 0.0)

    def test_marked_junctions_are_open_in_both_conditions(self):
        for condition_name in ("cond_005", "cond_006"):
            condition = importlib.import_module(f"cases.case_002.{condition_name}")
            scale = getattr(condition, "PATH_SCALE", 1.0)
            for center, radius, edge_ids in WALL_OPENINGS:
                scaled_center = scale * center
                clearance = scale * (radius if edge_ids is None else 2e-3)
                for start, end in condition.WALL_SEGMENTS:
                    direction = end - start
                    projection = np.clip(
                        np.dot(scaled_center - start, direction)
                        / np.dot(direction, direction),
                        0.0, 1.0,
                    )
                    closest = start + projection * direction
                    self.assertGreaterEqual(
                        np.linalg.norm(closest - scaled_center) + 1e-12,
                        clearance,
                    )

            for edge_id in (9, 42):
                start, end = CAD_EDGE_POLYLINES[edge_id][:2] * scale
                self.assertTrue(any(
                    np.allclose(wall_start, start) and np.allclose(wall_end, end)
                    for wall_start, wall_end in condition.WALL_SEGMENTS
                ))


if __name__ == "__main__":
    unittest.main()
