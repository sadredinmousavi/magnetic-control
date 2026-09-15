import importlib
import unittest

import numpy as np

from case_loader import build_common_config, load_case, validate_target_schedule
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

            for edge_id in (8, 16, 14, 43):
                junction = CAD_EDGE_POLYLINES[edge_id][0] * scale
                matching_ends = sum(
                    np.allclose(point, junction)
                    for start, end in condition.WALL_SEGMENTS
                    for point in (start, end)
                )
                self.assertGreaterEqual(matching_ends, 2)

    def test_both_paths_start_at_center_and_sample_zigzag_segments(self):
        for condition_name in ("cond_005", "cond_006"):
            condition = importlib.import_module(f"cases.case_002.{condition_name}")
            cfg = build_common_config(load_case(f"case_002.{condition_name}"))
            scale = getattr(condition, "PATH_SCALE", 1.0)
            entry_count = len(condition.ENTRY_PATH_POINTS)
            np.testing.assert_allclose(
                condition.PATH_POINTS[:entry_count], scale * condition.ENTRY_PATH_POINTS
            )
            np.testing.assert_allclose(condition.PATH_POINTS[0], [0.0, 0.0])
            self.assertEqual(entry_count, 6)
            np.testing.assert_allclose(condition.ENTRY_PATH_POINTS[-1], [-0.035, 0.0])
            np.testing.assert_allclose(np.diff(condition.ENTRY_PATH_POINTS[:, 0]), -0.007)
            self.assertTrue(all(
                target[2] == 5.0
                for target in condition.TARGET_SCHEDULE[:entry_count]
            ))

            lower_start = (len(condition.ENTRY_PATH_POINTS)
                           + len(condition.UPPER_PATH_POINTS)
                           + len(condition.TRANSITION_PATH_POINTS))
            lower = condition.PATH_POINTS[lower_start:]
            np.testing.assert_allclose(lower[::2], scale * condition.LOWER_PATH_POINTS)
            np.testing.assert_allclose(lower[1::2], 0.5 * (lower[:-2:2] + lower[2::2]))

            for segment_index, target_index in enumerate(
                range(lower_start + 1, len(condition.PATH_POINTS) - 1, 2)
            ):
                target = condition.TARGET_SCHEDULE[target_index]
                self.assertGreater(target[2], 1.0)
                tangent = lower[2 * segment_index + 2] - lower[2 * segment_index]
                tangent /= np.linalg.norm(tangent)
                long_axis = np.array([
                    np.cos(target[3] + np.pi / 2),
                    np.sin(target[3] + np.pi / 2),
                ])
                self.assertAlmostEqual(abs(np.dot(long_axis, tangent)), 1.0)

            walls = np.asarray(condition.WALL_SEGMENTS)
            starts, ends = walls[:, 0], walls[:, 1]
            direction = ends - starts
            length_squared = np.sum(direction**2, axis=1)
            for first, second in zip(
                condition.PATH_POINTS[:entry_count],
                condition.PATH_POINTS[1:entry_count + 1],
            ):
                samples = first + np.linspace(0.0, 1.0, 101)[:, None] * (second - first)
                projection = np.clip(
                    np.sum((samples[:, None, :] - starts) * direction, axis=2)
                    / length_squared,
                    0.0, 1.0,
                )
                closest = starts + projection[:, :, None] * direction
                clearance = np.min(np.linalg.norm(samples[:, None, :] - closest, axis=2))
                self.assertGreater(
                    clearance, cfg.ROBOT_RADIUS + cfg.WALL_INTERACTION_RANGE
                )

    def test_condition_005_matches_condition_006_shape_schedule(self):
        condition_005 = importlib.import_module("cases.case_002.cond_005")
        condition_006 = importlib.import_module("cases.case_002.cond_006")
        self.assertEqual(len(condition_005.TARGET_SCHEDULE), len(condition_006.TARGET_SCHEDULE))
        for target_005, target_006 in zip(
            condition_005.TARGET_SCHEDULE, condition_006.TARGET_SCHEDULE
        ):
            self.assertEqual(target_005[2], target_006[2])
            self.assertAlmostEqual(
                np.cos(2.0 * (target_005[3] - target_006[3])), 1.0
            )


if __name__ == "__main__":
    unittest.main()
