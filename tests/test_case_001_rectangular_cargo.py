import unittest

import numpy as np

from case_loader import build_common_config, load_case
from cases.case_001 import cond_009 as condition_009
from usage4 import _initial_state


class Case001RectangularCargoTests(unittest.TestCase):
    def test_condition_009_uses_rectangular_shape_and_tuned_contact_damping(self):
        circular = build_common_config(load_case("case_001.cond_003"))
        rectangular = build_common_config(load_case("case_001.cond_009"))

        np.testing.assert_allclose(rectangular.PAYLOAD_SIZE, [0.060, 0.015])
        self.assertEqual(rectangular.NUM_ROBOTS, 11)
        self.assertEqual(rectangular.TARGET_SCHEDULE[1][0], 1.0)
        pair_distances = np.linalg.norm(
            rectangular.INITIAL_ROBOT_POSITIONS[:, None, :]
            - rectangular.INITIAL_ROBOT_POSITIONS[None, :, :],
            axis=2,
        )
        pair_distances[pair_distances == 0.0] = np.inf
        self.assertAlmostEqual(np.min(pair_distances), 0.012)
        self.assertEqual(rectangular.PAYLOAD_RADIUS, circular.PAYLOAD_RADIUS)
        self.assertEqual(rectangular.PAYLOAD_MASS, circular.PAYLOAD_MASS)
        self.assertEqual(rectangular.PAYLOAD_DRAG, circular.PAYLOAD_DRAG)
        self.assertEqual(rectangular.CONTACT_STIFFNESS, circular.CONTACT_STIFFNESS)
        self.assertEqual(circular.CONTACT_DAMPING, 5e-5)
        self.assertEqual(rectangular.CONTACT_DAMPING, 1e-5)
        expected_inertia = rectangular.PAYLOAD_MASS * np.sum(
            rectangular.PAYLOAD_SIZE**2
        ) / 12.0
        self.assertAlmostEqual(rectangular.PAYLOAD_INERTIA, expected_inertia)
        initial_state = _initial_state(rectangular)
        self.assertEqual(len(initial_state), rectangular.NUM_ROBOTS * 4 + 6)
        self.assertEqual(initial_state[-2], 0.0)
        self.assertEqual(initial_state[-1], 0.0)
        infinity_and_center = rectangular.TARGET_SCHEDULE[
            -(condition_009.INFINITY_PATH_TARGETS + 1):
        ]
        infinity_points = np.array([entry[1] for entry in infinity_and_center[:-1]])
        self.assertGreater(np.ptp(infinity_points[:, 0]), np.ptp(infinity_points[:, 1]))
        np.testing.assert_allclose(infinity_points[0], [0.060, 0.0], atol=1e-15)
        np.testing.assert_allclose(
            infinity_points[len(infinity_points) // 2], [-0.060, 0.0], atol=1e-15
        )
        np.testing.assert_allclose(infinity_points[-1], [0.060, 0.0], atol=1e-15)
        path_steps = np.linalg.norm(np.diff(infinity_points, axis=0), axis=1)
        self.assertLess(np.max(path_steps), 0.021)

        infinity_angles = np.linspace(0.0, 2.0 * np.pi, len(infinity_points))
        expected_tangents = np.column_stack((
            -0.060 * np.sin(infinity_angles),
            2.0 * 0.025 * np.cos(2.0 * infinity_angles),
        ))
        expected_tangents /= np.linalg.norm(expected_tangents, axis=1, keepdims=True)
        cargo_directions = np.array([
            [np.cos(entry[3] - np.pi / 2.0), np.sin(entry[3] - np.pi / 2.0)]
            for entry in infinity_and_center[:-1]
        ])
        tangent_alignment = np.abs(np.sum(cargo_directions * expected_tangents, axis=1))
        np.testing.assert_allclose(tangent_alignment, 1.0, atol=1e-12)

        np.testing.assert_allclose(infinity_and_center[-1][1], [0.0, 0.0])
        growing_waits = np.diff([entry[0] for entry in infinity_and_center])
        np.testing.assert_allclose(growing_waits[1:] / growing_waits[:-1], 1.10)
        self.assertAlmostEqual(
            rectangular.T_SPAN[1] - infinity_and_center[-1][0],
            growing_waits[-1] * 1.10,
        )
        self.assertEqual(
            rectangular.PAYLOAD_CAPILLARY_GAIN,
            circular.PAYLOAD_CAPILLARY_GAIN,
        )
        self.assertEqual(
            rectangular.PAYLOAD_CAPILLARY_RANGE,
            circular.PAYLOAD_CAPILLARY_RANGE,
        )


if __name__ == "__main__":
    unittest.main()
