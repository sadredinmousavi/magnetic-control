import unittest

import numpy as np

from functions_main import calculate_robot_rectangular_payload_interaction_force
from functions_pm_microrobots import microrobot_payload_dynamics


class RectangularPayloadCollisionTests(unittest.TestCase):
    def calculate_force(
        self,
        robot_pos,
        robot_vel=(0.0, 0.0),
        payload_angle=0.0,
        return_contact_point=False,
    ):
        return calculate_robot_rectangular_payload_interaction_force(
            robot_pos=np.asarray(robot_pos),
            robot_vel=np.asarray(robot_vel),
            payload_pos=np.zeros(2),
            payload_vel=np.zeros(2),
            robot_radius=0.001,
            payload_size=np.array([0.060, 0.015]),
            k_contact=2.0,
            c_contact=0.0,
            capillary_gain=0.0,
            capillary_range=0.007,
            capillary_cutoff=0.021,
            payload_angle=payload_angle,
            return_contact_point=return_contact_point,
        )

    def test_contact_uses_long_rectangle_edge(self):
        force = self.calculate_force([0.0305, 0.0])
        self.assertLess(force[0], 0.0)
        self.assertAlmostEqual(force[1], 0.0)
        self.assertAlmostEqual(force[0], -0.001, places=12)

    def test_robot_beyond_contact_range_has_no_force(self):
        force = self.calculate_force([0.0311, 0.0])
        np.testing.assert_allclose(force, np.zeros(2))

    def test_corner_contact_uses_diagonal_normal(self):
        offset = 0.0005 / np.sqrt(2.0)
        force = self.calculate_force([0.030 + offset, 0.0075 + offset])
        self.assertLess(force[0], 0.0)
        self.assertLess(force[1], 0.0)
        self.assertAlmostEqual(force[0], force[1])

    def test_inside_robot_is_pushed_toward_nearest_exit(self):
        force_on_payload = self.calculate_force([0.0295, 0.0])
        self.assertLess(force_on_payload[0], 0.0)
        self.assertAlmostEqual(force_on_payload[1], 0.0)

    def test_rotated_rectangle_uses_its_local_edges(self):
        force = self.calculate_force([0.0, 0.0305], payload_angle=np.pi / 2.0)
        self.assertAlmostEqual(force[0], 0.0, places=12)
        self.assertAlmostEqual(force[1], -0.001, places=12)

    def test_contact_point_produces_off_center_torque(self):
        force, contact_point = self.calculate_force(
            [0.0305, 0.003], return_contact_point=True
        )
        torque = contact_point[0] * force[1] - contact_point[1] * force[0]
        self.assertGreater(torque, 0.0)

    def test_off_center_contact_accelerates_payload_rotation(self):
        state = np.array([
            0.0305, 0.003, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        target_controls = [
            (0.0, np.zeros(2), 1.0, 0.0, np.array([])),
        ]
        derivative = microrobot_payload_dynamics(
            0.0,
            state,
            np.empty((0, 2)),
            target_controls,
            0.0,
            0.0,
            1.0,
            1.0,
            0.001,
            0.0,
            0.0,
            0.015,
            1.0,
            1.0,
            2.0,
            0.0,
            payload_capillary_gain=0.0,
            payload_size=np.array([0.060, 0.015]),
            payload_inertia=1e-4,
            payload_angular_drag=0.0,
        )
        self.assertGreater(derivative[-1], 0.0)

if __name__ == "__main__":
    unittest.main()
