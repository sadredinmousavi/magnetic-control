import unittest

import numpy as np

from functions_main import calculate_capillary_force
from functions_pm_microrobots import calculate_robot_robot_forces_batch


class RobotCapillaryForceTests(unittest.TestCase):
    def test_pairwise_capillary_force_is_attractive(self):
        positions = np.array([[0.01, 0.0], [0.0, 0.0]])
        radius = 0.001
        gamma = 0.072
        sin_c = 0.3
        coefficient = 2 * np.pi * gamma * radius**2 * sin_c**2
        expected = np.array([-coefficient / 0.01, 0.0])

        batch = calculate_robot_robot_forces_batch(
            positions, m_robot=0.0, robot_radius=radius,
            gamma=gamma, capillary_sin_C=sin_c,
        )
        standalone = calculate_capillary_force(
            positions[1], positions[0], radius, gamma=gamma, sin_C=sin_c,
        )

        np.testing.assert_allclose(batch[0], expected)
        np.testing.assert_allclose(batch[1], -expected)
        np.testing.assert_allclose(standalone, expected)

    def test_pairwise_magnetic_force_remains_repulsive(self):
        positions = np.array([[0.01, 0.0], [0.0, 0.0]])
        batch = calculate_robot_robot_forces_batch(
            positions, m_robot=1e-4, robot_radius=0.001,
            gamma=0.0, capillary_sin_C=0.3,
        )

        self.assertGreater(batch[0, 0], 0.0)
        np.testing.assert_allclose(batch[0], -batch[1])


if __name__ == "__main__":
    unittest.main()
