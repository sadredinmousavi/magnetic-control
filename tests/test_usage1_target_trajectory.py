import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from usage1 import create_target_trajectory_figure


class Usage1TargetTrajectoryTests(unittest.TestCase):
    def test_creates_field_free_target_overview_with_workspace_geometry(self):
        params = {
            "NUM_SOURCES": 4,
            "RADIUS": 0.25,
            "GRID_MIN": -0.3,
            "GRID_MAX": 0.3,
            "TARGET_SCHEDULE": [
                (0.0, np.array([-0.1, 0.0]), 1.0, 0.0),
                (1.0, np.array([0.0, 0.1]), 1.0, 0.0),
                (2.0, np.array([0.1, 0.0]), 1.0, 0.0),
            ],
            "DISH_RADIUS": 0.15,
            "WALL_SEGMENTS": [
                (np.array([-0.1, -0.1]), np.array([0.1, -0.1])),
            ],
        }

        figure = create_target_trajectory_figure(params)
        axis = figure.axes[0]
        labels = axis.get_legend_handles_labels()[1]

        self.assertIn("Target trajectory", labels)
        self.assertIn("Source magnets", labels)
        self.assertIn("Walls", labels)
        self.assertIn("Petri dish", labels)
        self.assertEqual(axis.get_xlim(), (-0.3, 0.3))
        self.assertEqual(axis.get_ylim(), (-0.3, 0.3))


if __name__ == "__main__":
    unittest.main()
