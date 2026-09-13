import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")

from usage1 import create_target_trajectory_figure, save_all_target_trajectories


class Usage1TargetTrajectoryTests(unittest.TestCase):
    @staticmethod
    def _params():
        return {
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

    def test_creates_field_free_target_overview_with_workspace_geometry(self):
        params = self._params()

        figure = create_target_trajectory_figure(params)
        axis = figure.axes[0]
        labels = axis.get_legend_handles_labels()[1]

        self.assertIn("Target trajectory", labels)
        self.assertIn("Source magnets", labels)
        self.assertIn("Walls", labels)
        self.assertIn("Petri dish", labels)
        self.assertEqual(axis.get_xlim(), (-0.3, 0.3))
        self.assertEqual(axis.get_ylim(), (-0.3, 0.3))

    def test_batch_saves_named_plots_without_showing_them(self):
        cases = ["case_001.cond_001", "case_002.cond_006"]
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "usage1.load_case", return_value=self._params()
        ), patch("usage1.plt.show") as show:
            paths = save_all_target_trajectories(cases, Path(temp_dir))

            self.assertEqual([path.name for path in paths], [
                "case_001_cond_001_target_trajectory.png",
                "case_002_cond_006_target_trajectory.png",
            ])
            self.assertTrue(all(path.is_file() for path in paths))
            show.assert_not_called()


if __name__ == "__main__":
    unittest.main()
