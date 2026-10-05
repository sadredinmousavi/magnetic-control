import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import usage5


class Usage5DwellPointTests(unittest.TestCase):
    def test_schedule_matches_forward_reverse_grid_passes(self):
        schedule = usage5.build_expected_schedule(3)
        self.assertEqual(schedule["Grid Point"].tolist(), [
            *range(1, 10), *range(9, 0, -1), *range(1, 10),
        ])

    def test_loader_keeps_one_swarm_center_per_video_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.csv"
            pd.DataFrame({
                "time_s": [0.0, 0.0, 1.0, 1.0],
                "robot_center_x_px": [10, 10, 11, 11],
                "robot_center_y_px": [20, 20, 21, 21],
                "accepted": [1, 1, 1, 1],
            }).to_csv(path, index=False)
            loaded = usage5.load_tracking_file(path)
        self.assertEqual(len(loaded), 2)
        np.testing.assert_allclose(loaded[usage5.X_COLUMN], [10, 11])

    def test_rejects_grid_other_than_three_or_five(self):
        with self.assertRaisesRegex(ValueError, "must be 3 or 5"):
            usage5.main(grid_size=4, input_filename="unused.csv")

    def test_saves_dwell_visits_as_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dwells.csv"
            dwells = pd.DataFrame({
                "Dwell ID": [1], "Pass": [1], "Grid Point": [1],
                "Schedule ID": [1], "Start (s)": [0.0], "End (s)": [2.0],
                "Duration (s)": [2.0], "Median X (px)": [10.0],
                "Median Y (px)": [20.0], "Mean X (px)": [10.0],
                "Mean Y (px)": [20.0], "Std X (px)": [0.1],
                "Std Y (px)": [0.2], "Frames": [60],
                "Distance to Reference (px)": [0.0],
            })
            usage5.save_results(dwells, output)
            saved = pd.read_csv(output)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.loc[0, "Grid Point"], 1)


if __name__ == "__main__":
    unittest.main()
