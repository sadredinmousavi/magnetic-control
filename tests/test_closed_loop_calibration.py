import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from closed_loop_calibration import CalibrationTab, MovingROICalibration


class MovingROICalibrationTests(unittest.TestCase):
    def tracker(self):
        return MovingROICalibration(
            [(10, 10), (190, 10), (190, 190), (10, 190)],
            18, 18, (80, 100, 30), 2, 2,
        )

    def frame(self, centers):
        frame = np.full((200, 200, 3), 240, np.uint8)
        for point in centers:
            cv2.circle(frame, point, 3, (20, 20, 20), -1)
        return frame

    def test_roi_follows_robots_and_ignores_distant_noise(self):
        tracker = self.tracker()
        for step in range(12):
            x = 75 + 5 * step
            detections, center, xy = tracker.process(self.frame([
                (x, 95), (x + 10, 105), (25, 170),
            ]))
            self.assertEqual(len(detections), 2)
            np.testing.assert_allclose(center, (x + 5, 100), atol=1)
            np.testing.assert_allclose(tracker.center, center)
            np.testing.assert_allclose(xy, ((x + 5 - 100) / 10, 0), atol=.1)

    def test_missing_robot_holds_roi_and_leaves_group_xy_empty(self):
        tracker = self.tracker()
        tracker.process(self.frame([(75, 95), (85, 105)]))
        previous = tracker.center.copy()
        detections, center, xy = tracker.process(self.frame([(80, 95)]))
        self.assertEqual(len(detections), 1)
        self.assertIsNone(center)
        self.assertIsNone(xy)
        np.testing.assert_allclose(tracker.center, previous)
        self.assertIsNotNone(tracker.process(self.frame([(80, 95), (90, 105)]))[1])

    def test_physical_coordinates_have_y_up(self):
        tracker = self.tracker()
        detections, _, _ = tracker.process(self.frame([(75, 95), (85, 105)]))
        top_robot = min(detections, key=lambda item: item["center"][1])
        self.assertGreater(top_robot["xy_cm"][1], 0)

    def test_rejects_invalid_calibration(self):
        with self.assertRaises(ValueError):
            MovingROICalibration([(10, 10)] * 4, 10, 10, (50, 50, 20), 2, 1)
        with self.assertRaises(ValueError):
            MovingROICalibration([(10, 10), (100, 10), (100, 100), (10, 100)],
                                 10, 10, (50, 50, -1), 2, 1)

    def test_movie_tracks_every_frame_through_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "robots.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (200, 200))
            self.assertTrue(writer.isOpened())
            for step in range(10):
                writer.write(self.frame([(75 + step * 4, 95), (85 + step * 4, 105)]))
            writer.release()
            capture = cv2.VideoCapture(path)
            try:
                app = CalibrationTab.__new__(CalibrationTab)
                app.parent, app.status, app.render = Mock(), Mock(), Mock()
                app.capture = capture
                ok, app.frame = capture.read()
                self.assertTrue(ok)
                app.tracker = self.tracker()
                app.frame_number, app.fps = 0, 20
                app.playing, app.after_id = True, None
                app.rows, app.detections = [], []
                app.accept_frame(app.frame)
                while app.playing:
                    app.movie_tick()
                self.assertEqual(sorted({row[0] for row in app.rows}), list(range(10)))
                self.assertEqual(len(app.rows), 20)
                self.assertTrue(all(row[2] == 1 for row in app.rows))
                self.assertAlmostEqual(app.rows[-1][1], 9 / 20)
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
