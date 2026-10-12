import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from closed_loop_calibration import CalibrationTab, MovingROICalibration
from robot_vision import RobotDetector
from gui004 import circle_detection_mask, workspace_detection_mask


class MovingROICalibrationTests(unittest.TestCase):
    def test_unannotated_detector_does_not_copy_or_modify_source(self):
        frame = self.frame([(80, 100)])
        original = frame.copy()
        output, detections = RobotDetector().process(frame, mode='Color blobs', color='Dark contrast',
            minimum_area=1, morphology_kernel_size=1, draw_annotations=False, dark_contrast_threshold=10)
        self.assertIs(output, frame)
        self.assertTrue(detections)
        np.testing.assert_array_equal(frame, original)

    def test_hidden_idle_axis_tracker_skips_detection_but_keeps_latest_frame(self):
        app = CalibrationTab.__new__(CalibrationTab)
        app.frame = None
        app.visible = lambda: False
        app.axis_busy, app.axis_pending = False, None
        app.axis_tracker, app.update_axis_tracking = Mock(), Mock()
        app.tracker, app.render, app.frame_number = None, Mock(), 0
        frame = self.frame([])
        app.accept_live_frame(frame)
        app.update_axis_tracking.assert_not_called()
        self.assertIs(app.frame, frame)
        app.axis_pending = 'Origin'
        app.accept_live_frame(frame)
        app.update_axis_tracking.assert_called_once()

    def test_native_roi_detection_matches_full_frame_including_edges_and_coordinates(self):
        rng = np.random.default_rng(123)
        frame = np.clip(210 + rng.normal(0, 3, (700, 900, 3)), 0, 255).astype(np.uint8)
        for point in ((450, 350), (529, 350), (373, 350), (450, 272), (20, 20), (3, 80)):
            cv2.circle(frame, point, 3, (20, 20, 20), -1)
        corners = [(10.5, 10.5), (850.5, 40.5), (840.5, 650.5), (20.5, 660.5)]
        for circle in ((450.25, 350.25, 80), (30, 40, 60), (850, 630, 80)):
            for limit_to_pins in (True, False):
                tracker = MovingROICalibration(corners, 10, 10, circle, 1, 1, limit_to_pins=limit_to_pins)
                mask = circle_detection_mask(frame.shape[:2], circle)
                if limit_to_pins:
                    mask = cv2.bitwise_and(mask, workspace_detection_mask(frame.shape[:2], corners))
                _, expected = RobotDetector().process(frame, mode='Color blobs', color='Dark contrast',
                    minimum_area=1, morphology_kernel_size=1, draw_annotations=False,
                    detection_mask=mask, dark_contrast_threshold=10)
                actual = tracker.detect_candidates(frame)
                self.assertEqual(actual, expected)

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

    def test_faint_one_pixel_robot_is_trackable_with_lower_contrast(self):
        tracker = MovingROICalibration(
            [(10, 10), (190, 10), (190, 190), (10, 190)],
            18, 18, (80, 100, 30), 1, 1, contrast_threshold=10,
        )
        for x in range(80, 100):
            frame = np.full((200, 200, 3), 200, np.uint8)
            frame[100, x] = 182
            detections, center, xy = tracker.process(frame)
            self.assertEqual(len(detections), 1)
            self.assertEqual(center, (x, 100))
            self.assertEqual(detections[0]["area"], 1)

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

    def test_live_frames_use_elapsed_time_and_never_pause(self):
        app = CalibrationTab.__new__(CalibrationTab)
        app.parent, app.status, app.render = Mock(), Mock(), Mock()
        app.frame = None
        app.tracker = self.tracker()
        app.frame_number, app.live_started = 0, 100
        app.rows, app.detections = [], []
        for step in range(10):
            with patch('closed_loop_calibration.time.monotonic', return_value=100 + step / 20):
                app.accept_live_frame(self.frame([(75 + step * 4, 95), (85 + step * 4, 105)]))
        self.assertEqual(sorted({row[0] for row in app.rows}), list(range(10)))
        self.assertEqual(len(app.rows), 20)
        self.assertTrue(all(row[2] == 1 for row in app.rows))
        self.assertAlmostEqual(app.rows[-1][1], 9 / 20)
        app.accept_live_frame(self.frame([]))
        self.assertEqual(len(app.rows), 21)
        self.assertEqual(app.frame_number, 11)

    def test_movie_playback_api_is_removed(self):
        for method in ('open_movie', 'toggle_play', 'movie_tick', 'rewind', 'close_capture', 'use_live', 'freeze_live'):
            self.assertFalse(hasattr(CalibrationTab, method), method)

    def test_setup_save_uses_setup_folder_and_contains_four_points_and_roi(self):
        app = CalibrationTab.__new__(CalibrationTab)
        app.frame = self.frame([])
        app.corners = [(10, 10), (190, 10), (190, 190), (10, 190)]
        app.circle = (80, 100, 30)
        app.tracker = None
        app.status = Mock()
        for name, value in (("width_var", "18"), ("height_var", "18"),
                            ("radius_var", "30"), ("count_var", "2"), ("area_var", "2")):
            setattr(app, name, Mock(get=Mock(return_value=value)))
        with tempfile.TemporaryDirectory() as directory:
            setup_dir = Path(directory) / "setups"
            path = setup_dir / "20261011_143025.json"
            with patch("closed_loop_calibration.SETUP_DIR", setup_dir), \
                 patch("closed_loop_calibration.default_setup_name", return_value=path.name), \
                 patch("closed_loop_calibration.filedialog.asksaveasfilename", return_value=str(path)) as dialog:
                app.save_setup()
            self.assertEqual(dialog.call_args.kwargs["initialdir"], str(setup_dir))
            self.assertEqual(dialog.call_args.kwargs["initialfile"], path.name)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["corners_px"], [list(point) for point in app.corners])
            self.assertEqual(data["roi_circle_px"], [80, 100, 30])
            self.assertEqual(data["rectangle_width_cm"], 18)


if __name__ == "__main__":
    unittest.main()
