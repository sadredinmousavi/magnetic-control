import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from axis_calibration import AXIS_TARGETS_M, RobotAxes
from control_workflow import calculate_point_control
from closed_loop_calibration import (
    CalibrationTab, MovingROICalibration, DEFAULT_ROBOT_ROI_RADIUS, DEFAULT_PIN_ROI_RADIUS,
)
from gui005 import ClosedLoopGUI
from robot_vision import RobotDetector


CORNERS = [(10, 10), (210, 10), (210, 210), (10, 210)]


class RobotAxisTests(unittest.TestCase):
    def test_rotated_robot_axes_and_physical_scale(self):
        samples = {"Origin": [100, 100], "+X": [100, 80], "+Y": [98, 100]}
        axes = RobotAxes(CORNERS, 10, 10, samples)
        np.testing.assert_allclose(axes.coordinates(list(samples.values())),
                                   [[0, 0], [1, 0], [0, .1]], atol=1e-6)
        np.testing.assert_allclose(axes.coordinates([[90, 60]]), [[2, .5]], atol=1e-6)

    def test_mirrored_camera_and_scale_not_fitted_from_command_length(self):
        samples = {"Origin": [100, 100], "+X": [124, 100], "+Y": [100, 102]}
        axes = RobotAxes(CORNERS, 10, 10, samples)
        np.testing.assert_allclose(axes.coordinates([[124, 102]]), [[1.2, .1]], atol=1e-6)

    def test_unresolved_or_parallel_movements_rejected(self):
        for samples in (
            {"Origin": [100, 100], "+X": [110, 100], "+Y": [100, 101]},
            {"Origin": [100, 100], "+X": [110, 100], "+Y": [120, 100]},
        ):
            with self.assertRaises(ValueError):
                RobotAxes(CORNERS, 10, 10, samples)

    def app(self):
        app = CalibrationTab.__new__(CalibrationTab)
        app.frame = np.full((220, 220, 3), 240, np.uint8)
        app.corners = CORNERS.copy()
        app.circle = (100, 100, 30)
        app.tracker = None
        app.status, app.axis_status, app.render = Mock(), Mock(), Mock()
        app.axis_samples = {"Origin": [100, 100], "+X": [100, 80], "+Y": [98, 100]}
        app.axis_pending, app.axis_busy, app.axis_job = None, False, 0
        app.capture, app.frozen = None, False
        app.frame_received_at = time.monotonic()
        app.axis_ready_at = app.frame_received_at - .1
        app.axis_capture_button = Mock()
        app.axis_buttons = [Mock(), Mock(), Mock()]
        for name, value in (("width_var", "10"), ("height_var", "10"), ("radius_var", "30"),
                            ("count_var", "1"), ("area_var", "2"),
                            ("control_case_var", "case_100.cond_016_calibration_cartesian")):
            setattr(app, name, Mock(get=Mock(return_value=value)))
        return app

    def test_save_load_restores_axis_mapping_and_legacy_setup_is_accepted(self):
        app = self.app()
        data = app.setup_data()
        self.assertEqual(data["version"], 2)
        self.assertEqual(data["control_case"], "case_100")
        self.assertEqual(data["axis_samples_px"], app.axis_samples)
        expected = RobotAxes(CORNERS, 10, 10, app.axis_samples).coordinates([[90, 60]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "setup.json"
            with path.open("w", encoding="utf-8") as stream:
                json.dump(data, stream)
            with patch("closed_loop_calibration.SETUP_DIR", Path(directory)), \
                 patch("closed_loop_calibration.filedialog.askopenfilename", return_value=str(path)):
                app.axis_samples = {}
                app.load_setup()
                restored = MovingROICalibration(CORNERS, 10, 10, app.circle, 1, 2, app.axis_samples)
                np.testing.assert_allclose(restored.coordinates([[90, 60]]), expected)
                data["version"] = 1
                data["control_case"] = "case_100.cond_016_calibration_cartesian"
                data.pop("axis_samples_px")
                with path.open("w", encoding="utf-8") as stream:
                    json.dump(data, stream)
                app.load_setup()
                app.control_case_var.set.assert_called_with("case_100")
                self.assertEqual(app.axis_samples, {})

    def test_roi_capture_records_single_robot_without_moving_roi(self):
        app = self.app()
        app.axis_samples = {}
        app.axis_pending = "Origin"
        app.detector = RobotDetector()
        cv2.circle(app.frame, (100, 100), 3, (20, 20, 20), -1)
        circle = app.circle
        app.record_axis_from_roi()
        self.assertEqual(app.axis_samples["Origin"], [100, 100])
        self.assertIsNone(app.axis_pending)
        self.assertEqual(app.circle, circle)
        app.axis_capture_button.config.assert_called_with(state="disabled")

    def test_roi_capture_rejects_missing_or_multiple_candidates_without_saving(self):
        app = self.app()
        app.axis_samples = {}
        app.axis_pending = "Origin"
        app.detector = RobotDetector()
        with self.assertRaisesRegex(ValueError, "No robot detected"):
            app.record_axis_from_roi()
        cv2.circle(app.frame, (100, 100), 3, (20, 20, 20), -1)
        cv2.circle(app.frame, (115, 100), 3, (20, 20, 20), -1)
        app.frame_received_at += .01
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            app.record_axis_from_roi()
        self.assertEqual(app.axis_samples, {})
        self.assertEqual(app.axis_pending, "Origin")

    def test_axis_tracking_captures_robot_despite_smaller_noise_blob(self):
        app = self.app()
        app.axis_samples = {}
        app.axis_pending = "Origin"
        cv2.circle(app.frame, (100, 100), 4, (20, 20, 20), -1)
        cv2.circle(app.frame, (115, 100), 1, (20, 20, 20), -1)
        normal = MovingROICalibration(CORNERS, 10, 10, app.circle, 1, 2)
        expected, _, _ = normal.process(app.frame)
        app.record_axis_from_roi()
        self.assertEqual(app.axis_samples["Origin"], list(expected[0]["center"]))

    def test_axis_move_acquires_robot_before_starting_worker(self):
        app = self.app()
        app.axis_tracker = None
        app.axis_after = None
        app.axis_wait_var = Mock(get=Mock(return_value="3"))
        app.axis_move = Mock()
        app.parent = Mock()
        cv2.circle(app.frame, (100, 100), 3, (20, 20, 20), -1)
        with patch("closed_loop_calibration.threading.Thread") as thread:
            app.move_axis("Origin")
        self.assertTrue(app.axis_busy)
        self.assertIsNotNone(app.axis_tracker)
        self.assertEqual(app.axis_tracker.references, [(100, 100)])
        thread.return_value.start.assert_called_once()
        app.axis_move.assert_not_called()

    def test_axis_roi_follows_movement_outside_initial_roi_and_never_captures_lost_frame(self):
        app = self.app()
        app.axis_samples = {"Origin": [100, 100]}
        app.axis_pending = "+X"
        app.frame_number = 0
        app.axis_tracker = MovingROICalibration(CORNERS, 10, 10, app.circle, 1, 2, strict_single=True)
        for x in (100, 120, 140, 160):
            frame = np.full((220, 220, 3), 240, np.uint8)
            cv2.circle(frame, (x, 100), 3, (20, 20, 20), -1)
            cv2.circle(frame, (x + 10, 110), 1, (20, 20, 20), -1)
            app.accept_frame(frame)
            self.assertEqual(tuple(app.axis_tracker.center), (x, 100))
        app.accept_frame(np.full((220, 220, 3), 240, np.uint8))
        with self.assertRaisesRegex(ValueError, "No robot detected"):
            app.record_axis_from_roi()
        self.assertNotIn("+X", app.axis_samples)
        self.assertEqual(tuple(app.axis_tracker.center), (160, 100))
        frame = np.full((220, 220, 3), 240, np.uint8)
        cv2.circle(frame, (160, 100), 3, (20, 20, 20), -1)
        app.accept_frame(frame)
        app.record_axis_from_roi()
        self.assertEqual(app.axis_samples["+X"], [160, 100])

    def test_axis_tracking_blocks_ambiguous_identity_without_moving_roi(self):
        tracker = MovingROICalibration(CORNERS, 10, 10, (100, 100, 30), 1, 2, strict_single=True)
        frame = np.full((220, 220, 3), 240, np.uint8)
        cv2.circle(frame, (100, 100), 3, (20, 20, 20), -1)
        tracker.process(frame)
        frame = np.full((220, 220, 3), 240, np.uint8)
        cv2.circle(frame, (90, 100), 3, (20, 20, 20), -1)
        cv2.circle(frame, (110, 100), 3, (20, 20, 20), -1)
        selected, center, _ = tracker.process(frame)
        self.assertEqual(selected, [])
        self.assertIsNone(center)
        self.assertIn("ambiguous", tracker.selection_error)
        self.assertEqual(tuple(tracker.center), (100, 100))

    def test_roi_capture_rejects_stale_frozen_unmoved_and_uncommanded_frames(self):
        app = self.app()
        app.detector = RobotDetector()
        app.axis_pending = "Origin"
        app.frame_received_at = time.monotonic() - 3
        with self.assertRaisesRegex(ValueError, "fresh live frame"):
            app.record_axis_from_roi()
        app.frame_received_at = time.monotonic()
        app.axis_ready_at = app.frame_received_at + 1
        with self.assertRaisesRegex(ValueError, "new camera frame"):
            app.record_axis_from_roi()
        app.axis_ready_at = app.frame_received_at - .1
        app.frozen = True
        with self.assertRaisesRegex(ValueError, "running live camera"):
            app.record_axis_from_roi()
        app.frozen = False
        app.axis_pending = "+X"
        cv2.circle(app.frame, (100, 100), 3, (20, 20, 20), -1)
        original = dict(app.axis_samples)
        with self.assertRaisesRegex(ValueError, "not moved enough"):
            app.record_axis_from_roi()
        self.assertEqual(app.axis_samples, original)
        app.axis_pending = None
        with self.assertRaisesRegex(ValueError, "Move to a calibration position"):
            app.record_axis_from_roi()

    def test_capture_button_reports_errors_and_keeps_retry_available(self):
        app = self.app()
        app.axis_pending = "Origin"
        app.detector = RobotDetector()
        with patch("closed_loop_calibration.messagebox.showerror") as error:
            app.capture_axis()
        error.assert_called_once()
        self.assertEqual(app.axis_pending, "Origin")

    def test_save_rejects_pending_or_incomplete_axis_capture(self):
        app = self.app()
        app.axis_pending = "+Y"
        with self.assertRaisesRegex(ValueError, "Capture the current"):
            app.setup_data()
        app.axis_pending = None
        app.axis_samples.pop("+Y")
        with self.assertRaisesRegex(ValueError, "Incomplete axes"):
            app.setup_data()

    def test_three_roi_captures_save_and_restore_complete_metric_calibration(self):
        app = self.app()
        app.axis_samples = {}
        app.detector = RobotDetector()
        app.frame_number = 0
        for name, center in (("Origin", (100, 100)), ("+X", (120, 100)), ("+Y", (100, 120))):
            app.axis_pending = name
            app.axis_ready_at = time.monotonic() - .1
            frame = np.full((220, 220, 3), 240, np.uint8)
            cv2.circle(frame, center, 3, (20, 20, 20), -1)
            app.accept_frame(frame)
            app.record_axis_from_roi()
        data = json.loads(json.dumps(app.setup_data()))
        self.assertEqual(data["coordinate_system"], "robot_axes")
        self.assertEqual(data["roi_circle_px"], [100, 120, 30])
        self.assertEqual(data["control_case"], "case_100")
        restored = MovingROICalibration(data["corners_px"], data["rectangle_width_cm"],
                                        data["rectangle_height_cm"], data["roi_circle_px"],
                                        data["robot_count"], data["minimum_area"], data["axis_samples_px"],
                                        contrast_threshold=data["dark_contrast_threshold"])
        np.testing.assert_allclose(restored.coordinates([[100, 100], [120, 100], [100, 120]]),
                                   [[0, 0], [1, 0], [0, 1]], atol=1e-6)

    def pin_app(self):
        app = self.app()
        app.corners = []
        app.detector = RobotDetector()
        app.pin_radius_var = Mock(get=Mock(return_value=str(DEFAULT_PIN_ROI_RADIUS)))
        app.reset_axes = Mock()
        return app

    def test_default_robot_roi_is_300_and_pin_roi_is_small(self):
        self.assertEqual(DEFAULT_ROBOT_ROI_RADIUS, 300)
        self.assertEqual(DEFAULT_PIN_ROI_RADIUS, 35)

    def test_overlay_strokes_use_one_screen_pixel_without_black_halo_at_half_scale(self):
        app = self.app()
        app.frame = np.full((960, 1280, 3), 240, np.uint8)
        app.canvas = Mock()
        app.canvas.winfo_width.return_value = 640
        app.canvas.winfo_height.return_value = 480
        app.detections = []
        with patch("closed_loop_calibration.ImageTk.PhotoImage"), \
             patch("closed_loop_calibration.cv2.circle", wraps=cv2.circle) as circles:
            CalibrationTab.render(app)
        widths = [call.kwargs["thickness"] for call in circles.call_args_list]
        self.assertTrue(widths)
        self.assertEqual(set(widths), {2})
        self.assertTrue(all(call.kwargs['color'] != (0, 0, 0) for call in circles.call_args_list))

    def test_pin_roi_finds_exact_centers_and_orders_four_points(self):
        app = self.pin_app()
        expected = [(40, 40), (180, 40), (180, 180), (40, 180)]
        for point in expected:
            cv2.circle(app.frame, point, 3, (0, 255, 0), -1)
        for x, y in (expected[2], expected[0], expected[3], expected[1]):
            app.capture_pin((x + 8, y + 5))
        np.testing.assert_allclose(app.corners, expected)
        self.assertEqual(app.reset_axes.call_count, 4)
        with self.assertRaisesRegex(ValueError, "already captured"):
            app.capture_pin((40, 40))

    def test_pin_roi_rejects_missing_ambiguous_and_duplicate_pins(self):
        app = self.pin_app()
        with self.assertRaisesRegex(ValueError, "No green pin"):
            app.capture_pin((100, 100))
        self.assertEqual(app.corners, [])
        cv2.circle(app.frame, (100, 100), 3, (0, 255, 0), -1)
        app.capture_pin((108, 100))
        with self.assertRaisesRegex(ValueError, "already captured"):
            app.capture_pin((100, 105))
        cv2.circle(app.frame, (120, 100), 3, (0, 255, 0), -1)
        with self.assertRaisesRegex(ValueError, "2 green candidates"):
            app.capture_pin((110, 100))
        self.assertEqual(app.corners, [(100, 100)])

    def test_manual_pin_placement_bypasses_detection_only_when_explicit(self):
        app = self.pin_app()
        app.detector = Mock()
        app.capture_pin((40, 40), manual=True)
        self.assertEqual(app.corners, [(40, 40)])
        app.detector.process.assert_not_called()
        self.assertIn("manually placed", app.status.config.call_args.kwargs["text"])
        with self.assertRaisesRegex(ValueError, "already captured"):
            app.capture_pin((40, 40), manual=True)
        with self.assertRaisesRegex(ValueError, "inside the camera frame"):
            app.capture_pin((-1, 40), manual=True)

    def test_manual_and_auto_pins_can_be_mixed_and_saved(self):
        app = self.pin_app()
        app.axis_samples = {}
        for point in ((40, 40), (180, 40), (180, 180)):
            app.capture_pin(point, manual=True)
        cv2.circle(app.frame, (40, 180), 3, (0, 255, 0), -1)
        app.capture_pin((45, 180))
        data = app.setup_data()
        np.testing.assert_allclose(data["corners_px"], [(40, 40), (180, 40), (180, 180), (40, 180)])

    def test_pin_roi_moves_in_camera_pixels_without_changing_robot_roi(self):
        app = self.pin_app()
        app.mode = Mock(get=Mock(return_value="Corners"))
        app.transform = (10, 20, .5)
        app.pin_search_circle = None
        circle = app.circle
        app.move_pin_roi(Mock(x=60, y=70))
        self.assertEqual(app.pin_search_circle, (100, 100, 35))
        self.assertEqual(app.circle, circle)
        self.assertEqual(app.corners, [])
        app.clear_pin_roi()
        self.assertIsNone(app.pin_search_circle)

    def test_synthesis_uses_requested_one_robot_target_and_strict_failure_policy(self):
        with patch("control_workflow.build_common_config") as config, \
             patch("control_workflow.run_control_workflow") as workflow:
            workflow.return_value.opt_infos = [{"angles_deg": [60] * 8}]
            self.assertEqual(AXIS_TARGETS_M["+X"], (.01, 0))
            self.assertEqual(AXIS_TARGETS_M["+Y"], (0, .01))
            original = {"TARGET_SCHEDULE": [(9, [2, 3], 2, 1)]}
            result = calculate_point_control(original, AXIS_TARGETS_M["+Y"])
            self.assertEqual(len(result["angles_deg"]), 8)
            params = config.call_args.args[0]
            np.testing.assert_allclose(params["TARGET_SCHEDULE"][0][1], [0, .01])
            self.assertEqual(params["OPTIMIZATION_FAILURE_MODE"], "raise")
            self.assertEqual(params["TARGET_SCHEDULE"][0][2], 1)
            self.assertIsNone(params["TARGET_SCHEDULE"][0][3])
            self.assertEqual(original["TARGET_SCHEDULE"], [(9, [2, 3], 2, 1)])

    def test_calibration_commands_all_three_points_with_ratio_one(self):
        import queue
        app = ClosedLoopGUI.__new__(ClosedLoopGUI)
        app.point_commander = Mock()
        app.point_commander.move.return_value = {"angles_deg": [60] * 8}
        app.camera_event_queue = queue.Queue()
        app.ssh_camera = Mock(is_running=True)
        active = lambda: True
        for target in AXIS_TARGETS_M.values():
            app.move_axis_target(target, "case_100.cond_016_calibration_cartesian", 3, active)
            app.point_commander.move.assert_called_with(
                "case_100.cond_016_calibration_cartesian", target,
                ratio=1.0, angle_deg=None, settle_seconds=3, active=active,
            )
        app.point_commander.move.reset_mock()
        app.move_axis_target((.01, 0), "case_100.cond_016_calibration_cartesian", 0, lambda: False)
        app.point_commander.move.assert_not_called()


if __name__ == "__main__":
    unittest.main()
