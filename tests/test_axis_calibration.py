import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from axis_calibration import AXIS_TARGETS_M, RobotAxes
from control_workflow import calculate_point_control
from closed_loop_calibration import CalibrationTab, MovingROICalibration
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
                data.pop("axis_samples_px")
                with path.open("w", encoding="utf-8") as stream:
                    json.dump(data, stream)
                app.load_setup()
                self.assertEqual(app.axis_samples, {})

    def test_measurement_click_snaps_to_robot_and_rejects_empty_click(self):
        app = self.app()
        app.axis_samples = {}
        app.axis_pending = "Origin"
        app.detector = RobotDetector()
        cv2.circle(app.frame, (100, 100), 3, (20, 20, 20), -1)
        app.mark_axis((105, 102))
        self.assertEqual(app.axis_samples["Origin"], [100, 100])
        self.assertIsNone(app.axis_pending)
        app.axis_pending = "+X"
        with self.assertRaises(ValueError):
            app.mark_axis((170, 170))
        self.assertNotIn("+X", app.axis_samples)

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
