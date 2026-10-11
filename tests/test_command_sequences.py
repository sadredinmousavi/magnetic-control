from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from axis_calibration import RobotAxes
from command_sequences import XYStep, read_xy_sequence, project_sequence, execute_xy_sequence
from command_tab import CommandTab
from point_control import PointCommander


CORNERS = [(10, 10), (210, 10), (210, 210), (10, 210)]


def setup(samples=None):
    return {
        "coordinate_system": "robot_axes", "corners_px": CORNERS,
        "rectangle_width_cm": 10, "rectangle_height_cm": 10,
        "axis_samples_px": samples or {"Origin": [110, 110], "+X": [130, 110], "+Y": [110, 90]},
        "video_size_px": [220, 220], "control_case": "case_100",
    }


class CommandSequenceTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xy.txt"
            path.write_text(content, encoding="utf-8")
            return read_xy_sequence(path)

    def test_existing_export_format_units_wait_and_zero(self):
        steps = self.parse("# targets_xy_m | wait | zero\n\n[0, 0] | 3 | 180\n[0.01, -0.02] | .5 | 90 # note\n")
        self.assertEqual(steps, [XYStep((0, 0), 3, 180), XYStep((.01, -.02), .5, 90)])

    def test_invalid_rows_fail_before_any_execution(self):
        for row in ("[1, 2, 3] | 3 | 180", "[[0, 0], [1, 1]] | 3 | 180",
                    "[0, 1] | -1 | 180", "[0, 1e999] | 1 | 180", "[0, 1] | 1 | 400",
                    "[0, 1] | 1", "__import__('os').getcwd() | 1 | 180"):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "Line 1"):
                self.parse(row)
        with self.assertRaisesRegex(ValueError, "no points"):
            self.parse("# empty\n")

    def test_inverse_mapping_round_trip_rotations_reflections_and_perspective(self):
        for samples in ({"Origin": [100, 100], "+X": [120, 100], "+Y": [100, 80]},
                        {"Origin": [100, 100], "+X": [100, 80], "+Y": [80, 100]},
                        {"Origin": [100, 100], "+X": [120, 100], "+Y": [100, 120]}):
            axes = RobotAxes([(20, 15), (205, 30), (190, 205), (5, 185)], 10, 8, samples)
            coordinates = np.array([[0, 0], [1, 0], [0, 1], [-2, 1]])
            np.testing.assert_allclose(axes.coordinates(axes.pixel_positions(coordinates)), coordinates, atol=1e-6)

    def test_projection_uses_meters_and_robot_axes_not_image_axes(self):
        pixels = project_sequence([XYStep((0, 0), 0, 180), XYStep((.01, .01), 0, 180)], setup())
        np.testing.assert_allclose(pixels, [[110, 110], [130, 90]], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "Point 1.*outside"):
            project_sequence([XYStep((.2, 0), 0, 180)], setup())
        data = setup()
        data["coordinate_system"] = "rectangle_center"
        with self.assertRaisesRegex(ValueError, "Capture Center"):
            project_sequence([XYStep((0, 0), 0, 180)], data)

    def test_sequence_calculates_fresh_and_sends_every_point_with_row_zero(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        steps = [XYStep((0, 0), 0, 180), XYStep((.01, 0), 0, 90), XYStep((0, .01), 0, 180)]
        report = Mock()
        with patch.object(commander, "calculate", side_effect=[{"angles_deg": [60, 70]},
                                                               {"angles_deg": [61, 71]},
                                                               {"angles_deg": [62, 72]}]) as calculate, \
             patch("helpers.moveArray") as move:
            outcome = execute_xy_sequence(steps, commander, "case_100", active=lambda: True, report=report)
        self.assertEqual(outcome, "finished")
        self.assertEqual(calculate.call_count, 3)
        self.assertEqual([call.args[3] for call in move.call_args_list], [180, 90, 180])
        self.assertEqual([call.args[0] for call in report.call_args_list], ["calculating", "sent", "completed"] * 3)

    def test_stop_during_calculation_prevents_send_and_later_points(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        stop = threading.Event()

        def calculate(*args, **kwargs):
            stop.set()
            return {"angles_deg": [60, 70]}

        with patch.object(commander, "calculate", side_effect=calculate), patch("helpers.moveArray") as move:
            outcome = execute_xy_sequence([XYStep((0, 0), 0, 180)] * 3, commander, "case_100",
                                          active=lambda: not stop.is_set(), report=Mock())
        self.assertEqual(outcome, "stopped")
        move.assert_not_called()

    def test_solver_failure_stops_without_sending(self):
        commander = PointCommander(lambda: Mock(is_open=True))
        with patch.object(commander, "calculate", side_effect=RuntimeError("no stable solution")) as calculate, \
             patch("helpers.moveArray") as move:
            with self.assertRaisesRegex(RuntimeError, "no stable"):
                execute_xy_sequence([XYStep((0, 0), 0, 180)] * 3, commander, "case_100",
                                    active=lambda: True, report=Mock())
        self.assertEqual(calculate.call_count, 1)
        move.assert_not_called()

    def test_stop_after_sent_interrupts_dwell_and_skips_next_point(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        stop = threading.Event()

        def report(kind, index, result):
            if kind == "sent":
                stop.set()

        with patch.object(commander, "calculate", return_value={"angles_deg": [60, 70]}), \
             patch("helpers.moveArray") as move:
            outcome = execute_xy_sequence([XYStep((0, 0), 60, 180)] * 3, commander, "case_100",
                                          active=lambda: not stop.is_set(), report=report)
        self.assertEqual(outcome, "stopped")
        move.assert_called_once()

    def test_gui_event_poll_updates_angles_and_finishes_on_ui_thread(self):
        app = CommandTab.__new__(CommandTab)
        app.events, app.job, app.running = queue.Queue(), 1, True
        app.steps = [XYStep((0, 0), 3, 180)]
        app.active_index = None
        for name in ("table", "status", "render", "set_running", "load_button", "calibration_button", "run_button", "stop_button"):
            setattr(app, name, Mock())
        app.events.put((1, "sent", 0, {"angles_deg": [60, 70]}))
        app.events.put((1, "completed", 0, {"angles_deg": [60, 70]}))
        app.events.put((1, "finished", None, None))
        app.poll()
        self.assertFalse(app.running)
        app.table.set.assert_any_call("0", "angles", "60.00, 70.00")
        app.set_running.assert_called_with(False)

    def test_command_preview_requires_matching_live_calibration(self):
        app = CommandTab.__new__(CommandTab)
        app.steps = [XYStep((0, 0), 0, 180)]
        app.calibration = Mock(capture=None)
        app.calibration.setup_data.return_value = setup()
        app.frame = np.zeros((220, 220, 3), np.uint8)
        data, pixels = app.prepare()
        self.assertEqual(data["control_case"], "case_100")
        np.testing.assert_allclose(pixels, [[110, 110]], atol=1e-6)
        app.frame = np.zeros((480, 640, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "size differs"):
            app.prepare()


if __name__ == "__main__":
    unittest.main()
