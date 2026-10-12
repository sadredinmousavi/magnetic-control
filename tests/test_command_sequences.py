from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from axis_calibration import RobotAxes
from command_sequences import (
    XYStep, read_xy_sequence, resolve_xy_units, project_sequence, execute_xy_sequence,
    point_limits, sequence_diagnostics, validate_projected_points,
)
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
    def parse(self, content, units="auto"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xy.txt"
            path.write_text(content, encoding="utf-8")
            return read_xy_sequence(path, units=units)

    def test_existing_export_format_units_wait_and_zero(self):
        steps = self.parse("# targets_xy_m | wait | zero\n\n[0, 0] | 3 | 180\n[0.01, -0.02] | .5 | 90 # note\n")
        self.assertEqual(steps, [XYStep((0, 0), 3, 180), XYStep((.01, -.02), .5, 90)])

    def test_mm_sequence_projects_into_cm_calibration_and_solves_in_metres(self):
        steps = self.parse("[10, -20] | 3 | 180\n")
        np.testing.assert_allclose(steps[0].target_xy_m, [.01, -.02])
        np.testing.assert_allclose(project_sequence(steps, setup()), [[130, 150]], atol=1e-6)
        commander = Mock()
        commander.move.return_value = {"angles_deg": [60, 70]}
        execute_xy_sequence(steps, commander, "case_100", active=lambda: True, report=Mock())
        np.testing.assert_allclose(commander.move.call_args.args[1], [.01, -.02])

    def test_unit_headers_and_explicit_overrides_are_equivalent(self):
        mm = self.parse("# targets_xy_mm | wait | zero\n[10, -20] | 3 | 180")
        cm = self.parse("# targets_xy_cm | wait | zero\n[1, -2] | 3 | 180")
        meters = self.parse("# targets_xy_m | wait | zero\n[.01, -.02] | 3 | 180")
        self.assertEqual(mm, cm)
        self.assertEqual(mm, meters)
        override = self.parse("# targets_xy_m | wait | zero\n[10, -20] | 3 | 180", units="mm")
        self.assertEqual(override, mm)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            resolve_xy_units("# targets_xy_mm\n# targets_xy_m")

    def test_unit_change_reload_updates_table_without_moving_hardware(self):
        app = CommandTab.__new__(CommandTab)
        app.units_var = Mock(get=Mock(return_value="mm"))
        app.file_label, app.table, app.refresh_calibration = Mock(), Mock(), Mock()
        app.table.get_children.return_value = ()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xy.txt"
            path.write_text("[10, -20] | 3 | 180", encoding="utf-8")
            app.load_path(path)
        values = app.table.insert.call_args.kwargs["values"]
        self.assertEqual(values[:2], ("10", "-20"))
        self.assertEqual(app.input_units, "mm")
        np.testing.assert_allclose(app.steps[0].target_xy_m, [.01, -.02])

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
        app.arrival_gate, app.run_indices, app.stop_event = None, (0,), threading.Event()
        for name in ("table", "status", "render", "set_running", "load_button", "calibration_button", "run_button", "run_selected_button", "extrapolation_check", "stop_button", "units_combo", "next_button", "arrival_tolerance_entry", "arrival_hold_entry"):
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
        app.calibration = Mock()
        app.calibration.setup_data.return_value = setup()
        app.frame = np.zeros((220, 220, 3), np.uint8)
        data, pixels = app.prepare()
        self.assertEqual(data["control_case"], "case_100")
        np.testing.assert_allclose(pixels, [[110, 110]], atol=1e-6)
        app.frame = np.zeros((480, 640, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "size differs"):
            app.prepare()

    def test_preview_remains_visible_when_some_points_are_outside_pins(self):
        steps = [XYStep((0, 0), 0, 180), XYStep((.07, 0), 0, 180)]
        data = setup()
        data["video_size_px"] = [400, 400]
        pixels = project_sequence(steps, data, validate=False)
        outside, off_camera = point_limits(pixels, data)
        np.testing.assert_array_equal(outside, [False, True])
        np.testing.assert_array_equal(off_camera, [False, False])
        validate_projected_points(steps, data, pixels, indices=(0,))
        with self.assertRaisesRegex(ValueError, "Point 2.*four-pin"):
            validate_projected_points(steps, data, pixels)
        validate_projected_points(steps, data, pixels, allow_outside_pins=True)
        self.assertIn("Sequence range", sequence_diagnostics(steps, data, pixels))

    def test_extrapolation_permission_never_allows_outside_camera(self):
        steps = [XYStep((.2, 0), 0, 180)]
        data = setup()
        pixels = project_sequence(steps, data, validate=False)
        with self.assertRaisesRegex(ValueError, "outside the camera frame"):
            validate_projected_points(steps, data, pixels, allow_outside_pins=True)

    def test_shifted_origin_shifts_limits_not_units(self):
        data = setup({"Origin": [120, 110], "+X": [140, 110], "+Y": [120, 90]})
        steps = [XYStep((.05, 0), 0, 180)]
        pixels = project_sequence(steps, data, validate=False)
        outside, _ = point_limits(pixels, data)
        self.assertTrue(outside[0])
        self.assertIn("X [-55.00, 45.00]", sequence_diagnostics(steps, data, pixels))

    def run_app(self):
        app = CommandTab.__new__(CommandTab)
        app.running, app.job, app.active_index = False, 0, None
        app.events, app.stop_event = queue.Queue(), threading.Event()
        app.steps = [XYStep((.07, 0), 0, 180), XYStep((.01, 0), 0, 180)]
        app.calibration = Mock()
        data = setup()
        data["video_size_px"] = [400, 400]
        app.calibration.setup_data.return_value = data
        app.frame = np.zeros((400, 400, 3), np.uint8)
        app.frame_received_at = time.monotonic()
        app.camera_running = lambda: True
        app.commander = PointCommander(lambda: app.port)
        app.port = Mock(is_open=True)
        app.input_units = "mm"
        app.allow_extrapolation_var = Mock(get=Mock(return_value=False))
        app.configure_tracking = Mock()
        app.arrival_gate = None
        app.arrival_tolerance_var = Mock(get=Mock(return_value="1"))
        app.arrival_hold_var = Mock(get=Mock(return_value="0.5"))
        app.tracking_tracker, app.tracking_enabled = Mock(), True
        for name in ("parent", "table", "status", "render", "set_running", "load_button", "calibration_button",
                     "run_button", "run_selected_button", "stop_button", "units_combo", "extrapolation_check", "next_button", "arrival_tolerance_entry", "arrival_hold_entry"):
            setattr(app, name, Mock())
        app.table.selection.return_value = ("1",)
        return app

    def test_run_selected_sends_only_that_row_and_uses_original_row_number(self):
        app = self.run_app()
        with patch("command_tab.messagebox.askyesno", return_value=True), \
             patch("command_tab.ArrivalGate", return_value=Mock(tolerance_mm=1, stable_seconds=.5, waiting_for_next=False)), \
             patch("command_tab.threading.Thread") as thread, \
             patch("command_tab.load_hardware_case", return_value={}), \
             patch.object(app.commander, "calculate", return_value={"angles_deg": [60, 70]}) as calculate, \
             patch("helpers.moveArray") as move:
            app.run_selected()
            thread.call_args.kwargs["target"]()
            app.poll()
        calculate.assert_called_once_with("case_100", (.01, 0), ratio=1.0, angle_deg=None)
        move.assert_called_once()
        app.table.set.assert_any_call("1", "state", "Completed")
        self.assertEqual(app.run_indices, (1,))

    def test_run_all_blocks_unapproved_outside_points_before_starting_worker(self):
        app = self.run_app()
        with patch("command_tab.messagebox.showerror") as error, patch("command_tab.threading.Thread") as thread:
            app.run()
        error.assert_called_once()
        thread.assert_not_called()
        self.assertFalse(app.running)

    def test_run_all_can_extrapolate_only_with_explicit_permission_and_warning(self):
        app = self.run_app()
        app.allow_extrapolation_var.get.return_value = True
        with patch("command_tab.messagebox.askyesno", return_value=True) as confirm, \
             patch("command_tab.ArrivalGate", return_value=Mock(tolerance_mm=1, stable_seconds=.5, waiting_for_next=False)), \
             patch("command_tab.threading.Thread") as thread, \
             patch("command_tab.load_hardware_case", return_value={}), \
             patch.object(app.commander, "calculate", return_value={"angles_deg": [60, 70]}) as calculate, \
             patch("helpers.moveArray") as move:
            app.run()
            thread.call_args.kwargs["target"]()
            app.poll()
        self.assertIn("WARNING", confirm.call_args.args[1])
        self.assertEqual(calculate.call_count, 2)
        self.assertEqual(move.call_count, 2)
        self.assertEqual(app.run_indices, (0, 1))

    def test_show_all_fits_entire_image_and_even_off_frame_targets(self):
        app = CommandTab.__new__(CommandTab)
        app.frame = np.full((400, 400, 3), 240, np.uint8)
        app.setup = setup()
        app.pixels = np.array([[-40., 100.], [440., 100.]])
        app.active_index, app.show_all = 1, True
        app.canvas = Mock()
        app.canvas.winfo_width.return_value = 800
        app.canvas.winfo_height.return_value = 500
        with patch("command_tab.ImageTk.PhotoImage"):
            app.render()
        self.assertTrue(np.all(app.screen_points >= 0))
        self.assertTrue(np.all(app.screen_points < [800, 500]))
        colors = [call.kwargs.get("outline") for call in app.canvas.create_oval.call_args_list]
        self.assertIn("#ff9900", colors)

    def tracking_app(self):
        app = self.run_app()
        app.configure_tracking = CommandTab.configure_tracking.__get__(app)
        app.setup = app.calibration.setup_data.return_value
        app.setup.update(robot_count=1, minimum_area=2, dark_contrast_threshold=10, roi_circle_px=[110, 110, 90])
        app.visible = lambda: False
        app.tracking_tracker = None
        app.tracking_enabled, app.tracking_signature = True, None
        app.tracking_detections, app.tracking_group_xy = [], None
        app.tracking_rows, app.robot_trails = [], {}
        app.tracking_started_at, app.tracking_frame_number = None, 0
        app.tracking_status = Mock()
        for name, value in (("tracking_count_var", "1"), ("tracking_area_var", "2"),
                            ("tracking_contrast_var", "10"), ("tracking_radius_var", "90")):
            setattr(app, name, Mock(get=Mock(return_value=value)))
        app.frame = np.full((400, 400, 3), 240, np.uint8)
        app.steps, app.active_index = [XYStep((.01, 0), 0, 180)], 0
        return app

    def robot_frame(self, points):
        frame = np.full((400, 400, 3), 240, np.uint8)
        for point in points:
            cv2.circle(frame, point, 3, (20, 20, 20), -1)
        return frame

    def test_command_tracking_follows_robot_and_records_xy_and_target_error(self):
        app = self.tracking_app()
        app.start_tracking()
        for x in (110, 130, 160, 190, 230):
            app.accept_live_frame(self.robot_frame([(x, 110)]))
            self.assertEqual(app.tracking_detections[0]["center"], (x, 110))
            np.testing.assert_allclose(app.tracking_tracker.center, [x, 110])
        row = app.tracking_rows[-1]
        self.assertAlmostEqual(row["robot_x_mm"], 60)
        self.assertAlmostEqual(row["desired_x_mm"], 10)
        self.assertAlmostEqual(row["error_mm"], 50)
        self.assertEqual(row["group_valid"], 1)
        self.assertFalse(app.tracking_tracker.limit_to_pins)

    def test_missing_robot_leaves_gap_and_holds_roi_until_reacquisition(self):
        app = self.tracking_app()
        app.start_tracking()
        app.accept_live_frame(self.robot_frame([(130, 110)]))
        previous = app.tracking_tracker.center.copy()
        app.accept_live_frame(self.robot_frame([]))
        np.testing.assert_allclose(app.tracking_tracker.center, previous)
        self.assertEqual(app.tracking_detections, [])
        self.assertEqual(app.tracking_rows[-1]["group_valid"], 0)
        self.assertEqual(app.tracking_rows[-1]["robot_x_mm"], "")
        self.assertEqual(app.tracking_rows[-1]["error_mm"], "")
        app.accept_live_frame(self.robot_frame([(140, 110)]))
        self.assertEqual(app.tracking_rows[-1]["group_valid"], 1)

    def test_multiple_robots_keep_individual_ids_and_partial_group_is_invalid(self):
        app = self.tracking_app()
        app.tracking_count_var.get.return_value = "2"
        app.start_tracking()
        app.accept_live_frame(self.robot_frame([(110, 110), (130, 110)]))
        self.assertEqual({item["robot_id"] for item in app.tracking_detections}, {1, 2})
        self.assertEqual(len(app.tracking_rows[-2:]), 2)
        app.accept_live_frame(self.robot_frame([(115, 110)]))
        self.assertEqual(app.tracking_rows[-1]["group_valid"], 0)
        self.assertEqual(app.tracking_rows[-1]["detected_robots"], 1)

    def test_camera_loss_clears_current_positions_and_tracking_csv_keeps_measurements(self):
        app = self.tracking_app()
        app.start_tracking()
        app.accept_live_frame(self.robot_frame([(130, 110)]))
        app.camera_running = lambda: False
        app.check_camera_health()
        self.assertEqual(app.tracking_detections, [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.csv"
            with patch("command_tab.filedialog.asksaveasfilename", return_value=str(path)):
                app.export_tracking()
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(app.tracking_rows))
        self.assertAlmostEqual(float(rows[-1]["robot_x_mm"]), 10)

    def test_command_view_shows_robot_even_without_loaded_sequence_and_hides_path_by_default(self):
        app = self.tracking_app()
        app.start_tracking()
        app.accept_live_frame(self.robot_frame([(130, 110)]))
        app.pixels, app.active_index, app.show_all = None, None, True
        app.canvas = Mock()
        app.canvas.winfo_width.return_value = 800
        app.canvas.winfo_height.return_value = 500
        with patch("command_tab.ImageTk.PhotoImage"):
            CommandTab.render(app)
        colors = [call.kwargs.get("fill") for call in app.canvas.create_line.call_args_list]
        self.assertEqual(colors, ["#00ff66", "#00ff66"])
        self.assertEqual(app.canvas.create_oval.call_count, 0)
        labels = [call.kwargs["text"] for call in app.canvas.create_text.call_args_list]
        self.assertIn("R1", labels)


if __name__ == "__main__":
    unittest.main()
