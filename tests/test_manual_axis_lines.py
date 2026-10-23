"""Pen-based axes: metric geometry, click workflow and setup compatibility."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from axis_calibration import RobotAxes, manual_line_axes
from closed_loop_calibration import CalibrationTab


class Variable:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class ManualAxisTests(unittest.TestCase):
    def fixture(self):
        tab = CalibrationTab.__new__(CalibrationTab)
        tab.frame = np.zeros((240, 240, 3), np.uint8)
        tab.corners = [(10, 10), (230, 10), (230, 230), (10, 230)]
        tab.circle, tab.tracker = None, None
        tab.axis_busy = tab.axis_prepare_busy = tab.command_running = False
        tab.axis_pending, tab.axis_job = None, 0
        tab.axis_samples, tab.axis_capture_moves, tab.axis_raw_captures, tab.axis_line_points = {}, {}, {}, {}
        tab.axis_buttons = [Mock() for _ in range(7)]
        tab.status, tab.axis_status, tab.axis_capture_button, tab.render = Mock(), Mock(), Mock(), Mock()
        tab.axis_move, tab.axis_prepare = Mock(), Mock()
        tab.transform = (20, 30, 2.)
        for key, value in dict(width_var='10', height_var='10', radius_var='30', count_var='1',
                               area_var='1', control_case_var='case_100',
                               axis_method_var='Manual axis lines', mode='AxisLines').items():
            setattr(tab, key, Variable(value))
        return tab

    def capture(self, tab):
        for point in ((50, 120), (190, 120), (120, 200), (120, 40)):
            tab.record_line_point(point)

    def test_four_clicks_in_order_no_robot_or_motor_and_undo(self):
        tab = self.fixture()
        for index, point in enumerate(((50, 120), (190, 120), (120, 200), (120, 40))):
            tab.click(SimpleNamespace(x=20+2*point[0], y=30+2*point[1]))
            self.assertEqual(list(tab.axis_line_points), ['-X', '+X', '-Y', '+Y'][:index+1])
            self.assertEqual(bool(tab.axis_samples), index == 3)
        np.testing.assert_allclose(tab.axis_samples['Origin'], [120, 120], atol=1e-5)
        tab.move_axis('+X')
        tab.prepare_axis_moves()
        tab.axis_move.assert_not_called()
        tab.axis_prepare.assert_not_called()
        self.assertTrue(tab.calibration_progress()[1][2])  # No robot required.
        tab.undo_line_point()
        self.assertEqual(list(tab.axis_line_points), ['-X', '+X', '-Y'])
        self.assertEqual(tab.axis_samples, {})
        tab.record_line_point((120, 40))
        self.assertEqual(len(tab.axis_samples), 3)

    def test_metric_intersection_and_signs_with_perspective_and_mirrored_y(self):
        corners = [(20, 15), (205, 30), (190, 205), (5, 185)]
        project = RobotAxes(corners, 10, 10, {'Origin': [100, 100], '+X': [125, 100], '+Y': [100, 125]})
        # Endpoints need not be symmetric, nor bracket the origin.
        physical = np.array([[.5, 0], [2., 0], [0, -1.7], [0, .7]])
        pixels = project.pixel_positions(physical)
        samples, report = manual_line_axes(corners, 10, 10, dict(zip(('-X', '+X', '-Y', '+Y'), pixels)))
        fitted = RobotAxes(corners, 10, 10, samples)
        np.testing.assert_allclose(fitted.coordinates(pixels), physical, atol=1e-5)
        self.assertIn('90.00', report)
        self.assertNotIn('Center', report)

    def test_bad_last_point_keeps_previous_clicks_for_retry(self):
        tab = self.fixture()
        for point in ((50, 120), (190, 120), (120, 120)):
            tab.record_line_point(point)
        tab.record_line_point((180, 120))
        self.assertEqual(len(tab.axis_line_points), 3)
        self.assertEqual(tab.axis_samples, {})
        self.assertIn('parallel', tab.axis_status.config.call_args.kwargs['text'])
        tab.record_line_point((120, 40))
        self.assertEqual(len(tab.axis_samples), 3)

    def test_no_roi_setup_save_load_and_tracking_requires_roi(self):
        tab = self.fixture()
        self.capture(tab)
        data = tab.setup_data()
        self.assertEqual(data['axis_method'], 'manual_lines')
        self.assertIsNone(data['roi_circle_px'])
        self.assertEqual(data['coordinate_system'], 'robot_axes')
        with self.assertRaisesRegex(ValueError, 'ROI'):
            tab.setup_data(require_capture=False)
        preview = tab.setup_data(require_capture=False, require_roi=False)
        self.assertEqual(preview['coordinate_system'], 'robot_axes')
        self.assertIsNone(preview['roi_circle_px'])
        # The raw clicks, not a stale derived origin, are authoritative.
        data['axis_samples_px']['Origin'] = [80, 90]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manual.json'
            path.write_text(json.dumps(data))
            restored = self.fixture()
            with patch('closed_loop_calibration.show_view_dialog') as dialog:
                restored.load_setup(path)
                dialog.assert_not_called()
        self.assertEqual(restored.axis_method_var.get(), 'Manual axis lines')
        self.assertIsNone(restored.circle)
        np.testing.assert_allclose(restored.axis_samples['Origin'], [120, 120], atol=1e-5)
        restored.circle = (120, 120, 30)
        self.assertEqual(restored.setup_data(require_capture=False)['axis_method'], 'manual_lines')

    def test_incomplete_and_outside_points_rejected_and_operations_locked(self):
        tab = self.fixture()
        tab.record_line_point((-1, 100))
        self.assertFalse(tab.axis_line_points)
        with self.assertRaisesRegex(ValueError, 'Complete manual'):
            tab.setup_data()
        tab.command_running = True
        tab.record_line_point((50, 120))
        self.assertFalse(tab.axis_line_points)
        tab.command_running = False
        self.capture(tab)
        previous = dict(tab.axis_line_points)
        tab.axis_prepare_busy = True
        tab.undo_line_point()
        tab.start_manual_lines()
        self.assertEqual(tab.axis_line_points, previous)

    def test_malformed_saved_lines_do_not_replace_current_calibration(self):
        tab = self.fixture()
        self.capture(tab)
        data = tab.setup_data()
        del data['axis_line_points_px']['+Y']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'incomplete.json'
            path.write_text(json.dumps(data))
            with patch('closed_loop_calibration.show_view_dialog') as dialog:
                tab.load_setup(path)
        self.assertIn('incomplete', dialog.call_args.args[-1])
        self.assertEqual(len(tab.axis_samples), 3)


if __name__ == '__main__':
    unittest.main()
