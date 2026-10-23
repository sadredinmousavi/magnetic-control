"""Coil alignment remains independent from the workspace origin and robot ROI."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from axis_calibration import RobotAxes
from closed_loop_calibration import CalibrationTab
from gui005 import ClosedLoopGUI
from gui005_radial_control import HelmholtzRadialModel
from point_control import PointCommander


def var(value):
    result = Mock()
    result.get.return_value = value
    return result


def calibration():
    tab = CalibrationTab.__new__(CalibrationTab)
    tab.frame = np.zeros((300, 400, 3), np.uint8)
    tab.corners = [(0, 0), (399, 0), (399, 299), (0, 299)]
    tab.width_var, tab.height_var = var('10'), var('10')
    tab.axis_samples, tab.axis_capture_moves, tab.axis_raw_captures = {}, {}, {}
    tab.circle = (200, 150, 40)
    tab.helmholtz_circle = (240., 120., 100.)
    tab.helmholtz_radius_var = var('100')
    tab.helmholtz_status, tab.status, tab.axis_status = Mock(), Mock(), Mock()
    tab.helmholtz_center_changed = Mock()
    tab.transform = (10, 20, .5)
    tab.camera_popout = None
    tab.mode = var('Helmholtz')
    tab.axis_busy = tab.axis_prepare_busy = tab.command_running = False
    tab.axis_pending = None
    tab.helmholtz_edit_allowed = lambda: True
    tab.setup_revision = 0
    tab.render = Mock()
    return tab


class HelmholtzAlignmentTests(unittest.TestCase):
    def test_center_is_projected_into_robot_axes_not_assumed_origin(self):
        tab = calibration()
        metric = np.array([[1., -1.], [1., 0.], [0., -1.]])
        pixels = RobotAxes.metric_pixel_positions(metric, tab.corners, 10., 10.)
        tab.axis_samples = dict(zip(('Origin', '+X', '+Y'), pixels.tolist()))
        axes = RobotAxes(tab.corners, 10., 10., tab.axis_samples)
        expected = axes.coordinates([tab.helmholtz_circle[:2]])[0]*.01
        np.testing.assert_allclose(tab.helmholtz_center_m(), expected)
        self.assertGreater(np.linalg.norm(expected), .001)
        tab.refresh_helmholtz_center()
        np.testing.assert_allclose(tab.helmholtz_center_changed.call_args.args[0], expected)

    def test_drag_maps_display_coordinates_and_preserves_robot_roi(self):
        tab = calibration()
        original_roi = tab.circle
        tab.drag_helmholtz_circle(SimpleNamespace(x=150, y=100))
        self.assertEqual(tab.helmholtz_circle, (280, 160, 100.))
        self.assertEqual(tab.circle, original_roi)
        self.assertEqual(tab.setup_revision, 1)
        tab.command_running = True
        tab.drag_helmholtz_circle(SimpleNamespace(x=120, y=70))
        self.assertEqual(tab.helmholtz_circle, (280, 160, 100.))
        tab.command_running = False
        tab.axis_prepare_busy = True
        tab.drag_helmholtz_circle(SimpleNamespace(x=120, y=70))
        self.assertEqual(tab.helmholtz_circle, (280, 160, 100.))

    def test_popout_transform_and_resize_do_not_change_center(self):
        tab = calibration()
        tab.camera_popout = SimpleNamespace(is_open=True)
        tab.drag_helmholtz_circle(SimpleNamespace(x=50, y=60))
        self.assertEqual(tab.setup_revision, 0)
        tab.drag_helmholtz_circle(SimpleNamespace(x=50, y=60), transform=(0, 0, .5))
        self.assertEqual(tab.helmholtz_circle, (100, 120, 100.))
        tab.helmholtz_radius_var = var('180')
        tab.resize_helmholtz_circle()
        self.assertEqual(tab.helmholtz_circle, (100, 120, 180.))

    def test_center_updates_both_commanders_and_invalidates_preparation(self):
        gui = ClosedLoopGUI.__new__(ClosedLoopGUI)
        gui.radial_base_model = gui.radial_model = HelmholtzRadialModel()
        gui.point_commander = PointCommander(lambda: None, radial_model=gui.radial_model)
        gui.trajectory_commander = PointCommander(lambda: None, radial_model=gui.radial_model)
        gui.commands = SimpleNamespace(precalculated_signature='old')
        gui.trajectory = SimpleNamespace(precalculated_signature='old')
        gui.calibration = SimpleNamespace(prepared_case='case_100')
        for commander in (gui.point_commander, gui.trajectory_commander):
            commander.prepared_points['old'] = 123
        gui.set_helmholtz_center((.01, -.005))
        for commander in (gui.point_commander, gui.trajectory_commander):
            self.assertEqual(commander.radial_control.model.center_xy_m, (.01, -.005))
            self.assertEqual(commander.prepared_points, {})
        self.assertIsNone(gui.calibration.prepared_case)
        self.assertIsNone(gui.commands.precalculated_signature)
        gui.set_helmholtz_center(None)
        self.assertEqual(gui.radial_model.center_xy_m, (0., 0.))

    def test_setup_round_trip_and_old_setup_fallback(self):
        tab = calibration()
        tab.tracker = None
        tab.radius_var, tab.area_var = var('40'), var('1')
        tab.count_var, tab.contrast_var = var('1'), var('10')
        tab.control_case_var = var('case_100')
        tab.camera_source_key = 'Pi SSH'
        tab.axis_method_var = var('Center reference')
        data = tab.setup_data()
        np.testing.assert_allclose(data['helmholtz_center_xy_m'], tab.helmholtz_center_m())
        target = calibration()
        target.reset_axes, target.clear_tracking = Mock(), Mock()
        target.radius_var, target.count_var, target.area_var = var('40'), var('1'), var('1')
        target.control_case_var, target.contrast_var = var('case_100'), var('10')
        target.axis_method_var, target.camera_source_key = var('Center reference'), 'Pi SSH'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'setup.json'
            path.write_text(json.dumps(data))
            with patch('closed_loop_calibration.show_view_dialog') as dialog:
                target.load_setup(path)
                dialog.assert_not_called()
            self.assertEqual(target.helmholtz_circle, tab.helmholtz_circle)
            del data['helmholtz_circle_px']
            data.pop('helmholtz_center_xy_m')
            path.write_text(json.dumps(data))
            target.load_setup(path)
            self.assertIsNone(target.helmholtz_circle)


if __name__ == '__main__':
    unittest.main()
