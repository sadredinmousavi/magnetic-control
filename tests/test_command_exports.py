import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from command_sequences import XYStep, read_xy_sequence, write_xy_sequence, execute_xy_sequence
from command_recording import CommandRecording, draw_command_overlay
from command_tab import CommandTab


class CommandExportTests(unittest.TestCase):
    def test_rectangle_marker_and_center_trail_draw_without_spread_axes(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        geometry = dict(center_px=np.array([50., 50.]), center_mm=[0., 0.], ratio=1.32,
                        lines=[('Strong spread', np.array([[20., 70.], [80., 70.]]))])
        draw_command_overlay(frame, None, None, None,
                             [dict(center=[30, 30], bbox=(25, 26, 10, 8), robot_id=1)],
                             robot_marker='Rectangle', geometry=geometry, show_spread_axes=False,
                             show_labels=False, center_trail=[(10, 10), (20, 10), None, (40, 10), (50, 10)])
        self.assertTrue(frame[26, 25].any())  # Detection bounding box.
        self.assertFalse(frame[30, 30].any())  # No cross through the robot.
        self.assertTrue(frame[50, 50].any())  # Group center remains visible.
        self.assertFalse(frame[70, 30].any())  # Spread line hidden independently.
        self.assertTrue(frame[10, 15].all())
        self.assertTrue(frame[10, 45].all())
        self.assertFalse(frame[10, 30].any())  # No line across tracking loss.

    def test_center_trail_samples_centroid_and_breaks_on_loss_or_gap(self):
        app = CommandTab.__new__(CommandTab)
        app.update_center_trail(dict(center_px=[10., 20.]), 1.)
        app.update_center_trail(dict(center_px=[11., 21.]), 1.1)
        app.update_center_trail(dict(center_px=[99., 99.]), 1.1)  # Duplicate observation.
        self.assertEqual(len(app.center_trail), 2)
        app.update_center_trail(None, None)
        app.update_center_trail(dict(center_px=[30., 40.]), 1.2)
        self.assertIsNone(app.center_trail[2])
        app.update_center_trail(dict(center_px=[50., 60.]), 3.)
        self.assertIsNone(app.center_trail[-2])
        np.testing.assert_allclose(app.center_trail[-1], [50., 60.])
        for index in range(400):
            app.update_center_trail(dict(center_px=[index, index]), 4.+index*.1)
        self.assertEqual(len(app.center_trail), 300)
        app.clear_center_trail()
        self.assertFalse(app.center_trail)

    def test_label_free_overlay_keeps_markers_and_geometry(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        geometry = dict(center_px=np.array([50., 50.]), center_mm=[0., 0.], ratio=1.32,
                        lines=[('Strong spread', np.array([[20., 20.], [70., 70.]]))])
        cargo = dict(bbox=(60, 10, 20, 20), center=[70, 20], shape='Circle')
        with patch('cv2.putText') as text:
            draw_command_overlay(frame, None, np.array([[10., 10.]]), 0,
                                 [dict(center=[40, 40], robot_id=1, xy_cm=[0., 0.])],
                                 commanded_pixel=[80., 80.], geometry=geometry, cargo=cargo,
                                 show_labels=False)
        text.assert_not_called()
        for x, y in ((10, 10), (40, 40), (50, 50), (60, 10), (80, 76)):
            self.assertTrue(frame[y, x].any())

    def test_circle_crop_automatically_encloses_pins_despite_small_manual_radius(self):
        app = CommandTab.__new__(CommandTab)
        app.frame = np.zeros((600, 800, 3), np.uint8)
        app.crop_enabled_var = Mock(get=Mock(return_value=True))
        app.crop_radius_var = Mock(get=Mock(return_value='10'))
        app.crop_center = (20, 20)
        app.setup = {'corners_px': [(100, 100), (700, 100), (700, 500), (100, 500)]}
        x, y, radius = app.recording_circle()
        np.testing.assert_allclose([x, y], [400, 300])
        self.assertTrue(np.all(np.linalg.norm(np.array(app.setup['corners_px']) - [x, y], axis=1) <= radius - 35))
        app.crop_radius_var.get.assert_not_called()

    def test_extended_sequence_roundtrip_and_solver_settings(self):
        steps = [XYStep((.01, -.02), 3, 180, 2.5, 45), XYStep((0, 0), 0, 90)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'xy.txt'
            write_xy_sequence(path, steps)
            self.assertEqual(read_xy_sequence(path), steps)
            path.write_text('# targets_xy_mm\n[10, 20] | 2 | 180 | 0 | 45', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Ratio'):
                read_xy_sequence(path)
        commander = Mock()
        execute_xy_sequence(steps, commander, 'case_100', active=lambda: True, report=Mock())
        self.assertEqual(commander.move.call_args_list[0].kwargs['ratio'], 2.5)
        self.assertEqual(commander.move.call_args_list[0].kwargs['angle_deg'], 45)

    def test_selected_settings_only_change_requested_row_and_reject_invalid(self):
        app = CommandTab.__new__(CommandTab)
        app.running = False
        app.table = Mock()
        app.table.selection.return_value = ('1',)
        app.status = Mock()
        app.point_ratio_var = Mock(get=Mock(return_value='2'))
        app.point_angle_var = Mock(get=Mock(return_value='30'))
        app.steps = [XYStep((0, 0), 0, 180), XYStep((.01, 0), 0, 180)]
        app.apply_point_settings()
        self.assertEqual(app.steps[0].ratio, 1)
        self.assertEqual(app.steps[1].ratio, 2)
        self.assertEqual(app.steps[1].angle_deg, 30)
        app.point_ratio_var.get.return_value = '-1'
        app.apply_point_settings()
        self.assertEqual(app.steps[1].ratio, 2)

    def test_recording_writes_three_readable_videos_and_gap_safe_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = CommandRecording(Path(directory) / 'test.mp4', (81, 61), fps=15)
            frame = np.full((61, 81, 3), 220, np.uint8)
            overlay = np.zeros_like(frame)
            overlay[:] = (255, 0, 0)
            detections = [{'center': (40, 30), 'robot_id': 1}]
            annotated = draw_command_overlay(frame.copy(), None, np.array([[20, 20]]), 0, detections)
            draw_command_overlay(overlay, None, np.array([[20, 20]]), 0, detections)
            for index in range(3):
                rows = [] if index == 0 else [{'robot_x_mm': 10, 'group_valid': 1}]
                recording.write(frame, annotated, overlay, rows)
            recording.close()
            for path in recording.paths[:3]:
                capture = cv2.VideoCapture(str(path))
                try:
                    ok, decoded = capture.read()
                    self.assertTrue(ok, str(path))
                    self.assertEqual(decoded.shape[:2], (62, 82))
                    self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                finally:
                    capture.release()
            with recording.paths[3].open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['robot_x_mm'], '')
            self.assertEqual(rows[1]['robot_x_mm'], '10')
            self.assertEqual(rows[-1]['recording_frame'], '2')
            with self.assertRaisesRegex(ValueError, 'already exist'):
                CommandRecording(Path(directory) / 'test.mp4', (81, 61))


if __name__ == '__main__':
    unittest.main()
