import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from command_sequences import XYStep, read_xy_sequence, write_xy_sequence, execute_xy_sequence
from command_recording import CommandRecording, draw_command_overlay
from command_tab import CommandTab


class CommandExportTests(unittest.TestCase):
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
