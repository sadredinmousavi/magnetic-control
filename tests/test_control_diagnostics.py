import csv
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from control_diagnostics import ControlDiagnostics
from command_sequences import XYStep
from command_tab import CommandTab
from point_control import PointCommander


class ControlDiagnosticTests(unittest.TestCase):
    def test_frames_events_settings_and_failed_detections_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ControlDiagnostics(directory, dict(matrix=np.eye(2), step=XYStep((.01, 0), 0, 180)))
            for index, rows in enumerate(([dict(robot_id=1, robot_x_mm=2)], [])):
                recorder.record('frame', dict(diagnostic_frame=index, tracking_rows=rows,
                    control=dict(calculated_angles=[60]*8, sent_angles=[50]*8),
                    active_step=XYStep((.01, .02), 0, 180)))
            recorder.record('motor_send_completed', dict(angles_deg=np.ones(8)*50))
            recorder.close(dict(setting='final'))
            recorder.close()  # Idempotent.
            manifest = json.loads((recorder.path / 'session.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['frames_written'], 2)
            self.assertEqual(manifest['initial']['matrix'], [[1, 0], [0, 1]])
            self.assertEqual(manifest['final']['setting'], 'final')
            records = [json.loads(line) for line in (recorder.path / 'records.jsonl').read_text().splitlines()]
            self.assertEqual([item['record_number'] for item in records], [0, 1, 2])
            with (recorder.path / 'frames.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]['group_valid'], '0')
            self.assertEqual(float(rows[1]['desired_x_mm']), 10)
            self.assertEqual(json.loads(rows[1]['sent_motor_angles_deg']), [50]*8)

    def test_queue_overflow_is_explicit_and_pending_records_are_drained(self):
        release = threading.Event()
        original = ControlDiagnostics._write
        def slow_write(recorder):
            release.wait(3)
            original(recorder)
        with tempfile.TemporaryDirectory() as directory, patch.object(ControlDiagnostics, '_write', slow_write):
            recorder = ControlDiagnostics(directory, {}, capacity=1)
            try:
                recorder.record('event', {})
                recorder.record('event', {})
                self.assertIn('overflow', recorder.error)
            finally:
                release.set()
                recorder.close()
            manifest = json.loads((recorder.path / 'session.json').read_text())
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['records_rejected'], 1)
            self.assertEqual(manifest['records_written'], 1)

    def test_write_failure_is_reported_and_does_not_hang_close(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ControlDiagnostics(directory, {})
            with patch.object(recorder.stream, 'write', side_effect=OSError('disk full')):
                recorder.record('event', {})
                recorder.worker.join(3)
            recorder.close()
            self.assertIn('disk full', recorder.error)
            self.assertFalse(json.loads((recorder.path / 'session.json').read_text())['complete'])

    def test_payloads_are_copied_before_background_write(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ControlDiagnostics(directory, {})
            array = np.ones(2)
            recorder.record('event', dict(value=array, invalid=float('nan')))
            array[:] = 9
            recorder.close()
            data = json.loads((recorder.path / 'records.jsonl').read_text())['data']
            self.assertEqual(data['value'], [1, 1])
            self.assertIsNone(data['invalid'])

    def test_motor_events_distinguish_preview_success_and_failure(self):
        commander = PointCommander(lambda: Mock(is_open=True))
        port = commander.port_provider()
        commander.port_provider = lambda: port
        events = []
        commander.diagnostic_sink = lambda kind, data: events.append(kind)
        with patch.object(commander, 'calculate', return_value=dict(angles_deg=[60]*8)), patch('helpers.moveArray') as move:
            commander.move('case_100', (0, 0), preview_only=True)
            self.assertIn('calculation_finished', events)
            self.assertNotIn('motor_send_completed', events)
            move.assert_not_called()
            commander.move('case_100', (0, 0))
            self.assertLess(events.index('motor_send_started'), events.index('motor_send_completed'))
            events.clear()
            move.side_effect = OSError('serial failure')
            with self.assertRaises(OSError):
                commander.move('case_100', (0, 0))
            self.assertIn('move_failed', events)
            self.assertNotIn('motor_send_completed', events)

    def test_broken_diagnostic_sink_cannot_change_motor_behavior(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        commander.diagnostic_sink = Mock(side_effect=RuntimeError('logger failed'))
        with patch.object(commander, 'calculate', return_value=dict(angles_deg=[60]*8)), patch('helpers.moveArray') as move:
            commander.move('case_100', (0, 0))
            move.assert_called_once()

    def test_camera_hook_records_frames_even_without_tracker(self):
        tab = CommandTab.__new__(CommandTab)
        tab.setup = None
        tab.tracking_rows = []
        tab.tracking_tracker = None
        tab.diagnostics = Mock()
        tab.record_diagnostic_frame = Mock()
        tab.visible = lambda: False
        tab.accept_live_frame(np.zeros((4, 5, 3), np.uint8))
        tab.record_diagnostic_frame.assert_called_once_with([])


if __name__ == '__main__':
    unittest.main()
