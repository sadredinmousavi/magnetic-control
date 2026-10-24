"""Raw recording timing, data completeness, queue bounds and offline replay."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from command_recording import (RawCommandRecording, annotate_recording, RecordingPreview,
                               render_saved_annotations, render_time_annotation)
from command_tab import CommandTab


class RawRecordingTests(unittest.TestCase):
    def test_preview_and_export_use_identical_options_frame_timing_and_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawCommandRecording(Path(directory)/'run.mp4', (80, 60), fps=5., clock=lambda: 0.)
            for index, stamp in enumerate((0., .1, .2, .4)):
                snapshot = self.snapshot()
                snapshot['detections'][0]['center'][0] += index
                snapshot['detections'][0]['bbox'][0] += index
                recorder.submit(np.zeros((60, 80, 3), np.uint8), [], snapshot, stamp)
            recorder.close()
            recording = RecordingPreview(recorder.paths[0])
            self.assertEqual(len(recording.frames), 3)
            options = {key: False for key in ('show_path', 'show_points', 'show_roi', 'show_pins',
                'show_robots', 'show_trails', 'show_center_trail', 'show_group_center', 'show_spread_axes',
                'show_cargo', 'show_commanded', 'show_waypoint', 'show_labels')}
            plain, stamp = recording.render(1, options=options, apply_crop=False)
            self.assertAlmostEqual(stamp, .2)
            self.assertFalse(plain.any())
            options.update(show_robots=True, robot_marker='Rectangle')
            preview, _ = recording.render(1, options=options, apply_crop=False)
            self.assertTrue(preview.any())
            exported = []
            def render(frame, *args, **kwargs):
                result = render_saved_annotations(frame, *args, **kwargs)
                exported.append(result.copy())
                return result
            with patch('command_recording.render_saved_annotations', side_effect=render):
                count = annotate_recording(recorder.paths[0], Path(directory)/'annotated.mp4',
                                           options=options, apply_crop=False)
            self.assertEqual(count, 3)
            np.testing.assert_array_equal(exported[1], preview)
            # An export-specific crop works even when no crop was saved live.
            circle = (30, 25, 20)
            cropped, _ = recording.render(1, options=options, crop_circle=circle)
            self.assertEqual(cropped.shape, (40, 40, 3))
            self.assertFalse(cropped[0, 0].any())
            cropped_path = Path(directory)/'cropped.mp4'
            annotate_recording(recorder.paths[0], cropped_path, options=options, crop_circle=circle)
            capture = cv2.VideoCapture(str(cropped_path))
            try:
                self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 40)
                self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 40)
            finally:
                capture.release()
            # Time uses capture timestamps, survives cropping, and matches export pixels.
            options.update(show_time=True, time_position=(.5, .8))
            timed, elapsed = recording.render(1, options=options, crop_circle=circle)
            self.assertAlmostEqual(elapsed, .2)
            timed_frames, times = [], []
            def draw_time(frame, stamp, settings):
                times.append(stamp)
                result = render_time_annotation(frame, stamp, settings)
                timed_frames.append(frame.copy())
                return result
            with patch('command_recording.render_time_annotation', side_effect=draw_time):
                annotate_recording(recorder.paths[0], Path(directory)/'timed.mp4',
                                   options=options, crop_circle=circle)
            np.testing.assert_allclose(times, [0., .2, .4])
            np.testing.assert_array_equal(timed_frames[1], timed)
            # Trim at source frame 1: its real capture time becomes the new zero.
            first, zero = recording.render(1, options=options, crop_circle=circle, start_frame=1)
            second, later = recording.render(2, options=options, crop_circle=circle, start_frame=1)
            self.assertEqual(zero, 0.)
            self.assertAlmostEqual(later, .2)
            timed_frames.clear()
            times.clear()
            trimmed_path = Path(directory)/'trimmed.mp4'
            with patch('command_recording.render_time_annotation', side_effect=draw_time):
                count = annotate_recording(recorder.paths[0], trimmed_path,
                    options=options, crop_circle=circle, start_frame=1)
            self.assertEqual(count, 2)
            np.testing.assert_allclose(times, [0., .2])
            np.testing.assert_array_equal(timed_frames[0], first)
            np.testing.assert_array_equal(timed_frames[1], second)
            capture = cv2.VideoCapture(str(trimmed_path))
            try:
                self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            finally:
                capture.release()
            with self.assertRaisesRegex(ValueError, 'start frame'):
                recording.render(1, start_frame=len(recording.frames))
            with self.assertRaisesRegex(ValueError, 'start frame'):
                annotate_recording(recorder.paths[0], Path(directory)/'invalid.mp4', start_frame=-1)

    def test_time_label_clamps_inside_output_and_can_be_disabled(self):
        frame = np.zeros((60, 80, 3), np.uint8)
        self.assertIsNone(render_time_annotation(frame, 12.345))
        self.assertFalse(frame.any())
        with patch('command_recording.cv2.putText', wraps=cv2.putText) as text:
            x, y, width, height = render_time_annotation(frame, 12.345,
                dict(show_time=True, time_position=(1., 1.)))
        self.assertEqual(text.call_args.args[1], 'Time : 12.3 s')
        self.assertLessEqual(x+width, 80)
        self.assertLessEqual(y+height, 60)
        self.assertTrue(frame.any())

    def test_gui_submits_capture_timestamp_without_live_annotation_work(self):
        app = CommandTab.__new__(CommandTab)
        app.recording = Mock(frame_count=0, observation_count=0)
        app.recording_status = Mock()
        app.tracking_rows = [dict(robot_id=1)]
        app.recording_snapshot = Mock(return_value={'geometry': None})
        app.frame_received_at = 99.  # A newer frame arrived while vision was busy.
        app.visible = lambda: False
        frame = np.zeros((60, 80, 3), np.uint8)
        with patch('command_recording.draw_command_overlay') as draw:
            app.finish_frame_io(frame, 0, observed_at=42.)
        draw.assert_not_called()
        args = app.recording.submit.call_args.args
        self.assertIs(args[0], frame)
        self.assertEqual(args[1], [dict(robot_id=1)])
        self.assertEqual(args[3], 42.)

    def snapshot(self):
        return dict(
            setup={'corners_px': [[1, 1], [78, 1], [78, 58], [1, 58]]},
            pixels=np.array([[10., 10.], [30., 30.]]), active_index=0,
            detections=[dict(robot_id=1, center=[20., 20.], bbox=[16, 17, 8, 6], xy_cm=[1., 2.]),
                        dict(robot_id=2, center=[40., 30.], bbox=[37, 27, 6, 6], xy_cm=[3., 4.])],
            geometry=dict(center_px=np.array([30., 25.]), center_mm=[20., 30.], ratio=1.5,
                          strong_mm=3., weak_mm=2., angle_deg=45.,
                          lines=[('Strong spread', np.array([[20., 15.], [40., 35.]]))]),
            cargo=dict(shape='Rectangle', bbox=[50, 30, 10, 10], center=[55., 35.], xy_cm=[5., 3.],
                       outline_px=np.array([[50, 30], [60, 30], [60, 40], [50, 40]]),
                       direction_px=np.array([[50, 35], [60, 35]]), angle_deg=0.),
            waypoint_mm=np.array([4., 5.]), waypoint_pixel=[25., 25.],
            commanded_pixel=[35., 35.], roi=[30., 25., 20.],
            control=dict(pid_gains=dict(kp=.25, ki=.02), sent_angles=[60.]*8),
            options=dict(show_labels=False, robot_marker='Rectangle', show_group_center=True,
                         show_spread_axes=True, show_trails=True, show_center_trail=True))

    def test_raw_video_and_every_observation_csv_replay_with_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawCommandRecording(Path(directory)/'run.mp4', (80, 60), fps=5., clock=lambda: 0.)
            snapshot = self.snapshot()
            rows = [dict(robot_id=index, group_x_mm=20., group_y_mm=30., spread_ratio=1.5,
                         orientation_measured_deg=45., desired_ratio=2., robot_speed_mm_s=.2)
                    for index in (1, 2)]
            frame = np.full((60, 80, 3), 80, np.uint8)
            for stamp in (0., .1, .2, .4):
                recorder.submit(frame, rows, snapshot, stamp)
            recorder.submit(frame, rows, snapshot, .4)  # Repeated observation is ignored.
            snapshot['geometry']['ratio'] = 99  # Queue owns its snapshot.
            frame[:] = 0  # Queue also owns the camera pixels.
            recorder.close()
            self.assertEqual(recorder.frame_count, 3)
            self.assertEqual(recorder.observation_count, 4)
            self.assertFalse((Path(directory)/'run_annotated.mp4').exists())
            with recorder.paths[1].open(newline='', encoding='utf-8') as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), 8)
            self.assertEqual([r['recording_frame'] for r in saved[::2]], ['0', '', '1', '2'])
            self.assertEqual([float(r['capture_monotonic_s']) for r in saved[::2]], [0., .1, .2, .4])
            self.assertEqual(saved[0]['robot_bbox_width_px'], '8')
            self.assertEqual(saved[0]['group_x_px'], '30.0')
            self.assertEqual(saved[0]['waypoint_x_mm'], '4.0')
            self.assertEqual(saved[0]['desired_ratio'], '2.0')
            first = json.loads(saved[0]['annotation_snapshot_json'])
            self.assertEqual(first['geometry']['ratio'], 1.5)
            self.assertEqual(first['cargo']['outline_px'][0], [50, 30])
            self.assertEqual(first['control']['sent_angles'], [60.]*8)
            self.assertEqual(json.loads(recorder.paths[2].read_text())['logged_observations'], 4)
            capture = cv2.VideoCapture(str(recorder.paths[0]))
            ok, decoded = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(decoded.mean(), 70)  # Raw frame was copied before caller mutation.
            output = Path(directory)/'render.mp4'
            self.assertEqual(annotate_recording(recorder.paths[0], output), 3)
            capture = cv2.VideoCapture(str(output))
            self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            ok, annotated = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(np.abs(annotated.astype(float)-decoded.astype(float)).sum(), 1000)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                annotate_recording(recorder.paths[0], output)

    def test_queue_overflow_reports_loss_without_blocking_control(self):
        entered, release = threading.Event(), threading.Event()
        writer = Mock()
        writer.isOpened.return_value = True
        def encode(frame):
            entered.set()
            release.wait(5.)
        writer.write.side_effect = encode
        with tempfile.TemporaryDirectory() as directory, patch('command_recording.cv2.VideoWriter', return_value=writer):
            recorder = RawCommandRecording(Path(directory)/'run.mp4', (80, 60), fps=10.,
                                           clock=lambda: 0., queue_size=1)
            try:
                frame = np.zeros((60, 80, 3), np.uint8)
                recorder.submit(frame, [], {}, 0.)
                self.assertTrue(entered.wait(2.))
                recorder.submit(frame, [], {}, .2)
                with self.assertRaisesRegex(OSError, 'queue is full'):
                    recorder.submit(frame, [], {}, .4)
            finally:
                release.set()
                recorder.close()
            metadata = json.loads(recorder.paths[2].read_text())
            self.assertEqual(metadata['logged_observations'], 2)
            self.assertEqual(metadata['rejected_observations'], 1)
            self.assertIn('queue is full', metadata['error'])
            writer.release.assert_called_once()

    def test_offline_crop_only_and_frame_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawCommandRecording(Path(directory)/'run.mp4', (80, 60), clock=lambda: 0.,
                                           metadata={'crop_circle': [40, 30, 10]})
            recorder.submit(np.zeros((60, 80, 3), np.uint8), [], self.snapshot(), 0.)
            recorder.close()
            output = Path(directory)/'crop.mp4'
            annotate_recording(recorder.paths[0], output)
            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (20, 20))
            with recorder.paths[1].open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]['recording_frame'] = '1'
            with recorder.paths[1].open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                annotate_recording(recorder.paths[0], Path(directory)/'bad.mp4')


if __name__ == '__main__':
    unittest.main()
