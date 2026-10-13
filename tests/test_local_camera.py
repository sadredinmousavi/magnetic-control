from pathlib import Path
import queue
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from local_camera import LocalCameraClient
from gui005 import ClosedLoopGUI


class LocalCameraTests(unittest.TestCase):
    def test_resolution_requests_driver_properties_and_reports_actual_mismatch(self):
        import cv2
        frames, events = queue.Queue(maxsize=1), queue.Queue()
        capture = Mock(isOpened=Mock(return_value=True))
        capture.read.side_effect = [(True,np.zeros((480,640,3),np.uint8)),(False,None)]
        client = LocalCameraClient(frames, events, capture_factory=Mock(return_value=capture))
        client.start(2, width=2596, height=1111)
        client.thread.join(2)
        capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 2596)
        capture.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 1111)
        kind, text = events.get_nowait()
        self.assertEqual(kind, 'connected')
        self.assertIn('640 × 480', text)
        self.assertIn('2596 × 1111', text)
        self.assertIn('different size', text)
        self.assertEqual(frames.get_nowait().shape, (480,640,3))

    def test_resolution_input_auto_presets_custom_and_invalid(self):
        app = ClosedLoopGUI.__new__(ClosedLoopGUI)
        app.webcam_resolution_var = Mock(get=Mock(return_value='Auto'))
        self.assertIsNone(app.read_webcam_resolution())
        app.webcam_resolution_var.get.return_value = '1920x1080'
        self.assertEqual(app.read_webcam_resolution(), (1920,1080))
        app.webcam_resolution_var.get.return_value = 'Custom'
        app.webcam_width_var = Mock(get=Mock(return_value='2596'))
        app.webcam_height_var = Mock(get=Mock(return_value='1111'))
        self.assertEqual(app.read_webcam_resolution(), (2596,1111))
        app.webcam_width_var.get.return_value = '0'
        with self.assertRaises(ValueError):
            app.read_webcam_resolution()
        client = LocalCameraClient(queue.Queue(), queue.Queue(), capture_factory=Mock())
        with self.assertRaises(ValueError):
            client.start(0, width=640)
        with self.assertRaises(ValueError):
            client.start(0, width=-1, height=480)

    def test_background_capture_publishes_latest_frame_and_releases_device(self):
        frames, events = queue.Queue(maxsize=1), queue.Queue()
        first, newest = np.zeros((10,20,3),np.uint8), np.ones((10,20,3),np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True,first),(True,newest),(False,None)]
        factory = Mock(return_value=capture)
        client = LocalCameraClient(frames, events, capture_factory=factory)
        client.start(2)
        client.thread.join(2)
        self.assertFalse(client.thread.is_alive())
        self.assertIs(frames.get_nowait(), newest)
        self.assertEqual(events.get_nowait()[0], 'connected')
        self.assertEqual(events.get_nowait()[0], 'error')
        capture.release.assert_called_once()
        self.assertFalse(client.is_running)

    def test_invalid_or_unavailable_device_fails_without_frames(self):
        frames, events = queue.Queue(maxsize=1), queue.Queue()
        capture = Mock(isOpened=Mock(return_value=False))
        client = LocalCameraClient(frames, events, capture_factory=Mock(return_value=capture))
        with self.assertRaises(ValueError):
            client.start(-1)
        client.start(0)
        client.thread.join(2)
        self.assertEqual(events.get_nowait()[0], 'error')
        self.assertTrue(frames.empty())
        capture.release.assert_called_once()

    def test_selected_webcam_frames_feed_both_phases_not_pi_queue(self):
        app = ClosedLoopGUI.__new__(ClosedLoopGUI)
        app.selected_camera_source = 'Iriun / Local webcam'
        app.camera_event_queue, app.camera_frame_queue = queue.Queue(), queue.Queue()
        app.local_event_queue, app.local_frame_queue = queue.Queue(), queue.Queue()
        pi, webcam = object(), object()
        app.camera_frame_queue.put(pi)
        app.local_frame_queue.put(webcam)
        app.calibration, app.commands, app.root = Mock(camera_popout=None), Mock(camera_popout=None), Mock()
        app.calibration.live_fps_var.get.return_value = '5'
        app.connection_visible = lambda: False
        with patch('gui005.time.monotonic', side_effect=[100,100]):
            app.poll_camera()
        app.calibration.accept_live_frame.assert_called_once_with(webcam)
        app.commands.accept_live_frame.assert_called_once_with(webcam)
        self.assertIs(app.camera_frame_queue.get_nowait(), pi)

    def test_source_switch_cancels_run_recording_and_discards_calibration(self):
        app = ClosedLoopGUI.__new__(ClosedLoopGUI)
        app.calibration, app.commands = Mock(), Mock()
        app.calibration.setup_revision = 1
        app.commands.render_revision = 1
        app.remote_camera_canvas = Mock()
        app.reset_camera_geometry('Iriun / Local webcam:1')
        app.commands.stop_event.set.assert_called_once()
        app.commands.stop_recording.assert_called_once()
        self.assertEqual(app.calibration.corners, [])
        self.assertIsNone(app.commands.setup)
        self.assertEqual(app.calibration.camera_source_key, 'Iriun / Local webcam:1')


if __name__ == '__main__':
    unittest.main()
