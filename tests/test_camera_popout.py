from pathlib import Path
import sys
import unittest
import queue
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from camera_popout import CameraPopout, show_view_dialog
from closed_loop_calibration import CalibrationTab
from command_tab import CommandTab
from gui005 import ClosedLoopGUI


def canvas(width, height):
    result = Mock()
    result.winfo_width.return_value = width
    result.winfo_height.return_value = height
    return result


class CameraPopoutTests(unittest.TestCase):
    def test_dialog_is_owned_by_open_popout_and_restores_it_after_error(self):
        view = CameraPopout.__new__(CameraPopout)
        view.closed, view.window, view.show = False, Mock(), Mock()
        owner = Mock(camera_popout=view)
        dialog = Mock(return_value=True)
        self.assertTrue(show_view_dialog(owner, dialog, 'Error', 'Message'))
        dialog.assert_called_once_with('Error', 'Message', parent=view.window)
        view.show.assert_called_once()
        dialog.side_effect = RuntimeError('Dialog failed')
        with self.assertRaises(RuntimeError):
            show_view_dialog(owner, dialog, 'Error', 'Message')
        self.assertEqual(view.show.call_count, 2)

    def test_camera_processing_caps_fps_and_uses_only_latest_frame(self):
        app = ClosedLoopGUI.__new__(ClosedLoopGUI)
        app.camera_event_queue, app.camera_frame_queue = queue.Queue(), queue.Queue()
        old, newest = object(), object()
        app.camera_frame_queue.put(old)
        app.camera_frame_queue.put(newest)
        app.calibration, app.commands, app.root, app.show_camera_frame = Mock(), Mock(), Mock(), Mock()
        app.calibration.live_fps_var.get.return_value = '5'
        app.connection_visible = lambda: False
        with patch('gui005.time.monotonic', side_effect=[100, 100.01]):
            app.poll_camera()
        app.calibration.accept_live_frame.assert_called_once_with(newest)
        app.commands.accept_live_frame.assert_called_once_with(newest)
        app.show_camera_frame.assert_not_called()
        app.root.after.assert_called_once_with(190, app.poll_camera)

    def test_unchanged_frames_skip_redrawing_but_still_sync_buttons(self):
        revision = [1]
        render = Mock()
        with patch('camera_popout.tk.Toplevel') as window, patch('camera_popout.tk.Frame'), \
             patch('camera_popout.tk.Button'), patch('camera_popout.tk.Label'), patch('camera_popout.tk.Canvas') as surface:
            window.return_value.winfo_width.return_value = 1000
            surface.return_value.winfo_width.return_value = 800
            surface.return_value.winfo_height.return_value = 600
            view = CameraPopout(Mock(), 'Camera', render, render_version=lambda: revision[0])
            view.refresh()
            self.assertEqual(render.call_count, 1)
            revision[0] += 1
            view.refresh()
            self.assertEqual(render.call_count, 2)
            surface.return_value.winfo_width.return_value = 900
            view.refresh()
            self.assertEqual(render.call_count, 3)
            view.close()

    def test_calibration_progress_and_capture_hint_follow_workflow(self):
        app = CalibrationTab.__new__(CalibrationTab)
        app.corners, app.detections, app.axis_samples = [], [], {}
        app.circle, app.axis_pending, app.axis_busy = None, None, False
        app.command_running = False
        app.count_var = Mock(get=Mock(return_value='1'))
        labels = [Mock(), Mock(), Mock()]
        app.refresh_step_status(labels)
        self.assertIn('0/4', labels[0].config.call_args.kwargs['text'])
        self.assertIn('Move to Center', app.capture_explanation())
        app.corners = [(0, 0), (100, 0), (100, 100), (0, 100)]
        app.circle = (50, 50, 30)
        app.detections = [{'center': (50, 50)}]
        app.refresh_step_status(labels)
        self.assertEqual(labels[0].config.call_args.kwargs['foreground'], '#176637')
        self.assertEqual(labels[1].config.call_args.kwargs['foreground'], '#176637')
        app.axis_busy = True
        self.assertIn('still being sent', app.capture_explanation())
        app.axis_busy, app.axis_pending = False, 'Origin'
        self.assertIn('Capture ready for Center', app.capture_explanation())
        app.command_running = True
        self.assertIn('Capture locked', app.capture_explanation())

    def test_compact_labels_preserved_and_all_actions_lock_during_command_run(self):
        enabled = [True]
        source = Mock()
        source.cget.return_value = 'normal'
        with patch('camera_popout.tk.Toplevel') as window, patch('camera_popout.tk.Frame'), \
             patch('camera_popout.tk.Button'), patch('camera_popout.tk.Label'), patch('camera_popout.tk.Canvas'):
            window.return_value.winfo_width.return_value = 1000
            view = CameraPopout(Mock(), 'Camera', Mock(), actions=(('Center', Mock(), source),),
                                mirror_action_text=False, controls_enabled=lambda: enabled[0])
            button = view.actions[0][0]
            button.config.assert_called_with(state='normal')
            enabled[0] = False
            view.refresh()
            button.config.assert_called_with(state='disabled')
            view.close()

    def test_close_cancels_only_view_timer_and_is_idempotent(self):
        render, command, source = Mock(), Mock(), Mock()
        source.cget.side_effect = lambda key: {'state': 'disabled', 'text': 'Finish'}[key]
        with patch('camera_popout.tk.Toplevel') as window, patch('camera_popout.tk.Frame'), \
             patch('camera_popout.tk.Button'), patch('camera_popout.tk.Label'), patch('camera_popout.tk.Canvas'):
            window.return_value.winfo_width.return_value = 1000
            window.return_value.after.return_value = 'timer1'
            view = CameraPopout(Mock(), 'Camera', render, actions=(('Next', command, source),))
            self.assertTrue(view.is_open)
            view.show()
            view.actions[0][0].config.assert_any_call(state='disabled')
            view.actions[0][0].config.assert_any_call(text='Finish')
            view.close()
            view.close()
            self.assertFalse(view.is_open)
            window.return_value.after_cancel.assert_called_once_with('timer1')
            window.return_value.destroy.assert_called_once()
            command.assert_not_called()

    def test_calibration_popout_does_not_replace_main_image_or_mouse_transform(self):
        app = CalibrationTab.__new__(CalibrationTab)
        app.frame = np.full((400, 600, 3), 220, np.uint8)
        app.canvas = canvas(300, 200)
        app.corners, app.detections = [], []
        app.circle, app.tracker = (300, 200, 100), None
        with patch('closed_loop_calibration.ImageTk.PhotoImage', side_effect=[Mock(), Mock()]):
            app.render()
            transform, photo = app.transform, app.photo
            popup_canvas = canvas(900, 600)
            app.render(popup_canvas)
        self.assertEqual(app.transform, transform)
        self.assertIs(app.photo, photo)
        self.assertEqual(popup_canvas.camera_transform, (0, 0, 1.5))
        self.assertIsNot(popup_canvas.camera_photo, photo)

    def test_command_popout_keeps_independent_target_hit_positions(self):
        app = CommandTab.__new__(CommandTab)
        app.frame = np.full((400, 600, 3), 220, np.uint8)
        app.canvas = canvas(300, 200)
        app.setup = {'corners_px': [(10, 10), (590, 10), (590, 390), (10, 390)], 'video_size_px': [600, 400]}
        app.pixels = np.array([[100, 100], [200, 200]], dtype=float)
        app.show_all, app.active_index = False, 1
        with patch('command_tab.ImageTk.PhotoImage', side_effect=[Mock(), Mock()]):
            app.render()
            positions, photo = app.screen_points.copy(), app.photo
            popup_canvas = canvas(900, 600)
            app.render(popup_canvas)
        np.testing.assert_allclose(app.screen_points, positions)
        np.testing.assert_allclose(popup_canvas.screen_points, app.pixels * 1.5)
        self.assertIs(app.photo, photo)
        app.running = False
        app.table, app.select_point = Mock(), Mock()
        app.click_target(Mock(x=300, y=300), screen_points=popup_canvas.screen_points)
        app.table.selection_set.assert_called_once_with('1')

    def test_embedded_views_do_not_draw_images_while_popout_is_open(self):
        for cls in (CalibrationTab, CommandTab):
            app = cls.__new__(cls)
            app.canvas = canvas(300, 200)
            app.camera_popout = Mock(is_open=True)
            app.frame = np.full((400, 600, 3), 220, np.uint8)
            with patch(f'{cls.__module__}.ImageTk.PhotoImage') as photo:
                app.render()
                photo.assert_not_called()
            app.canvas.create_image.assert_not_called()
            self.assertIn('pop-out', app.canvas.create_text.call_args.kwargs['text'])


if __name__ == '__main__':
    unittest.main()
