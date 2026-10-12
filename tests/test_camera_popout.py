from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from camera_popout import CameraPopout
from closed_loop_calibration import CalibrationTab
from command_tab import CommandTab


def canvas(width, height):
    result = Mock()
    result.winfo_width.return_value = width
    result.winfo_height.return_value = height
    return result


class CameraPopoutTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
