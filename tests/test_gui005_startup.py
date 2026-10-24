"""Exercise real Tk startup in a timed child so layout hangs cannot stall tests."""
from pathlib import Path
import subprocess
import sys
import unittest


class GUI005StartupTests(unittest.TestCase):
    def test_startup_and_tab4_layout_complete_without_callback_errors(self):
        project = Path(__file__).resolve().parents[1]
        script = '''
import sys, tkinter as tk, traceback
sys.path.insert(0, 'experimental')
from gui005 import ClosedLoopGUI
root = tk.Tk()
root.attributes('-alpha', 0)
errors = []
def report(*args):
    errors.append(''.join(traceback.format_exception(*args)))
root.report_callback_exception = report
try:
    app = ClosedLoopGUI(root)
    assert len(app.workflow_notebook.tabs()) == 5
    assert app.exports.recording is None
    assert not hasattr(app.trajectory, 'annotate_recording_button')
    def check_controller_controls():
        tab = app.trajectory
        assert tab.force_pid_settings['I max µN'].get() == '10'
        assert tab.arrival_tolerance_var.get() == '1.5'
        assert tab.incremental_settings['Step mm'].get() == '1'
        assert str(tab.skip_waypoint_button.cget('state')) == 'disabled'
        assert 'Skip waypoint' in [item[0] for item in tab.path_shortcuts()]
        assert tab.controller_var.get() == 'Incremental XY PID'
        assert all(str(entry.cget('state')) == 'normal' for entry in tab.incremental_pid_entries)
        assert str(next(entry for key, entry in tab.force_pid_entries.items() if key.startswith('I max')).cget('state')) == 'disabled'
        assert tab.diagnostic_settings()['controller'] == 'anchored_xy_incremental_pid'
        from control_workflow import ForcePIDFeedback
        from incremental_pid_xy_control import IncrementalPIDXYFeedback
        assert isinstance(tab.configure_run_feedback(ForcePIDFeedback()), IncrementalPIDXYFeedback)
        assert tab.speed_settings['Speed limit mm/s'].get() == '.5'
        assert all(str(entry.cget('state')) == 'normal' for entry in tab.speed_entries)
        tab.controller_var.set('Direction-weighted XY PID')
        tab.refresh_controller_controls()
        assert str(tab.force_pid_entries['Kp'].cget('state')) == 'normal'
        assert all(str(entry.cget('state')) == 'normal' for entry in tab.direction_entries)
        assert all(str(entry.cget('state')) == 'disabled' for entry in tab.incremental_entries)
        assert tab.diagnostic_settings()['controller'] == 'anchored_xy_optimizer_direction_weighted_pid'
        tab.running = True
        tab.refresh_controller_controls()
        assert all(str(entry.cget('state')) == 'disabled' for entry in tab.direction_entries)
        assert all(str(entry.cget('state')) == 'disabled' for entry in tab.speed_entries)
        tab.running = False
        tab.controller_var.set('Incremental move-and-settle')
        tab.refresh_controller_controls()
        assert str(tab.force_pid_entries['Kp'].cget('state')) == 'disabled'
        assert all(str(entry.cget('state')) == 'normal' for entry in tab.incremental_entries)
        tab.controller_var.set('Online assistant')
        tab.refresh_controller_controls()
        assert str(tab.force_pid_entries['Kp'].cget('state')) == 'disabled'
        assert all(str(entry.cget('state')) == 'normal' for entry in tab.incremental_entries)
        tab.running = True
        tab.refresh_controller_controls()
        assert str(tab.controller_selector.cget('state')) == 'disabled'
        assert all(str(entry.cget('state')) == 'disabled' for entry in tab.incremental_entries)
        tab.running = False
        tab.controller_var.set('XY PID (current)')
        tab.refresh_controller_controls()
        assert str(tab.force_pid_entries['Kp'].cget('state')) == 'normal'
        assert str(tab.controller_selector.cget('state')) == 'readonly'
        tab.manual_mode_var.set(True)
        tab.change_manual_mode()
        assert tab.manual_mode_active
        assert not tab.control_requested
        assert str(tab.controller_selector.cget('state')) == 'disabled'
        assert str(tab.force_pid_entries['Kp'].cget('state')) == 'disabled'
        assert not tab.make_arrival_gate().require_arrival
        tab.manual_mode_var.set(False)
        tab.change_manual_mode()
        assert str(tab.controller_selector.cget('state')) == 'readonly'
        tab.control_target_var.set('Cargo center')
        tab.change_control_target()
        assert tab.cargo_enabled_var.get()
        assert str(tab.cargo_roi_button.cget('state')) == 'normal'
        assert not tab.tracking_detections and tab.group_geometry is None
        tab.control_target_var.set('Robots')
        tab.change_control_target()
        assert not tab.cargo_enabled_var.get()
        assert str(tab.cargo_roi_button.cget('state')) == 'disabled'
        tab.begin_cargo_selection()
        assert tab.control_target_var.get() == 'Robots'
    def check_manual_axis_lines():
        import numpy as np
        tab = app.calibration
        app.workflow_notebook.select(tab.parent)
        tab.frame = np.full((240, 240, 3), 220, np.uint8)
        tab.corners = [(10,10), (230,10), (230,230), (10,230)]
        tab.axis_method_var.set('Manual axis lines')
        tab.change_axis_method()
        assert all(str(button.cget('state')) == 'disabled' for button in tab.axis_buttons)
        assert str(tab.prepare_button.cget('state')) == 'disabled'
        assert str(tab.manual_lines_button.cget('state')) == 'normal'
        tab.start_manual_lines()
        for point in ((50,120), (190,120), (120,200), (120,40)):
            tab.record_line_point(point)
        assert len(tab.axis_samples) == 3
        np.testing.assert_allclose(tab.axis_samples['Origin'], [120,120], atol=1e-5)
        tab.render()
        assert tab.setup_data()['axis_method'] == 'manual_lines'
        tab.undo_line_point()
        assert len(tab.axis_line_points) == 3 and not tab.axis_samples
        tab.axis_method_var.set('Symmetric pairs')
        tab.change_axis_method()
        assert not tab.axis_line_points
        assert all(str(button.cget('state')) == 'normal' for button in tab.axis_buttons)
    root.after(100, lambda: app.workflow_notebook.select(app.trajectory.parent))
    root.after(200, check_controller_controls)
    root.after(350, check_manual_axis_lines)
    root.after(550, lambda: app.workflow_notebook.select(app.trajectory.parent))
    root.after(700, root.quit)
    root.mainloop()
    if errors:
        raise AssertionError(''.join(errors))
    root.update()
    assert app.trajectory.recording_start_button.winfo_ismapped()
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import numpy as np
    from command_recording import RawCommandRecording
    with tempfile.TemporaryDirectory() as directory:
        # Preview and recording work before COM, hardware selection, or robot ROI.
        import time, json, csv
        from command_sequences import XYStep
        from command_recording import RecordingPreview
        cal, view = app.calibration, app.commands
        cal.control_case_var.set('')
        cal.circle = None
        cal.axis_method_var.set('Manual axis lines')
        cal.change_axis_method()
        for point in ((50,120), (190,120), (120,200), (120,40)):
            cal.record_line_point(point)
        view.camera_running = lambda: True
        view.frame = cal.frame.copy()
        view.frame_received_at = time.monotonic()
        view.steps = [XYStep((0.,0.),0.,180.), XYStep((.01,0.),0.,180.)]
        view.table.delete(*view.table.get_children())
        for index in range(2):
            view.table.insert('', 'end', iid=str(index))
        app.workflow_notebook.select(view.parent)
        root.update()
        view.refresh_calibration()
        assert view.pixels is not None, view.status.cget('text')
        np.testing.assert_allclose(view.pixels, [[120,120], [142,120]], atol=1e-4)
        assert view.tracking_tracker is None
        view.open_camera_popout()
        root.update()
        view.render(canvas=view.camera_popout.canvas)
        assert len(view.camera_popout.canvas.screen_points) == 2
        for calibrated in (True, False):
            if not calibrated:
                cal.corners = []
                cal.reset_axes()
            path = Path(directory)/('preview.mp4' if calibrated else 'camera_only.mp4')
            view.frame_received_at = time.monotonic()
            with patch('command_tab.filedialog.asksaveasfilename', return_value=str(path)):
                view.start_recording()
            assert view.recording is not None, view.recording_status.cget('text')
            recording = view.recording
            view.accept_live_frame(np.zeros_like(view.frame))
            view.stop_recording()
            view.recording_finalize_thread.join(timeout=5.)
            root.after(100, root.quit)
            root.mainloop()
            assert not view.recording_finalizing
            assert recording.frame_count == 1
            metadata = json.loads(recording.paths[2].read_text())
            assert not metadata['settings']['tracking_available']
            rows = list(csv.DictReader(recording.paths[1].open()))
            assert rows and rows[0]['group_valid'] == '0'
            saved = json.loads(rows[0]['annotation_snapshot_json'])
            assert (saved['pixels'] is not None) == calibrated
            preview, _ = RecordingPreview(recording.paths[0]).render(0, apply_crop=False)
            assert bool(preview.any()) == calibrated
        view.camera_popout.close()
        recorder = RawCommandRecording(Path(directory)/'export_test.mp4', (80, 60), fps=5., clock=lambda: 0.)
        snapshot = dict(detections=[dict(robot_id=1, center=[20., 20.], bbox=[17, 17, 6, 6])])
        for stamp in (0., .2):
            recorder.submit(np.zeros((60, 80, 3), np.uint8), [], snapshot, stamp)
        recorder.close()
        app.workflow_notebook.select(app.export_frame)
        def wait_for_export_task():
            def poll_done():
                if not app.exports.busy:
                    root.quit()
                else:
                    root.after(50, poll_done)
            root.after(50, poll_done)
            root.after(5000, root.quit)
            root.mainloop()
            assert not app.exports.busy, 'Export tab task did not finish'
            assert not errors, ''.join(errors)
        with patch('recording_export_tab.filedialog.askopenfilename', return_value=str(recorder.paths[0])):
            app.exports.load_recording()
        wait_for_export_task()
        assert app.exports.preview_image is not None, app.exports.status.cget('text')
        assert app.exports.preview_image.any()
        for variable in app.exports.options.values():
            variable.set(False)
        app.exports.preview()
        wait_for_export_task()
        assert not app.exports.preview_image.any()
        output = Path(directory)/'annotated.mp4'
        with patch('recording_export_tab.filedialog.asksaveasfilename', return_value=str(output)):
            app.exports.export_video()
        wait_for_export_task()
        assert output.exists(), app.exports.status.cget('text')
        png = Path(directory)/'preview.png'
        with patch('recording_export_tab.filedialog.asksaveasfilename', return_value=str(png)):
            app.exports.save_preview()
        assert png.exists(), app.exports.status.cget('text')
        from types import SimpleNamespace
        app.exports.options['show_time'].set(True)
        app.exports.preview()
        wait_for_export_task()
        assert app.exports.preview_image.any()
        left, top, scale = app.exports.preview_transform
        x, y, width, height = app.exports.time_bounds
        old_position = app.exports.time_position
        app.exports.begin_preview_drag(SimpleNamespace(x=left+(x+2)*scale, y=top+(y+2)*scale))
        app.exports.drag_time(SimpleNamespace(x=left+(x+2)*scale, y=top+50*scale))
        app.exports.end_preview_drag(SimpleNamespace(x=left+(x+2)*scale, y=top+50*scale))
        assert app.exports.time_position != old_position
        assert app.exports.time_drag_offset is None
        timed, _ = app.exports.recording.render(0, options=app.exports.selected_options())
        np.testing.assert_array_equal(app.exports.preview_image, timed)
        app.exports.crop_radius_var.set('20')
        app.exports.begin_crop_placement()
        wait_for_export_task()
        assert app.exports.crop_placing
        left, top, scale = app.exports.preview_transform
        app.exports.place_crop(SimpleNamespace(x=left+30*scale, y=top+25*scale))
        wait_for_export_task()
        assert not app.exports.crop_placing
        assert app.exports.preview_image.shape == (40, 40, 3)
        assert app.exports.selected_crop() == (True, (30, 25, 20))
        app.exports.frame_var.set(1)
        app.exports.use_current_as_start()
        wait_for_export_task()
        assert app.exports.selected_start_frame() == 1
        assert app.exports.preview_elapsed == 0.
        assert float(app.exports.slider.cget('from')) == 1.
        trimmed = Path(directory)/'trimmed.mp4'
        with patch('recording_export_tab.filedialog.asksaveasfilename', return_value=str(trimmed)):
            app.exports.export_video()
        wait_for_export_task()
        import cv2
        capture = cv2.VideoCapture(str(trimmed))
        try:
            assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
        finally:
            capture.release()
        app.exports.start_frame_var.set('bad')
        app.exports.preview()
        assert 'whole frame number' in app.exports.status.cget('text')
        app.exports.start_frame_var.set('0')
        app.exports.preview()
        wait_for_export_task()
        assert app.exports.preview_elapsed == .2
    app.on_close()
finally:
    try:
        root.destroy()
    except tk.TclError:
        pass
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=project,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
