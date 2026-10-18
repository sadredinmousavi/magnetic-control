import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from agent_control import AgentControl, atomic_json


class AgentControlTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.bridge = AgentControl.__new__(AgentControl)
        self.bridge.session = 'test-session'
        self.bridge.directory = Path(self.folder.name)
        self.bridge.mailbox = self.bridge.directory
        self.bridge.gui = Mock()
        self.bridge.closed = False
        self.bridge.stopped_at = 0
        self.bridge.camera_delay_s = 1.
        self.bridge.status = lambda: dict(session='test-session')

    def request(self, action, **extra):
        return dict(session='test-session', action=action, args={},
                    created_at=time.time(), expires_at=time.time() + 5, **extra)

    def test_expired_and_previous_session_requests_never_reach_controller(self):
        for key, value in [('session', 'old-session'), ('expires_at', time.time() - 1)]:
            request = self.request('run-all')
            request[key] = value
            with self.assertRaises(ValueError):
                self.bridge.dispatch(request)
        self.bridge.gui.commands.run.assert_not_called()

    def test_stop_invalidates_already_queued_motion(self):
        queued = self.request('run-all')
        self.bridge.dispatch(self.request('stop'))
        with self.assertRaisesRegex(ValueError, 'predates'):
            self.bridge.dispatch(queued)
        self.bridge.gui.commands.stop.assert_called_once()
        self.bridge.gui.commands.run.assert_not_called()

    def test_claimed_request_is_not_executed_again_on_next_poll(self):
        path = self.bridge.mailbox / 'stop-1.request.json'
        atomic_json(path, self.request('stop'))
        self.bridge.poll()
        self.bridge.poll()
        self.bridge.gui.commands.stop.assert_called_once()
        response = json.loads((self.bridge.mailbox / 'stop-1.response.json').read_text())
        self.assertTrue(response['ok'])
        self.assertFalse(path.exists())

    def test_stop_precedes_and_rejects_queued_run(self):
        atomic_json(self.bridge.mailbox / 'aaa.request.json', self.request('run-all'))
        atomic_json(self.bridge.mailbox / 'stop-z.request.json', self.request('stop'))
        self.bridge.poll()
        self.bridge.poll()
        self.bridge.gui.commands.run.assert_not_called()
        response = json.loads((self.bridge.mailbox / 'aaa.response.json').read_text())
        self.assertFalse(response['ok'])

    def test_stale_snapshot_is_not_returned_as_live(self):
        self.bridge.gui.commands.frame_received_at = time.monotonic() - 3
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.bridge.dispatch(self.request('snapshot'))

    def test_busy_gui_cannot_start_another_run(self):
        self.bridge.gui.commands.running = True
        with self.assertRaisesRegex(ValueError, 'Stop'):
            self.bridge.dispatch(self.request('run-all'))
        self.bridge.gui.commands.run.assert_not_called()

    def test_setup_filename_uses_existing_load_validation_and_restores_dialogs(self):
        import closed_loop_calibration as module
        from agent_control import gui_dialogs
        original = module.show_view_dialog
        with gui_dialogs(module, 'saved.json'):
            self.assertEqual(module.show_view_dialog(None, module.filedialog.askopenfilename,
                                                    title='Load'), 'saved.json')
            with self.assertRaisesRegex(ValueError, 'wrong camera'):
                module.show_view_dialog(None, module.messagebox.showerror, 'Setup', 'wrong camera')
        self.assertIs(module.show_view_dialog, original)

    def test_camera_opens_without_starting_control_or_changing_setup(self):
        self.bridge.dispatch(self.request('show-camera'))
        self.bridge.gui.commands.open_camera_popout.assert_called_once()
        self.bridge.gui.commands.run.assert_not_called()

    def test_recent_receipt_is_not_enough_to_advance_after_send_with_camera_delay(self):
        tab = self.bridge.gui.commands
        tab.feedback.snapshot.return_value = {'sent_time_s': 100.}
        tab.frame_received_at = 100.9
        with self.assertRaisesRegex(ValueError, 'estimated delay'):
            self.bridge.require_observation_after_send()
        tab.frame_received_at = 101.6
        self.bridge.require_observation_after_send()
        tab.run.assert_not_called()

    def test_transient_response_file_lock_does_not_stop_or_repeat_an_action(self):
        original = Path.replace
        failures = []

        def replace(path, destination):
            if str(destination).endswith('.response.json') and not failures:
                failures.append(True)
                raise PermissionError('Temporary Windows sharing violation')
            return original(path, destination)

        atomic_json(self.bridge.mailbox / 'camera.request.json', self.request('show-camera'))
        with patch.object(Path, 'replace', replace):
            self.bridge.poll()
            self.bridge.poll()
        self.bridge.gui.commands.open_camera_popout.assert_called_once()
        self.bridge.gui.commands.stop.assert_not_called()
        self.assertTrue(json.loads((self.bridge.mailbox / 'camera.response.json').read_text())['ok'])

    def test_shared_status_reader_closes_windows_handle_and_reads_replacements(self):
        from agent_control import read_json
        path = self.bridge.directory / 'reader.json'
        for value in range(3):
            atomic_json(path, {'value': value})
            self.assertEqual(read_json(path), {'value': value})


if __name__ == '__main__':
    unittest.main()
