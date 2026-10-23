import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from assistant_xy_control import AssistantXYFeedback, atomic_json
from command_sequences import XYStep
from functions_main import MagneticForcePID
from arrival_control import ArrivalGate


class AssistantXYTests(unittest.TestCase):
    def test_atomic_publication_retries_windows_sharing_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            path.write_text('{"old":true}')
            original = Path.replace
            attempts = []
            def replace(source, destination):
                attempts.append(source)
                if len(attempts) < 3:
                    raise PermissionError(5, 'Access is denied')
                return original(source, destination)
            with patch.object(Path, 'replace', replace):
                atomic_json(path, {'ready': True})
            self.assertEqual(json.loads(path.read_text()), {'ready': True})
            self.assertEqual(len(attempts), 3)
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_persistent_file_conflict_holds_and_recovers_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            control = self.controller(directory)
            self.request(control)
            old_request = control.request_id
            control.goal = np.array([.001, 0.])
            with patch('assistant_xy_control.atomic_json', side_effect=PermissionError(5, 'Access is denied')):
                control.publish_bridge()
            self.assertIsNone(control.goal)
            self.assertIsNone(control.request_id)
            self.assertFalse(control.awaiting_command)
            np.testing.assert_allclose(control.request_xy, control.applied_xy)
            self.assertIsNotNone(control.bridge_error)
            control.publish_bridge()
            self.assertIsNone(control.bridge_error)
            self.assertFalse(json.loads((Path(directory)/'state.json').read_text())['ready_for_command'])
            self.request(control)
            self.assertNotEqual(control.request_id, old_request)

    def test_live_loop_waits_then_applies_only_one_assistant_selected_nudge(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [0.]
            control = AssistantXYFeedback(bridge_directory=directory, clock=lambda: now[0],
                pid=MagneticForcePID(kp=0., ki=0., kv=0., period_s=.1, velocity_tau_s=0.))
            control.seed_anchor([0., 0.], dict(angles_deg=[60.]*8, sent_time_s=0.))
            step = XYStep((.004, 0.), 0., 180.)
            control.enabled.set()
            control.begin_point(step, [0., 0.])
            gate = ArrivalGate(tolerance_mm=.25, clock=lambda: now[0])
            gate.arm(step.target_xy_m)
            commander = Mock()
            commander.calculate.return_value = dict(angles_deg=[60.]*8)
            def move(case, target, **kwargs):
                self.assertTrue(kwargs['active']())
                kwargs['on_sent'](dict(angles_deg=[59.]*8, allocation_fraction=1.))
            commander.move.side_effect = move
            def tick(index):
                now[0] = round(index*.1, 3)
                gate.observe([[0., 0.]], now[0])
                control.pid.observe([0., 0.], now[0])
                control._continuous_update(step, commander, 'case', gate, lambda: True, Mock(),
                    np.zeros(2), now[0], gate.loss_generation, control.command_revision)
            for index in range(1, 31):
                tick(index)
            commander.move.assert_not_called()
            self.assertTrue(control.awaiting_command)
            self.command(control)
            for index in range(31, 81):
                tick(index)
            np.testing.assert_allclose(control.applied_xy, [.0005, 0.])
            self.assertTrue(control.awaiting_command)
            self.assertEqual(control.no_progress_steps, 1)
            count = commander.move.call_count
            for index in range(81, 101):
                tick(index)
            self.assertEqual(commander.move.call_count, count)
            self.assertEqual(control.no_progress_steps, 1)

    def controller(self, directory):
        control = AssistantXYFeedback(bridge_directory=directory)
        control.seed_anchor([0., 0.], dict(angles_deg=[60.]*8, sent_time_s=0.))
        control.enabled.set()
        control.begin_point(XYStep((.004, 0.), 0., 180.), [0., 0.])
        return control

    def request(self, control, measured=(0., 0.)):
        measured = np.asarray(measured)
        return control.choose_goal(np.array([.004, 0.]), np.array([.004, 0.])-measured, measured, time.monotonic())

    def command(self, control, **changes):
        command = dict(session_id=control.session_id, request_id=control.request_id,
                       issued_monotonic_s=time.monotonic(), delta_command_mm=[.5, 0.])
        command.update(changes)
        atomic_json(control.bridge_directory/'command.json', command)

    def test_waits_without_commands_and_accepts_one_matching_bounded_command(self):
        with tempfile.TemporaryDirectory() as directory:
            control = self.controller(directory)
            self.assertIsNone(self.request(control))
            control.publish_bridge()
            state = json.loads((Path(directory)/'state.json').read_text())
            self.assertTrue(state['ready_for_command'])
            self.command(control)
            np.testing.assert_allclose(self.request(control), [.0005, 0.])
            self.assertFalse(control.awaiting_command)
            self.assertIsNone(self.request(control))  # Old command cannot be replayed.
            self.assertEqual(len(control.history), 1)

    def test_rejects_stale_wrong_session_oversized_and_wrong_direction_commands(self):
        for changes in (dict(session_id='old'), dict(request_id='old'), dict(issued_monotonic_s=time.monotonic()-20),
                        dict(delta_command_mm=[1., 1.]), dict(delta_command_mm=[-.5, 0.]),
                        dict(delta_command_mm=[0., 0.]), dict(delta_command_mm=[float('nan'), 0.])):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                control = self.controller(directory)
                self.request(control)
                # Permit deliberately malformed NaN in this input test.
                command = dict(session_id=control.session_id, request_id=control.request_id,
                               issued_monotonic_s=time.monotonic(), delta_command_mm=[.5, 0.])
                command.update(changes)
                (Path(directory)/'command.json').write_text(json.dumps(command))
                self.assertIsNone(self.request(control))
                self.assertFalse(control.history)

    def test_tracking_motion_loss_and_new_waypoint_invalidate_old_request(self):
        with tempfile.TemporaryDirectory() as directory:
            control = self.controller(directory)
            self.request(control)
            self.command(control)
            old = control.request_id
            self.assertIsNone(self.request(control, measured=(.0005, 0.)))
            self.assertNotEqual(control.request_id, old)
            control._reset_slow_assist()
            self.assertFalse(control.awaiting_command)
            self.assertIsNone(control.request_id)
            self.request(control)
            self.command(control)
            control.begin_point(XYStep((.008, 0.), 0., 180.), [0., 0.])
            self.assertIsNone(control.request_id)
            self.assertIsNone(self.request(control))
            control.publish_bridge(stopped=True)
            self.assertFalse(json.loads((Path(directory)/'state.json').read_text())['ready_for_command'])


if __name__ == '__main__':
    unittest.main()
