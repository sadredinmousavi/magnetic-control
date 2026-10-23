"""Overspeed braking uses mocked actuators and fresh measured velocity."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from arrival_control import ArrivalGate
from command_sequences import XYStep
from control_workflow import OptimizerXYFeedback
from direction_xy_control import DirectionWeightedPID, DirectionWeightedXYFeedback
from incremental_xy_control import IncrementalXYFeedback
from assistant_xy_control import AssistantXYFeedback
from functions_main import MagneticForcePID


class SpeedBrakingTests(unittest.TestCase):
    def fixture(self, cls=OptimizerXYFeedback, **kwargs):
        now = [.1]
        pid_class = DirectionWeightedPID if cls is DirectionWeightedXYFeedback else MagneticForcePID
        control = cls(clock=lambda: now[0], speed_limit_mm_s=.5,
                      pid=pid_class(kp=.25, ki=.02, kv=1.5, period_s=.1,
                                    integral_limit_n=.01, velocity_tau_s=0.), **kwargs)
        control.seed_anchor([0., 0.], dict(angles_deg=[60.]*8, sent_time_s=0.))
        step = XYStep((.01, 0.), 0., 180.)
        control.begin_point(step, [0., 0.])
        control.enabled.set()
        control.pid.seed_integral([.003, .004])
        gate = ArrivalGate(tolerance_mm=.25, clock=lambda: now[0])
        gate.arm(step.target_xy_m)
        commander = Mock()
        commander.calculate.return_value = dict(angles_deg=[60.]*8)
        def tick(velocity, *, fraction=1., held=False, changed=None):
            now[0] = round(now[0]+.2, 6)
            measured = np.array([.001, .002])
            gate.observe([measured], now[0])
            control.pid.position, control.pid.observed_at = measured.copy(), now[0]
            control.pid.velocity, control.pid.velocity_ready = np.asarray(velocity), True
            def move(case, target, **options):
                self.assertTrue(options['active']())
                if changed:
                    changed()
                    self.assertFalse(options['active']())
                    return
                result = dict(angles_deg=[59.]*8, allocation_fraction=fraction)
                options['on_calculated'](result)
                options['on_held' if held else 'on_sent'](result)
            commander.move.side_effect = move
            control._continuous_update(step, commander, 'case', gate, lambda: True, Mock(),
                                       measured, now[0], gate.loss_generation, control.command_revision)
            return measured
        return control, commander, gate, tick

    def test_brake_opposes_velocity_and_freezes_large_integral_in_both_pid_modes(self):
        for cls in (OptimizerXYFeedback, DirectionWeightedXYFeedback):
            with self.subTest(controller=cls.__name__):
                control, commander, _, tick = self.fixture(cls)
                velocity = np.array([.0006, .0008])
                measured = tick(velocity)
                command = commander.move.call_args.args[1]
                np.testing.assert_allclose(command-measured, -1.5*velocity)
                np.testing.assert_allclose(control.pid.integral_force, [.003, .004])
                self.assertTrue(control.snapshot()['speed_braking'])
                self.assertIn('Speed braking', control.snapshot()['state'])

    def test_brake_offset_cap_hysteresis_and_resume(self):
        control, commander, _, tick = self.fixture()
        measured = tick([.003, .004])
        offset = commander.move.call_args.args[1]-measured
        self.assertAlmostEqual(np.linalg.norm(offset), .002)
        self.assertLess(np.dot(offset, [.003, .004]), 0.)
        tick([.00045, 0.])
        self.assertTrue(control.speed_braking)
        tick([.0004, 0.])
        self.assertFalse(control.speed_braking)
        self.assertGreater(commander.move.call_args.args[1][0], .01)

    def test_motor_deadband_does_not_claim_applied_brake_and_limits_are_reported(self):
        control, commander, _, tick = self.fixture()
        tick([.001, 0.], held=True)
        self.assertIsNone(control.snapshot().get('sent_optimizer_xy_mm'))
        self.assertIn('motor deadband', control.snapshot()['state'])
        measured = tick([.001, 0.], fraction=.5)
        desired = measured-np.array([.0015, 0.])
        np.testing.assert_allclose(control.snapshot()['sent_optimizer_xy_mm'], .5*desired*1000)
        self.assertIn('optimizer input limit', control.snapshot()['state'])

    def test_incremental_and_assistant_cancel_ramp_and_sync_actual_brake_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            for cls in (IncrementalXYFeedback, AssistantXYFeedback):
                with self.subTest(controller=cls.__name__):
                    extra = {'bridge_directory': directory} if cls is AssistantXYFeedback else {}
                    control, _, _, tick = self.fixture(cls, **extra)
                    control.goal = np.array([.002, 0.])
                    control.phase_started = .1
                    control.request_id, control.awaiting_command = 'old-request', True
                    measured = tick([.001, 0.])
                    self.assertIsNone(control.goal)
                    self.assertIsNone(control.phase_started)
                    self.assertFalse(control.completed_step)
                    np.testing.assert_allclose(control.applied_xy, measured-[.0015, 0.])
                    if cls is AssistantXYFeedback:
                        self.assertIsNone(control.request_id)
                        self.assertFalse(control.awaiting_command)

    def test_off_loss_or_new_frame_veto_braking_solution(self):
        for change in ('off', 'loss', 'new_frame'):
            control, _, gate, tick = self.fixture()
            def changed():
                if change == 'off':
                    control.set_enabled(False)
                elif change == 'loss':
                    gate.observe(None)
                else:
                    gate.observe([[.001, .002]], gate.observed_at+.01)
            tick([.001, 0.], changed=changed)
            self.assertIsNone(control.pid.committed_at)
            np.testing.assert_allclose(control.pid.integral_force, [.003, .004])

    def test_invalid_settings_are_rejected(self):
        for settings in ({'speed_limit_mm_s': 0}, {'speed_limit_mm_s': float('nan')},
                         {'brake_gain_s': -1}, {'max_brake_mm': float('inf')}):
            with self.assertRaises(ValueError):
                OptimizerXYFeedback(**settings)

    def test_braking_has_priority_inside_arrival_tolerance_and_respects_workspace(self):
        control, commander, gate, tick = self.fixture()
        gate.tolerance_mm = 20.
        tick([.001, 0.])
        self.assertTrue(control.speed_braking)
        self.assertEqual(commander.move.call_count, 1)
        control.validate = Mock(side_effect=ValueError('outside calibrated workspace'))
        tick([.001, 0.])
        self.assertEqual(commander.move.call_count, 1)
        self.assertIn('Speed braking blocked: outside calibrated workspace', control.snapshot()['state'])


if __name__ == '__main__':
    unittest.main()
