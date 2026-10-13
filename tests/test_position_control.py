from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from arrival_control import ArrivalGate
from position_control import PositionPID, PositionFeedback
from command_sequences import XYStep


class PositionControlTests(unittest.TestCase):
    def test_pid_bounds_slew_antiwindup_reset_and_invalid_gains(self):
        pid = PositionPID(kp=10, ki=1, limit_mm=2, slew_mm_s=1)
        np.testing.assert_allclose(pid.update([10, 0], 1), [1, 0])
        np.testing.assert_allclose(pid.update([10, 0], 2), [2, 0])
        for timestamp in range(3, 20):
            self.assertLessEqual(np.linalg.norm(pid.update([10, 10], timestamp)), 2 + 1e-9)
        np.testing.assert_allclose(pid.integral, [0, 0])
        pid.reset()
        np.testing.assert_allclose(pid.output, [0, 0])
        for gains in ({'kp': -1}, {'period_s': 0}, {'limit_mm': float('nan')}):
            with self.assertRaises(ValueError):
                PositionPID(**gains)

    def fixture(self):
        now, active = [.1], [True]
        clock = lambda: now[0]
        gate = ArrivalGate(clock=clock)
        now[0] = 0
        gate.arm((0, 0))
        now[0] = .1
        gate.observe([[-.01, 0]])
        feedback = PositionFeedback(PositionPID(), clock=clock)
        feedback.enabled.set()
        return feedback, gate, now, active

    def test_correction_uses_desired_plus_u_and_fixed_row_settings(self):
        feedback, gate, now, active = self.fixture()
        commander, report = Mock(), Mock()
        def move(case, target, **kwargs):
            np.testing.assert_allclose(target, [.001, 0])
            self.assertEqual(kwargs['ratio'], 2)
            self.assertEqual(kwargs['angle_deg'], 45)
            self.assertEqual(kwargs['zero_angle_deg'], 90)
            self.assertTrue(kwargs['active']())
            result = {'angles_deg': [60, 70]}
            kwargs['on_calculated'](result)
            kwargs['on_sent'](result)
            active[0] = False
            return result
        commander.move.side_effect = move
        feedback.wait_at_target(XYStep((0,0),0,90,2,45), commander, 'case_100', gate, lambda: active[0], report)
        self.assertEqual(commander.move.call_count, 1)
        self.assertEqual(feedback.snapshot()['commanded_xy_mm'], [1,0])
        self.assertEqual(feedback.snapshot()['sent_angles'], [60,70])
        report.assert_called_once()
        np.testing.assert_allclose(gate.target, [0,0])  # Arrival remains at original target.

    def test_stale_or_lost_then_reacquired_observation_discards_calculation(self):
        for loss in (False, True):
            feedback, gate, now, active = self.fixture()
            commander, report = Mock(), Mock()
            def move(case, target, **kwargs):
                kwargs['on_calculated']({'angles_deg':[60]})
                if loss:
                    gate.observe(None)
                    now[0] = .2
                    gate.observe([[-.01, 0]])
                else:
                    now[0] = 2
                self.assertFalse(kwargs['active']())
                active[0] = False
                return None
            commander.move.side_effect = move
            feedback.wait_at_target(XYStep((0,0),0,180), commander, 'case', gate, lambda: active[0], report)
            report.assert_not_called()
            self.assertEqual(feedback.snapshot()['sent_angles'], [])

    def test_outside_workspace_is_never_sent(self):
        feedback, gate, now, active = self.fixture()
        def validate(target):
            active[0] = False
            raise ValueError('Outside pins')
        feedback.validate = validate
        commander = Mock()
        feedback.wait_at_target(XYStep((0,0),0,180), commander, 'case', gate, lambda: active[0], Mock())
        commander.move.assert_not_called()
        self.assertIn('blocked', feedback.snapshot()['state'])

    def test_feedback_defaults_off(self):
        feedback = PositionFeedback(PositionPID())
        self.assertFalse(feedback.enabled.is_set())


if __name__ == '__main__':
    unittest.main()
