from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from settling_control import SettlingMonitor
from arrival_control import ArrivalGate
from position_control import PositionPID, PositionFeedback
from command_sequences import XYStep


class SettlingTests(unittest.TestCase):
    def test_requires_post_command_samples_and_continuous_low_speed(self):
        monitor = SettlingMonitor()
        monitor.reset(10.)
        monitor.observe([[5,0]], 9.)
        self.assertFalse(monitor.snapshot(10.)['ready'])
        for timestamp in (10.1,10.3,10.5,10.7,10.9,11.1):
            monitor.observe([[5,0]], timestamp)
            self.assertFalse(monitor.snapshot(timestamp)['ready'])
        monitor.observe([[5,0]], 11.3)
        result = monitor.snapshot(11.3)
        self.assertTrue(result['ready'])
        np.testing.assert_allclose(result['positions_mm'], [[5,0]])
        monitor.observe([[5,0]], 11.3)  # Same frame cannot advance time.
        self.assertFalse(monitor.snapshot(12.4)['ready'])
        monitor.reset(11.3)
        self.assertFalse(monitor.snapshot(11.3)['ready'])

    def test_opposing_robots_and_oscillation_are_not_stationary(self):
        for oscillates in (False, True):
            monitor = SettlingMonitor(minimum_wait_s=0)
            for index in range(12):
                t = index*.2
                x = (-1 if index % 2 else 1) if oscillates else t
                monitor.observe([[x,0],[-x,0]], t)
            result = monitor.snapshot(t)
            self.assertFalse(result['ready'])
            self.assertGreater(result['speed_mm_s'], .5)

    def test_loss_gap_and_robot_count_change_require_new_hold(self):
        for interruption in ('lost', 'gap', 'count'):
            monitor = SettlingMonitor()
            for index in range(8):
                monitor.observe([[0,0]], index*.2)
            self.assertTrue(monitor.snapshot(1.4)['ready'])
            if interruption == 'lost':
                monitor.observe(None, 1.5)
                monitor.observe([[0,0]], 1.6)
            elif interruption == 'gap':
                monitor.observe([[0,0]], 3.)
            else:
                monitor.observe([[0,0],[1,0]], 1.6)
            self.assertFalse(monitor.snapshot(3. if interruption == 'gap' else 1.6)['ready'])

    def test_initial_stillness_cannot_skip_minimum_wait(self):
        monitor = SettlingMonitor(minimum_wait_s=2)
        monitor.reset(0)
        for index in range(1,10):
            monitor.observe([[0,0]], index*.2)
            self.assertFalse(monitor.snapshot(index*.2)['ready'])
        monitor.observe([[2,0]], 2.)  # Delayed response starts moving.
        self.assertFalse(monitor.snapshot(2.)['ready'])

    def test_settled_measurement_averages_the_low_speed_window(self):
        monitor = SettlingMonitor()
        for index in range(8):
            t = index*.2
            monitor.observe([[.1*t,0]], t)
        result = monitor.snapshot(t)
        self.assertTrue(result['ready'])
        self.assertLess(result['positions_mm'][0,0], .1*t)
        self.assertGreaterEqual(result['positions_mm'][0,0], .1*(t-.5))

    def test_feedback_holds_during_motion_sends_once_settled_and_restarts_hold(self):
        now = [0.]
        monitor = SettlingMonitor()
        gate = ArrivalGate(clock=lambda: now[0], require_arrival=False)
        gate.settling = monitor
        gate.arm((0,0))
        feedback = PositionFeedback(PositionPID(incremental=True, period_s=.25, limit_mm=20),
                                    settling=monitor, clock=lambda: now[0])
        feedback.enabled.set()
        feedback.initial_sent()
        sent_times = []
        def advance(timeout):
            now[0] = round(now[0]+.2, 4)
            # Robot moves until t=1, then stays still at -4 mm.
            x = -5 + min(now[0], 1.)
            gate.observe([[x/1000,0]], now[0])
        gate.condition.wait = advance
        def move(case, target, **kwargs):
            self.assertTrue(kwargs['active']())
            self.assertTrue(monitor.snapshot(now[0])['ready'])
            self.assertGreaterEqual(now[0], 2.)
            result = {'angles_deg':[60]}
            kwargs['on_sent'](result)
            self.assertFalse(monitor.snapshot(now[0])['ready'])
            sent_times.append(now[0])
            return result
        feedback.wait_at_target(XYStep((0,0),0,180), Mock(move=move), 'case', gate,
                                lambda: now[0] < 4 and len(sent_times) < 2, Mock())
        self.assertEqual(len(sent_times), 2)
        self.assertGreaterEqual(sent_times[1]-sent_times[0], 1.)

    def test_solver_result_is_discarded_if_robot_moves_again(self):
        now = [0.]
        monitor = SettlingMonitor()
        gate = ArrivalGate(clock=lambda: now[0], require_arrival=False)
        gate.settling = monitor
        gate.arm((0,0))
        feedback = PositionFeedback(PositionPID(incremental=True, limit_mm=20),
                                    settling=monitor, clock=lambda: now[0])
        feedback.enabled.set()
        called = []
        def advance(timeout):
            now[0] += .2
            gate.observe([[-.004,0]], now[0])
        gate.condition.wait = advance
        def move(case, target, **kwargs):
            self.assertTrue(kwargs['active']())
            now[0] += .2
            gate.observe([[-.0032,0]], now[0])  # <1 mm drift: speed alone must veto it.
            self.assertFalse(monitor.snapshot(now[0])['ready'])
            self.assertFalse(kwargs['active']())
            called.append(True)
            return None
        report = Mock()
        feedback.wait_at_target(XYStep((0,0),0,180), Mock(move=move), 'case', gate,
                                lambda: now[0] < 5 and not called, report)
        self.assertEqual(len(called), 1)
        report.assert_not_called()
        np.testing.assert_allclose(feedback.last_sent_u, [0,0])

    def test_invalid_settings(self):
        for settings in ({'speed_mm_s':0}, {'hold_s':0}, {'minimum_wait_s':-1}, {'window_s':float('nan')}):
            with self.assertRaises(ValueError):
                SettlingMonitor(**settings)


if __name__ == '__main__':
    unittest.main()
