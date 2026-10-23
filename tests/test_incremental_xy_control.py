"""Exercise move/settle behavior with fresh observations and mocked motor IO."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from arrival_control import ArrivalGate
from command_sequences import XYStep
from functions_main import MagneticForcePID
from incremental_xy_control import IncrementalXYFeedback
from trajectory_tab import TrajectoryArrivalGate


class IncrementalXYTests(unittest.TestCase):
    def fixture(self, *, target=(.004, 0.), held=False, fraction=1., auto_arrival=False, **settings):
        now = [0.]
        settings.setdefault('step_mm', 1.)
        controller = IncrementalXYFeedback(clock=lambda: now[0], rate_mm_s=1., settle_seconds=.2,
            pid=MagneticForcePID(kp=0., ki=0., kv=0., period_s=.1, velocity_tau_s=0.), **settings)
        controller.seed_anchor([0., 0.], dict(angles_deg=[60.]*8, sent_time_s=0.))
        step = XYStep(target, 0, 180)
        controller.enabled.set()
        controller.begin_point(step, [0., 0.])
        gate = (TrajectoryArrivalGate(tolerance_mm=.25, stable_seconds=.2, max_speed_mm_s=.2,
                                      velocity_tau_s=0., clock=lambda: now[0]) if auto_arrival else
                ArrivalGate(tolerance_mm=.25, clock=lambda: now[0]))
        gate.arm(target)
        commander, calls = Mock(), []
        commander.calculate.return_value = dict(angles_deg=[60.]*8)
        def move(case, command, **kwargs):
            self.assertTrue(kwargs['active']())
            result = dict(angles_deg=[59.]*8, allocation_fraction=fraction)
            kwargs['on_calculated'](result)
            calls.append((now[0], np.asarray(command).copy(), controller.goal.copy()))
            kwargs['on_held' if held else 'on_sent'](result)
            return result
        commander.move.side_effect = move
        def tick(stamp, position=(0., 0.)):
            now[0] = float(stamp)
            measured = np.asarray(position)
            gate.observe([measured], now[0])
            controller.pid.observe(measured, now[0])
            controller._continuous_update(step, commander, 'case', gate, lambda: True, Mock(),
                                          measured, now[0], gate.loss_generation, controller.command_revision)
        return controller, commander, calls, tick, gate, step

    def test_diagonal_step_has_one_mm_norm_rate_limit_and_settling_pause(self):
        control, commander, calls, tick, _, _ = self.fixture(target=(.004, .004))
        for stamp in np.arange(.1, 2.7, .1):
            tick(round(float(stamp), 3))
        self.assertTrue(calls)
        first_goal = calls[0][2]
        self.assertAlmostEqual(np.linalg.norm(first_goal), .001)
        previous = np.zeros(2)
        previous_stamp = 0.
        for stamp, command, goal in calls:
            self.assertLessEqual(np.linalg.norm(command-previous), .001*(stamp-previous_stamp)+1e-12)
            self.assertAlmostEqual(np.linalg.norm(goal), .001 if np.allclose(goal, first_goal) else .002)
            previous = command
            previous_stamp = stamp
        first_ramp = [call for call in calls if np.allclose(call[2], first_goal)]
        second_ramp = [call for call in calls if not np.allclose(call[2], first_goal)]
        self.assertTrue(second_ramp)
        self.assertGreaterEqual(second_ramp[0][0]-first_ramp[-1][0], .2)
        commander.calculate.assert_called_once()
        np.testing.assert_allclose(commander.move.call_args.kwargs['anchored_xy_request']['anchor_xy_m'], [0., 0.])
        self.assertEqual(control.pid.ki, 0.)

    def test_stationary_robot_causes_no_progress_stop_instead_of_unbounded_commands(self):
        control, _, calls, tick, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'no progress after 3'):
            for stamp in np.arange(.1, 12., .1):
                tick(round(float(stamp), 3))
        self.assertLessEqual(max(np.linalg.norm(call[1]) for call in calls), .003+1e-12)
        self.assertEqual(control.snapshot()['incremental_phase'], 'stopped')

    def test_following_robot_reaches_waypoint_with_settled_automatic_arrival(self):
        control, _, calls, tick, gate, _ = self.fixture(auto_arrival=True)
        position = np.zeros(2)
        for index in range(1, 600):
            # A simple lagging plant; real hardware tuning remains necessary.
            position += .3*(control.applied_xy-position)
            tick(round(index*.1, 3), position=position)
            if gate.advance_due:
                break
        self.assertTrue(gate.advance_due)
        self.assertTrue(calls)
        self.assertLessEqual(np.linalg.norm(position-np.array([.004, 0.]))*1000, gate.tolerance_mm)
        self.assertFalse(control.halt_reason)

    def test_no_ramp_until_measured_speed_settles_and_timeout_when_it_does_not(self):
        _, _, calls, tick, _, _ = self.fixture(timeout_seconds=2.)
        with self.assertRaisesRegex(ValueError, 'timeout'):
            for stamp in np.arange(.1, 3., .1):
                tick(round(float(stamp), 3), position=(stamp*.001, 0.))
        self.assertFalse(calls)

    def test_motor_deadband_does_not_claim_unsent_command_or_completed_step(self):
        control, _, calls, tick, _, _ = self.fixture(held=True, timeout_seconds=3.)
        with self.assertRaisesRegex(ValueError, 'timeout'):
            for stamp in np.arange(.1, 4., .1):
                tick(round(float(stamp), 3))
        self.assertTrue(calls)
        np.testing.assert_allclose(control.applied_xy, [0., 0.])
        self.assertFalse(control.completed_step)
        self.assertTrue(all(np.linalg.norm(call[1]) <= .001+1e-12 for call in calls))

    def test_optimizer_limit_stops_and_reports_realizable_position(self):
        control, _, calls, tick, _, _ = self.fixture(fraction=.5)
        with self.assertRaisesRegex(ValueError, 'optimizer input limit'):
            for stamp in np.arange(.1, 1., .1):
                tick(round(float(stamp), 3))
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(control.applied_xy, calls[0][1]*.5)
        np.testing.assert_allclose(control.snapshot()['sent_optimizer_xy_mm'], calls[0][1]*500)

    def test_offset_limit_rejects_first_step_before_motor_write(self):
        _, _, calls, tick, _, _ = self.fixture(max_bias_mm=.5)
        with self.assertRaisesRegex(ValueError, 'maximum command lead'):
            for stamp in np.arange(.1, 1., .1):
                tick(round(float(stamp), 3))
        self.assertFalse(calls)

    def test_arrival_holds_and_waypoint_transition_keeps_applied_command(self):
        control, _, calls, tick, _, _ = self.fixture(target=(.0002, 0.))
        for stamp in (.1, .2, .5):
            tick(stamp)
        self.assertFalse(calls)
        control.applied_xy = np.array([.001, .0005])
        control.begin_point(XYStep((.008, 0.), 0, 180), [0., 0.])
        np.testing.assert_allclose(control.applied_xy, [.001, .0005])
        np.testing.assert_allclose(control.snapshot()['sent_optimizer_xy_mm'], [1., .5])
        np.testing.assert_allclose(control.last_sent_u, [-7., .5])
        self.assertIsNone(control.goal)

    def test_tracking_loss_cancels_pending_ramp_and_requires_settling_again(self):
        control, _, calls, tick, _, _ = self.fixture()
        for stamp in np.arange(.1, .9, .1):
            tick(round(float(stamp), 3))
        self.assertTrue(calls)
        applied = control.applied_xy.copy()
        control._reset_slow_assist()  # Parent feedback loop calls this on tracking loss.
        control.pid.reset()
        self.assertIsNone(control.goal)
        np.testing.assert_allclose(control.request_xy, applied)
        count = len(calls)
        for stamp in (1., 1.1, 1.2):
            tick(stamp)
        self.assertEqual(len(calls), count)

    def test_configuration_validation(self):
        for invalid in ({'step_mm': 1.1}, {'rate_mm_s': 0.}, {'settle_seconds': -1.},
                        {'max_bias_mm': np.nan}, {'timeout_seconds': .1}):
            with self.assertRaises(ValueError):
                IncrementalXYFeedback(**invalid)


if __name__ == '__main__':
    unittest.main()
