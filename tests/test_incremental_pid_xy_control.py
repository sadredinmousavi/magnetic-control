from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from incremental_pid_xy_control import IncrementalPIDXYFeedback
from functions_main import MagneticForcePID
from command_sequences import XYStep
from arrival_control import ArrivalGate


class IncrementalPIDTests(unittest.TestCase):
    def fixture(self, **kwargs):
        now = [0.]
        controller = IncrementalPIDXYFeedback(clock=lambda: now[0], speed_limit_mm_s=.5,
            pid=MagneticForcePID(kp=.25, ki=.02, kv=.25, period_s=.25), **kwargs)
        controller.seed_anchor([0., 0.], dict(angles_deg=[60.]*8, sent_time_s=0.))
        step = XYStep((.01, 0.), 0., 180.)
        controller.begin_point(step, [0., 0.])
        controller.enabled.set()
        gate = ArrivalGate(tolerance_mm=.25, clock=lambda: now[0])
        gate.arm(step.target_xy_m)
        commander = Mock()
        commander.calculate.return_value = dict(angles_deg=[60.]*8)
        def tick(position=(0., 0.), velocity=(0., 0.), outcome='sent', fraction=1., stale=False):
            now[0] += .25
            measured = np.asarray(position, float)
            gate.observe([measured], now[0])
            controller.pid.velocity = np.asarray(velocity, float)
            controller.pid.velocity_ready = True
            def move(case, target, **options):
                result = dict(angles_deg=[59.]*8, allocation_fraction=fraction)
                options['on_calculated'](result)
                if stale:
                    now[0] += 2.
                if not options['active']() or outcome == 'preview':
                    return
                if outcome == 'error':
                    raise ValueError('test optimizer failure')
                options['on_sent' if outcome == 'sent' else 'on_held'](result)
            commander.move.side_effect = move
            controller._continuous_update(step, commander, 'case', gate, lambda: True, Mock(),
                measured, now[0], gate.loss_generation, controller.command_revision)
        return controller, commander, tick

    def test_recorded_y_case_increment_uses_error_not_old_integral(self):
        c, _, _ = self.fixture(command_rate_mm_s=100.)
        c.applied_xy = c.request_xy = np.array([0., -.004274868674])
        e = np.array([0., -.003475417617])
        v = np.array([0., .0000125737])
        c.history_error, c.history_velocity = e.copy(), v.copy()
        proposal = c.propose_command(np.array([0., -.010526315789]),
            np.array([0., -.007050898172]), v, .25, .25)
        self.assertAlmostEqual(proposal['target'][1], c.applied_xy[1]-.000017377088085)

    def test_rate_is_vector_norm_not_per_axis(self):
        c, _, tick = self.fixture()
        tick(position=(-.01, -.01))
        self.assertAlmostEqual(np.linalg.norm(c.applied_xy), .000125)
        self.assertIn('command rate', c.snapshot()['command_limits'])

    def test_no_hidden_integral_windup_after_rate_clipping(self):
        c, _, tick = self.fixture()
        for _ in range(100):
            tick()
        previous = c.applied_xy.copy()
        tick(position=(.011, 0.))
        self.assertLess(c.applied_xy[0], previous[0])
        np.testing.assert_array_equal(c.pid.integral_force, [0., 0.])

    def test_brake_opposes_motion_relative_to_command_and_keeps_rate_bound(self):
        c, _, tick = self.fixture()
        c.applied_xy = c.request_xy = np.array([.008, .004])
        before = c.applied_xy.copy()
        tick(velocity=(.0006, .0008))
        delta = c.applied_xy-before
        self.assertLess(np.dot(delta, [.0006, .0008]), 0.)
        self.assertAlmostEqual(np.linalg.norm(delta), .000125)
        self.assertTrue(c.snapshot()['speed_braking'])

    def test_slowdown_brakes_below_global_speed_limit(self):
        c, _, tick = self.fixture()
        tick(position=(.009, 0.), velocity=(.0003, 0.))
        self.assertTrue(c.speed_braking)
        self.assertAlmostEqual(c.snapshot()['effective_speed_limit_mm_s'], .5/3.)
        tick(position=(.009, 0.), velocity=(.00012, 0.))
        self.assertFalse(c.speed_braking)

    def test_deadband_keeps_sent_command_but_can_accumulate_small_pending_steps(self):
        c, _, tick = self.fixture()
        c.history_error = np.array([.01, 0.])
        tick(outcome='held')
        np.testing.assert_array_equal(c.applied_xy, [0., 0.])
        self.assertAlmostEqual(c.request_xy[0], .00005)
        tick(outcome='held')
        self.assertAlmostEqual(c.request_xy[0], .0001)
        tick()
        self.assertAlmostEqual(c.applied_xy[0], .000125)

    def test_failed_stale_preview_and_partial_allocations_do_not_commit(self):
        for options in (dict(outcome='error'), dict(outcome='preview'), dict(stale=True), dict(fraction=.5)):
            with self.subTest(options=options):
                c, _, tick = self.fixture()
                tick(**options)
                np.testing.assert_array_equal(c.applied_xy, [0., 0.])
                np.testing.assert_array_equal(c.request_xy, [0., 0.])
                self.assertIsNone(c.history_sample)

    def test_offset_bound_and_reversal(self):
        c, _, tick = self.fixture(command_offset_mm=1.)
        c.applied_xy = c.request_xy = np.array([.011, 0.])
        c.history_error = np.array([.01, 0.])
        tick()
        self.assertAlmostEqual(c.applied_xy[0], .011)
        self.assertIn('command offset', c.snapshot()['command_limits'])
        tick(position=(.012, 0.))
        self.assertLess(c.applied_xy[0], .011)

    def test_arrival_holds_and_pause_clears_pending_without_losing_sent_command(self):
        c, commander, tick = self.fixture()
        tick(position=(.01, 0.))
        commander.move.assert_not_called()
        tick(outcome='held')
        c.set_enabled(False)
        np.testing.assert_array_equal(c.request_xy, c.applied_xy)
        self.assertIsNone(c.history_sample)

    def test_waypoint_preserves_command_and_pid_error_history(self):
        c, _, tick = self.fixture()
        tick()
        previous = c.applied_xy.copy()
        history = c.history_error.copy()
        c.begin_point(XYStep((.012, 0.), 0., 180.), [0., 0.])
        np.testing.assert_array_equal(c.applied_xy, previous)
        np.testing.assert_array_equal(c.history_error, history)

    def test_invalid_configuration(self):
        for setting in ('command_rate_mm_s', 'command_offset_mm', 'slowdown_mm'):
            for value in (0., -1., float('nan')):
                with self.subTest(setting=setting, value=value), self.assertRaises(ValueError):
                    self.fixture(**{setting: value})

    def test_delayed_nonunity_plant_reaches_target(self):
        # A synthetic plant, not a fit to this user's hardware: 1.8x position
        # response, 0.5 s transport delay and 0.8 s mechanical time constant.
        c, _, tick = self.fixture()
        position, velocity = np.zeros(2), np.zeros(2)
        delayed = [np.zeros(2), np.zeros(2)]
        peak = 0.
        for _ in range(1600):
            tick(position, velocity)
            delayed.append(c.applied_xy.copy())
            equilibrium = 1.8*delayed.pop(0)
            following = position+(equilibrium-position)*(1-np.exp(-.25/.8))
            velocity = (following-position)/.25
            position = following
            peak = max(peak, position[0])
        self.assertLess(abs(position[0]-.01), .0003)
        self.assertLess(peak, .0103)

    def test_slew_budget_uses_actual_send_time_as_well_as_camera_time(self):
        c, _, _ = self.fixture()
        proposal = c.propose_command(np.array([.01, 0.]), np.zeros(2), np.zeros(2), .5, .25,
                                     rate_elapsed_s=.05)
        self.assertAlmostEqual(np.linalg.norm(proposal['target']), .000025)


if __name__ == '__main__':
    unittest.main()
