"""Continuous force PID checks; all actuator calls are mocked."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from functions_main import MagneticForcePID, build_actuation_matrix, generate_circular_source_positions
from control_workflow import ForcePIDFeedback, calculate_force_pid_control, apply_integral_to_baseline
from case_loader import load_case, build_common_config
from arrival_control import ArrivalGate
from settling_control import SettlingMonitor
from command_sequences import XYStep, execute_xy_sequence
from point_control import PointCommander


class ForcePIDMathTests(unittest.TestCase):
    def test_default_integral_has_hundred_second_time_and_gradual_force_growth(self):
        pid = MagneticForcePID()
        self.assertAlmostEqual(pid.kp, 1.5e-5)
        self.assertAlmostEqual(pid.ki, 1.5e-7)
        self.assertAlmostEqual(pid.kv, 1.5e-5)
        self.assertAlmostEqual(pid.kp / pid.ki, 100.)
        pid.observe([-.01, 0], 0.)
        pid.observe([-.01, 0], .25)
        proposal = pid.propose([0, 0])
        pid.commit(proposal, proposal['pid_force_n'], proposal['pid_force_n'])
        pid.observe([-.01, 0], .5)
        proposal = pid.propose([0, 0])
        np.testing.assert_allclose(proposal['integral_force_n'], [3.75e-10, 0], atol=1e-20)

    def test_carried_bias_is_bounded_and_disabled_by_zero_ki(self):
        pid = MagneticForcePID(integral_limit_n=1e-7)
        pid.seed_integral([2e-7, -3e-7])
        np.testing.assert_allclose(pid.integral_force, [1e-7, -1e-7])
        pid.ki = 0
        pid.reset(keep_integral=True)
        np.testing.assert_array_equal(pid.integral_force, [0, 0])
        with self.assertRaisesRegex(ValueError, 'finite'):
            pid.seed_integral([np.nan, 0])

    def test_next_point_retains_integral_but_reacquires_velocity_without_catchup(self):
        feedback = ForcePIDFeedback()
        feedback.enabled.set()
        feedback.begin_point(XYStep((0, 0), 0, 180), [0, 0])
        feedback.pid.seed_integral([2e-8, -1e-8])
        feedback.pid.observe([0, 0], 1.)
        feedback.pid.observe([.001, 0], 1.25)
        feedback.pid.committed_at = 1.25
        feedback.begin_point(XYStep((.01, 0), 0, 180), [0, 0])
        np.testing.assert_allclose(feedback.pid.integral_force, [2e-8, -1e-8])
        self.assertFalse(feedback.pid.velocity_ready)
        feedback.pid.observe([.001, 0], 100.)
        feedback.pid.observe([.001, 0], 100.25)
        proposal = feedback.pid.propose([.01, 0])
        self.assertEqual(proposal['dt_s'], 0.)
        np.testing.assert_allclose(proposal['integral_force_n'], [2e-8, -1e-8])

    def test_off_manual_and_preview_do_not_send_a_carried_initial_bias(self):
        for mode in ('off', 'manual', 'preview'):
            with self.subTest(mode=mode):
                feedback = ForcePIDFeedback(manual=mode == 'manual', preview_only=mode == 'preview')
                if mode != 'off':
                    feedback.enabled.set()
                feedback.pid.seed_integral([2e-8, 0])
                feedback.begin_point(XYStep((.01, 0), 0, 180), [0, 0])
                self.assertEqual(feedback.initial_move_options(), {})

    def test_infeasible_carried_force_fails_before_any_motor_write(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        with patch.object(commander, 'calculate', return_value={'angles_deg': [60.]*8}), \
             patch('point_control.load_hardware_case', return_value={}), \
             patch('control_workflow.calculate_force_pid_control', return_value={'force_residual_n': 1.}), \
             patch('helpers.moveArray') as write:
            with self.assertRaisesRegex(ValueError, 'infeasible'):
                commander.move('case_100', (.01, 0), initial_force_bias_n=[1e-8, 0])
        write.assert_not_called()

    def test_two_point_sequence_sends_only_the_bias_corrected_second_baseline(self):
        params = load_case('case_100.case')
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        feedback = ForcePIDFeedback()
        feedback.enabled.set()
        steps = [XYStep((0., 0.), 0, 180), XYStep((.01, 0.), 0, 180)]
        bias = np.array([2e-8, -1e-8])
        completed = []
        def wait(step, *args, **kwargs):
            completed.append(step)
            if len(completed) == 1:
                feedback.pid.seed_integral(bias)
            else:
                np.testing.assert_allclose(feedback.last_result['predicted_force_n'], bias, atol=1e-12)
                self.assertTrue(feedback.last_result['force_compensated'])
                np.testing.assert_allclose(feedback.pid.integral_force, bias)
            return True
        feedback.wait_at_target = wait
        with patch('helpers.moveArray') as write:
            commander.prepare_points('case_100', steps)
            baseline = commander.prepared_points['case_100'][1][commander.point_key((.01, 0.), 1., None)]
            cached_angles = baseline['angles_deg'].copy()
            outcome = execute_xy_sequence(steps, commander, 'case_100', active=lambda: True,
                                          report=Mock(), arrival_gate=ArrivalGate(require_arrival=False),
                                          feedback=feedback, prepared=True)
        self.assertEqual(outcome, 'finished')
        self.assertEqual(write.call_count, 2)  # No extra uncorrected transition command.
        sent_angles = np.asarray(write.call_args_list[1].args[1])
        self.assertGreater(np.max(abs(sent_angles - cached_angles)), .01)
        np.testing.assert_array_equal(baseline['angles_deg'], cached_angles)  # Cache remains nominal.
        cfg = build_common_config(dict(params, TARGET_SCHEDULE=[(0., np.array([.01, 0.]), 1., None)],
                                       INITIAL_ROBOT_POSITIONS=np.zeros((1, 2))))
        sources = generate_circular_source_positions(8, cfg.RADIUS)
        force = build_actuation_matrix((.01, 0.), sources, cfg.C_F) @ np.cos(np.deg2rad(sent_angles))
        np.testing.assert_allclose(force, bias, atol=1e-12)

    def test_switching_control_off_during_initial_bias_solve_vetoes_transmission(self):
        feedback = ForcePIDFeedback()
        feedback.enabled.set()
        feedback.pid.seed_integral([2e-8, 0])
        commander = PointCommander(lambda: Mock(is_open=True))
        # Keep the port identity stable across pre-send validation.
        port = Mock(is_open=True)
        commander.port_provider = lambda: port
        def allocate(*args, **kwargs):
            feedback.set_enabled(False)
            return {'angles_deg': [59.]*8}
        with patch.object(commander, 'calculate', return_value={'angles_deg': [60.]*8}), \
             patch('point_control.load_hardware_case', return_value={}), \
             patch('point_control.apply_integral_to_baseline', side_effect=allocate), \
             patch('helpers.moveArray') as write:
            outcome = execute_xy_sequence([XYStep((.01, 0), 0, 180)], commander, 'case',
                                          active=lambda: True, report=Mock(), feedback=feedback)
        self.assertEqual(outcome, 'stopped')
        write.assert_not_called()

    def test_velocity_braking_remains_active_inside_position_tolerance(self):
        pid = MagneticForcePID(kp=2, ki=0, kv=3, velocity_tau_s=0)
        pid.observe([0, 0], 0.)
        pid.observe([.001, 0], .1)
        proposal = pid.propose([.001, 0], tolerance_m=.002)
        np.testing.assert_allclose(proposal['velocity_m_s'], [.01, 0])
        np.testing.assert_allclose(proposal['pid_force_n'], [-.03, 0])
        pid.observe([50, 50], .1)  # Repeated timestamp is not another velocity sample.
        np.testing.assert_allclose(pid.velocity, [.01, 0])

    def test_antiwindup_freezes_saturated_integral_but_allows_unwinding(self):
        pid = MagneticForcePID(kp=0, ki=1, kv=0, velocity_tau_s=0, integral_limit_n=.01)
        pid.observe([-.01, 0], 0.)
        pid.observe([-.01, 0], .1)
        pid.commit(pid.propose([0, 0]), [0, 0], [0, 0])
        pid.observe([-.01, 0], .2)
        proposal = pid.propose([0, 0])
        self.assertGreater(proposal['integral_force_n'][0], 0)
        pid.commit(proposal, proposal['pid_force_n'], [0, 0])
        np.testing.assert_allclose(pid.integral_force, [0, 0])
        pid.integral_force = np.array([.005, 0])
        pid.observe([-.01, 0], .3)
        proposal = pid.propose([-.02, 0])
        pid.commit(proposal, [.01, 0], [0, 0])
        self.assertLess(pid.integral_force[0], .005)

    def test_stale_gap_requires_velocity_reacquisition_and_no_integral_catchup(self):
        pid = MagneticForcePID(ki=1)
        pid.observe([0, 0], 0.)
        pid.observe([.001, 0], .1)
        pid.observe([.1, 0], 2.)
        self.assertFalse(pid.velocity_ready)
        with self.assertRaises(ValueError):
            pid.propose([0, 0])
        pid.observe([.1, 0], 2.1)
        self.assertEqual(pid.propose([0, 0])['dt_s'], 0.)

    def test_allocation_matches_only_pid_force_at_measured_position(self):
        params = load_case('case_100.case')
        measured = np.array([.002, -.001])
        requested = np.array([-1e-8, 2e-9])
        result = calculate_force_pid_control(params, (0, 0), measured, [60.]*8, [.5]*8,
                                              requested, max_angle_step_deg=30)
        self.assertNotIn('disturbance_n', result)
        self.assertNotIn('integral_rebase_n', result)
        np.testing.assert_allclose(result['requested_force_n'], requested)
        cfg = build_common_config(dict(params, TARGET_SCHEDULE=[(0., np.zeros(2), 1., None)],
                                       INITIAL_ROBOT_POSITIONS=np.zeros((1, 2))))
        A = build_actuation_matrix(measured, generate_circular_source_positions(8, cfg.RADIUS), cfg.C_F)
        np.testing.assert_allclose(result['predicted_force_n'], A @ result['u_values'])
        self.assertLess(np.linalg.norm(A @ result['u_values']-requested), 1e-10)

    def test_pid_gain_validation(self):
        for values in ({'kp': -1}, {'kv': np.nan}, {'period_s': 0}, {'integral_limit_n': 0}):
            with self.assertRaises(ValueError):
                MagneticForcePID(**values)


class ContinuousRuntimeTests(unittest.TestCase):
    def fixture(self, *, speed=.001, start=-.01):
        now = [0.]
        monitor = SettlingMonitor()
        pid = MagneticForcePID(ki=0, period_s=.1, velocity_tau_s=0)
        feedback = ForcePIDFeedback(pid=pid, clock=lambda: now[0])
        gate = ArrivalGate(require_arrival=False, clock=lambda: now[0])
        gate.settling = monitor
        step = XYStep((0, 0), 0, 180)
        feedback.begin_point(step, [0, 0])
        gate.arm(step.target_xy_m)
        feedback.initial_sent({'angles_deg': [60.]*8})
        feedback.enabled.set()
        def advance(timeout):
            now[0] = round(now[0]+.05, 5)
            gate.observe([[start+speed*now[0], 0]], now[0])
        gate.condition.wait = advance
        return feedback, gate, monitor, step, now

    def result(self, requested):
        return dict(angles_deg=[59.]*8, reference_inputs=np.full(8, .5), force_compensated=True,
                    requested_force_n=np.asarray(requested), predicted_force_n=np.asarray(requested),
                    force_residual_n=0., stiffness_eigenvalues=[1e-5, 1e-5], calculation_time_s=.001)

    def test_sends_while_moving_without_stationary_learning_or_integral(self):
        feedback, gate, monitor, step, now = self.fixture()
        calls, report = [], Mock()
        def move(case, target, measured, angles, reference, pid_force, **kwargs):
            self.assertTrue(kwargs['active']())
            self.assertFalse(monitor.snapshot(now[0])['ready'])
            self.assertNotIn('stationary_position_m', kwargs)
            self.assertLessEqual(kwargs['max_angle_step_deg'], feedback.angle_rate*(now[0]-feedback.last_sent_at)+1e-6)
            result = self.result(pid_force)
            kwargs['on_calculated'](result)
            kwargs['on_sent'](result)
            calls.append(now[0])
            return result
        feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                lambda: now[0] < 1 and len(calls) < 2, report)
        self.assertEqual(len(calls), 2)
        self.assertLess(calls[0], .5)  # Before the stationary speed window can complete.
        self.assertGreaterEqual(calls[1]-calls[0], .1-1e-8)
        np.testing.assert_allclose(feedback.pid.integral_force, [0, 0])

    def test_brakes_even_when_already_inside_arrival_tolerance(self):
        feedback, gate, monitor, step, now = self.fixture(speed=.005, start=-.0002)
        calls = []
        def move(*args, **kwargs):
            self.assertTrue(kwargs['active']())
            self.assertLess(np.linalg.norm(args[2]), .001)
            self.assertLess(args[5][0], 0)  # Moving +X: damping requests -X force.
            calls.append(1)
            return None
        feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                lambda: now[0] < .5 and not calls, Mock())
        self.assertEqual(len(calls), 1)

    def test_stationary_hold_cannot_learn_or_double_pid_force(self):
        feedback, gate, monitor, step, now = self.fixture(speed=0)
        forces = []
        def move(*args, **kwargs):
            self.assertNotIn('stationary_position_m', kwargs)
            self.assertNotIn('rebase_integral', kwargs)
            self.assertEqual(len(args), 6)
            forces.append(np.asarray(args[5]).copy())
            result = self.result(args[5])
            kwargs['on_calculated'](result)
            kwargs['on_held'](result)
            return result
        feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                lambda: now[0] < 2, Mock())
        self.assertGreater(len(forces), 8)
        np.testing.assert_allclose(forces, np.tile(forces[0], (len(forces), 1)))
        self.assertFalse(hasattr(feedback, 'disturbance_estimate_n'))

    def test_optional_slow_assist_boosts_stall_then_restores_normal_force_on_motion(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                feedback, gate, monitor, step, now = self.fixture(speed=0)
                feedback.slow_motion_assist = enabled
                base_kp, base_ki, base_kv = feedback.pid.kp, feedback.pid.ki, feedback.pid.kv
                def resume_motion(timeout):
                    now[0] = round(now[0]+.05, 5)
                    gate.observe([[-.01+.001*max(0., now[0]-7.), 0]], now[0])
                gate.condition.wait = resume_motion
                samples = []
                def move(*args, **kwargs):
                    error = np.asarray(step.target_xy_m)-args[2]
                    normal = base_kp*error*max(0., 1-gate.tolerance_mm/1000/np.linalg.norm(error))-base_kv*feedback.pid.velocity
                    samples.append((now[0], args[5].copy(), normal, feedback.snapshot()['slow_assist_scale']))
                    self.assertLessEqual(kwargs['max_angle_step_deg'], feedback.max_angle_step_deg)
                    kwargs['on_held'](self.result(args[5]))
                feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                        lambda: now[0] < 8., Mock())
                for t, force, normal, scale in samples:
                    if not enabled or t < 3. or t > 7.1:
                        np.testing.assert_allclose(force, normal)
                        self.assertEqual(scale, 1.)
                    if enabled and 6.5 < t < 7.:
                        np.testing.assert_allclose(force, 1.5*normal)
                        self.assertEqual(scale, 1.5)
                self.assertTrue(any(6.5 < t < 7. for t, *_ in samples))
                self.assertEqual((feedback.pid.kp, feedback.pid.ki, feedback.pid.kv), (base_kp, base_ki, base_kv))
                np.testing.assert_array_equal(feedback.pid.integral_force, [0, 0])

    def test_slow_assist_restarts_dwell_after_gap_off_or_new_point_and_excludes_near_target(self):
        feedback, gate, monitor, step, now = self.fixture(speed=0)
        feedback.slow_motion_assist = True
        for t in np.arange(0., 7., .25):
            scale = feedback._slow_motion_scale(10., 0., t, 1.)
        self.assertEqual(scale, 1.5)
        self.assertEqual(feedback._slow_motion_scale(10., 0., 9., 1.), 1.)
        self.assertEqual(feedback._slow_motion_scale(2., 0., 9.25, 1.), 1.)
        self.assertIsNone(feedback.slow_since)
        for reset in (lambda: feedback.set_enabled(False), lambda: feedback.set_manual_mode(True),
                      lambda: feedback.begin_point(step, [0, 0])):
            feedback.slow_since, feedback.slow_sample = 0., 6.
            reset()
            self.assertIsNone(feedback.slow_since)
            self.assertEqual(feedback.snapshot()['slow_assist_scale'], 1.)

    def test_mode_change_discards_solution_without_integral_commit(self):
        feedback, gate, monitor, step, now = self.fixture()
        running = [True]
        def move(*args, **kwargs):
            self.assertTrue(kwargs['active']())
            feedback.set_manual_mode(True)
            self.assertFalse(kwargs['active']())
            running[0] = False
            return None
        feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                lambda: running[0] and now[0] < .5, Mock())
        self.assertIsNone(feedback.pid.committed_at)

    def test_optional_integral_builds_gradually_without_stationary_force_jump(self):
        feedback, gate, monitor, step, now = self.fixture(speed=0)
        feedback.pid.ki = 1e-6
        forces = []
        def move(*args, **kwargs):
            forces.append(np.asarray(args[5]).copy())
            kwargs['on_held'](self.result(args[5]))
        feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                lambda: now[0] < 2, Mock())
        increments = np.diff(np.asarray(forces)[:, 0])
        self.assertGreater(feedback.pid.integral_force[0], 0)
        self.assertTrue(np.all(increments >= -1e-20))
        self.assertLessEqual(float(increments.max()), feedback.pid.ki*.01*2*feedback.pid.period+1e-20)
        self.assertLess(feedback.pid.integral_force[0], feedback.pid.integral_limit)

    def test_loss_off_next_and_stale_frame_veto_inflight_pid_allocation(self):
        for change in ('loss', 'off', 'next', 'stale', 'stop'):
            with self.subTest(change=change):
                feedback, gate, monitor, step, now = self.fixture()
                running = [True]
                def move(*args, **kwargs):
                    self.assertTrue(kwargs['active']())
                    if change == 'loss':
                        gate.observe(None)
                    elif change == 'off':
                        feedback.set_enabled(False)
                    elif change == 'next':
                        gate.request_next()
                    elif change == 'stale':
                        now[0] += 1.
                    else:
                        running[0] = False
                    self.assertFalse(kwargs['active']())
                    running[0] = False
                    return None
                feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                        lambda: running[0] and now[0] < 2, Mock())
                self.assertIsNone(feedback.pid.committed_at)

    def test_repeated_camera_frame_cannot_produce_multiple_control_commands(self):
        feedback, gate, monitor, step, now = self.fixture()
        old_wait = gate.condition.wait
        def advance(timeout):
            if now[0] < .1:
                old_wait(timeout)
            else:
                now[0] = round(now[0]+.05, 5)  # No new frame.
        gate.condition.wait = advance
        solver = Mock(return_value=None)
        feedback.wait_at_target(step, Mock(move_force_pid=solver), 'case', gate,
                                lambda: now[0] < .8, Mock())
        solver.assert_called_once()

    def test_queued_next_keeps_braking_until_minimum_point_wait_expires(self):
        feedback, gate, monitor, step, now = self.fixture()
        gate.next_not_before = .5
        gate.request_next()
        times = []
        def move(*args, **kwargs):
            self.assertTrue(kwargs['active']())
            times.append(now[0])
            result = self.result(args[5])
            kwargs['on_held'](result)
            return result
        outcome = feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                           lambda: now[0] < 1., Mock())
        self.assertTrue(outcome)
        self.assertTrue(times)
        self.assertTrue(all(t < .5 for t in times))
        self.assertGreaterEqual(now[0], .5)

    def test_sequence_starts_feedback_before_row_wait_and_gates_navigation_instead(self):
        feedback, gate, monitor, original, now = self.fixture()
        step = XYStep((0, 0), 2., 180)
        result = {'angles_deg': [60.]*8}
        commander = Mock()
        def move(*args, **kwargs):
            self.assertEqual(kwargs['settle_seconds'], 0)
            kwargs['on_sent'](result)
            return result
        commander.move.side_effect = move
        def control(*args, **kwargs):
            self.assertEqual(now[0], 0.)
            self.assertEqual(gate.next_not_before, 2.)
            gate.request_next()
            self.assertFalse(gate.advance_due)
            return True
        feedback.wait_at_target = Mock(side_effect=control)
        outcome = execute_xy_sequence([step], commander, 'case', active=lambda: True,
                                       report=Mock(), arrival_gate=gate, feedback=feedback)
        self.assertEqual(outcome, 'finished')
        feedback.wait_at_target.assert_called_once()

    def test_motor_deadband_uses_actual_held_force_for_antiwindup_without_serial_write(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        held, sent = Mock(), Mock()
        result = dict(angles_deg=[60.01]*8, applied_force_n=np.array([1e-9, 0]),
                      applied_force_residual_n=2e-9, predicted_force_n=[3e-9, 0])
        with patch('point_control.load_hardware_case', return_value={}), \
             patch('point_control.calculate_force_pid_control', return_value=result), \
             patch('helpers.moveArray') as write:
            commander.move_force_pid('case', (0, 0), (.001, 0), [60.]*8, [.5]*8, [0, 0],
                                     motor_deadband_deg=.1, on_held=held, on_sent=sent)
        write.assert_not_called()
        sent.assert_not_called()
        np.testing.assert_allclose(held.call_args.args[0]['predicted_force_n'], [1e-9, 0])


if __name__ == '__main__':
    unittest.main()
