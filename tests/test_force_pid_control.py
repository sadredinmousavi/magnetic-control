"""Continuous force PID checks; all actuator calls are mocked."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from functions_main import MagneticForcePID, build_actuation_matrix, generate_circular_source_positions
from control_workflow import EquilibriumForceFeedback, calculate_force_pid_control
from case_loader import load_case, build_common_config
from arrival_control import ArrivalGate
from settling_control import SettlingMonitor
from command_sequences import XYStep, execute_xy_sequence
from point_control import PointCommander


class ForcePIDMathTests(unittest.TestCase):
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

    def test_allocation_matches_total_force_at_measured_position_without_moving_observer(self):
        params = load_case('case_100.case')
        measured = np.array([.002, -.001])
        with patch('control_workflow.estimate_equilibrium_disturbance') as observer:
            result = calculate_force_pid_control(params, (0, 0), measured, [60.]*8, [.5]*8,
                                                  [-1e-8, 2e-9], [3e-9, -2e-9], max_angle_step_deg=30)
        observer.assert_not_called()
        requested = np.array([-1.3e-8, 4e-9])
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
        pid = MagneticForcePID(period_s=.1, velocity_tau_s=0)
        feedback = EquilibriumForceFeedback(settling=monitor, continuous=True, pid=pid, clock=lambda: now[0])
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

    def result(self, requested, stationary=None):
        return dict(angles_deg=[59.]*8, reference_inputs=np.full(8, .5), force_compensated=True,
                    disturbance_n=[1e-9, 0] if stationary is not None else [0., 0.],
                    requested_force_n=np.asarray(requested), predicted_force_n=np.asarray(requested),
                    integral_rebase_n=[0., 0.], disturbance_learned=stationary is not None,
                    force_residual_n=0., stiffness_eigenvalues=[1e-5, 1e-5], calculation_time_s=.001)

    def test_sends_while_moving_without_stationary_learning_or_integral(self):
        feedback, gate, monitor, step, now = self.fixture()
        calls, report = [], Mock()
        def move(case, target, measured, angles, reference, pid_force, disturbance, **kwargs):
            self.assertTrue(kwargs['active']())
            self.assertFalse(monitor.snapshot(now[0])['ready'])
            self.assertIsNone(kwargs['stationary_position_m'])
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

    def test_stationary_estimate_only_after_unchanged_command_hold(self):
        feedback, gate, monitor, step, now = self.fixture(speed=0)
        learned = []
        def move(*args, **kwargs):
            if kwargs['stationary_position_m'] is not None:
                self.assertTrue(monitor.snapshot(now[0])['ready'])
                learned.append(now[0])
            result = self.result(args[5], kwargs['stationary_position_m'])
            if kwargs['stationary_position_m'] is None:
                result['disturbance_n'] = args[6]
            kwargs['on_calculated'](result)
            kwargs['on_held'](result)
            return result
        feedback.wait_at_target(step, Mock(move_force_pid=move), 'case', gate,
                                lambda: now[0] < 2, Mock())
        self.assertEqual(len(learned), 1)
        self.assertGreaterEqual(learned[0], 1.)
        np.testing.assert_allclose(feedback.disturbance_estimate_n, [1e-9, 0])

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
            commander.move_force_pid('case', (0, 0), (.001, 0), [60.]*8, [.5]*8, [0, 0], [0, 0],
                                     motor_deadband_deg=.1, on_held=held, on_sent=sent)
        write.assert_not_called()
        sent.assert_not_called()
        np.testing.assert_allclose(held.call_args.args[0]['predicted_force_n'], [1e-9, 0])


if __name__ == '__main__':
    unittest.main()
