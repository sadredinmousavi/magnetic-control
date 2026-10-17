"""Model and runtime regression checks; motor IO is always mocked."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from functions_main import (generate_circular_source_positions, build_actuation_matrix,
                            calculate_potential_hessian, estimate_equilibrium_disturbance,
                            find_force_compensated_inputs)
from control_workflow import EquilibriumForceFeedback, calculate_equilibrium_force_control
from case_loader import load_case
from command_sequences import XYStep
from arrival_control import ArrivalGate
from settling_control import SettlingMonitor
from point_control import PointCommander


class EquilibriumMathTests(unittest.TestCase):
    def setUp(self):
        self.sources = generate_circular_source_positions(8, .18)
        self.cf = 1e-9
        self.c = np.full(8, .5)
        self.target = np.zeros(2)

    def test_stationary_force_estimate_and_cancellation_preserve_full_hessian(self):
        measured = np.array([.003, -.002])
        d = estimate_equilibrium_disturbance(measured, self.sources, self.cf, self.c)
        np.testing.assert_allclose(d, -build_actuation_matrix(measured, self.sources, self.cf) @ self.c)
        c, info = find_force_compensated_inputs(self.target, self.sources, self.cf, d, self.c, self.c)
        self.assertLess(info['force_residual_n'], info['initial_force_residual_n']*1e-3)
        H0 = calculate_potential_hessian(self.target, self.sources, self.cf, self.c)
        np.testing.assert_allclose(info['H'], H0, rtol=1e-5, atol=1e-15)
        self.assertTrue(np.all(info['stiffness_eigenvalues'] > 0))
        self.assertLessEqual(np.max(abs(np.rad2deg(np.arccos(c))-60)), 10+1e-6)

    def test_partial_compensation_respects_angle_step_and_reports_remaining_force(self):
        d = estimate_equilibrium_disturbance([.02, -.01], self.sources, self.cf, self.c)
        c, info = find_force_compensated_inputs(self.target, self.sources, self.cf, d, self.c, self.c,
                                               max_angle_step_deg=.1)
        self.assertGreater(info['force_residual_n'], 0)
        self.assertLess(info['force_residual_n'], info['initial_force_residual_n'])
        self.assertLessEqual(np.max(abs(np.rad2deg(np.arccos(c))-60)), .1+1e-5)

    def test_invalid_unstable_and_no_improvement_requests_do_not_return_angles(self):
        with self.assertRaises(ValueError):
            estimate_equilibrium_disturbance([np.nan, 0], self.sources, self.cf, self.c)
        with self.assertRaisesRegex(ValueError, 'positive restoring stiffness'):
            find_force_compensated_inputs(self.target, self.sources, self.cf, [0, 0], self.c, np.zeros(8))
        with self.assertRaisesRegex(ValueError, 'No useful force improvement'):
            find_force_compensated_inputs(self.target, self.sources, self.cf, [0, 0], self.c, self.c)

    def test_workflow_uses_original_target_ratio_for_shifted_initial_pose(self):
        with patch('control_workflow.calculate_point_control', return_value={'u_values': self.c}) as baseline:
            result = calculate_equilibrium_force_control(load_case('case_100.case'), (0, 0), (.003, -.002),
                                                         [60.]*8, None, ratio=1., angle_deg=45)
        baseline.assert_called_once()
        self.assertEqual(baseline.call_args.kwargs, {'ratio': 1., 'angle_deg': 45})
        np.testing.assert_allclose(result['reference_inputs'], self.c)
        self.assertEqual(result['angles_deg'].shape, (8,))


class EquilibriumRuntimeTests(unittest.TestCase):
    def fixture(self):
        now = [0.]
        monitor = SettlingMonitor(speed_mm_s=.5)
        feedback = EquilibriumForceFeedback(settling=monitor, clock=lambda: now[0])
        gate = ArrivalGate(require_arrival=False, clock=lambda: now[0])
        gate.settling = monitor
        step = XYStep((0, 0), 0, 180)
        feedback.begin_point(step, [0, 0])
        result = dict(angles_deg=[60.]*8)
        gate.arm(step.target_xy_m)
        feedback.initial_sent(result)
        feedback.enabled.set()
        return feedback, gate, monitor, step, now

    def result(self):
        return dict(angles_deg=[59.]*8, force_compensated=True,
                    disturbance_n=[1e-8, 0], requested_force_n=[-1e-8, 0],
                    predicted_force_n=[-1e-8, 0], force_residual_n=0.,
                    stiffness_eigenvalues=[1e-5, 1e-5], calculation_time_s=.001)

    def test_waits_for_stationarity_and_reestimates_from_latest_sent_angles(self):
        feedback, gate, monitor, step, now = self.fixture()
        calls = []
        def advance(timeout):
            now[0] = round(now[0]+.2, 4)
            x = -5+min(now[0], 1.)
            gate.observe([[x/1000, 0]], now[0])
        gate.condition.wait = advance
        def move(case, target, measured, applied_angles, reference, **kwargs):
            self.assertGreaterEqual(now[0], 2.)
            self.assertTrue(monitor.snapshot(now[0])['ready'])
            self.assertTrue(kwargs['active']())
            self.assertEqual(applied_angles, [60.]*8 if not calls else [59.]*8)
            np.testing.assert_allclose(measured, [-.004, 0])
            calls.append(now[0])
            result = self.result()
            kwargs['on_calculated'](result)
            kwargs['on_sent'](result)
            self.assertFalse(monitor.snapshot(now[0])['ready'])
            return result
        feedback.wait_at_target(step, Mock(move_equilibrium_compensation=move), 'case', gate,
                                lambda: now[0] < 7 and len(calls) < 2, Mock())
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(calls[1]-calls[0], 1.)
        self.assertIsNone(feedback.snapshot()['commanded_xy_mm'])
        self.assertEqual(feedback.snapshot()['correction_mm'], [0., 0.])

    def test_failure_or_preview_is_only_attempted_once_without_new_send(self):
        for preview in (False, True):
            with self.subTest(preview=preview):
                feedback, gate, monitor, step, now = self.fixture()
                feedback.preview_only = preview
                def advance(timeout):
                    now[0] = round(now[0]+.2, 4)
                    gate.observe([[-.004, 0]], now[0])
                gate.condition.wait = advance
                solver = Mock(return_value=self.result()) if preview else Mock(side_effect=ValueError('infeasible'))
                feedback.wait_at_target(step, Mock(move_equilibrium_compensation=solver), 'case', gate,
                                        lambda: now[0] < 4, Mock())
                solver.assert_called_once()
                self.assertEqual(feedback.last_result['angles_deg'], [60.]*8)
                self.assertFalse(feedback.snapshot().get('force_compensated', False))

    def test_movement_loss_and_mode_switch_veto_inflight_solution(self):
        for change in ('moving', 'loss', 'manual', 'off', 'next'):
            with self.subTest(change=change):
                feedback, gate, monitor, step, now = self.fixture()
                running = [True]
                def advance(timeout):
                    now[0] = round(now[0]+.2, 4)
                    gate.observe([[-.004, 0]], now[0])
                gate.condition.wait = advance
                def move(*args, **kwargs):
                    self.assertTrue(kwargs['active']())
                    if change == 'moving':
                        now[0] += .2
                        gate.observe([[-.01, 0]], now[0])
                    elif change == 'loss':
                        gate.observe(None)
                    elif change == 'manual':
                        feedback.set_manual_mode(True)
                    elif change == 'off':
                        feedback.set_enabled(False)
                    else:
                        gate.request_next()
                    self.assertFalse(kwargs['active']())
                    running[0] = False
                    return None
                report = Mock()
                feedback.wait_at_target(step, Mock(move_equilibrium_compensation=move), 'case', gate,
                                        lambda: running[0] and now[0] < 3, report)
                report.assert_not_called()

    def test_manual_clicks_still_use_xy_optimizer_and_never_force_solver(self):
        feedback, gate, monitor, step, now = self.fixture()
        feedback.set_manual_mode(True)
        feedback.request_manual(step, [4, -2])
        running = [True]
        commander, applied = Mock(), Mock()
        feedback.on_applied = applied
        def move(case, target, **kwargs):
            self.assertTrue(kwargs['active']())
            np.testing.assert_allclose(target, [.004, -.002])
            kwargs['on_sent'](dict(angles_deg=[58.]*8))
            running[0] = False
        commander.move.side_effect = move
        feedback.wait_at_target(step, commander, 'case', gate, lambda: running[0], Mock())
        applied.assert_called_once_with([4., -2.])
        commander.move_equilibrium_compensation.assert_not_called()

    def test_force_commander_preview_and_stale_calculation_never_write_serial(self):
        commander = PointCommander(lambda: Mock(is_open=True))
        for preview in (False, True):
            active = [True]
            with patch('point_control.load_hardware_case', return_value={}), \
                 patch('point_control.calculate_equilibrium_force_control', return_value=self.result()), \
                 patch('helpers.moveArray') as send:
                def calculated(result):
                    if not preview:
                        active[0] = False
                commander.move_equilibrium_compensation('case', (0, 0), (.004, 0), [60.]*8, [.5]*8,
                                                        preview_only=preview, active=lambda: active[0], on_calculated=calculated)
                send.assert_not_called()

    def test_successful_force_allocation_uses_shared_protocol_and_motor_zero(self):
        port = Mock(is_open=True)
        commander, sent = PointCommander(lambda: port), Mock()
        result = self.result()
        with patch('point_control.load_hardware_case', return_value={}), \
             patch('point_control.calculate_equilibrium_force_control', return_value=result), \
             patch('helpers.moveArray') as send:
            commander.move_equilibrium_compensation('case', (0, 0), (.004, 0), [60.]*8, [.5]*8,
                                                    zero_angle_deg=90, on_sent=sent)
        send.assert_called_once()
        self.assertIs(send.call_args.args[0], port)
        np.testing.assert_allclose(send.call_args.args[1], [59.]*8)
        self.assertEqual(send.call_args.args[2:], (0, 90.))
        sent.assert_called_once_with(result)


if __name__ == '__main__':
    unittest.main()
