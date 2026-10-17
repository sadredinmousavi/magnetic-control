"""Force allocation and transport regressions; hardware is always mocked."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from functions_main import (generate_circular_source_positions, build_actuation_matrix,
                            calculate_potential_hessian, allocate_force_inputs)
from control_workflow import ForcePIDFeedback, calculate_force_pid_control
from case_loader import load_case
from command_sequences import XYStep
from arrival_control import ArrivalGate
from point_control import PointCommander


class ForceAllocationTests(unittest.TestCase):
    def setUp(self):
        self.sources = generate_circular_source_positions(8, .18)
        self.cf = 1e-9
        self.c = np.full(8, .5)
        self.target = np.zeros(2)

    def test_requested_force_preserves_full_hessian_and_angle_step(self):
        requested = build_actuation_matrix([.003, -.002], self.sources, self.cf) @ self.c
        c, info = allocate_force_inputs(self.target, self.sources, self.cf, requested, self.c, self.c)
        self.assertLess(info['force_residual_n'], info['initial_force_residual_n']*1e-3)
        H0 = calculate_potential_hessian(self.target, self.sources, self.cf, self.c)
        np.testing.assert_allclose(info['H'], H0, rtol=1e-5, atol=1e-15)
        self.assertTrue(np.all(info['stiffness_eigenvalues'] > 0))
        self.assertLessEqual(np.max(abs(np.rad2deg(np.arccos(c))-60)), 10+1e-6)

    def test_partial_allocation_respects_angle_step_and_reports_remaining_force(self):
        force = build_actuation_matrix([.02, -.01], self.sources, self.cf) @ self.c
        c, info = allocate_force_inputs(self.target, self.sources, self.cf, force, self.c, self.c,
                                       max_angle_step_deg=.1)
        self.assertGreater(info['force_residual_n'], 0)
        self.assertLess(info['force_residual_n'], info['initial_force_residual_n'])
        self.assertLessEqual(np.max(abs(np.rad2deg(np.arccos(c))-60)), .1+1e-5)

    def test_invalid_unstable_and_nonimproving_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            allocate_force_inputs(self.target, self.sources, self.cf, [np.nan, 0], self.c, self.c)
        with self.assertRaisesRegex(ValueError, 'positive restoring stiffness'):
            allocate_force_inputs(self.target, self.sources, self.cf, [0, 0], self.c, np.zeros(8))
        with self.assertRaisesRegex(ValueError, 'No useful force improvement'):
            allocate_force_inputs(self.target, self.sources, self.cf, [0, 0], self.c, self.c)

    def test_original_target_ratio_is_used_for_shifted_initial_pose(self):
        with patch('control_workflow.calculate_point_control', return_value={'u_values': self.c}) as baseline:
            result = calculate_force_pid_control(load_case('case_100.case'), (0, 0), (.003, -.002),
                                                 [60.]*8, None, [1e-8, 0], ratio=1., angle_deg=45)
        baseline.assert_called_once()
        self.assertEqual(baseline.call_args.kwargs, {'ratio': 1., 'angle_deg': 45})
        np.testing.assert_allclose(result['reference_inputs'], self.c)

    def test_manual_clicks_keep_using_xy_optimizer(self):
        feedback = ForcePIDFeedback()
        gate = ArrivalGate(require_arrival=False)
        step = XYStep((0, 0), 0, 180)
        feedback.begin_point(step, [0, 0])
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
        commander.move_force_pid.assert_not_called()

    def result(self):
        return dict(angles_deg=[59.]*8, requested_force_n=[1e-8, 0], predicted_force_n=[1e-8, 0])

    def test_preview_and_cancelled_calculation_never_write_serial(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        for preview in (False, True):
            active = [True]
            with patch('point_control.load_hardware_case', return_value={}), \
                 patch('point_control.calculate_force_pid_control', return_value=self.result()), \
                 patch('helpers.moveArray') as send:
                def calculated(result):
                    if not preview:
                        active[0] = False
                commander.move_force_pid('case', (0, 0), (.004, 0), [60.]*8, [.5]*8, [1e-8, 0],
                    preview_only=preview, active=lambda: active[0], on_calculated=calculated)
                send.assert_not_called()

    def test_successful_allocation_uses_shared_protocol_and_motor_zero(self):
        port = Mock(is_open=True)
        commander, sent = PointCommander(lambda: port), Mock()
        result = self.result()
        with patch('point_control.load_hardware_case', return_value={}), \
             patch('point_control.calculate_force_pid_control', return_value=result), \
             patch('helpers.moveArray') as send:
            commander.move_force_pid('case', (0, 0), (.004, 0), [60.]*8, [.5]*8, [1e-8, 0],
                                     zero_angle_deg=90, on_sent=sent)
        self.assertIs(send.call_args.args[0], port)
        np.testing.assert_allclose(send.call_args.args[1], [59.]*8)
        self.assertEqual(send.call_args.args[2:], (0, 90.))
        sent.assert_called_once_with(result)
