"""Supplied coil polynomial, total equilibrium and GUI005 isolation; no hardware."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from gui005_radial_control import HelmholtzRadialModel, RadialControl, load_radial_model
from point_control import PointCommander, load_hardware_case
from control_workflow import calculate_point_control
from functions_main import calculate_potential_hessian, build_actuation_matrix


class RadialModelTests(unittest.TestCase):
    def test_polynomial_direction_sign_current_and_center(self):
        p = np.array([.012, .016])
        r, mz = .02, 1.25e-4
        expected = mz*(.053654*r-10.7146*r**3-1629.01*r**5)*p/r
        model = HelmholtzRadialModel(mz_a_m2=mz)
        force, _ = model.force_and_hessian(p, 123.)
        np.testing.assert_allclose(force, expected)
        reverse, _ = HelmholtzRadialModel(current_a=-2*12/5.31, mz_a_m2=mz).force_and_hessian(p, 123.)
        np.testing.assert_allclose(reverse, -2*force)
        shifted, _ = HelmholtzRadialModel(mz_a_m2=mz, center_xy_m=(.01, -.01)).force_and_hessian(p+[.01, -.01], 123.)
        np.testing.assert_allclose(shifted, force)
        signed, _ = HelmholtzRadialModel(moment_sign=-1).force_and_hessian(p, mz)
        np.testing.assert_allclose(signed, -force)

    def test_hessian_matches_negative_force_derivative_including_axis(self):
        model = HelmholtzRadialModel()
        mz = 1.25e-4
        for point in (np.zeros(2), np.array([.012, -.023])):
            force, H = model.force_and_hessian(point, mz)
            jac = np.column_stack([
                (model.force_and_hessian(point+delta, mz)[0]
                 - model.force_and_hessian(point-delta, mz)[0]) / 2e-7
                for delta in np.eye(2)*1e-7])
            np.testing.assert_allclose(H, -jac, atol=1e-15, rtol=1e-8)
        np.testing.assert_array_equal(model.force_and_hessian([0, 0], mz)[0], [0, 0])
        np.testing.assert_allclose(model.force_and_hessian([0, 0], mz)[1], -mz*.053654*np.eye(2))

    def test_polynomial_extrapolates_and_rejects_nonfinite_inputs(self):
        model = HelmholtzRadialModel()
        for point in ([.04, 0], [.040001, 0], [.03, .03], [.07, -.02]):
            point = np.asarray(point)
            s = point@point
            force, H = model.force_and_hessian(point, 1e-4)
            np.testing.assert_allclose(force, 1e-4*(.053654-10.7146*s-1629.01*s*s)*point)
            jac = np.column_stack([
                (model.force_and_hessian(point+delta, 1e-4)[0]
                 - model.force_and_hessian(point-delta, 1e-4)[0])/2e-7
                for delta in np.eye(2)*1e-7])
            np.testing.assert_allclose(H, -jac, atol=1e-15, rtol=1e-8)
        for point in ([np.nan, 0], [np.inf, 0], [0]):
            with self.assertRaises(ValueError):
                model.force_and_hessian(point, 1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'settings.json'
            path.write_text('{"enabled":false}')
            self.assertIsNone(load_radial_model(path))


class RadialAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_hardware_case('case_100')
        cls.control = RadialControl(HelmholtzRadialModel())

    def test_initial_equilibrium_preserves_total_stiffness_and_orientation(self):
        target = np.array([.02, -.01])
        result = self.control.synthesize(self.params, target, ratio=2., angle_deg=30.)
        cfg, sources = self.control.geometry(self.params, target)
        coil, coil_h = self.control.model.force_and_hessian(target, cfg.M_ROBOT_MAGNITUDE)
        magnetic = build_actuation_matrix(target, sources, cfg.C_F)@result['u_values']
        np.testing.assert_allclose(magnetic+coil, [0, 0], atol=1e-12)
        expected_h = calculate_potential_hessian(target, sources, cfg.C_F, result['nominal_magnetic_inputs'])
        actual_h = calculate_potential_hessian(target, sources, cfg.C_F, result['u_values'])+coil_h
        np.testing.assert_allclose(actual_h, expected_h, rtol=1e-5, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(actual_h)[0], 0)
        np.testing.assert_allclose(result['radial_force_n'], coil)

    def test_inward_coil_is_balanced_by_outward_magnetic_compensation(self):
        center = np.array([.005, .011])
        control = RadialControl(HelmholtzRadialModel(moment_sign=-1, center_xy_m=tuple(center)))
        for offset in ([.015, .008], [-.015, .008], [-.015, -.008], [.015, -.008]):
            offset = np.asarray(offset)
            target = center+offset
            result = control.synthesize(self.params, target)
            cfg, sources = control.geometry(self.params, target)
            coil, coil_h = control.model.force_and_hessian(target, cfg.M_ROBOT_MAGNITUDE)
            magnetic = build_actuation_matrix(target, sources, cfg.C_F)@result['u_values']
            self.assertLess(float(coil@offset), 0.)
            self.assertGreater(float(magnetic@offset), 0.)
            np.testing.assert_allclose(magnetic+coil, [0, 0], atol=1e-12)
            H = calculate_potential_hessian(target, sources, cfg.C_F, result['u_values'])+coil_h
            self.assertGreater(np.linalg.eigvalsh(H)[0], 0.)

    def test_pid_and_carried_integral_count_coil_once(self):
        target = np.array([.01, 0])
        baseline = self.control.synthesize(self.params, target)
        original = baseline['angles_deg'].copy()
        bias = np.array([2e-8, -1e-8])
        result = self.control.apply_integral(self.params, target, baseline, bias)
        np.testing.assert_allclose(result['predicted_force_n'], bias, atol=1e-12)
        np.testing.assert_array_equal(baseline['angles_deg'], original)
        measured = target+[.001, -.001]
        feedback = self.control.allocate(self.params, target, measured, result['angles_deg'],
                                         baseline['u_values'], bias, max_angle_step_deg=2.)
        cfg, sources = self.control.geometry(self.params, target)
        A = build_actuation_matrix(measured, sources, cfg.C_F)
        coil, _ = self.control.model.force_and_hessian(measured, cfg.M_ROBOT_MAGNITUDE)
        np.testing.assert_allclose(feedback['predicted_force_n'], A@feedback['u_values']+coil)
        np.testing.assert_allclose(feedback['applied_force_n'], A@result['u_values']+coil)
        self.assertLessEqual(np.max(abs(feedback['angles_deg']-result['angles_deg'])), 2.+1e-8)

    def test_zero_current_reproduces_nominal_equilibrium(self):
        target = [.01, .005]
        nominal = calculate_point_control(self.params, target)
        corrected = RadialControl(HelmholtzRadialModel(current_a=0)).synthesize(self.params, target)
        np.testing.assert_allclose(corrected['angles_deg'], nominal['angles_deg'], atol=1e-6)

    def test_incremental_needs_no_equilibrium_solve(self):
        target = [.01, 0]
        initial = self.control.synthesize(self.params, target)
        with patch('gui005_radial_control.calculate_point_control', side_effect=AssertionError('unexpected solve')):
            result = self.control.allocate(self.params, target, target, initial['angles_deg'],
                                           None, [1e-9, 0], incremental=True)
        self.assertTrue(np.isfinite(result['predicted_force_n']).all())

    def test_gui_only_injection_signatures_and_transmission_beyond_fit(self):
        port = Mock(is_open=True)
        plain = PointCommander(lambda: port)
        gui = PointCommander(lambda: port, radial_model=HelmholtzRadialModel())
        reverse = PointCommander(lambda: port, radial_model=HelmholtzRadialModel(moment_sign=-1))
        self.assertIsNone(plain.radial_control)
        self.assertNotEqual(plain.hardware_signature(self.params), gui.hardware_signature(self.params))
        self.assertNotEqual(gui.hardware_signature(self.params), reverse.hardware_signature(self.params))
        with patch('helpers.moveArray') as send:
            result = gui.move('case_100', [.041, 0])
            self.assertLess(result['force_residual_n'], 1e-12)
            gui.move('case_100', [.01, 0])
            self.assertEqual(send.call_count, 2)

    def test_infeasible_bias_rejected(self):
        target = [.01, 0]
        baseline = self.control.synthesize(self.params, target)
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            self.control.apply_integral(self.params, target, baseline, [1., 0])

    def test_deadband_hold_reports_total_force_of_actual_pose(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port, radial_model=HelmholtzRadialModel())
        target = np.array([.01, 0])
        initial = commander.calculate('case_100', target)
        measured = target+[.001, 0]
        held = Mock()
        with patch('helpers.moveArray') as send:
            result = commander.move_force_pid('case_100', target, measured, initial['angles_deg'],
                initial['u_values'], [1e-8, 0], motor_deadband_deg=90., on_held=held)
            send.assert_not_called()
        held.assert_called_once()
        cfg, sources = self.control.geometry(self.params, target)
        coil, _ = self.control.model.force_and_hessian(measured, cfg.M_ROBOT_MAGNITUDE)
        actual = build_actuation_matrix(measured, sources, cfg.C_F)@initial['u_values']+coil
        np.testing.assert_allclose(held.call_args.args[0]['predicted_force_n'], actual)
        self.assertAlmostEqual(held.call_args.args[0]['force_residual_n'], np.linalg.norm(actual-[1e-8, 0]))
        np.testing.assert_array_equal(result['requested_force_n'], [1e-8, 0])

    def test_anchored_path_uses_compensated_baseline(self):
        anchor = [.01, 0]
        baseline = self.control.synthesize(self.params, anchor)
        same = self.control.anchored(self.params, anchor, anchor_xy_m=anchor,
            anchor_angles_deg=baseline['angles_deg'], baseline_angles_deg=baseline['angles_deg'])
        np.testing.assert_allclose(same['angles_deg'], baseline['angles_deg'])
        shifted = self.control.anchored(self.params, [.012, 0], anchor_xy_m=anchor,
            anchor_angles_deg=baseline['angles_deg'], baseline_angles_deg=baseline['angles_deg'])
        expected = self.control.synthesize(self.params, [.012, 0])
        np.testing.assert_allclose(shifted['angles_deg'], expected['angles_deg'], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
