"""Direction priority, integral state and allocation limits for optional XY PID."""
import unittest

import numpy as np

from direction_xy_control import DirectionWeightedPID


class DirectionWeightedPIDTests(unittest.TestCase):
    def pid(self, memory):
        pid = DirectionWeightedPID(kp=.25, ki=.01, kv=.5, integral_limit_n=.01,
                                   period_s=.25, velocity_tau_s=0.)
        pid.observe([0., 0.], 0.)
        pid.observe([0., 0.], .25)
        pid.seed_integral(memory)
        pid.committed_at = 0.
        return pid

    def test_sideways_memory_is_reduced_without_changing_parallel_correction(self):
        pid = self.pid([.004, .006])
        proposal = pid.propose([.002, 0.])
        np.testing.assert_allclose(proposal['pid_force_n'], [.004505, .0012])
        # Preview/discard/repeated proposals do not mutate memory.
        pid.propose([0., .002])
        np.testing.assert_allclose(pid.integral_force, [.004, .006])
        pid.commit(proposal, proposal['pid_force_n'], proposal['pid_force_n'])
        np.testing.assert_allclose(pid.integral_force, [.004005, .006])

    def test_rotated_error_uses_vector_projection(self):
        pid = self.pid([.004, 0.])
        result = pid.propose([.002, .002])['pid_force_n']
        np.testing.assert_allclose(result, [.002905, .002105])

    def test_opposing_memory_unwinds_only_on_commit_even_under_saturation(self):
        pid = self.pid([-.004, .006])
        proposal = pid.propose([.002, 0.])
        decayed = -.004 * np.exp(-.25 / 2.)
        self.assertAlmostEqual(proposal['pid_force_n'][0], .0005 + decayed + .000005)
        np.testing.assert_allclose(pid.integral_force, [-.004, .006])
        requested = proposal['pid_force_n']
        pid.commit(proposal, requested, requested - np.array([.001, 0.]))
        # Saturation blocks new positive integral, but allows memory decay.
        np.testing.assert_allclose(pid.integral_force, [decayed, .006])

    def test_zero_error_is_finite_and_keeps_velocity_damping(self):
        pid = self.pid([.004, .006])
        pid.velocity = np.array([.001, -.002])
        result = pid.propose([0., 0.])['pid_force_n']
        np.testing.assert_allclose(result, [.0035, .007])
        self.assertTrue(np.isfinite(pid.propose([1e-12, 0.])['pid_force_n']).all())

    def test_integral_limit_and_parameter_validation(self):
        pid = self.pid([.01, .01])
        proposal = pid.propose([1., 1.])
        self.assertTrue((np.abs(proposal['integral_force_n']) <= .01).all())
        for kwargs in ({'sideways_weight': -1}, {'sideways_weight': 1.1},
                       {'sideways_weight': float('nan')}, {'opposing_decay_s': 0},
                       {'opposing_decay_s': float('inf')}):
            with self.assertRaises(ValueError):
                DirectionWeightedPID(**kwargs)


if __name__ == '__main__':
    unittest.main()
