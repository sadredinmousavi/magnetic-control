import importlib
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from case_loader import validate_target_schedule
from functions_main import calculate_total_field, calculate_total_force_from_sources
from functions_main import _optimization_failure
from functions_utility import compute_grid_fields, extract_optimization_info, plot_field


class RefactorTests(unittest.TestCase):
    def test_usage_modules_are_import_safe(self):
        for module_name in ("usage1", "usage2", "usage3", "usage4"):
            importlib.import_module(module_name)

    def test_schedule_requires_increasing_times(self):
        with self.assertRaises(ValueError):
            validate_target_schedule([
                (1.0, np.zeros(2), 1.0, 0.0),
                (0.0, np.zeros(2), 1.0, 0.0),
            ])

    def test_optimization_failure_policy(self):
        fallback = np.array([0.25])
        with self.assertWarns(RuntimeWarning):
            returned = _optimization_failure("failed", fallback, "warn")
        self.assertIs(returned, fallback)
        with self.assertRaises(RuntimeError):
            _optimization_failure("failed", fallback, "error")

    def test_vectorized_grid_matches_scalar_models(self):
        sources = np.array([[2.0, 0.0], [0.0, 2.0]])
        controls = np.array([0.3, -0.4])
        moments = np.array([[1.0, 0.0], [0.0, 1.0]])
        X, Y, Fx, Fy, _, Bx, By = compute_grid_fields(
            sources, controls, moments, -0.5, 0.5, 3, 2.0, 3.0
        )
        point = np.array([X[1, 1], Y[1, 1]])
        self.assertTrue(np.allclose(
            [Fx[1, 1], Fy[1, 1]],
            calculate_total_force_from_sources(sources, controls, point, 2.0, 3.0),
        ))
        self.assertTrue(np.allclose(
            [Bx[1, 1], By[1, 1]], calculate_total_field(sources, moments, point)
        ))

    def test_all_plot_types_return_figures(self):
        axis = np.linspace(-1.0, 1.0, 5)
        X, Y = np.meshgrid(axis, axis)
        values = 1.0 + X**2 + Y**2
        field = {
            "X": X, "Y": Y, "Fx": values, "Fy": values,
            "U_pot": values, "Bx": values, "By": values,
            "target_pos": np.zeros(2),
        }
        info = extract_optimization_info(
            np.array([0.5, 0.5]), np.zeros(2), np.eye(2), np.ones(2),
            np.eye(2), desired_pos=np.zeros(2),
        )
        sources = np.array([[2.0, 0.0], [0.0, 2.0]])
        for plot_type in ("force_info", "force_potential", "force_magnetic"):
            options = {"block": False} if plot_type == "force_info" else {}
            figure = plot_field(plot_type, field, sources, info, **options)
            self.assertIsNotNone(figure)


if __name__ == "__main__":
    unittest.main()
