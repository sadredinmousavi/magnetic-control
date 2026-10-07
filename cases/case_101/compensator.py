"""Case 101's calibrated 2D feed-forward error model (all model units: mm)."""

import argparse
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


CALIBRATION_CSV = Path(__file__).with_name("assets") / "calibration_xy.csv"
CALIBRATION_SOURCE = "20261007_012328_dwell_points_compensation_xy.csv"
COMMAND_RADIUS_LIMIT_MM = 50.0


class FeedForwardCompensator2D:
    """Fit measured = command + error(command), then invert within calibration."""

    def __init__(self, csv_path=CALIBRATION_CSV, tolerance_mm=0.05, max_iterations=100):
        if not np.isfinite(tolerance_mm) or tolerance_mm <= 0 or max_iterations < 1:
            raise ValueError("Tolerance and iteration count must be positive.")
        frame = pd.read_csv(csv_path)
        columns = ["target_x_mm", "target_y_mm", "measured_x_mm", "measured_y_mm"]
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Calibration columns missing: {sorted(missing)}")
        if "role" in frame:
            frame = frame[frame["role"] == "cartesian"]
        frame = frame[columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Calibration coordinates must be finite.")
        # Average repeated commanded locations instead of giving an exact RBF
        # duplicate nodes (the three initial axis references are not fit rows).
        frame = frame.groupby(columns[:2], as_index=False, sort=False).mean()
        if len(frame) < 6:
            raise ValueError("At least six distinct Cartesian calibration points are needed.")
        self.command_xy = frame[columns[:2]].to_numpy()
        self.measured_xy = frame[columns[2:]].to_numpy()
        if np.linalg.matrix_rank(self.command_xy - self.command_xy.mean(axis=0)) < 2:
            raise ValueError("Calibration commands must span two dimensions.")
        if np.linalg.norm(self.command_xy, axis=1).max() > COMMAND_RADIUS_LIMIT_MM + 1e-8:
            raise ValueError("Calibration commands exceed Case 101's 50 mm radius limit.")
        self.error_model = RBFInterpolator(
            self.command_xy, self.measured_xy - self.command_xy,
            kernel="thin_plate_spline", neighbors=min(20, len(frame)), smoothing=0.0,
        )
        self.hull_equations = ConvexHull(self.command_xy).equations
        self.tolerance_mm = float(tolerance_mm)
        self.max_iterations = int(max_iterations)
        self.bounds = list(zip(self.command_xy.min(axis=0), self.command_xy.max(axis=0)))

    def is_inside_calibration_region(self, xy_mm):
        point = np.asarray(xy_mm, dtype=float)
        return bool(point.shape == (2,) and np.isfinite(point).all()
                    and np.all(self.hull_equations[:, :2] @ point
                               + self.hull_equations[:, 2] <= 1e-7)
                    and np.linalg.norm(point) <= COMMAND_RADIUS_LIMIT_MM + 1e-7)

    def forward(self, xy_command_mm):
        points = np.asarray(xy_command_mm, dtype=float)
        if points.shape[-1:] != (2,) or not np.isfinite(points).all():
            raise ValueError("Command coordinates must be finite XY pairs in mm.")
        flat = points.reshape(-1, 2)
        return (flat + self.error_model(flat)).reshape(points.shape)

    def compensate(self, xy_desired_mm):
        """Return a bounded candidate plus explicit convergence diagnostics.

        The hull bounds COMMAND space, not desired/measured space. An
        unreachable desired position is reported, never extrapolated to an
        oversized command. The caller decides whether to use a failed fit.
        """
        desired = np.asarray(xy_desired_mm, dtype=float)
        if desired.shape != (2,) or not np.isfinite(desired).all():
            raise ValueError("Desired position must be a finite XY pair in mm.")
        constraints = {
            "type": "ineq",
            "fun": lambda command: -(
                self.hull_equations[:, :2] @ command + self.hull_equations[:, 2]
            ),
            "jac": lambda command: -self.hull_equations[:, :2],
        }
        closest = np.argsort(np.linalg.norm(self.measured_xy - desired, axis=1))[:3]
        best = None
        # Multiple nearby calibration seeds reduce dependence on a local
        # inverse branch in this nonlinear, locally interpolated error field.
        for index in closest:
            result = minimize(
                lambda command: float(np.sum((self.forward(command) - desired) ** 2)),
                self.command_xy[index].copy(), method="SLSQP", bounds=self.bounds,
                constraints=constraints,
                options={"maxiter": self.max_iterations, "ftol": 1e-10},
            )
            if not self.is_inside_calibration_region(result.x):
                continue
            predicted = self.forward(result.x)
            residual = float(np.linalg.norm(predicted - desired))
            if best is None or residual < best[1]["residual_mm"]:
                best = (result.x.copy(), {
                    "converged": residual <= self.tolerance_mm,
                    "residual_mm": residual,
                    "predicted_xy_mm": predicted.tolist(),
                    "iterations": int(result.nit),
                })
            if residual <= self.tolerance_mm:
                break
        if best is None:
            raise ValueError("No inverse candidate stayed inside the calibrated command region.")
        return best


@lru_cache(maxsize=1)
def get_compensator():
    return FeedForwardCompensator2D()


def apply_case_101_compensation(params, condition_name, *, report_warnings=True):
    """Correct Case 101 commands only; retain unsupported targets with a report."""
    if not condition_name.startswith("case_101.cond_"):
        raise ValueError("This compensator is restricted to Case 101 conditions.")
    params = deepcopy(params)
    desired_schedule = deepcopy(params["TARGET_SCHEDULE"])
    model = get_compensator()
    commands = []
    report = []
    cache = {}
    for index, (time, point, ratio, angle) in enumerate(desired_schedule, start=1):
        desired = np.asarray(point, dtype=float) * 1000
        if np.linalg.norm(desired) > COMMAND_RADIUS_LIMIT_MM + 1e-7:
            raise ValueError("Scale Case 101 targets within 50 mm before compensation.")
        key = tuple(desired)
        if key not in cache:
            cache[key] = model.compensate(desired)
        candidate, info = cache[key]
        applied = info["converged"]
        used = candidate.copy() if applied else desired.copy()
        command_m = used * 1e-3 if applied else np.asarray(point, dtype=float).copy()
        commands.append((time, command_m, ratio, angle))
        predicted_used = model.forward(used) if model.is_inside_calibration_region(used) else [np.nan, np.nan]
        report.append({
            "point_id": index, "time_s": time,
            "desired_x_mm": desired[0], "desired_y_mm": desired[1],
            "command_x_mm": used[0], "command_y_mm": used[1],
            "applied": applied,
            "status": "compensated" if applied else "original_command_retained",
            "candidate_x_mm": candidate[0], "candidate_y_mm": candidate[1],
            "candidate_residual_mm": info["residual_mm"],
            "predicted_used_x_mm": predicted_used[0], "predicted_used_y_mm": predicted_used[1],
            "iterations": info["iterations"],
        })
    uncorrected = sum(not row["applied"] for row in report)
    params["DESIRED_TARGET_SCHEDULE"] = desired_schedule
    params["TARGET_SCHEDULE"] = commands
    params["FEEDFORWARD_COMPENSATION_ENABLED"] = True
    params["FEEDFORWARD_COMPENSATION_REPORT"] = report
    params["FEEDFORWARD_UNCORRECTED_COUNT"] = uncorrected
    params["FEEDFORWARD_CALIBRATION_SOURCE"] = CALIBRATION_SOURCE
    if uncorrected and report_warnings:
        warnings.warn(
            f"{condition_name}: {uncorrected}/{len(report)} targets could not be compensated "
            "within calibration to 0.05 mm tolerance; their original commands are retained. "
            "Inspect the Case 101 compensation report.", RuntimeWarning, stacklevel=2,
        )
    return params


def save_case_reports(output_dir):
    from case_loader import load_case

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, 9):
        name = f"case_101.cond_{number:03d}"
        params = load_case(name)
        report = pd.DataFrame(params["FEEDFORWARD_COMPENSATION_REPORT"])
        filename = output_dir / f"case_101_cond_{number:03d}_compensation.csv"
        report.to_csv(filename, index=False)
        print(f"{name}: {report.applied.sum()}/{len(report)} compensated; report: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "outputs" / "case_101_compensation")
    args = parser.parse_args()
    save_case_reports(args.output_dir)
