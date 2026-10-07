"""Recover robot-frame XY and radial compensation from GUI004 dwell exports."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from case_loader import load_case, normalize_case_name


PROJECT_DIR = Path(__file__).resolve().parent
COMPENSATION_CONDITIONS = tuple(
    f"case_100.cond_{number:03d}_test_extract_compensator"
    for number in (13, 14, 15)
) + ("case_100.cond_016_calibration_cartesian",)
CONDITION_CHOICES = (*COMPENSATION_CONDITIONS, "all")
CONDITION_LABELS = (
    "Condition 013 - 0.06 m square",
    "Condition 014 - 0.10 m square",
    "Condition 015 - 0.14 m square",
    "Condition 016 - Cartesian calibration, r < 5 cm",
    "All three conditions - combined compensation",
)


def resolve_condition(case_name):
    text = str(case_name).strip()
    if text.lower() == "all":
        return "all"
    if text in ("13", "14", "15", "16"):
        return COMPENSATION_CONDITIONS[int(text) - 13]
    name = normalize_case_name(text)
    if name not in COMPENSATION_CONDITIONS:
        raise ValueError("Usage 5 supports conditions 013-016 or 'all' (the three square conditions).")
    return name


def choose_condition():
    from usage_launcher import select_menu

    selected = select_menu("Choose compensation experiment", CONDITION_LABELS)
    if selected is None:
        raise SystemExit("No compensation experiment selected.")
    return CONDITION_CHOICES[selected]


def choose_dwell_file(condition):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            parent=root, title=f"Select GUI004 dwell CSV for {condition}",
            initialdir=str(PROJECT_DIR / "outputs" / "offline_detection"),
            filetypes=[("Dwell CSV files", "*.csv")],
        )
    finally:
        root.destroy()
    if not selected:
        raise SystemExit("No dwell CSV selected.")
    return Path(selected)


def load_dwell_positions(filename, expected_count=28):
    """Keep visits in acquisition order and prefer perspective-corrected cm."""
    frame = pd.read_csv(filename)
    if len(frame) != expected_count:
        raise ValueError(
            f"{Path(filename).name}: expected exactly {expected_count} dwells (3 calibration + "
            f"{expected_count - 3} path points), found {len(frame)}. Correct missing/extra dwells "
            "in GUI004 first; row-to-target matching would otherwise be ambiguous."
        )
    if "dwell_id" in frame:
        ids = pd.to_numeric(frame["dwell_id"], errors="raise").to_numpy()
        if not np.array_equal(ids, np.arange(1, expected_count + 1)):
            raise ValueError(f"Dwell IDs must be 1-{expected_count} in acquisition order.")
    if "start_s" in frame:
        times = pd.to_numeric(frame["start_s"], errors="raise").to_numpy()
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Dwell start times must be finite and strictly increasing.")
    for columns, scale, space in (
        (("x_cm", "y_cm"), 1.0, "Phase 1 perspective-corrected cm"),
        (("x_mm", "y_mm"), 0.1, "Phase 1 perspective-corrected mm"),
        (("median_x_px", "median_y_px"), 1.0, "pixels (affine calibration only)"),
    ):
        if all(column in frame for column in columns):
            positions = frame[list(columns)].apply(pd.to_numeric, errors="raise").to_numpy() * scale
            if not np.isfinite(positions).all():
                raise ValueError("Dwell coordinates must all be finite.")
            return frame, positions, space
    raise ValueError("Select a GUI004 Phase 5 dwell export, not the per-frame detection CSV.")


def recover_xy_mm(positions):
    """Map measured [origin, +Y, +X] to [(0,0), (0,10), (10,0)] mm."""
    positions = np.asarray(positions, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 3
            or not np.isfinite(positions).all()):
        raise ValueError("At least three finite 2D calibration positions are required.")
    origin = positions[0]
    # Column order is X then Y, although the recorded order is origin, Y, X.
    basis = np.column_stack((positions[2] - origin, positions[1] - origin))
    if np.linalg.cond(basis) > 1000:
        raise ValueError("The first three dwells have coincident or nearly collinear axes.")
    return np.linalg.solve(basis, (positions - origin).T).T * 10.0


def analyze_condition(case_name, filename):
    case_name = resolve_condition(case_name)
    if case_name == "all":
        raise ValueError("Analyze one condition at a time.")
    params = load_case(case_name)
    targets = np.asarray([entry[1] for entry in params["TARGET_SCHEDULE"]]) * 1000.0
    if (targets.ndim != 2 or targets.shape[1] != 2 or len(targets) < 4
            or not np.allclose(targets[:3], [[0, 0], [0, 10], [10, 0]])):
        raise ValueError(f"{case_name} must start with the three calibration targets.")
    point_count = len(targets)
    source, positions, space = load_dwell_positions(filename, point_count)
    roles = ["center", "y_axis", "x_axis"]
    if case_name == COMPENSATION_CONDITIONS[3]:
        roles += ["cartesian"] * (point_count - 3)
    else:
        roles += ["square"] * (point_count - 4) + ["closure"]
    measured = recover_xy_mm(positions)
    errors = measured - targets
    commanded_r = np.linalg.norm(targets, axis=1)
    measured_r = np.linalg.norm(measured, axis=1)
    factors = np.divide(commanded_r, measured_r, out=np.ones(point_count), where=measured_r > 1e-9)
    factors[(commanded_r > 0) & (measured_r <= 1e-9)] = np.nan
    result = pd.DataFrame({
        "condition": case_name, "source_csv": str(Path(filename).resolve()),
        "point_id": np.arange(1, point_count + 1),
        "role": roles,
        "target_x_mm": targets[:, 0], "target_y_mm": targets[:, 1],
        "measured_x_mm": measured[:, 0], "measured_y_mm": measured[:, 1],
        "measured_x_cm": measured[:, 0] / 10, "measured_y_cm": measured[:, 1] / 10,
        "error_x_mm": errors[:, 0], "error_y_mm": errors[:, 1],
        "position_error_mm": np.linalg.norm(errors, axis=1),
        "commanded_radius_mm": commanded_r, "measured_radius_mm": measured_r,
        "radial_error_mm": measured_r - commanded_r,
        "radial_compensation_factor": factors,
        "tangential_error_mm": np.divide(
            targets[:, 0] * measured[:, 1] - targets[:, 1] * measured[:, 0],
            commanded_r, out=np.zeros(point_count), where=commanded_r > 0,
        ),
    })
    for column in ("start_s", "end_s", "duration_s", "std_px", "frames"):
        if column in source:
            result[column] = source[column].to_numpy()
    calibration = {
        "condition": case_name, "source_csv": str(Path(filename).resolve()),
        "input_coordinates": space,
        "measured_origin": positions[0].tolist(),
        "measured_positive_y": positions[1].tolist(),
        "measured_positive_x": positions[2].tolist(),
    }
    return result, calibration


def fit_radial_compensation(points):
    """Fit the inverse response: commanded radius = a*r + b*r**3 (mm)."""
    # The final corner is a repeated visit, retained for error reporting only.
    samples = points[points["role"] != "closure"].copy()
    samples["commanded_radius_mm"] = samples["commanded_radius_mm"].round(6)
    table = samples.groupby("commanded_radius_mm", as_index=False).agg(
        measured_radius_mm=("measured_radius_mm", "mean"),
        radial_spread_mm=("measured_radius_mm", "std"),
        sample_count=("measured_radius_mm", "size"),
    )
    table["radial_spread_mm"] = table["radial_spread_mm"].fillna(0)
    x = table["measured_radius_mm"].to_numpy()
    y = table["commanded_radius_mm"].to_numpy()
    nonzero = x > 1e-9
    if np.count_nonzero(nonzero) < 2:
        raise ValueError("Too few nonzero measured radii to fit compensation.")
    scale = float(x.max())
    normalized = x[nonzero] / scale
    design = np.column_stack((normalized, normalized ** 3))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y[nonzero], rcond=None)
    if rank < 2:
        raise ValueError("Measured radii do not span enough distinct values for a cubic fit.")
    a, b = float(coefficients[0] / scale), float(coefficients[1] / scale ** 3)
    predicted = a * x + b * x ** 3
    monotonic = a > 0 and a + 3 * b * scale ** 2 > 0
    table["fitted_commanded_radius_mm"] = predicted
    table["fit_residual_mm"] = predicted - y
    measured_r = samples["measured_radius_mm"].to_numpy()
    sample_residual = a * measured_r + b * measured_r ** 3 - samples["commanded_radius_mm"].to_numpy()
    model = {
        "formula": "commanded_radius_mm = a * desired_radius_mm + b * desired_radius_mm**3",
        "a": a, "b_per_mm_squared": b,
        "desired_radius_range_mm": [0.0, scale],
        "monotonic_over_calibrated_range": bool(monotonic),
        "sample_fit_rmse_mm": float(np.sqrt(np.mean(sample_residual ** 2))),
        "tangential_error_rmse_mm": float(np.sqrt(np.mean(samples["tangential_error_mm"] ** 2))),
        "calibration_assumption": "First three measured dwells are assigned exactly (0,0), (0,10), (10,0) mm; their error is absorbed into the coordinate frame.",
        "closure_policy": "Square closing corners are excluded from the fit; all Cartesian grid visits are retained.",
    }
    return table, model


def compensate_xy_mm(desired_xy_mm, model):
    """Apply the fitted radial correction, preserving angle and the origin."""
    points = np.asarray(desired_xy_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Desired points must be finite Nx2 coordinates in mm.")
    if not model["monotonic_over_calibrated_range"]:
        raise ValueError("This fitted response is not monotonic; inspect the calibration data.")
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii > model["desired_radius_range_mm"][1] + 1e-6):
        raise ValueError("Desired radius exceeds the measured calibration range.")
    return points * (model["a"] + model["b_per_mm_squared"] * radii ** 2)[:, None]


def save_error_plot(points, filename):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition, section in points.groupby("condition", sort=False):
        path = section[section["role"].isin(("square", "closure", "cartesian"))]
        label = "_".join(condition.split(".")[1].split("_")[:2])
        axes[0].plot(path["target_x_mm"], path["target_y_mm"], "--", alpha=0.5)
        axes[0].scatter(path["measured_x_mm"], path["measured_y_mm"], s=18, label=label)
        axes[0].quiver(path["target_x_mm"], path["target_y_mm"],
                       path["error_x_mm"], path["error_y_mm"],
                       angles="xy", scale_units="xy", scale=1, width=0.003)
        axes[1].scatter(path["commanded_radius_mm"], path["radial_error_mm"], s=18, label=label)
    axes[0].set(xlabel="X (mm)", ylabel="Y (mm)", title="Targets to measured dwell positions", aspect="equal")
    axes[1].set(xlabel="Commanded radius (mm)", ylabel="Measured minus commanded radius (mm)", title="Radial position error")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(filename, dpi=180)
    plt.close(figure)


def main(case_name=None, input_filename=None, output_filename=None):
    selected = resolve_condition(case_name) if case_name else choose_condition()
    conditions = COMPENSATION_CONDITIONS[:3] if selected == "all" else (selected,)
    if input_filename is None:
        files = [choose_dwell_file(condition) for condition in conditions]
    else:
        files = [Path(input_filename)] if isinstance(input_filename, (str, Path)) else list(map(Path, input_filename))
    if len(files) != len(conditions):
        raise ValueError("Provide one dwell CSV per condition, in order 013, 014, 015 for 'all'.")
    if len(set(path.resolve() for path in files)) != len(files):
        raise ValueError("Each condition needs its own distinct dwell CSV.")
    analyzed = [analyze_condition(condition, filename) for condition, filename in zip(conditions, files)]
    points = pd.concat([item[0] for item in analyzed], ignore_index=True)
    table, model = fit_radial_compensation(points)
    output = Path(output_filename) if output_filename else files[0].with_name(
        "all_three_compensation_xy.csv" if selected == "all" else f"{files[0].stem}_compensation_xy.csv"
    )
    if output.suffix.lower() != ".csv":
        raise ValueError("The XY output filename must end in .csv.")
    radial_file = output.with_name(output.stem + "_radial.csv")
    coefficient_file = output.with_name(output.stem + "_coefficients.json")
    plot_file = output.with_name(output.stem + "_errors.png")
    if any(path.resolve() in {source.resolve() for source in files}
           for path in (output, radial_file, coefficient_file, plot_file)):
        raise ValueError("Output paths must not overwrite input dwell CSVs.")
    output.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(output, index=False)
    table.to_csv(radial_file, index=False)
    coefficient_file.write_text(json.dumps({
        "model": model, "calibrations": [item[1] for item in analyzed],
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_error_plot(points, plot_file)
    print(f"Analyzed {len(points)} points across {len(conditions)} condition(s).")
    print(f"Radial correction (mm): command_r = {model['a']:.8g} * r + {model['b_per_mm_squared']:.8g} * r^3")
    print(f"Fit RMSE: {model['sample_fit_rmse_mm']:.3f} mm; tangential RMSE: {model['tangential_error_rmse_mm']:.3f} mm")
    if not model["monotonic_over_calibrated_range"]:
        print("Fit is not monotonic: inspect the samples before applying compensation.")
    for path in (output, radial_file, coefficient_file, plot_file):
        print(f"Saved: {path.resolve()}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", nargs="?", help="13, 14, 15, 16, all (three squares), or a full case name")
    parser.add_argument("dwell_csv", nargs="*", help="GUI004 Phase 5 export(s), in condition order")
    parser.add_argument("--output", type=Path, help="Output XY CSV path")
    args = parser.parse_args()
    try:
        main(args.condition, args.dwell_csv or None, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Usage 5 error: {exc}\n")
