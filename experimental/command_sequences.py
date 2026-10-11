"""XY sequence parsing, calibrated projection and cancellable point execution."""

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from axis_calibration import RobotAxes


XY_SEQUENCE_DIR = Path(__file__).resolve().parent / "xy_sequences"


@dataclass(frozen=True)
class XYStep:
    target_xy_m: tuple
    wait_seconds: float
    zero_angle_deg: float


def read_xy_sequence(path):
    """Read the existing export format: [x_m, y_m] | wait_s | zero_deg."""
    steps = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            columns = [column.strip() for column in line.split("|")]
            if len(columns) != 3:
                raise ValueError("Expected [x, y] | wait | zero.")
            target = np.asarray(ast.literal_eval(columns[0]), dtype=float)
            if target.shape != (2,):
                raise ValueError("Commands currently require one XY target per row, not magnet angles or simultaneous multiple targets.")
            wait, zero = float(columns[1]), float(columns[2])
            if not np.isfinite(target).all() or not np.isfinite([wait, zero]).all():
                raise ValueError("Coordinates, wait and zero must be finite.")
            if wait < 0 or not 0 <= zero <= 360:
                raise ValueError("Wait must be nonnegative and zero must be between 0 and 360 degrees.")
            steps.append(XYStep(tuple(target), wait, zero))
        except (SyntaxError, ValueError, TypeError) as error:
            raise ValueError(f"Line {line_number}: {error}") from error
    if not steps:
        raise ValueError("The XY sequence contains no points.")
    return steps


def project_sequence(steps, setup):
    if setup.get("coordinate_system") != "robot_axes":
        raise ValueError("Capture Center, +X and +Y in Calibration, or load a complete saved setup first.")
    axes = RobotAxes(setup["corners_px"], setup["rectangle_width_cm"],
                     setup["rectangle_height_cm"], setup["axis_samples_px"])
    pixels = axes.pixel_positions(np.asarray([step.target_xy_m for step in steps]) * 100)
    polygon = np.asarray(setup["corners_px"], np.float32)
    width, height = setup["video_size_px"]
    for index, point in enumerate(pixels, 1):
        if (np.any(point < 0) or np.any(point >= [width, height])
                or cv2.pointPolygonTest(polygon, tuple(point), True) < -1e-3):
            raise ValueError(f"Point {index} is outside the calibrated four-pin workspace. Check sequence units (metres), axes and rectangle dimensions.")
    return pixels


def execute_xy_sequence(steps, commander, case_name, *, active, report):
    """Fresh synthesis per point; dwell begins after that motor pose is sent."""
    for index, step in enumerate(steps):
        if not active():
            return "stopped"
        report("calculating", index, None)
        result = commander.move(
            case_name, step.target_xy_m, ratio=1.0, angle_deg=None,
            settle_seconds=step.wait_seconds, zero_angle_deg=step.zero_angle_deg,
            active=active, on_sent=lambda result: report("sent", index, result),
        )
        if result is None or not active():
            return "stopped"
        report("completed", index, result)
    return "finished"
