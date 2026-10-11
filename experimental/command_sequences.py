"""XY sequence parsing, calibrated projection and cancellable point execution."""

import ast
import re
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


def resolve_xy_units(content, units="auto"):
    units = units.lower()
    if units not in ("auto", "mm", "cm", "m"):
        raise ValueError("Sequence units must be Auto, mm, cm or m.")
    if units != "auto":
        return units
    headers = set(re.findall(r"^\s*#\s*targets_xy_(mm|cm|m)\b", content, re.MULTILINE))
    if len(headers) > 1:
        raise ValueError("Conflicting XY unit headers. Select the correct sequence units explicitly.")
    return next(iter(headers), "mm")


def read_xy_sequence(path, units="auto"):
    """Convert file XY units to solver metres; unlabelled sequences use mm."""
    content = Path(path).read_text(encoding="utf-8-sig")
    factor = {"mm": .001, "cm": .01, "m": 1.0}[resolve_xy_units(content, units)]
    steps = []
    for line_number, line in enumerate(content.splitlines(), 1):
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
            steps.append(XYStep(tuple(target * factor), wait, zero))
        except (SyntaxError, ValueError, TypeError) as error:
            raise ValueError(f"Line {line_number}: {error}") from error
    if not steps:
        raise ValueError("The XY sequence contains no points.")
    return steps


def project_sequence(steps, setup, *, validate=True):
    if setup.get("coordinate_system") != "robot_axes":
        raise ValueError("Capture Center, +X and +Y in Calibration, or load a complete saved setup first.")
    axes = RobotAxes(setup["corners_px"], setup["rectangle_width_cm"],
                     setup["rectangle_height_cm"], setup["axis_samples_px"])
    pixels = axes.pixel_positions(np.asarray([step.target_xy_m for step in steps]) * 100)
    if validate:
        validate_projected_points(steps, setup, pixels)
    return pixels


def point_limits(pixels, setup):
    polygon = np.asarray(setup["corners_px"], np.float32)
    width, height = setup["video_size_px"]
    outside_pins = np.array([cv2.pointPolygonTest(polygon, tuple(point), True) < -1e-3 for point in pixels])
    outside_frame = np.array([np.any(point < 0) or np.any(point >= [width, height]) for point in pixels])
    return outside_pins, outside_frame


def sequence_diagnostics(steps, setup, pixels):
    axes = RobotAxes(setup["corners_px"], setup["rectangle_width_cm"], setup["rectangle_height_cm"], setup["axis_samples_px"])
    corners_mm = axes.coordinates(setup["corners_px"]) * 10
    xy_mm = np.asarray([step.target_xy_m for step in steps]) * 1000
    return (f"Sequence range: X [{xy_mm[:, 0].min():g}, {xy_mm[:, 0].max():g}], Y [{xy_mm[:, 1].min():g}, {xy_mm[:, 1].max():g}] mm. "
            f"Pin extents in robot XY: X [{corners_mm[:, 0].min():.2f}, {corners_mm[:, 0].max():.2f}], "
            f"Y [{corners_mm[:, 1].min():.2f}, {corners_mm[:, 1].max():.2f}] mm. "
            "XY zero is captured Center, not necessarily the pin rectangle's center.")


def validate_projected_points(steps, setup, pixels, *, indices=None, allow_outside_pins=False):
    outside_pins, outside_frame = point_limits(pixels, setup)
    for index in range(len(steps)) if indices is None else indices:
        target = np.asarray(steps[index].target_xy_m) * 1000
        description = f"Point {index + 1}, XY ({target[0]:g}, {target[1]:g}) mm"
        if outside_frame[index]:
            raise ValueError(f"{description} is outside the camera frame. Check Sequence units, origin and calibration. "
                             + sequence_diagnostics(steps, setup, pixels))
        if outside_pins[index] and not allow_outside_pins:
            raise ValueError(f"{description} is outside the four-pin rectangle. "
                             "Choose an inside point, use a smaller sequence, or explicitly allow extrapolation beyond pins. "
                             + sequence_diagnostics(steps, setup, pixels))


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
