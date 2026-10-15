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
    ratio: float = 1.0
    angle_deg: float | None = None


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
            if len(columns) not in (3, 5):
                raise ValueError("Expected [x, y] | wait | zero [| ratio | angle_deg].")
            target = np.asarray(ast.literal_eval(columns[0]), dtype=float)
            if target.shape != (2,):
                raise ValueError("Commands currently require one XY target per row, not magnet angles or simultaneous multiple targets.")
            wait, zero = float(columns[1]), float(columns[2])
            ratio = float(columns[3]) if len(columns) == 5 else 1.0
            angle = (None if len(columns) == 3 or columns[4].lower() in ('none', 'auto')
                     else float(columns[4]))
            if not np.isfinite(ratio) or ratio <= 0 or (angle is not None and not np.isfinite(angle)):
                raise ValueError("Ratio must be positive and angle finite or Auto.")
            if not np.isfinite(target).all() or not np.isfinite([wait, zero]).all():
                raise ValueError("Coordinates, wait and zero must be finite.")
            if wait < 0 or not 0 <= zero <= 360:
                raise ValueError("Wait must be nonnegative and zero must be between 0 and 360 degrees.")
            steps.append(XYStep(tuple(target * factor), wait, zero, ratio, angle))
        except (SyntaxError, ValueError, TypeError) as error:
            raise ValueError(f"Line {line_number}: {error}") from error
    if not steps:
        raise ValueError("The XY sequence contains no points.")
    return steps


def write_xy_sequence(path, steps):
    """Save explicit SI units and every point's equilibrium settings."""
    with open(path, "w", encoding="utf-8") as output:
        output.write("# targets_xy_m | wait | zero | ratio | angle_deg\n")
        for step in steps:
            xy = ', '.join(f'{value:.12g}' for value in step.target_xy_m)
            angle = 'Auto' if step.angle_deg is None else f'{step.angle_deg:.12g}'
            output.write(f'[{xy}] | {step.wait_seconds:g} | {step.zero_angle_deg:g} | {step.ratio:.12g} | {angle}\n')


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


def execute_xy_sequence(steps, commander, case_name, *, active, report, arrival_gate=None, prepared=False, feedback=None,
                        initial_corrections=None, on_correction=lambda index, correction: None):
    """Fresh synthesis; optionally require visual arrival and manual progression."""
    for index, step in enumerate(steps):
        if not active():
            return "stopped"
        report("calculating", index, None)
        correction = np.zeros(2) if initial_corrections is None else np.asarray(initial_corrections[index], float)
        command = tuple(np.asarray(step.target_xy_m) + correction / 1000)
        if feedback is not None:
            command = feedback.begin_point(step, correction,
                                           lambda value, point=index: on_correction(point, value))
            feedback.publish(desired_xy_mm=(np.asarray(step.target_xy_m) * 1000).tolist(),
                             ratio=step.ratio, angle_deg=step.angle_deg, state='Calculating initial pose')
        def sent(result):
            on_correction(index, correction.tolist())
            if arrival_gate is not None:
                arrival_gate.arm(step.target_xy_m)
            report("sent", index, result)
            if feedback is not None:
                feedback.initial_sent()
                feedback.publish(commanded_xy_mm=(np.asarray(command) * 1000).tolist(), correction_mm=correction.tolist(),
                                 sent_angles=[float(v) for v in result['angles_deg']], sent_time_s=feedback.clock(), state='Sent')

        result = commander.move(
            case_name, command, ratio=step.ratio, angle_deg=step.angle_deg,
            settle_seconds=step.wait_seconds, zero_angle_deg=step.zero_angle_deg,
            active=active, on_sent=sent,
            **({'prepared_point': True} if prepared else {}),
            **({'on_calculated': lambda result: feedback.publish(calculated_angles=[float(v) for v in result['angles_deg']])}
               if feedback is not None else {}),
        )
        if result is None or not active():
            return "stopped"
        if arrival_gate is not None:
            report("waiting_arrival", index, None)
            if feedback is not None:
                if not feedback.wait_at_target(step, commander, case_name, arrival_gate, active,
                                              lambda result: report('corrected', index, result), initial_result=result):
                    return 'stopped'
                result = dict(result, angles_deg=feedback.snapshot()['sent_angles'])
            else:
                if not arrival_gate.wait_for_arrival(active):
                    return "stopped"
                report("arrived", index, None)
                if not arrival_gate.wait_for_next(active):
                    return "stopped"
        report("completed", index, result)
    return "finished"
