"""Reusable point synthesis and serialized motor commands for GUI005 tabs."""

from pathlib import Path
import sys
import threading
import time
import json
import hashlib

import numpy as np

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from case_loader import load_case, normalize_case_name
from control_workflow import calculate_point_control


def hardware_case_name(case_name):
    """Return the case folder, including for older condition-based setups."""
    return normalize_case_name(case_name).split(".")[0]


def load_hardware_case(case_name):
    """Load only the base hardware model; never import a condition/sequence."""
    name = hardware_case_name(case_name)
    if not name:
        raise ValueError("Select the case matching your physical hardware first.")
    return load_case(f"{name}.case")


class PointCommander:
    """Shared across tabs; call move from a worker, not the Tk UI thread."""

    def __init__(self, port_provider, zero_angle_deg=180):
        self.port_provider = port_provider
        self.zero_angle_deg = zero_angle_deg
        self.lock = threading.Lock()
        self.prepared_axes = {}
        self.prepared_points = {}

    @staticmethod
    def point_key(target, ratio, angle):
        return (tuple(target), float(ratio), None if angle is None else float(angle))

    def prepare_points(self, case_name, steps, *, active=lambda: True, report=lambda index, result: None):
        """Prepare a complete sequence atomically; no motor IO, partial results unused."""
        name = hardware_case_name(case_name)
        with self.lock:
            params = load_hardware_case(name)
            signature = self.hardware_signature(params)
            results, rows = {}, []
            for index, step in enumerate(steps):
                if not active():
                    return None
                key = self.point_key(step.target_xy_m, step.ratio, step.angle_deg)
                if key not in results:
                    result = calculate_point_control(params, step.target_xy_m, ratio=step.ratio, angle_deg=step.angle_deg)
                    angles = np.asarray(result['angles_deg'], dtype=float)
                    if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
                        raise ValueError('Precalculation produced invalid motor angles.')
                    results[key] = dict(result, angles_deg=angles.copy())
                rows.append(results[key])
                report(index, results[key])
            if not active():
                return None
            self.prepared_points[name] = (signature, results)
            return rows

    @staticmethod
    def hardware_signature(params):
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return repr(value)
        return hashlib.sha256(json.dumps(params, sort_keys=True, default=encode).encode()).hexdigest()

    def prepare_axes(self, case_name):
        """Compute all three fixed ratio-1 poses without connecting or moving motors."""
        name = hardware_case_name(case_name)
        with self.lock:
            params = load_hardware_case(name)
            signature = self.hardware_signature(params)
            results = {}
            for target in ((0., 0.), (.01, 0.), (0., .01)):
                result = calculate_point_control(params, target, ratio=1., angle_deg=None)
                angles = np.asarray(result['angles_deg'], dtype=float)
                if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
                    raise ValueError('Axis preparation did not produce valid motor angles.')
                results[target] = dict(result, angles_deg=angles.copy())
            self.prepared_axes[name] = (signature, results)
        return results

    def move_prepared_axis(self, case_name, target_xy_m, *, settle_seconds=0, active=lambda: True):
        return self.move(case_name, target_xy_m, settle_seconds=settle_seconds,
                         active=active, prepared_axis=True)

    def calculate(self, case_name, target_xy_m, *, ratio=1.0, angle_deg=None):
        return calculate_point_control(load_hardware_case(case_name), target_xy_m,
                                       ratio=ratio, angle_deg=angle_deg)

    def move(self, case_name, target_xy_m, *, ratio=1.0, angle_deg=None,
             settle_seconds=0, active=lambda: True, zero_angle_deg=None, on_sent=None, prepared_axis=False, prepared_point=False,
             on_calculated=None):
        """Calculate fresh motor angles, send them, then wait interruptibly.

        Cancellation skips unsent commands; it cannot undo an already sent pose.
        Returns optimization information for logging/preview in any caller tab.
        """
        from helpers import moveArray

        settle_seconds = float(settle_seconds)
        if not np.isfinite(settle_seconds) or settle_seconds < 0:
            raise ValueError("Settle time must be finite and nonnegative.")
        zero_angle_deg = self.zero_angle_deg if zero_angle_deg is None else float(zero_angle_deg)
        if not np.isfinite(zero_angle_deg) or not 0 <= zero_angle_deg <= 360:
            raise ValueError("Motor zero angle must be between 0 and 360 degrees.")
        with self.lock:
            if not active():
                return None
            port = self.port_provider()
            if port is None or not getattr(port, "is_open", False):
                raise RuntimeError("Connect the serial controller in Connections first.")
            if prepared_point:
                name = hardware_case_name(case_name)
                cached = self.prepared_points.get(name)
                signature = self.hardware_signature(load_hardware_case(name))
                key = self.point_key(target_xy_m, ratio, angle_deg)
                if cached is None or cached[0] != signature or key not in cached[1]:
                    raise ValueError('Precalculation no longer matches this point/hardware. Precalculate all again.')
                result = cached[1][key]
            elif prepared_axis:
                name = hardware_case_name(case_name)
                signature = self.hardware_signature(load_hardware_case(name))
                cached = self.prepared_axes.get(name)
                if cached is None or cached[0] != signature or tuple(target_xy_m) not in cached[1]:
                    raise ValueError('Axis poses are not prepared for this hardware. Press Prepare axis moves again.')
                result = cached[1][tuple(target_xy_m)]
            else:
                result = self.calculate(case_name, target_xy_m, ratio=ratio, angle_deg=angle_deg)
            if on_calculated is not None:
                on_calculated(result)
            if not active():
                return None
            if self.port_provider() is not port or not getattr(port, "is_open", False):
                raise RuntimeError("The controller disconnected before the move.")
            angles = np.asarray(result["angles_deg"], dtype=float)
            if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
                raise ValueError("Calculation did not produce valid motor angles.")
            moveArray(port, angles, 0, zero_angle_deg)
            if on_sent is not None:
                on_sent(result)
            deadline = time.monotonic() + settle_seconds
            while active() and time.monotonic() < deadline:
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
            return result
