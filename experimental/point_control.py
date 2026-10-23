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
from control_workflow import calculate_point_control, calculate_force_pid_control, apply_integral_to_baseline, calculate_anchored_xy_control


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

    def __init__(self, port_provider, zero_angle_deg=180, *, applied_state=None, radial_model=None):
        self.port_provider = port_provider
        self.zero_angle_deg = zero_angle_deg
        self.lock = threading.Lock()
        self.prepared_axes = {}
        self.prepared_points = {}
        self.motor_repetitions = 3
        self.diagnostic_sink = None
        self.applied_state = applied_state if applied_state is not None else {'lock': threading.Lock(), 'pose': None}
        self.radial_control = None
        if radial_model is not None:
            from gui005_radial_control import RadialControl
            self.radial_control = RadialControl(radial_model)

    def synthesize(self, params, target, **options):
        solve = self.radial_control.synthesize if self.radial_control else calculate_point_control
        return solve(params, target, **options)

    def applied_pose(self, case_name, zero_angle_deg):
        port = self.port_provider()
        signature = self.hardware_signature(load_hardware_case(case_name))
        with self.applied_state['lock']:
            pose = self.applied_state['pose']
            if (pose is None or pose['port'] is not port or not getattr(port, 'is_open', False)
                    or pose['case'] != hardware_case_name(case_name) or pose['signature'] != signature
                    or pose['zero_angle_deg'] != zero_angle_deg):
                raise ValueError('Current magnet pose is unknown for this connection/hardware. Send Center or a calibration move first.')
            angles = pose['angles_deg']
            if (angles.ndim != 1 or not angles.size or not np.isfinite(angles).all()
                    or np.any(angles < 0) or np.any(angles > 90)):
                raise ValueError('The last sent pose is outside the supported model-angle range [0,90].')
            return dict(angles_deg=pose['angles_deg'].copy(), sent_time_s=pose['sent_time_s'], force_compensated=True)

    def diagnostic_event(self, kind, **data):
        sink = self.diagnostic_sink
        if sink is not None:
            try:
                sink(kind, data)
            except Exception:
                # Diagnostic failures must never interrupt a hardware command.
                pass

    def set_motor_repetitions(self, repetitions):
        if repetitions not in (1, 2, 3) or int(repetitions) != repetitions:
            raise ValueError('Motor command repetitions must be 1, 2 or 3.')
        # UI updates never wait for the solver lock. Each transmission takes a
        # single snapshot, so changing this cannot alter a batch halfway through.
        self.motor_repetitions = int(repetitions)

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
                    result = self.synthesize(params, step.target_xy_m, ratio=step.ratio, angle_deg=step.angle_deg)
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

    def hardware_signature(self, params):
        if self.radial_control is not None:
            params = dict(params, gui005_radial_model=self.radial_control.signature())
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return repr(value)
        return hashlib.sha256(json.dumps(params, sort_keys=True, default=encode).encode()).hexdigest()

    def prepare_axes(self, case_name):
        """Compute Center and -10/+10/+20 mm axis poses without moving motors."""
        from axis_calibration import AXIS_MOVE_TARGETS_M
        name = hardware_case_name(case_name)
        with self.lock:
            params = load_hardware_case(name)
            signature = self.hardware_signature(params)
            results = {}
            for target in AXIS_MOVE_TARGETS_M.values():
                result = self.synthesize(params, target, ratio=1., angle_deg=None)
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
        return self.synthesize(load_hardware_case(case_name), target_xy_m,
                                       ratio=ratio, angle_deg=angle_deg)

    def move_force_pid(self, case_name, target_xy_m, measured_xy_m, applied_angles_deg,
                       reference_inputs, pid_force_n, *,
                       max_angle_step_deg=10., ratio=1., angle_deg=None,
                       zero_angle_deg=None, active=lambda: True, on_sent=None, on_calculated=None,
                       on_held=None, preview_only=False, motor_deadband_deg=.1, incremental=False):
        return self.move(case_name, target_xy_m, zero_angle_deg=zero_angle_deg, active=active,
                         on_sent=on_sent, on_calculated=on_calculated, preview_only=preview_only,
                         on_held=on_held, unchanged_angles_deg=applied_angles_deg,
                         motor_deadband_deg=motor_deadband_deg,
                         force_request=dict(measured_xy_m=measured_xy_m, applied_angles_deg=applied_angles_deg,
                            reference_inputs=reference_inputs, pid_force_n=pid_force_n,
                            max_angle_step_deg=max_angle_step_deg, ratio=ratio, angle_deg=angle_deg,
                            **({'incremental': True} if incremental else {})))

    def move(self, case_name, target_xy_m, **options):
        self.diagnostic_event('move_requested', case=case_name, target_xy_m=target_xy_m,
                              options={key: value for key, value in options.items() if not callable(value)})
        try:
            return self._move(case_name, target_xy_m, **options)
        except Exception as error:
            self.diagnostic_event('move_failed', case=case_name, target_xy_m=target_xy_m,
                                  error=repr(error))
            raise

    def _move(self, case_name, target_xy_m, *, ratio=1.0, angle_deg=None,
             settle_seconds=0, active=lambda: True, zero_angle_deg=None, on_sent=None, prepared_axis=False, prepared_point=False,
             on_calculated=None, force_request=None, preview_only=False, on_held=None,
             unchanged_angles_deg=None, motor_deadband_deg=0., initial_force_bias_n=None, anchored_xy_request=None):
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
                self.diagnostic_event('move_cancelled', case=case_name, target_xy_m=target_xy_m)
                return None
            port = self.port_provider()
            if port is None or not getattr(port, "is_open", False):
                raise RuntimeError("Connect the serial controller in Connections first.")
            self.diagnostic_event('calculation_started', case=case_name, target_xy_m=target_xy_m)
            if anchored_xy_request is not None:
                solve = self.radial_control.anchored if self.radial_control else calculate_anchored_xy_control
                result = solve(load_hardware_case(case_name), target_xy_m,
                                                       ratio=ratio, angle_deg=angle_deg, **anchored_xy_request)
            elif force_request is not None:
                solve = self.radial_control.allocate if self.radial_control else calculate_force_pid_control
                result = solve(load_hardware_case(case_name), target_xy_m, **force_request)
            elif prepared_point:
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
            if initial_force_bias_n is not None:
                solve = self.radial_control.apply_integral if self.radial_control else apply_integral_to_baseline
                result = solve(
                    load_hardware_case(case_name), target_xy_m, result, initial_force_bias_n)
            self.diagnostic_event('calculation_finished', case=case_name, target_xy_m=target_xy_m,
                                  result=result, preview_only=preview_only)
            if on_calculated is not None:
                on_calculated(result)
            if preview_only:
                return result
            if not active():
                self.diagnostic_event('move_cancelled', case=case_name, target_xy_m=target_xy_m)
                return None
            if self.port_provider() is not port or not getattr(port, "is_open", False):
                raise RuntimeError("The controller disconnected before the move.")
            angles = np.asarray(result["angles_deg"], dtype=float)
            if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
                raise ValueError("Calculation did not produce valid motor angles.")
            if (unchanged_angles_deg is not None
                    and np.max(np.abs(angles-np.asarray(unchanged_angles_deg))) <= motor_deadband_deg):
                self.diagnostic_event('motor_pose_held', case=case_name, target_xy_m=target_xy_m,
                                      angles_deg=unchanged_angles_deg,
                                      proposed_angles_deg=angles, deadband_deg=motor_deadband_deg)
                if on_held is not None:
                    # Use the ACTUAL held pose for anti-windup, not the proposed angles.
                    held = (dict(result) if anchored_xy_request is not None else
                            dict(result, predicted_force_n=result['applied_force_n'],
                                 force_residual_n=result['applied_force_residual_n']))
                    on_held(held)
                return result
            repetitions = self.motor_repetitions
            self.diagnostic_event('motor_send_started', case=case_name, target_xy_m=target_xy_m,
                                  angles_deg=angles, zero_angle_deg=zero_angle_deg,
                                  serial_target_angles_deg=angles+zero_angle_deg,
                                  repetitions=repetitions)
            # Invalidate before transmission: a partial/failed write cannot seed feedback.
            with self.applied_state['lock']:
                self.applied_state['pose'] = None
            moveArray(port, angles, 0, zero_angle_deg, repetitions=repetitions)
            with self.applied_state['lock']:
                self.applied_state['pose'] = dict(port=port, case=hardware_case_name(case_name),
                    signature=self.hardware_signature(load_hardware_case(case_name)),
                    zero_angle_deg=zero_angle_deg, angles_deg=angles.copy(), sent_time_s=time.monotonic())
            self.diagnostic_event('motor_send_completed', case=case_name, target_xy_m=target_xy_m,
                                  angles_deg=angles, zero_angle_deg=zero_angle_deg,
                                  serial_target_angles_deg=angles+zero_angle_deg,
                                  repetitions=repetitions)
            if on_sent is not None:
                on_sent(result)
            deadline = time.monotonic() + settle_seconds
            while active() and time.monotonic() < deadline:
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
            return result
