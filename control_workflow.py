"""Shared control synthesis and field-preparation workflow."""

from dataclasses import dataclass
import threading
import time

import numpy as np
from scipy.optimize import lsq_linear

from case_loader import build_common_config, unpack_target_schedule_entry, validate_target_schedule
from functions_main import (
    calculate_potential_hessian,
    calculate_total_force_from_sources,
    find_four_stable_equilibrium_inputs,
    find_stable_equilibrium_inputs,
    find_two_equilibrium_inputs,
    find_two_equilibrium_with_center_repulsion_inputs,
    find_two_stable_equilibrium_inputs,
    generate_circular_source_positions,
    allocate_force_inputs,
    MagneticForcePID,
    build_actuation_matrix,
)
from functions_utility import compute_grid_fields, extract_optimization_info


@dataclass(frozen=True)
class ScheduledControl:
    start_time: float
    target_positions: tuple
    control_inputs: np.ndarray
    eig_ratio: float | None = None
    eig_angle: float | None = None


@dataclass
class ControlWorkflowResult:
    source_positions: np.ndarray
    scheduled_controls: list
    opt_infos: list
    field_data: list

    @property
    def target_controls(self):
        """Legacy tuples consumed by the dynamics function."""
        return [
            (
                item.start_time,
                item.target_positions[0],
                item.eig_ratio,
                item.eig_angle,
                item.control_inputs,
            )
            for item in self.scheduled_controls
        ]


def calculate_point_control(params, target_xy_m, *, ratio=1.0, angle_deg=None):
    """Synthesize one target using the same workflow as Usage 2/3, without IO.

    XY is in metres; angle is the equilibrium orientation, not motor angles.
    Isotropic ratio=1 requires no orientation. The case's trajectory is replaced
    in a copy, so repeated interactive commands never modify the source model.
    """
    target = np.asarray(target_xy_m, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("Target must contain finite X and Y coordinates in metres.")
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Equilibrium ratio must be positive.")
    if angle_deg is not None and not np.isfinite(float(angle_deg)):
        raise ValueError("Equilibrium angle must be finite.")
    angle_rad = None if np.isclose(ratio, 1) or angle_deg is None else np.deg2rad(float(angle_deg))
    point_params = dict(params)
    point_params["TARGET_SCHEDULE"] = [(0.0, target.copy(), ratio, angle_rad)]
    point_params["INITIAL_ROBOT_POSITIONS"] = np.zeros((1, 2))
    point_params["OPTIMIZATION_FAILURE_MODE"] = "raise"
    result = run_control_workflow(build_common_config(point_params), point_params, compute_fields=False)
    return result.opt_infos[0]


def calculate_anchored_xy_control(params, target_xy_m, *, anchor_xy_m, anchor_angles_deg,
                                  baseline_angles_deg, ratio=1., angle_deg=None):
    """Apply the original optimizer's input difference to the real starting pose."""
    target, anchor = np.asarray(target_xy_m, float), np.asarray(anchor_xy_m, float)
    angles, baseline = np.asarray(anchor_angles_deg, float), np.asarray(baseline_angles_deg, float)
    if (target.shape != (2,) or anchor.shape != (2,) or angles.ndim != 1 or not angles.size
            or angles.shape != baseline.shape or not np.isfinite([target, anchor]).all()
            or not np.isfinite([angles, baseline]).all()
            or np.any(angles < 0) or np.any(angles > 90)
            or np.any(baseline < 0) or np.any(baseline > 90)):
        raise ValueError('Anchored XY control requires finite XY and matching model-angle vectors.')
    result = (dict(angles_deg=baseline) if np.array_equal(target, anchor) else
              calculate_point_control(params, target, ratio=ratio, angle_deg=angle_deg))
    nominal = np.asarray(result['angles_deg'], float)
    if (nominal.shape != angles.shape or not np.isfinite(nominal).all()
            or np.any(nominal < 0) or np.any(nominal > 90)):
        raise ValueError('Optimizer returned invalid angles for anchored XY control.')
    actual_inputs = np.cos(np.deg2rad(angles))
    delta = np.cos(np.deg2rad(nominal))-np.cos(np.deg2rad(baseline))
    fraction = 1.
    for value, change in zip(actual_inputs, delta):
        if change > 0:
            fraction = min(fraction, (1-value)/change)
        elif change < 0:
            fraction = min(fraction, -value/change)
    fraction = float(np.clip(fraction, 0., 1.))
    inputs = np.clip(actual_inputs+fraction*delta, 0., 1.)
    return dict(angles_deg=np.rad2deg(np.arccos(inputs)), u_values=inputs,
                nominal_angles_deg=nominal, anchor_xy_m=anchor, anchor_angles_deg=angles,
                optimizer_xy_m=target, allocation_fraction=fraction,
                allocation_mode='anchored_xy_optimizer', force_compensated=False)


def calculate_force_pid_control(params, target_xy_m, measured_xy_m, applied_angles_deg,
                                reference_inputs, pid_force_n, *, max_angle_step_deg=10.,
                                ratio=1., angle_deg=None, incremental=False):
    """Allocate only the PID force at measured XY; no disturbance/feedforward."""
    started = time.monotonic()
    point_params = dict(params, TARGET_SCHEDULE=[(0., np.asarray(target_xy_m), 1., None)],
                        INITIAL_ROBOT_POSITIONS=np.zeros((1, 2)))
    cfg = build_common_config(point_params)
    sources = generate_circular_source_positions(cfg.NUM_SOURCES, cfg.RADIUS)
    angles = np.asarray(applied_angles_deg, float)
    if angles.shape != (cfg.NUM_SOURCES,) or not np.isfinite(angles).all() or np.any(angles < 0) or np.any(angles > 90):
        raise ValueError('Force PID needs last sent model angles in [0,90] degrees.')
    applied = np.cos(np.deg2rad(angles))
    if incremental:
        requested = np.asarray(pid_force_n, float)
        measured = np.asarray(measured_xy_m, float)
        if (requested.shape != (2,) or measured.shape != (2,)
                or not np.isfinite([requested, measured]).all()
                or not np.isfinite(max_angle_step_deg) or not 0 < max_angle_step_deg <= 90):
            raise ValueError('Incremental force PID requires finite XY/force and a positive angle limit.')
        A = build_actuation_matrix(measured, sources, cfg.C_F)
        if not np.isfinite(A).all():
            raise ValueError('Force coefficients are invalid at the measured position.')
        current_force = A @ applied
        change = requested-current_force
        lower = np.cos(np.deg2rad(np.minimum(90., angles+max_angle_step_deg)))-applied
        upper = np.cos(np.deg2rad(np.maximum(0., angles-max_angle_step_deg)))-applied
        scale = max(float(np.linalg.norm(A)), float(np.linalg.norm(change)), 1e-30)
        # Solve for input CHANGES around the applied pose, using A at measured XY.
        # The small regularizer selects a nearby solution in the underdetermined system.
        solve = lsq_linear(np.vstack((A/scale, 1e-4*np.eye(len(applied)))),
                           np.concatenate((change/scale, np.zeros(len(applied)))),
                           bounds=(lower, upper), tol=1e-12, max_iter=100)
        controls = applied+solve.x
        if (not solve.success or not np.isfinite(controls).all()
                or np.any(solve.x < lower-1e-9) or np.any(solve.x > upper+1e-9)):
            raise ValueError('Incremental force allocation failed; last pose is held.')
        predicted = A @ controls
        H = calculate_potential_hessian(measured, sources, cfg.C_F, controls)
        return dict(angles_deg=np.rad2deg(np.arccos(np.clip(controls, 0, 1))),
                    u_values=controls, reference_inputs=applied, current_force_n=current_force,
                    force_adjustment_requested_n=change, delta_inputs=solve.x,
                    requested_force_n=requested, predicted_force_n=predicted,
                    applied_force_n=current_force, applied_force_residual_n=float(np.linalg.norm(change)),
                    force_residual_n=float(np.linalg.norm(predicted-requested)),
                    stiffness_eigenvalues=np.linalg.eigvalsh(H), allocation_position_m=measured,
                    allocation_mode='incremental_force', allocation_iterations=solve.nit,
                    force_compensated=True, calculation_time_s=time.monotonic()-started)
    if reference_inputs is None:
        reference_inputs = calculate_point_control(params, target_xy_m, ratio=ratio, angle_deg=angle_deg)['u_values']
    requested = np.asarray(pid_force_n, float)
    controls, diagnostics = allocate_force_inputs(
        target_xy_m, sources, cfg.C_F, requested, applied, reference_inputs,
        max_angle_step_deg=max_angle_step_deg, allocation_position_m=measured_xy_m,
        allow_no_improvement=True)
    diagnostics.update(applied_force_residual_n=float(np.linalg.norm(
                           build_actuation_matrix(measured_xy_m, sources, cfg.C_F) @ applied-requested)))
    return dict(diagnostics, angles_deg=np.rad2deg(np.arccos(np.clip(controls, 0, 1))),
                u_values=controls, reference_inputs=np.asarray(reference_inputs, float),
                applied_force_n=build_actuation_matrix(measured_xy_m, sources, cfg.C_F) @ applied,
                force_compensated=True, calculation_time_s=time.monotonic()-started)


def apply_integral_to_baseline(params, target_xy_m, baseline, integral_force_n):
    """Include carried bias in the first pose, without sending an uncorrected pose."""
    bias = np.asarray(integral_force_n, float)
    if bias.shape != (2,) or not np.isfinite(bias).all():
        raise ValueError('Carried integral must be a finite X,Y force.')
    reference = np.cos(np.deg2rad(np.asarray(baseline['angles_deg'], float)))
    allocated = calculate_force_pid_control(
        params, target_xy_m, target_xy_m, baseline['angles_deg'], reference, bias,
        max_angle_step_deg=90.)
    if allocated['force_residual_n'] > max(1e-12, np.linalg.norm(bias)*.01):
        raise ValueError('Carried PID bias is infeasible at the next target. Stop and reset PID memory.')
    return dict(allocated, desired_pos=np.asarray(target_xy_m, float),
                baseline=baseline, inherited_integral_force_n=bias.copy())


class ForcePIDFeedback:
    """Camera-independent continuous force PID; manual XY remains separate."""
    realtime = True

    def __init__(self, *, validate=lambda target: None, manual=False,
                 max_angle_step_deg=10., preview_only=False, clock=time.monotonic,
                 pid=None, angle_rate_deg_s=20., motor_deadband_deg=.02, hold_speed_mm_s=.5,
                 slow_motion_assist=False, incremental_angles=False, allocation_position=None):
        if not np.isfinite(max_angle_step_deg) or not 0 < max_angle_step_deg <= 90:
            raise ValueError('Maximum angle step must be in (0, 90] degrees.')
        self.validate, self.manual = validate, bool(manual)
        self.max_angle_step_deg, self.preview_only, self.clock = float(max_angle_step_deg), bool(preview_only), clock
        self.enabled, self.lock = threading.Event(), threading.RLock()
        self.telemetry, self.current_step = {}, None
        self.command_revision = 0
        self.pending_manual = None
        self.last_sent_u = np.zeros(2)
        self.last_sent_at, self.last_result, self.reference_inputs = None, None, None
        self.on_applied = lambda correction: None
        if (not np.isfinite(angle_rate_deg_s) or angle_rate_deg_s < 0
                or not np.isfinite(motor_deadband_deg) or motor_deadband_deg < 0
                or not np.isfinite(hold_speed_mm_s) or hold_speed_mm_s <= 0):
            raise ValueError('Angle rate/deadband must be nonnegative; hold speed must be positive. Rate 0 disables limiting.')
        self.pid = pid or MagneticForcePID()
        self.hold_speed_mm_s = float(hold_speed_mm_s)
        self.angle_rate, self.motor_deadband = float(angle_rate_deg_s), float(motor_deadband_deg)
        self.last_control_at = None
        self.last_control_sample = None
        self.slow_motion_assist = bool(slow_motion_assist)
        self.incremental_angles = bool(incremental_angles)
        self.allocation_position = allocation_position
        self._reset_slow_assist()

    def _reset_slow_assist(self):
        self.slow_since = self.slow_sample = None
        self.publish(slow_assist_active=False, slow_assist_scale=1.)

    def _slow_motion_scale(self, error_mm, speed_mm_s, sample, tolerance_mm):
        # Use successive camera observations, never repeated idle-loop polls.
        # No integral boost is retained after motion resumes or at the next point.
        if (not self.slow_motion_assist or speed_mm_s >= .3
                or error_mm <= max(3., 2*tolerance_mm)):
            self._reset_slow_assist()
            return 1.
        if self.slow_sample is None or not 0 <= sample-self.slow_sample <= 1.:
            self.slow_since = sample
        self.slow_sample = sample
        scale = 1. + min(.5, max(0., (sample-self.slow_since-3.)/6.))
        self.publish(slow_assist_active=scale > 1., slow_assist_scale=scale)
        return scale

    def publish(self, **values):
        with self.lock:
            self.telemetry.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.telemetry)

    def set_enabled(self, enabled):
        with self.lock:
            self.command_revision += 1
            (self.enabled.set if enabled else self.enabled.clear)()
        self.pid.reset(keep_integral=True)
        self._reset_slow_assist()

    def set_manual_mode(self, manual):
        with self.lock:
            self.manual = bool(manual)
            self.command_revision += 1
            self.pending_manual = None
            if manual:
                self.enabled.clear()
        self.pid.reset(keep_integral=True)
        self._reset_slow_assist()

    def request_manual(self, step, correction):
        vector = np.asarray(correction, float)
        if vector.shape != (2,) or not np.isfinite(vector).all():
            raise ValueError('Manual u must be a finite X,Y pair.')
        with self.lock:
            if not self.manual or self.current_step is not step:
                raise ValueError('Wait for the active point to start before changing its command.')
            self.command_revision += 1
            self.pending_manual = (self.command_revision, vector.copy())

    def begin_point(self, step, correction, on_applied=lambda correction: None):
        command = np.asarray(step.target_xy_m) + np.asarray(correction)/1000
        self.validate(command)
        if self.incremental_angles:
            if self.last_result is None:
                raise ValueError('Current magnet pose is unknown. Send Center or a calibration move once first.')
            with self.lock:
                self.current_step, self.pending_manual = step, None
                self.command_revision += 1
                self.on_applied = on_applied
                self.last_sent_u = np.zeros(2)
                self.publish(desired_xy_mm=(np.asarray(step.target_xy_m)*1000).tolist(),
                             commanded_xy_mm=None, correction_mm=[0., 0.],
                             integral_force_n=self.pid.integral_force.tolist(),
                             inherited_integral_force_n=self.pid.integral_force.tolist())
            return np.asarray(step.target_xy_m)
        with self.lock:
            self.current_step, self.pending_manual = step, None
            self.command_revision += 1
            self.reference_inputs, self.last_result = None, None
            self.last_sent_u, self.last_sent_at = np.asarray(correction, float).copy(), None
            self.on_applied = on_applied
            self.telemetry = {}
            self._reset_slow_assist()
            self.last_control_at = None
            self.last_control_sample = None
            self.pid.reset(keep_integral=self.enabled.is_set() and not self.manual)
            self.publish(integral_force_n=self.pid.integral_force.tolist(),
                         inherited_integral_force_n=self.pid.integral_force.tolist())
        return command

    def initial_move_options(self):
        with self.lock:
            if (not self.enabled.is_set() or self.manual or self.preview_only
                    or not np.any(self.pid.integral_force)):
                return {}
            return dict(initial_force_bias_n=self.pid.integral_force.copy())

    def initial_sent(self, result=None):
        self.last_sent_at = self.clock()
        if result is not None:
            self.last_result = dict(result)
            angles = np.asarray(result['angles_deg'], float)
            if self.reference_inputs is None:
                if result.get('reference_inputs') is not None:
                    self.reference_inputs = np.asarray(result['reference_inputs'], float)
                elif (result.get('desired_pos') is None or
                      np.allclose(result['desired_pos'], self.current_step.target_xy_m, atol=1e-10, rtol=0)):
                    self.reference_inputs = np.cos(np.deg2rad(angles))
            self.publish(sent_angles=angles.tolist(), magnet_inputs=np.cos(np.deg2rad(angles)).tolist(),
                         sent_time_s=self.last_sent_at, force_compensated=bool(result.get('force_compensated', False)))

    def wait_at_target(self, step, commander, case_name, gate, active, report, initial_result=None):
        if self.last_result is None and initial_result is not None:
            self.initial_sent(initial_result)
        target_mm = np.asarray(step.target_xy_m)*1000
        self.publish(desired_xy_mm=target_mm.tolist(), commanded_xy_mm=None if self.incremental_angles else (target_mm+self.last_sent_u).tolist(),
                     correction_mm=self.last_sent_u.tolist(), ratio=step.ratio, angle_deg=step.angle_deg)
        while active():
            with gate.condition:
                if gate.advance_due and (gate.skip_requested or not gate.require_arrival or gate._ready()):
                    gate.consume_advance()
                    return True
                positions = None if gate.positions is None else gate.positions.copy()
                sample, loss_generation = gate.observed_at, gate.loss_generation
            with self.lock:
                pending, revision = self.pending_manual, self.command_revision
            enabled = self.enabled.is_set() and not self.manual
            self.publish(mode='manual' if self.manual else getattr(self, 'control_mode',
                         'incremental_force_pid' if self.incremental_angles else 'force_pid'))
            def request_active():
                with gate.condition:
                    return active() and not gate.advance_due and self.command_revision == revision
            if pending is not None and self.manual:
                request_revision, correction = pending
                command = (target_mm+correction)/1000
                def manual_active():
                    return request_active() and self.manual and self.command_revision == request_revision
                try:
                    self.validate(command)
                    self.publish(state='Calculating manual command', proposed_xy_mm=(command*1000).tolist())
                    def manual_sent(result):
                        self.last_sent_u = correction.copy()
                        self.initial_sent(result)
                        self.pid.reset()
                        self.on_applied(correction.tolist())
                        self.publish(commanded_xy_mm=(command*1000).tolist(), correction_mm=correction.tolist(),
                                     state='Manual command sent', requested_force_n=None, force_residual_n=None)
                        report(result)
                    commander.move(case_name, command, ratio=step.ratio, angle_deg=step.angle_deg,
                                   settle_seconds=0, zero_angle_deg=step.zero_angle_deg,
                                   active=manual_active, on_sent=manual_sent,
                                   on_calculated=lambda result: self.publish(calculated_angles=list(result['angles_deg'])))
                except ValueError as error:
                    self.publish(state=f'Manual command blocked: {error}')
                finally:
                    with self.lock:
                        if self.command_revision == request_revision:
                            self.pending_manual = None
                continue
            fresh = positions is not None and sample is not None and self.clock()-sample <= 1.
            if fresh and len(positions) == 1:
                self.pid.observe(positions[0], sample)
                self.publish(velocity_xy_mm_s=(self.pid.velocity*1000).tolist())
            else:
                self.pid.reset(keep_integral=True)
            self.publish(measurement_monotonic_s=sample, measurement_fresh=bool(fresh),
                         measured_xy_mm=None if not fresh else (positions.mean(axis=0)*1000).tolist(),
                         robot_speed_mm_s=(float(np.linalg.norm(self.pid.velocity)*1000)
                                          if self.pid.velocity_ready else None),
                         hold_speed_mm_s=self.hold_speed_mm_s)
            if not fresh:
                self._reset_slow_assist()
                self.publish(state='Tracking lost — holding last motor pose')
            elif not enabled:
                self._reset_slow_assist()
                self.publish(state='Manual — holding' if self.manual else 'Control off — holding')
            elif len(positions) != 1:
                self._reset_slow_assist()
                self.publish(state='Force PID requires one tracked robot')
            else:
                self._continuous_update(step, commander, case_name, gate, active, report,
                                        positions[0], sample, loss_generation, revision)
            with gate.condition:
                # Fresh camera samples notify the condition immediately; bound
                # idle polling even if an unrealistically short period is set.
                gate.condition.wait(max(.005, min(.05, self.pid.period)))
        return False

    def _continuous_update(self, step, commander, case_name, gate, active, report,
                           measured, sample, loss_generation, revision):
        if not self.pid.velocity_ready:
            self.publish(state='Measuring velocity — holding')
            return
        if sample == self.last_control_sample:
            return
        if self.last_control_at is not None and self.clock()-self.last_control_at < self.pid.period:
            return
        if self.last_result is None:
            self.publish(state='Waiting for applied magnet angles')
            return
        error = np.asarray(step.target_xy_m)-measured
        speed = float(np.linalg.norm(self.pid.velocity)*1000)
        self.publish(position_error_mm=(error*1000).tolist(), velocity_xy_mm_s=(self.pid.velocity*1000).tolist())
        if np.linalg.norm(error)*1000 <= gate.tolerance_mm and speed <= self.hold_speed_mm_s:
            self._reset_slow_assist()
            self.publish(state='Reached — holding; damping remains active if moving')
            return
        self.last_control_at = self.clock()
        self.last_control_sample = sample
        control_tolerance = 0. if self.incremental_angles else gate.tolerance_mm/1000
        proposal = self.pid.propose(step.target_xy_m, tolerance_m=control_tolerance)
        scale = self._slow_motion_scale(float(np.linalg.norm(error)*1000), speed, sample, gate.tolerance_mm)
        control_error = error*max(0., 1-control_tolerance/max(float(np.linalg.norm(error)), 1e-30))
        proposal['pid_force_n'] = proposal['pid_force_n'] + (scale-1.)*self.pid.kp*control_error
        elapsed = self.pid.period if self.last_sent_at is None else max(.001, self.clock()-self.last_sent_at)
        max_step = (self.max_angle_step_deg if self.angle_rate == 0
                    else min(self.max_angle_step_deg, self.angle_rate*elapsed))
        def valid():
            with gate.condition:
                return (active() and self.enabled.is_set() and not self.manual
                        and self.command_revision == revision and not gate.advance_due
                        and gate.loss_generation == loss_generation and gate.positions is not None
                        and len(gate.positions) == 1 and self.clock()-sample <= min(1., 2*self.pid.period))
        accepted = [False]
        def commit(result, *, sent):
            accepted[0] = True
            if self.preview_only:
                return
            self.pid.commit(proposal, result['requested_force_n'], result['predicted_force_n'])
            if sent:
                self.initial_sent(result)
                state = 'Force PID sent' if scale == 1. else f'Force PID sent; slow-motion assist ×{scale:.2f}'
                self.publish(commanded_xy_mm=None, state=state, observation_age_s=self.clock()-sample)
                report(result)
            else:
                self.publish(state='Force PID holding — below motor deadband')
            self.publish(integral_force_n=self.pid.integral_force.tolist(),
                         predicted_force_n=np.asarray(result['predicted_force_n']).tolist(),
                         force_residual_n=float(result['force_residual_n']))
        def calculated(result):
            if result.get('reference_inputs') is not None:
                self.reference_inputs = np.asarray(result['reference_inputs'], float)
            self.publish(calculated_angles=list(result['angles_deg']), pid_force_n=proposal['pid_force_n'].tolist(),
                         **{key: np.asarray(result[key]).tolist() for key in (
                             'requested_force_n', 'predicted_force_n', 'force_residual_n',
                             'stiffness_eigenvalues', 'calculation_time_s')})
            self.publish(**{key: np.asarray(result[key]).tolist() for key in (
                'current_force_n', 'force_adjustment_requested_n', 'delta_inputs', 'allocation_position_m') if key in result})
        self.publish(state='Calculating force PID', force_observation_monotonic_s=sample,
                     observation_age_s=self.clock()-sample,
                     force_kp_n_m=float(self.pid.kp*scale), force_ki_n_m_s=float(self.pid.ki), force_kv_n_s_m=float(self.pid.kv))
        try:
            allocation_position = measured if self.allocation_position is None else self.allocation_position()
            if allocation_position is None:
                raise ValueError('Robot position is unavailable for force allocation.')
            commander.move_force_pid(
                case_name, step.target_xy_m, allocation_position, self.last_result['angles_deg'], self.reference_inputs,
                proposal['pid_force_n'],
                max_angle_step_deg=max_step, ratio=step.ratio, angle_deg=step.angle_deg,
                zero_angle_deg=step.zero_angle_deg, active=valid, on_calculated=calculated,
                on_sent=lambda result: commit(result, sent=True),
                on_held=lambda result: commit(result, sent=False), motor_deadband_deg=self.motor_deadband,
                preview_only=self.preview_only,
                **({'incremental': True} if self.incremental_angles else {}))
            if self.preview_only:
                self.publish(state='Preview only — force PID not sent')
            elif not accepted[0]:
                self.publish(state='Force PID discarded — stale frame, tracking or mode changed')
        except ValueError as error:
            self.publish(state=f'Force PID held: {error}')
        finally:
            self.publish(control_cycle_time_s=self.clock()-self.last_control_at)


class OptimizerXYFeedback(ForcePIDFeedback):
    """Position PID output in metres, followed by anchored equilibrium optimization."""
    control_mode = 'anchored_xy_optimizer_pid'

    def __init__(self, *, speed_limit_mm_s=None, brake_gain_s=1.5, max_brake_mm=2., **kwargs):
        if (speed_limit_mm_s is not None and (not np.isfinite(speed_limit_mm_s) or speed_limit_mm_s <= 0)
                or not np.isfinite(brake_gain_s) or brake_gain_s <= 0
                or not np.isfinite(max_brake_mm) or max_brake_mm <= 0):
            raise ValueError('Speed limit, brake gain and maximum brake offset must be finite and positive.')
        self.speed_limit_mm_s = speed_limit_mm_s
        self.brake_gain_s, self.max_brake_m = float(brake_gain_s), float(max_brake_mm)*.001
        self.speed_braking = False
        super().__init__(incremental_angles=True, **kwargs)
        self.anchor_xy = self.anchor_angles = None
        self.baselines = {}

    def _reset_slow_assist(self):
        super()._reset_slow_assist()
        self.speed_braking = False
        self.publish(speed_braking=False, speed_limit_mm_s=self.speed_limit_mm_s)
        self.stall_since = self.stall_sample = None
        self.publish(stalled=False, stall_duration_s=0., stall_reason=None)

    def _observe_stall(self, error_mm, speed_mm_s, sample, tolerance_mm):
        # Detect lack of measured motion, not lack of changes in the command.
        # Only consecutive fresh observations may extend this timer.
        if error_mm <= tolerance_mm or speed_mm_s >= .1:
            self._reset_slow_assist()
            return
        if self.stall_sample is None or not 0 <= sample-self.stall_sample <= 1.:
            self.stall_since = sample
        self.stall_sample = sample
        duration = sample-self.stall_since
        self.publish(stalled=duration >= 3., stall_duration_s=duration, stall_reason=None)

    def _stall_status(self, fraction, sent, error):
        if not self.snapshot().get('stalled'):
            return ''
        if fraction < 1.-1e-9:
            reason = 'optimizer input limit; integral constrained'
        elif self.pid.ki == 0:
            reason = 'Ki is zero; integral disabled'
        elif np.any((np.abs(self.pid.integral_force) >= self.pid.integral_limit-1e-12)
                    & (self.pid.integral_force*error > 0)):
            reason = 'integral limit reached'
        else:
            reason = 'building integral' if sent else 'motor deadband; building integral'
        self.publish(stall_reason=reason)
        return '; stalled: '+reason

    def seed_anchor(self, xy_m, pose):
        self._reset_slow_assist()
        self.anchor_xy = np.asarray(xy_m, float).copy()
        self.anchor_angles = np.asarray(pose['angles_deg'], float).copy()
        self.baselines.clear()
        self.pid.reset()
        self.initial_sent(pose)
        self.last_sent_at = pose['sent_time_s']
        self.publish(anchor_xy_mm=(self.anchor_xy*1000).tolist(),
                     anchor_angles_deg=self.anchor_angles.tolist(), integral_xy_mm=[0., 0.])

    def _speed_brake_needed(self):
        if self.speed_limit_mm_s is None or not self.pid.velocity_ready:
            self.speed_braking = False
        else:
            speed = float(np.linalg.norm(self.pid.velocity)*1000)
            if speed > self.speed_limit_mm_s:
                self.speed_braking = True
            elif speed <= .8*self.speed_limit_mm_s:
                self.speed_braking = False
        self.publish(speed_braking=self.speed_braking, speed_limit_mm_s=self.speed_limit_mm_s)
        return self.speed_braking

    def _on_speed_brake(self):
        """Subclasses cancel unfinished steps before a braking command."""

    def _speed_brake_applied(self, applied_xy):
        """Subclasses synchronize their command state after a braking write."""

    def begin_point(self, step, correction, on_applied=lambda correction: None):
        self._reset_slow_assist()
        result = super().begin_point(step, correction, on_applied)
        with self.lock:
            self.telemetry.pop('integral_force_n', None)
            self.telemetry.pop('inherited_integral_force_n', None)
        self.publish(mode=self.control_mode, integral_xy_mm=(self.pid.integral_force*1000).tolist())
        return result

    def _continuous_update(self, step, commander, case_name, gate, active, report,
                           measured, sample, loss_generation, revision):
        if (not self.pid.velocity_ready or self.anchor_xy is None
                or sample == self.last_control_sample
                or (self.last_control_at is not None and self.clock()-self.last_control_at < self.pid.period)):
            return
        error = np.asarray(step.target_xy_m)-measured
        braking = self._speed_brake_needed()
        if not braking:
            self._observe_stall(float(np.linalg.norm(error)*1000),
                                float(np.linalg.norm(self.pid.velocity)*1000), sample, gate.tolerance_mm)
        self.publish(position_error_mm=(error*1000).tolist(),
                     velocity_xy_mm_s=(self.pid.velocity*1000).tolist())
        if not braking and np.linalg.norm(error)*1000 <= gate.tolerance_mm and np.linalg.norm(self.pid.velocity)*1000 <= self.hold_speed_mm_s:
            self.publish(state='Reached — holding optimizer pose')
            return
        self.last_control_at, self.last_control_sample = self.clock(), sample
        if braking:
            self._on_speed_brake()
            offset = -self.brake_gain_s*self.pid.velocity
            offset *= min(1., self.max_brake_m/max(float(np.linalg.norm(offset)), 1e-30))
            target = measured+offset
            correction = target-np.asarray(step.target_xy_m)
            proposal = None  # Neither waypoint pull nor integral accumulation during braking.
        else:
            proposal = self.pid.propose(step.target_xy_m, tolerance_m=0.)
            correction = proposal['pid_force_n']  # Generic PID arithmetic; units here are metres.
            target = np.asarray(step.target_xy_m)+correction
        self.publish(mode=self.control_mode, state='Speed braking — calculating optimizer' if braking else 'Calculating XY optimizer',
                     correction_mm=(correction*1000).tolist(), proposed_xy_mm=(target*1000).tolist(),
                     integral_xy_mm=(self.pid.integral_force*1000).tolist(),
                     observation_age_s=self.clock()-sample,
                     pid_gains=dict(kp=self.pid.kp, ki=self.pid.ki, kd=self.pid.kv))
        def valid():
            with gate.condition:
                return (active() and self.enabled.is_set() and not self.manual
                        and revision == self.command_revision and not gate.advance_due
                        and gate.loss_generation == loss_generation and gate.positions is not None
                        and len(gate.positions) == 1 and gate.observed_at is not None
                        and self.clock()-gate.observed_at <= 1.
                        and np.linalg.norm(gate.positions[0]-measured) <= .0005
                        and (not braking or gate.observed_at == sample))
        accepted = [False]
        def commit(result, sent):
            accepted[0] = True
            fraction = result['allocation_fraction']
            unmet = (1-fraction)*(target-self.anchor_xy)
            if braking:
                self.pid.committed_at = sample  # Freeze memory and prevent catch-up integration.
            else:
                self.pid.commit(proposal, correction, correction-unmet)
            if sent:
                self.initial_sent(result)
                applied = self.anchor_xy+fraction*(target-self.anchor_xy) if braking else target
                self.last_sent_u = (applied-np.asarray(step.target_xy_m))*1000
                self.publish(commanded_xy_mm=(applied*1000).tolist(),
                             sent_optimizer_xy_mm=(applied*1000).tolist())
                if braking:
                    self._speed_brake_applied(applied)
                self.on_applied(self.last_sent_u.tolist())
                report(result)
            self.publish(integral_xy_mm=(self.pid.integral_force*1000).tolist(),
                         allocation_fraction=result['allocation_fraction'],
                         state=(('Speed braking — sent' if sent else 'Speed braking — motor deadband')
                                + ('; optimizer input limit' if fraction < 1.-1e-9 else '') if braking else
                                ('XY optimizer sent' if sent else 'XY optimizer holding — motor deadband')
                                +self._stall_status(fraction, sent, error)))
        try:
            self.validate(target)
            key = (step.ratio, step.angle_deg)
            if key not in self.baselines:
                self.baselines[key] = commander.calculate(case_name, self.anchor_xy,
                                                          ratio=step.ratio, angle_deg=step.angle_deg)
            if not valid():
                self.publish(state='XY optimizer discarded — observation changed')
                return
            commander.move(case_name, target, ratio=step.ratio, angle_deg=step.angle_deg,
                zero_angle_deg=step.zero_angle_deg, active=valid, preview_only=self.preview_only,
                unchanged_angles_deg=self.last_result['angles_deg'], motor_deadband_deg=self.motor_deadband,
                anchored_xy_request=dict(anchor_xy_m=self.anchor_xy, anchor_angles_deg=self.anchor_angles,
                                         baseline_angles_deg=self.baselines[key]['angles_deg']),
                on_calculated=lambda result: self.publish(calculated_angles=list(result['angles_deg']),
                    allocation_fraction=result['allocation_fraction']),
                on_sent=lambda result: commit(result, True), on_held=lambda result: commit(result, False))
            if not accepted[0]:
                self.publish(state='XY optimizer preview' if self.preview_only else 'XY optimizer discarded — observation changed')
        except ValueError as error:
            self.publish(state=f'{"Speed braking blocked" if braking else "XY optimizer held"}: {error}')
        finally:
            self.publish(control_cycle_time_s=self.clock()-self.last_control_at)


def _solve_entry(entry, source_positions, cfg, params):
    start_time, target, additional, ratio, angle = unpack_target_schedule_entry(entry)
    if additional is None:
        extra_positions = []
    elif isinstance(additional, list):
        extra_positions = additional
    else:
        extra_positions = [additional]

    positions = [np.asarray(target, dtype=float)] + [
        np.asarray(position, dtype=float) for position in extra_positions
    ]
    failure_mode = params.get("OPTIMIZATION_FAILURE_MODE", "warn")

    if len(positions) == 1:
        controls = find_stable_equilibrium_inputs(
            positions[0], source_positions, cfg.C_F,
            ratio=ratio, eig_angle_rad=angle,
            trace_margin=cfg.STABILITY_TRACE_MARGIN,
            det_margin=cfg.STABILITY_DET_MARGIN,
            failure_mode=failure_mode,
        )
    elif len(positions) == 2:
        solver = params.get("TWO_EQUILIBRIUM_SOLVER", "stable")
        solvers = {
            "plain": find_two_equilibrium_inputs,
            "center_repulsion": find_two_equilibrium_with_center_repulsion_inputs,
            "stable": find_two_stable_equilibrium_inputs,
        }
        if solver not in solvers:
            raise ValueError(
                "TWO_EQUILIBRIUM_SOLVER must be 'stable', 'plain', "
                "or 'center_repulsion'."
            )
        kwargs = {"failure_mode": failure_mode}
        if solver == "stable":
            kwargs.update({
                "trace_margin": cfg.STABILITY_TRACE_MARGIN,
                "det_margin": cfg.STABILITY_DET_MARGIN,
            })
        controls = solvers[solver](positions[0], positions[1], source_positions, cfg.C_F, **kwargs)
    elif len(positions) == 4:
        controls = find_four_stable_equilibrium_inputs(
            positions, source_positions, cfg.C_F, failure_mode=failure_mode
        )
    else:
        raise ValueError("Only one, two, or four equilibrium targets are supported.")

    return start_time, positions, ratio, angle, controls


def run_control_workflow(cfg, params, *, compute_fields=True, report=None):
    """Optimize every scheduled target and optionally sample its fields once."""
    validate_target_schedule(cfg.TARGET_SCHEDULE)
    source_positions = generate_circular_source_positions(cfg.NUM_SOURCES, cfg.RADIUS)
    scheduled_controls = []
    opt_infos = []
    field_data = []

    for entry in cfg.TARGET_SCHEDULE:
        start_time, positions, ratio, angle, controls = _solve_entry(
            entry, source_positions, cfg, params
        )
        net_forces = [
            calculate_total_force_from_sources(
                source_positions, controls, position,
                cfg.M_SOURCE_MAGNITUDE, cfg.M_ROBOT_MAGNITUDE,
            )
            for position in positions
        ]
        stability = []
        for position in positions:
            hessian = calculate_potential_hessian(position, source_positions, cfg.C_F, controls)
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            stability.append({
                "position": position,
                "H": hessian,
                "trace": np.trace(hessian),
                "determinant": np.linalg.det(hessian),
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
            })

        opt_info = extract_optimization_info(
            control_inputs_u=controls,
            net_force=net_forces[0],
            H=stability[0]["H"],
            eigenvalues=stability[0]["eigenvalues"],
            eigenvectors=stability[0]["eigenvectors"],
            desired_pos=positions[0],
            equilibrium_positions=positions,
            equilibrium_net_forces=net_forces,
            equilibrium_stability=stability,
            microrobot_positions=cfg.INITIAL_ROBOT_POSITIONS,
        )
        scheduled_controls.append(ScheduledControl(
            start_time=start_time,
            target_positions=tuple(positions),
            control_inputs=controls,
            eig_ratio=ratio,
            eig_angle=angle,
        ))
        opt_infos.append(opt_info)

        if compute_fields:
            radial_units = source_positions / np.linalg.norm(
                source_positions, axis=1, keepdims=True
            )
            moments = controls[:, None] * cfg.M_SOURCE_MAGNITUDE * radial_units
            X, Y, Fx, Fy, U_pot, Bx, By = compute_grid_fields(
                source_positions, controls, moments,
                cfg.GRID_MIN, cfg.GRID_MAX, cfg.RESOLUTION,
                cfg.M_SOURCE_MAGNITUDE, cfg.M_ROBOT_MAGNITUDE,
            )
            field_data.append({
                "start_time": start_time,
                "target_pos": positions[0],
                "u": controls,
                "X": X, "Y": Y, "Fx": Fx, "Fy": Fy,
                "U_pot": U_pot, "Bx": Bx, "By": By,
            })

        if report is not None:
            report(opt_info)

    return ControlWorkflowResult(
        source_positions=source_positions,
        scheduled_controls=scheduled_controls,
        opt_infos=opt_infos,
        field_data=field_data,
    )
