"""Shared control synthesis and field-preparation workflow."""

from dataclasses import dataclass
import threading
import time

import numpy as np

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


def calculate_force_pid_control(params, target_xy_m, measured_xy_m, applied_angles_deg,
                                reference_inputs, pid_force_n, *, max_angle_step_deg=10.,
                                ratio=1., angle_deg=None):
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


class ForcePIDFeedback:
    """Camera-independent continuous force PID; manual XY remains separate."""
    realtime = True

    def __init__(self, *, validate=lambda target: None, manual=False,
                 max_angle_step_deg=10., preview_only=False, clock=time.monotonic,
                 pid=None, angle_rate_deg_s=20., motor_deadband_deg=.1, hold_speed_mm_s=.5):
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
        if (not np.isfinite(angle_rate_deg_s) or angle_rate_deg_s <= 0
                or not np.isfinite(motor_deadband_deg) or motor_deadband_deg < 0
                or not np.isfinite(hold_speed_mm_s) or hold_speed_mm_s <= 0):
            raise ValueError('Angle rate/hold speed must be positive; motor deadband nonnegative.')
        self.pid = pid or MagneticForcePID()
        self.hold_speed_mm_s = float(hold_speed_mm_s)
        self.angle_rate, self.motor_deadband = float(angle_rate_deg_s), float(motor_deadband_deg)
        self.last_control_at = None
        self.last_control_sample = None

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

    def set_manual_mode(self, manual):
        with self.lock:
            self.manual = bool(manual)
            self.command_revision += 1
            self.pending_manual = None
            if manual:
                self.enabled.clear()
        self.pid.reset(keep_integral=True)

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
        with self.lock:
            self.current_step, self.pending_manual = step, None
            self.command_revision += 1
            self.reference_inputs, self.last_result = None, None
            self.last_sent_u, self.last_sent_at = np.asarray(correction, float).copy(), None
            self.on_applied = on_applied
            self.telemetry = {}
            self.last_control_at = None
            self.last_control_sample = None
            self.pid.reset()
        return command

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
        self.publish(desired_xy_mm=target_mm.tolist(), commanded_xy_mm=(target_mm+self.last_sent_u).tolist(),
                     correction_mm=self.last_sent_u.tolist(), ratio=step.ratio, angle_deg=step.angle_deg)
        while active():
            with gate.condition:
                if gate.advance_due and (not gate.require_arrival or gate._ready()):
                    gate.next_requested = gate.waiting_for_next = False
                    return True
                positions = None if gate.positions is None else gate.positions.copy()
                sample, loss_generation = gate.observed_at, gate.loss_generation
            with self.lock:
                pending, revision = self.pending_manual, self.command_revision
            enabled = self.enabled.is_set() and not self.manual
            self.publish(mode='manual' if self.manual else 'force_pid')
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
                self.publish(state='Tracking lost — holding last motor pose')
            elif not enabled:
                self.publish(state='Manual — holding' if self.manual else 'Control off — holding')
            elif len(positions) != 1:
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
            self.publish(state='Reached — holding; damping remains active if moving')
            return
        self.last_control_at = self.clock()
        self.last_control_sample = sample
        proposal = self.pid.propose(step.target_xy_m, tolerance_m=gate.tolerance_mm/1000)
        elapsed = self.pid.period if self.last_sent_at is None else max(.001, self.clock()-self.last_sent_at)
        max_step = min(self.max_angle_step_deg, self.angle_rate*elapsed)
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
                self.publish(commanded_xy_mm=None, state='Force PID sent', observation_age_s=self.clock()-sample)
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
        self.publish(state='Calculating force PID', force_observation_monotonic_s=sample,
                     observation_age_s=self.clock()-sample,
                     force_kp_n_m=float(self.pid.kp), force_ki_n_m_s=float(self.pid.ki), force_kv_n_s_m=float(self.pid.kv))
        try:
            commander.move_force_pid(
                case_name, step.target_xy_m, measured, self.last_result['angles_deg'], self.reference_inputs,
                proposal['pid_force_n'],
                max_angle_step_deg=max_step, ratio=step.ratio, angle_deg=step.angle_deg,
                zero_angle_deg=step.zero_angle_deg, active=valid, on_calculated=calculated,
                on_sent=lambda result: commit(result, sent=True),
                on_held=lambda result: commit(result, sent=False), motor_deadband_deg=self.motor_deadband,
                preview_only=self.preview_only)
            if self.preview_only:
                self.publish(state='Preview only — force PID not sent')
            elif not accepted[0]:
                self.publish(state='Force PID discarded — stale frame, tracking or mode changed')
        except ValueError as error:
            self.publish(state=f'Force PID held: {error}')
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
