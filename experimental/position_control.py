"""Bounded position-only feedback in millimetres; no orientation inference."""

import threading
import time
import numpy as np


class PositionPID:
    def __init__(self, kp=.2, ki=0., kd=0., limit_mm=2., slew_mm_s=1., period_s=1., *,
                 adaptive=False, tolerance_mm=1., kp_min=.05, kp_max=1.,
                 adaptive_limit=False, limit_max_mm=20., adapt_percent=50., incremental=False):
        values = np.asarray([kp, ki, kd, limit_mm, slew_mm_s, period_s], float)
        if not np.isfinite(values).all() or np.any(values[:3] < 0) or np.any(values[3:] <= 0):
            raise ValueError('PID gains must be finite/nonnegative; limits and period positive.')
        self.kp, self.ki, self.kd, self.limit, self.slew, self.period = values
        bounds = np.asarray([tolerance_mm, kp_min, kp_max], float)
        if not np.isfinite(bounds).all() or tolerance_mm <= 0 or not 0 < kp_min <= kp_max:
            raise ValueError('Adaptive tolerance and Kp bounds must be positive and finite; min <= max.')
        if adaptive and not kp_min <= kp <= kp_max:
            raise ValueError('Starting Kp must be within adaptive bounds (0.05 to Kp max).')
        self.initial_kp = self.kp
        self.adaptive, self.tolerance = bool(adaptive), tolerance_mm
        self.kp_min, self.kp_max = kp_min, kp_max
        if not np.isfinite(limit_max_mm) or limit_max_mm <= 0 or (adaptive_limit and limit_max_mm < limit_mm):
            raise ValueError('Limit max must be positive/finite and at least the starting limit when adaptive.')
        self.initial_limit = self.limit
        self.adaptive_limit, self.limit_max = bool(adaptive_limit), float(limit_max_mm)
        if not np.isfinite(adapt_percent) or not 0 < adapt_percent <= 100:
            raise ValueError('Adapt % must be finite and between 0 and 100 (exclusive of zero).')
        self.adapt_rate = float(adapt_percent) / 100
        self.incremental = bool(incremental)
        self.reset()

    def reset(self):
        self.kp = self.initial_kp
        self.limit = self.initial_limit
        self.sent_error = None
        self.sent_at = None
        self.sent_limited = False
        self.sent_at_cap = False
        self.limit_adaptation = 'Waiting for applied response' if self.adaptive_limit else 'Fixed limit'
        self.adaptation = 'Waiting for applied response' if self.adaptive else 'Fixed Kp'
        self.integral = np.zeros(2)
        self.derivative = np.zeros(2)
        self.previous_error = None
        self.previous_time = None
        self.output = np.zeros(2)
        self.applied_output = np.zeros(2)
        self.output_limited = self.output_at_cap = False

    def seed_output(self, correction):
        vector = np.asarray(correction, float)
        ceiling = self.limit_max if self.adaptive_limit else self.limit
        if vector.shape != (2,) or not np.isfinite(vector).all() or np.linalg.norm(vector) > ceiling + 1e-9:
            raise ValueError('Learned u exceeds correction limit/max; increase the limit or clear learned u.')
        self.output = self.applied_output = vector.copy()
        self.limit = max(self.limit, float(np.linalg.norm(vector)))

    def mark_sent(self, error_mm, timestamp):
        """Only successfully sent corrections may inform the next gain update."""
        self.applied_output = self.output.copy()
        if self.sent_error is not None:
            return  # Keep the response anchor; a newer send must not postpone it forever.
        self.sent_error = np.asarray(error_mm, float).copy()
        self.sent_at = timestamp
        self.sent_limited = self.output_limited
        self.sent_at_cap = self.output_at_cap

    def adapt_response(self, error, timestamp):
        if not (self.adaptive or self.adaptive_limit) or self.sent_error is None:
            return
        if timestamp - self.sent_at < self.period:
            return
        previous = self.sent_error
        self.sent_error = None  # Consume each applied response only once.
        size, old_size = np.linalg.norm(error), np.linalg.norm(previous)
        worsening = np.dot(previous, error) < 0 or size > old_size + max(.2, old_size * .1)
        if self.adaptive_limit:
            if size <= self.tolerance:
                self.limit_adaptation = 'Within tolerance; limit held'
            elif worsening:
                # Do not suddenly clip an already-applied correction on shrink.
                self.limit = max(self.initial_limit, float(np.linalg.norm(self.output)), self.limit * .8)
                self.limit_adaptation = 'Limit reduced/held: overshoot / error grew'
            elif self.sent_at_cap:
                self.limit = min(self.limit_max, self.limit + max(.25, self.limit * self.adapt_rate * .5))
                self.limit_adaptation = 'Limit max reached' if self.limit >= self.limit_max else 'Limit increased: persistent error'
            else:
                self.limit_adaptation = 'Below limit / slew ramp; limit held'
        if not self.adaptive:
            return
        if size <= self.tolerance:
            self.adaptation = 'Within tolerance'
        elif worsening:
            self.kp = max(self.kp_min, self.kp * .8)
            self.adaptation = 'Kp reduced: overshoot / error grew'
        elif self.sent_limited:
            self.adaptation = 'At correction / slew limit; no gain increase'
        elif old_size - size < max(.05, old_size * .05):
            # Semi-static plant: act on each applied response, not a four-cycle vote.
            # Taper aggressiveness near tolerance to avoid chasing pixel noise.
            rate = self.adapt_rate * float(np.clip((size / self.tolerance - 1) / 4, .2, 1))
            self.kp = min(self.kp_max, self.kp * (1 + rate))
            self.adaptation = 'Kp max reached' if self.kp >= self.kp_max else 'Kp increased: slow progress'
        else:
            self.adaptation = 'Error improving; Kp held'

    def update(self, error_mm, timestamp):
        error = np.asarray(error_mm, float)
        if error.shape != (2,) or not np.isfinite(error).all():
            raise ValueError('PID needs a valid measured XY error.')
        if not np.isfinite(timestamp):
            raise ValueError('PID needs a finite observation timestamp.')
        dt = self.period if self.previous_time is None else timestamp - self.previous_time
        if dt <= 0:
            return self.output.copy()
        self.adapt_response(error, timestamp)
        dt = min(dt, self.period * 2)
        derivative = np.zeros(2) if self.previous_error is None else (error - self.previous_error) / dt
        self.derivative += dt / (.5 + dt) * (derivative - self.derivative)
        proposed_integral = self.integral + error * dt
        raw = (self.applied_output + self.kp * error if self.incremental else
               self.kp * error + self.ki * proposed_integral + self.kd * self.derivative)
        norm = np.linalg.norm(raw)
        bounded = raw * min(1., self.limit / max(norm, 1e-12))
        base = self.applied_output if self.incremental else self.output
        delta = bounded - base
        bounded = base + delta * min(1., self.slew * dt / max(np.linalg.norm(delta), 1e-12))
        self.output_limited = norm >= self.limit or np.linalg.norm(delta) > self.slew * dt
        self.output_at_cap = norm >= self.limit and np.linalg.norm(bounded) >= self.limit - 1e-9
        if np.linalg.norm(raw - bounded) < 1e-9:
            self.integral = proposed_integral
        self.output = bounded
        self.previous_error, self.previous_time = error.copy(), timestamp
        return bounded.copy()


class PositionFeedback:
    def __init__(self, pid, *, validate=lambda target: None, clock=time.monotonic, settling=None, manual=False):
        self.manual = manual
        self.pid, self.validate, self.clock = pid, validate, clock
        self.enabled = threading.Event()  # Opt-in, never enabled automatically.
        self.lock = threading.Lock()
        self.telemetry = {}
        self.last_sent_u = np.zeros(2)
        self.last_sent_at = None
        self.on_applied = lambda correction: None
        self.settling = settling
        self.current_step = None
        self.command_revision = 0
        self.pending_manual = None

    def set_manual_mode(self, manual):
        with self.lock:
            self.manual = bool(manual)
            self.command_revision += 1
            self.pending_manual = None
        if manual:
            self.enabled.clear()

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
        with self.lock:
            self.current_step = step
            self.pending_manual = None
            self.command_revision += 1
        self.pid.reset()
        if not self.manual:
            self.pid.seed_output(correction)
        command = np.asarray(step.target_xy_m) + np.asarray(correction) / 1000
        if np.linalg.norm(correction) > 0:
            self.validate(command)
        self.last_sent_u = np.asarray(correction, float).copy()
        self.last_sent_at = None
        self.on_applied = on_applied
        return command

    def initial_sent(self, result=None):
        self.last_sent_at = self.clock()
        if self.settling is not None:
            self.settling.reset(self.last_sent_at)
            self.publish(robot_speed_mm_s=None, settling_state='Minimum wait', settle_hold_elapsed_s=0., settled=0,
                         settled_xy_mm=None)

    def reset_history(self):
        self.pid.reset()
        if not self.manual:
            # Resuming control must start at the applied manual pose, not clip it.
            magnitude = float(np.linalg.norm(self.last_sent_u))
            self.pid.limit = max(self.pid.limit, magnitude)
            if self.pid.adaptive_limit:
                self.pid.limit_max = max(self.pid.limit_max, magnitude)
            self.pid.seed_output(self.last_sent_u)

    def publish(self, **values):
        with self.lock:
            self.telemetry.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.telemetry)

    def wait_at_target(self, step, commander, case_name, gate, active, report, initial_result=None):
        self.reset_history()
        target_mm = np.asarray(step.target_xy_m) * 1000
        self.publish(desired_xy_mm=target_mm.tolist(), commanded_xy_mm=(target_mm + self.last_sent_u).tolist(),
                     correction_mm=self.last_sent_u.tolist(),
                     mode='manual' if self.manual else ('incremental' if self.pid.incremental else 'pid'),
                     kp=float(self.pid.kp), adaptive=self.pid.adaptive, adaptation=self.pid.adaptation,
                     limit_mm=float(self.pid.limit), adaptive_limit=self.pid.adaptive_limit, limit_adaptation=self.pid.limit_adaptation,
                     ratio=step.ratio, angle_deg=step.angle_deg, state='Waiting', calculated_angles='',
                     sent_angles=[] if initial_result is None else [float(value) for value in initial_result['angles_deg']])
        last_update = last_sample = None
        was_enabled = self.enabled.is_set()
        was_manual = self.manual
        while active():
            with gate.condition:
                ready = gate._ready()
                if not gate.require_arrival and gate.next_requested:
                    gate.next_requested = gate.waiting_for_next = False
                    return True
                if ready:
                    gate.waiting_for_next = True
                    if gate.next_requested:
                        gate.next_requested = gate.waiting_for_next = False
                        return True
                elif gate.next_requested and gate.require_arrival:
                    gate.next_requested = False
                positions = None if gate.positions is None else gate.positions.copy()
                sample = gate.observed_at
                generation = gate.loss_generation
            with self.lock:
                pending = self.pending_manual
                revision = self.command_revision
            if pending is not None and self.manual:
                request_revision, correction = pending
                command = (target_mm + correction) / 1000
                def manual_active():
                    with gate.condition:
                        with self.lock:
                            return (active() and self.manual and self.current_step is step
                                    and self.command_revision == request_revision and not gate.next_requested)
                try:
                    self.validate(command)
                    self.publish(proposed_xy_mm=(command*1000).tolist(), state='Calculating manual command', calculated_angles='')
                    def manual_sent(result):
                        self.last_sent_u = correction.copy()
                        self.initial_sent()
                        self.on_applied(correction.tolist())
                        self.publish(commanded_xy_mm=(command*1000).tolist(), correction_mm=correction.tolist(),
                                     sent_angles=[float(v) for v in result['angles_deg']],
                                     sent_time_s=self.clock(), state='Manual command sent', mode='manual')
                        report(result)
                    commander.move(case_name, command, ratio=step.ratio, angle_deg=step.angle_deg,
                                   settle_seconds=0, zero_angle_deg=step.zero_angle_deg,
                                   active=manual_active, on_sent=manual_sent,
                                   on_calculated=lambda result: self.publish(calculated_angles=[float(v) for v in result['angles_deg']]))
                except ValueError as error:
                    self.publish(state=f'Manual command blocked: {error}')
                finally:
                    with self.lock:
                        if self.command_revision == request_revision:
                            self.pending_manual = None
                continue
            enabled = self.enabled.is_set() and not self.manual
            now = self.clock()
            if enabled != was_enabled or self.manual != was_manual:
                self.reset_history()
                if self.settling is not None:
                    self.settling.reset(self.last_sent_at)
                was_enabled = enabled
                was_manual = self.manual
            self.publish(mode='manual' if self.manual else ('incremental' if self.pid.incremental else 'pid'))
            fresh = positions is not None and sample is not None and now - sample <= 1.
            if not fresh:
                self.reset_history()
                if self.settling is not None:
                    self.settling.reset(self.last_sent_at)
                self.publish(state='Tracking lost — corrections paused', measured_xy_mm=None,
                             robot_speed_mm_s=None, settling_state='Waiting for tracking',
                             settle_hold_elapsed_s=0., settled=0, settled_xy_mm=None)
            else:
                measured = positions.mean(axis=0) * 1000
                self.publish(measured_xy_mm=measured.tolist())
                motion = None if self.settling is None else self.settling.snapshot(now)
                settled = motion is None or motion['ready']
                if motion is not None:
                    self.publish(robot_speed_mm_s=motion['speed_mm_s'], settling_state=motion['state'],
                                 settle_hold_elapsed_s=motion['hold_elapsed_s'],
                                 settle_hold_s=float(self.settling.hold), settle_speed_limit_mm_s=float(self.settling.speed_limit),
                                 settled=int(settled))
                    if settled:
                        measured = motion['positions_mm'].mean(axis=0)
                        self.publish(settled_xy_mm=measured.tolist())
                    else:
                        self.publish(settled_xy_mm=None)
                if not settled:
                    self.publish(state=f"{motion['state']} — {'holding u' if enabled else 'Control off'}")
                elif ready:
                    self.publish(state='Reached — waiting for Next')
                elif not enabled:
                    self.publish(state='Control off — tracking only')
                elif np.linalg.norm(target_mm - measured) <= gate.tolerance_mm:
                    self.publish(state='Within XY tolerance — waiting for complete-group arrival')
                elif (sample != last_sample and (last_update is None or now - last_update >= self.pid.period)
                      and (not self.pid.incremental or self.last_sent_at is None
                           or sample - self.last_sent_at >= self.pid.period)):
                    last_sample, last_update = sample, now
                    correction = self.pid.update(target_mm - measured, sample)
                    self.publish(kp=float(self.pid.kp), adaptation=self.pid.adaptation,
                                 limit_mm=float(self.pid.limit), limit_adaptation=self.pid.limit_adaptation)
                    command = (target_mm + correction) / 1000
                    self.publish(proposed_xy_mm=(command * 1000).tolist(), state='Calculating', calculated_angles='')
                    try:
                        self.validate(command)
                    except ValueError as error:
                        self.reset_history()
                        self.publish(state=f'Correction blocked: {error}')
                        with gate.condition:
                            gate.condition.wait(.1)
                        continue

                    def correction_active():
                        with gate.condition:
                            return (active() and self.enabled.is_set() and gate.positions is not None
                                    and not self.manual and self.command_revision == revision
                                    and gate.observed_at is not None and self.clock() - sample <= 1.
                                    and gate.loss_generation == generation
                                    and (self.settling is None or self.settling.snapshot(self.clock())['ready'])
                                    and np.linalg.norm(gate.positions.mean(axis=0) * 1000 - measured) <= gate.tolerance_mm
                                    and not gate.next_requested and not gate._ready())

                    def sent(result):
                        self.pid.mark_sent(target_mm - measured, self.clock())
                        self.last_sent_u = correction.copy()
                        self.last_sent_at = self.clock()
                        if self.settling is not None:
                            self.settling.reset(self.last_sent_at)
                            self.publish(robot_speed_mm_s=None, settling_state='Minimum wait', settle_hold_elapsed_s=0.,
                                         settled=0, settled_xy_mm=None)
                        self.on_applied(correction.tolist())
                        self.publish(commanded_xy_mm=(command * 1000).tolist(), correction_mm=correction.tolist(),
                                     sent_angles=[float(v) for v in result['angles_deg']],
                                     sent_time_s=self.clock(), state='Sent')
                        report(result)

                    result = commander.move(case_name, command, ratio=step.ratio, angle_deg=step.angle_deg,
                        settle_seconds=0, zero_angle_deg=step.zero_angle_deg, active=correction_active, on_sent=sent,
                        on_calculated=lambda result: self.publish(calculated_angles=[float(v) for v in result['angles_deg']]))
                    if result is None:
                        self.publish(state='Correction discarded: stale observation / Control off / arrival')
            with gate.condition:
                self.publish(kp=float(self.pid.kp), adaptation=self.pid.adaptation,
                             limit_mm=float(self.pid.limit), limit_adaptation=self.pid.limit_adaptation)
                gate.condition.wait(.1)
        return False
