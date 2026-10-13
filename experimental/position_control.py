"""Bounded position-only feedback in millimetres; no orientation inference."""

import threading
import time
import numpy as np


class PositionPID:
    def __init__(self, kp=.2, ki=0., kd=0., limit_mm=2., slew_mm_s=1., period_s=1.):
        values = np.asarray([kp, ki, kd, limit_mm, slew_mm_s, period_s], float)
        if not np.isfinite(values).all() or np.any(values[:3] < 0) or np.any(values[3:] <= 0):
            raise ValueError('PID gains must be finite/nonnegative; limits and period positive.')
        self.kp, self.ki, self.kd, self.limit, self.slew, self.period = values
        self.reset()

    def reset(self):
        self.integral = np.zeros(2)
        self.derivative = np.zeros(2)
        self.previous_error = None
        self.previous_time = None
        self.output = np.zeros(2)

    def update(self, error_mm, timestamp):
        error = np.asarray(error_mm, float)
        if error.shape != (2,) or not np.isfinite(error).all():
            raise ValueError('PID needs a valid measured XY error.')
        dt = self.period if self.previous_time is None else timestamp - self.previous_time
        if dt <= 0:
            return self.output.copy()
        dt = min(dt, self.period * 2)
        derivative = np.zeros(2) if self.previous_error is None else (error - self.previous_error) / dt
        self.derivative += dt / (.5 + dt) * (derivative - self.derivative)
        proposed_integral = self.integral + error * dt
        raw = self.kp * error + self.ki * proposed_integral + self.kd * self.derivative
        norm = np.linalg.norm(raw)
        bounded = raw * min(1., self.limit / max(norm, 1e-12))
        delta = bounded - self.output
        bounded = self.output + delta * min(1., self.slew * dt / max(np.linalg.norm(delta), 1e-12))
        if np.linalg.norm(raw - bounded) < 1e-9:
            self.integral = proposed_integral
        self.output = bounded
        self.previous_error, self.previous_time = error.copy(), timestamp
        return bounded.copy()


class PositionFeedback:
    def __init__(self, pid, *, validate=lambda target: None, clock=time.monotonic):
        self.pid, self.validate, self.clock = pid, validate, clock
        self.enabled = threading.Event()  # Opt-in, never enabled automatically.
        self.lock = threading.Lock()
        self.telemetry = {}

    def publish(self, **values):
        with self.lock:
            self.telemetry.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.telemetry)

    def wait_at_target(self, step, commander, case_name, gate, active, report, initial_result=None):
        self.pid.reset()
        target_mm = np.asarray(step.target_xy_m) * 1000
        self.publish(desired_xy_mm=target_mm.tolist(), commanded_xy_mm=target_mm.tolist(), correction_mm=[0., 0.],
                     ratio=step.ratio, angle_deg=step.angle_deg, state='Waiting', calculated_angles='',
                     sent_angles=[] if initial_result is None else [float(value) for value in initial_result['angles_deg']])
        last_update = last_sample = None
        was_enabled = False
        while active():
            with gate.condition:
                ready = gate._ready()
                if ready:
                    gate.waiting_for_next = True
                    if gate.next_requested:
                        gate.next_requested = gate.waiting_for_next = False
                        return True
                elif gate.next_requested:
                    gate.next_requested = False
                positions = None if gate.positions is None else gate.positions.copy()
                sample = gate.observed_at
                generation = gate.loss_generation
            enabled = self.enabled.is_set()
            now = self.clock()
            if enabled != was_enabled:
                self.pid.reset()
                was_enabled = enabled
            fresh = positions is not None and sample is not None and now - sample <= 1.
            if not fresh:
                self.pid.reset()
                self.publish(state='Tracking lost — corrections paused', measured_xy_mm=None)
            else:
                measured = positions.mean(axis=0) * 1000
                self.publish(measured_xy_mm=measured.tolist())
                if ready:
                    self.publish(state='Reached — waiting for Next')
                elif not enabled:
                    self.publish(state='Control off — tracking only')
                elif np.linalg.norm(target_mm - measured) <= gate.tolerance_mm:
                    self.publish(state='Within XY tolerance — waiting for complete-group arrival')
                elif sample != last_sample and (last_update is None or now - last_update >= self.pid.period):
                    last_sample, last_update = sample, now
                    correction = self.pid.update(target_mm - measured, sample)
                    command = (target_mm + correction) / 1000
                    self.publish(proposed_xy_mm=(command * 1000).tolist(), state='Calculating', calculated_angles='')
                    try:
                        self.validate(command)
                    except ValueError as error:
                        self.pid.reset()
                        self.publish(state=f'Correction blocked: {error}')
                        with gate.condition:
                            gate.condition.wait(.1)
                        continue

                    def correction_active():
                        with gate.condition:
                            return (active() and self.enabled.is_set() and gate.positions is not None
                                    and gate.observed_at is not None and self.clock() - sample <= 1.
                                    and gate.loss_generation == generation
                                    and np.linalg.norm(gate.positions.mean(axis=0) * 1000 - measured) <= gate.tolerance_mm
                                    and not gate.next_requested and not gate._ready())

                    def sent(result):
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
                gate.condition.wait(.1)
        return False
