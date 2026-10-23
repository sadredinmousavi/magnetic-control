"""Bounded XY command ramps separated by measured settling, using the same optimizer."""
import numpy as np

from control_workflow import OptimizerXYFeedback


class IncrementalXYFeedback(OptimizerXYFeedback):
    control_mode = 'anchored_xy_optimizer_step_settle'

    def __init__(self, *, step_mm=1., rate_mm_s=.5, settle_speed_mm_s=.2,
                 settle_seconds=.75, max_bias_mm=5., timeout_seconds=10., **kwargs):
        values = np.asarray([step_mm, rate_mm_s, settle_speed_mm_s, settle_seconds,
                             max_bias_mm, timeout_seconds], float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Incremental controller settings must be finite and positive.')
        if step_mm > 1.:
            raise ValueError('Incremental command step must be at most 1 mm.')
        if timeout_seconds <= step_mm/rate_mm_s+settle_seconds:
            raise ValueError('Step timeout must exceed the full ramp time plus settling hold.')
        self.step_m, self.rate_m_s = step_mm*.001, rate_mm_s*.001
        self.settle_speed, self.settle_seconds = settle_speed_mm_s, settle_seconds
        self.max_bias_m, self.timeout = max_bias_mm*.001, timeout_seconds
        self.min_progress_mm = min(.1, step_mm*.1)
        self.applied_xy = self.request_xy = None
        self.goal = self.settle_since = self.phase_started = self.previous_sample = None
        self.no_progress_steps, self.completed_step, self.step_error = 0, False, None
        self.halt_reason = None
        super().__init__(**kwargs)

    def _reset_slow_assist(self):
        super()._reset_slow_assist()
        self.settle_since = self.phase_started = self.previous_sample = None
        self.goal = None
        applied = getattr(self, 'applied_xy', None)
        self.request_xy = None if applied is None else applied.copy()
        self.completed_step = False

    def seed_anchor(self, xy_m, pose):
        super().seed_anchor(xy_m, pose)
        self.applied_xy = self.request_xy = np.asarray(xy_m, float).copy()
        self.halt_reason = None

    def begin_point(self, step, correction, on_applied=lambda correction: None):
        result = super().begin_point(step, correction, on_applied)
        self.no_progress_steps, self.step_error = 0, None
        if self.applied_xy is not None:
            self.last_sent_u = (self.applied_xy-np.asarray(step.target_xy_m))*1000
            self.publish(sent_optimizer_xy_mm=(self.applied_xy*1000).tolist(),
                         commanded_xy_mm=(self.applied_xy*1000).tolist(),
                         correction_mm=self.last_sent_u.tolist(), integral_xy_mm=[0., 0.])
        return result

    def _halt(self, reason):
        self.halt_reason = reason
        self.publish(state='Incremental controller stopped: '+reason, incremental_phase='stopped')
        raise ValueError('Incremental controller stopped: '+reason)

    def choose_goal(self, waypoint, error, measured, sample):
        return self.applied_xy+error*min(1., self.step_m/max(float(np.linalg.norm(error)), 1e-30))

    def _on_speed_brake(self):
        self.goal = None
        self.request_xy = self.applied_xy.copy()
        self.settle_since = self.phase_started = self.previous_sample = None
        self.completed_step = False
        self.publish(incremental_phase='braking')

    def _speed_brake_applied(self, applied_xy):
        self.applied_xy = self.request_xy = applied_xy.copy()

    def _continuous_update(self, step, commander, case_name, gate, active, report,
                           measured, sample, loss_generation, revision):
        if self.halt_reason:
            self._halt(self.halt_reason)
        if not self.pid.velocity_ready or self.applied_xy is None or sample == self.last_control_sample:
            return
        if self._speed_brake_needed():
            # Shared brake takes priority over ramps, settling and supervisor commands.
            super()._continuous_update(step, commander, case_name, gate, active, report,
                                       measured, sample, loss_generation, revision)
            return
        if self.last_control_at is not None and self.clock()-self.last_control_at < self.pid.period-1e-9:
            return
        if self.previous_sample is not None and sample-self.previous_sample > 1.:
            self._reset_slow_assist()
        dt = self.pid.period if self.previous_sample is None else min(sample-self.previous_sample, 2*self.pid.period)
        self.previous_sample = sample
        self.last_control_sample, self.last_control_at = sample, self.clock()
        waypoint = np.asarray(step.target_xy_m)
        error = waypoint-measured
        distance = float(np.linalg.norm(error))
        speed = float(np.linalg.norm(self.pid.velocity)*1000)
        self.publish(mode=self.control_mode, position_error_mm=(error*1000).tolist(),
                     velocity_xy_mm_s=(self.pid.velocity*1000).tolist(), integral_xy_mm=[0., 0.],
                     incremental_step_mm=self.step_m*1000, incremental_rate_mm_s=self.rate_m_s*1000,
                     no_progress_steps=self.no_progress_steps)
        if distance*1000 <= gate.tolerance_mm:
            self.goal = None
            self.request_xy = self.applied_xy.copy()
            self.settle_since = self.phase_started = None
            self.publish(state='Incremental arrival — holding command for settled confirmation', incremental_phase='arrival')
            return
        if self.goal is not None and np.dot(self.goal-self.request_xy, error) < 0:
            # Reassess after settling if the robot passes the waypoint mid-ramp.
            self.goal = None
            self.request_xy = self.applied_xy.copy()
            self.settle_since = None
            self.completed_step = False
        if self.phase_started is None:
            self.phase_started = sample
        if sample-self.phase_started >= self.timeout:
            self._halt('step/settling timeout; check motion, tracking and motor deadband')
        if self.goal is None:
            self.publish(state='Incremental — waiting for measured settling', incremental_phase='settling')
            if speed > self.settle_speed:
                self.settle_since = None
                return
            if self.settle_since is None:
                self.settle_since = sample
            if sample-self.settle_since < self.settle_seconds:
                return
            if self.completed_step:
                self.completed_step = False  # Evaluate each applied step once, including when a supervisor waits.
                improvement_mm = (self.step_error-distance)*1000
                self.no_progress_steps = self.no_progress_steps+1 if improvement_mm < self.min_progress_mm else 0
                self.publish(no_progress_steps=self.no_progress_steps, step_improvement_mm=improvement_mm)
                if self.no_progress_steps >= 3:
                    self._halt(f'no progress after 3 settled steps (less than {self.min_progress_mm:g} mm improvement each)')
            self.goal = self.choose_goal(waypoint, error, measured, sample)
            if self.goal is None:
                self.phase_started = None  # A supervisor may wait without issuing a command.
                return
            bias = float(np.linalg.norm(self.goal-waypoint))
            old_bias = float(np.linalg.norm(self.applied_xy-waypoint))
            if bias > self.max_bias_m and bias >= old_bias:
                self._halt('maximum command offset from waypoint reached')
            if np.linalg.norm(self.goal-measured) > self.max_bias_m:
                self._halt('maximum command lead over measured position reached')
            self.step_error = distance
            self.request_xy = self.applied_xy.copy()
            self.settle_since = None
            self.completed_step = False
            self.phase_started = sample
        delta = self.goal-self.request_xy
        candidate = self.request_xy+delta*min(1., self.rate_m_s*max(0., dt)/max(float(np.linalg.norm(delta)), 1e-30))
        if np.linalg.norm(candidate-measured) > self.max_bias_m+1e-12:
            self._halt('maximum command lead over measured position reached during ramp')
        self.publish(state='Incremental — ramping command', incremental_phase='ramping',
                     step_goal_xy_mm=(self.goal*1000).tolist(), proposed_xy_mm=(candidate*1000).tolist(),
                     observation_age_s=self.clock()-sample)

        def valid():
            with gate.condition:
                return (active() and self.enabled.is_set() and not self.manual and not gate.advance_due
                        and revision == self.command_revision and gate.loss_generation == loss_generation
                        and gate.positions is not None and len(gate.positions) == 1
                        and gate.observed_at is not None and self.clock()-gate.observed_at <= 1.
                        and np.linalg.norm(gate.positions[0]-measured) <= .0005)

        accepted = [False]
        limited = [False]
        def commit(result, sent):
            accepted[0] = True
            fraction = float(result['allocation_fraction'])
            limited[0] = fraction < 1.-1e-9
            self.request_xy = candidate.copy()
            if sent:
                self.applied_xy = self.anchor_xy+fraction*(candidate-self.anchor_xy)
                self.initial_sent(result)
                self.last_sent_u = (self.applied_xy-waypoint)*1000
                self.on_applied(self.last_sent_u.tolist())
                self.publish(sent_optimizer_xy_mm=(self.applied_xy*1000).tolist(),
                             commanded_xy_mm=(self.applied_xy*1000).tolist(), correction_mm=self.last_sent_u.tolist())
                report(result)
                if np.linalg.norm(self.goal-candidate) < 1e-10:
                    self.goal = None
                    self.completed_step = True
                    self.settle_since = None
            self.publish(allocation_fraction=fraction,
                         state='Incremental command sent' if sent else 'Incremental — motor deadband; waiting to send')
        try:
            self.validate(candidate)
            key = (step.ratio, step.angle_deg)
            if key not in self.baselines:
                self.baselines[key] = commander.calculate(case_name, self.anchor_xy, ratio=step.ratio, angle_deg=step.angle_deg)
            if not valid():
                self.publish(state='Incremental — discarded changed observation')
                return
            commander.move(case_name, candidate, ratio=step.ratio, angle_deg=step.angle_deg,
                zero_angle_deg=step.zero_angle_deg, active=valid, preview_only=self.preview_only,
                unchanged_angles_deg=self.last_result['angles_deg'], motor_deadband_deg=self.motor_deadband,
                anchored_xy_request=dict(anchor_xy_m=self.anchor_xy, anchor_angles_deg=self.anchor_angles,
                                         baseline_angles_deg=self.baselines[key]['angles_deg']),
                on_calculated=lambda result: self.publish(calculated_angles=list(result['angles_deg']),
                                                          allocation_fraction=result['allocation_fraction']),
                on_sent=lambda result: commit(result, True), on_held=lambda result: commit(result, False))
        except ValueError as error:
            self._halt(str(error))
        finally:
            self.publish(control_cycle_time_s=self.clock()-self.last_control_at)
        if limited[0]:
            self._halt('optimizer input limit reached')
        if not accepted[0]:
            self.publish(state='Incremental — preview or discarded observation')
