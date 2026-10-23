"""GUI005 velocity-form position PID with bounded command slew and speed feedback.

Coordinates are metres internally. The optimizer command is a model coordinate,
not a measured equilibrium; only a full, transmitted allocation advances it.
"""
import numpy as np

from control_workflow import OptimizerXYFeedback


def bounded_vector(vector, radius):
    return vector * min(1., radius / max(float(np.linalg.norm(vector)), 1e-30))


class IncrementalPIDXYFeedback(OptimizerXYFeedback):
    control_mode = 'anchored_xy_incremental_pid'

    def __init__(self, *, command_rate_mm_s=.5, command_offset_mm=100., slowdown_mm=3., **kwargs):
        values = np.asarray([command_rate_mm_s, command_offset_mm, slowdown_mm], float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Command rate, offset limit and slowdown distance must be finite and positive.')
        self.command_rate = command_rate_mm_s * .001
        self.command_offset = command_offset_mm * .001
        self.slowdown_m = slowdown_mm * .001
        self.applied_xy = self.request_xy = None
        self.history_error = self.history_velocity = self.history_sample = None
        self.sent_sample = None
        super().__init__(**kwargs)
        if self.speed_limit_mm_s is None:
            raise ValueError('Incremental XY PID requires a measured speed limit.')

    def _reset_slow_assist(self):
        super()._reset_slow_assist()
        self.history_error = self.history_velocity = self.history_sample = None
        if self.applied_xy is not None:
            self.request_xy = self.applied_xy.copy()

    def seed_anchor(self, xy_m, pose):
        super().seed_anchor(xy_m, pose)
        # The anchored solver defines this pose's command coordinate as measured XY.
        self.applied_xy = np.asarray(xy_m, float).copy()
        self.request_xy = self.applied_xy.copy()
        self.sent_sample = None
        self.history_error = np.zeros(2)  # Initial proportional response to the target.
        self.history_velocity = np.zeros(2)

    def begin_point(self, step, correction, on_applied=lambda correction: None):
        # Parent resets auxiliary state at each waypoint. Preserve PID differences
        # across waypoints, but not across tracking loss or a pause.
        history = self.history_error, self.history_velocity, self.history_sample
        result = super().begin_point(step, correction, on_applied)
        self.history_error, self.history_velocity, self.history_sample = history
        self._publish_command(step)
        return result

    def _publish_command(self, step):
        if self.applied_xy is not None:
            self.last_sent_u = (self.applied_xy-np.asarray(step.target_xy_m))*1000
            self.publish(commanded_xy_mm=(self.applied_xy*1000).tolist(),
                         sent_optimizer_xy_mm=(self.applied_xy*1000).tolist(),
                         correction_mm=self.last_sent_u.tolist(), integral_xy_mm=None,
                         command_offset_xy_mm=self.last_sent_u.tolist())

    def propose_command(self, waypoint, measured, velocity, sample, tolerance_mm, *, rate_elapsed_s=None):
        """Side-effect-free proposal. Clipped increments are discarded, not stored as I."""
        error = np.asarray(waypoint)-measured
        distance = float(np.linalg.norm(error))
        dt = (self.pid.period if self.history_sample is None else
              np.clip(sample-self.history_sample, 0., 2*self.pid.period))
        allowed_speed = self.speed_limit_mm_s * np.clip(distance/self.slowdown_m, .1, 1.)
        speed = float(np.linalg.norm(velocity)*1000)
        braking = speed > allowed_speed or (self.speed_braking and speed > .8*allowed_speed)
        holding = distance*1000 <= tolerance_mm and speed <= min(allowed_speed, self.hold_speed_mm_s)
        old_error = error if self.history_error is None else self.history_error
        old_velocity = velocity if self.history_velocity is None else self.history_velocity
        p = self.pid.kp*(error-old_error)
        i = self.pid.ki*error*dt
        d = -self.pid.kv*(velocity-old_velocity)
        if holding:
            raw = self.applied_xy.copy()
        elif braking:
            # Brake around the sent command, retaining the learned command/position
            # offset. Do not jump back to measured XY or accumulate waypoint pull.
            raw = self.applied_xy + bounded_vector(-self.brake_gain_s*velocity, self.max_brake_m)
        else:
            raw = self.request_xy+p+i+d
        offset = raw-np.asarray(waypoint)
        bounded = np.asarray(waypoint)+bounded_vector(offset, self.command_offset)
        # Do not demand an abrupt return if a new waypoint moves the offset bound.
        elapsed = (dt if self.sent_sample is None else
                   np.clip(sample-self.sent_sample, 0., 2*self.pid.period))
        if rate_elapsed_s is not None:
            elapsed = min(elapsed, max(0., rate_elapsed_s))
        target = self.applied_xy+bounded_vector(bounded-self.applied_xy, self.command_rate*elapsed)
        if holding:
            target = self.applied_xy.copy()
        limits = []
        if np.linalg.norm(offset) > self.command_offset+1e-12:
            limits.append('command offset')
        if np.linalg.norm(bounded-self.applied_xy) > self.command_rate*elapsed+1e-12:
            limits.append('command rate')
        return dict(target=target, error=error, velocity=velocity.copy(), sample=sample,
                    braking=braking, holding=holding, limits=limits,
                    allowed_speed=allowed_speed, p=p, i=i, d=d)

    def _remember(self, proposal):
        self.history_error = proposal['error'].copy()
        self.history_velocity = proposal['velocity'].copy()
        self.history_sample = proposal['sample']

    def _continuous_update(self, step, commander, case_name, gate, active, report,
                           measured, sample, loss_generation, revision):
        if (not self.pid.velocity_ready or self.applied_xy is None
                or sample == self.last_control_sample
                or (self.last_control_at is not None and self.clock()-self.last_control_at < self.pid.period)):
            return
        if self.history_sample is not None and sample-self.history_sample > 1.:
            self._reset_slow_assist()
        self.last_control_at, self.last_control_sample = self.clock(), sample
        proposal = self.propose_command(np.asarray(step.target_xy_m), measured,
            self.pid.velocity, sample, gate.tolerance_mm,
            rate_elapsed_s=self.clock()-self.last_sent_at)
        target = proposal['target']
        self.speed_braking = proposal['braking']
        self._publish_command(step)
        self.publish(mode=self.control_mode, position_error_mm=(proposal['error']*1000).tolist(),
                     velocity_xy_mm_s=(self.pid.velocity*1000).tolist(),
                     speed_braking=self.speed_braking, speed_limit_mm_s=self.speed_limit_mm_s,
                     effective_speed_limit_mm_s=proposal['allowed_speed'],
                     command_increment_mm=((target-self.applied_xy)*1000).tolist(),
                     pid_p_increment_mm=(proposal['p']*1000).tolist(),
                     pid_i_increment_mm=(proposal['i']*1000).tolist(),
                     pid_d_increment_mm=(proposal['d']*1000).tolist(),
                     command_limits=proposal['limits'], proposed_xy_mm=(target*1000).tolist(),
                     observation_age_s=self.clock()-sample)
        if proposal['holding']:
            self.request_xy = self.applied_xy.copy()
            self._remember(proposal)
            self.publish(state='Incremental PID: arrived, holding for settled confirmation')
            return
        feasible = [True]
        accepted = [False]
        def valid():
            with gate.condition:
                return (feasible[0] and active() and self.enabled.is_set() and not self.manual
                        and revision == self.command_revision and not gate.advance_due
                        and gate.loss_generation == loss_generation and gate.positions is not None
                        and len(gate.positions) == 1 and gate.observed_at is not None
                        and self.clock()-gate.observed_at <= 1.
                        and np.linalg.norm(gate.positions[0]-measured) <= .0005
                        and (not self.speed_braking or gate.observed_at == sample))
        def calculated(result):
            fraction = result['allocation_fraction']
            feasible[0] = np.isfinite(fraction) and fraction >= 1.-1e-9
            self.publish(calculated_angles=list(result['angles_deg']), allocation_fraction=fraction)
            # Partial cosine allocation is NOT a linear interpolation of XY.
            # Reject before writing rather than inventing an achieved command.
        def commit(result, sent):
            accepted[0] = True
            self._remember(proposal)
            self.request_xy = target.copy()  # Bounded pending request can cross motor deadband.
            if sent:
                self.applied_xy = target.copy()
                self.sent_sample = sample
                self.initial_sent(result)
                self._publish_command(step)
                self.on_applied(self.last_sent_u.tolist())
                report(result)
            status = 'braking' if self.speed_braking else 'correcting'
            limits = ', '.join(proposal['limits']) or 'none'
            increment = (target-self.applied_xy)*1000 if not sent else np.zeros(2)
            delta = self.snapshot()['command_increment_mm']
            self.publish(state=f'Incremental PID {status}: {"sent" if sent else "motor deadband"}; '
                               f'limits: {limits}; speed cap {proposal["allowed_speed"]:.3g} mm/s; '
                               f'delta XY ({delta[0]:.3f}, {delta[1]:.3f}) mm',
                         pending_increment_mm=increment.tolist())
        self.publish(state='Incremental PID: calculating brake' if self.speed_braking else
                           'Incremental PID: calculating command')
        try:
            self.validate(target)
            key = (step.ratio, step.angle_deg)
            if key not in self.baselines:
                self.baselines[key] = commander.calculate(case_name, self.anchor_xy,
                                                          ratio=step.ratio, angle_deg=step.angle_deg)
            if not valid():
                self.publish(state='Incremental PID: discarded, observation changed')
                return
            commander.move(case_name, target, ratio=step.ratio, angle_deg=step.angle_deg,
                zero_angle_deg=step.zero_angle_deg, active=valid, preview_only=self.preview_only,
                unchanged_angles_deg=self.last_result['angles_deg'], motor_deadband_deg=self.motor_deadband,
                anchored_xy_request=dict(anchor_xy_m=self.anchor_xy, anchor_angles_deg=self.anchor_angles,
                                         baseline_angles_deg=self.baselines[key]['angles_deg']),
                on_calculated=calculated, on_sent=lambda result: commit(result, True),
                on_held=lambda result: commit(result, False))
            if not accepted[0]:
                reason = ('optimizer input limit' if not feasible[0] else
                          'preview' if self.preview_only else 'observation changed')
                self.publish(state='Incremental PID: held; '+reason)
        except ValueError as error:
            self.publish(state=f'Incremental PID: held; {error}')
        finally:
            self.publish(control_cycle_time_s=self.clock()-self.last_control_at)
