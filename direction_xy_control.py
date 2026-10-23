"""Optional XY PID that gives the current position-error direction priority."""
import numpy as np

from functions_main import MagneticForcePID
from control_workflow import OptimizerXYFeedback


class DirectionWeightedPID(MagneticForcePID):
    def __init__(self, *, sideways_weight=.2, opposing_decay_s=2., **kwargs):
        if not np.isfinite(sideways_weight) or not 0 <= sideways_weight <= 1:
            raise ValueError('Sideways integral weight must be between 0 and 1.')
        if not np.isfinite(opposing_decay_s) or opposing_decay_s <= 0:
            raise ValueError('Opposing integral decay time must be positive.')
        self.sideways_weight = float(sideways_weight)
        self.opposing_decay_s = float(opposing_decay_s)
        super().__init__(**kwargs)

    def propose(self, desired_m, *, tolerance_m=0.):
        proposal = super().propose(desired_m, tolerance_m=tolerance_m)
        error, dt = proposal['error_m'], proposal['dt_s']
        length = float(np.linalg.norm(error))
        memory = self.integral_force.copy()
        direction = None
        # No direction is defined at zero; avoid amplifying sub-micron noise.
        if length > max(tolerance_m, 1e-6):
            direction = error / length
            opposing = min(0., float(memory @ direction)) * direction
            memory -= opposing * (-np.expm1(-dt / self.opposing_decay_s))
        increment = self.ki * error * dt if length > tolerance_m else np.zeros(2)
        candidate = np.clip(memory + increment, -self.integral_limit, self.integral_limit)
        effective = candidate.copy()
        if direction is not None:
            parallel = float(candidate @ direction) * direction
            effective = parallel + self.sideways_weight * (candidate - parallel)
            effective = np.clip(effective, -self.integral_limit, self.integral_limit)
        control_error = error * max(0., 1 - tolerance_m / max(length, 1e-30))
        proposal.update(integral_force_n=candidate, decayed_integral=memory,
                        pid_force_n=self.kp * control_error + effective - self.kv * self.velocity)
        return proposal

    def commit(self, proposal, requested_force_n, achieved_force_n):
        # Accepted allocations unwind old memory even if saturation prevents
        # accumulating more. Discarded proposals never change integral state.
        self.integral_force = proposal['decayed_integral'].copy()
        super().commit(proposal, requested_force_n, achieved_force_n)


class DirectionWeightedXYFeedback(OptimizerXYFeedback):
    control_mode = 'anchored_xy_optimizer_direction_weighted_pid'
