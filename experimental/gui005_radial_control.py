"""GUI005-only compensation for the supplied z=24 mm Helmholtz force fit.

No usage script or shared physics/optimizer imports this module. Positions are
metres, forces newtons, moments A*m^2 and stiffnesses N/m.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from case_loader import build_common_config
from control_workflow import calculate_point_control
from functions_main import (build_actuation_matrix, calculate_potential_hessian,
                            generate_circular_source_positions)


@dataclass(frozen=True)
class HelmholtzRadialModel:
    current_a: float = 12 / 5.31
    mz_a_m2: float | None = None
    moment_sign: int = 1
    center_xy_m: tuple = (0., 0.)

    def __post_init__(self):
        center = np.asarray(self.center_xy_m, float)
        if (center.shape != (2,) or not np.isfinite(center).all()
                or not np.isfinite(self.current_a)
                or self.moment_sign not in (-1, 1)
                or (self.mz_a_m2 is not None and not np.isfinite(self.mz_a_m2))):
            raise ValueError('Invalid GUI005 radial-force settings.')
        object.__setattr__(self, 'center_xy_m', tuple(center))

    def force_and_hessian(self, xy_m, hardware_moment):
        """Return force and -force Jacobian; extrapolate beyond the 40 mm fit."""
        xy = np.asarray(xy_m, float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Radial force needs finite XY metres.')
        q = xy - self.center_xy_m
        s = float(q @ q)
        mz = (self.moment_sign * hardware_moment
              if self.mz_a_m2 is None else self.mz_a_m2)
        scale = mz * self.current_a / (12 / 5.31)
        # F = mz * I/Iref * (a + b*r^2 + c*r^4) * [x,y].
        g = .053654 - 10.7146*s - 1629.01*s*s
        dg_ds = -10.7146 - 2*1629.01*s
        force = scale*g*q
        hessian = -scale*(g*np.eye(2) + 2*dg_ds*np.outer(q, q))
        if not np.isfinite(force).all() or not np.isfinite(hessian).all():
            raise ValueError('GUI005 radial force or stiffness is nonfinite.')
        return force, hessian


def load_radial_model(path=None):
    """Read once at GUI startup; restart after changing settings."""
    path = Path(path) if path is not None else Path(__file__).with_name('gui005_radial_force.json')
    settings = json.loads(path.read_text(encoding='utf-8'))
    enabled = settings.pop('enabled')
    if not isinstance(enabled, bool):
        raise ValueError('GUI005 radial-force enabled must be true or false.')
    model = HelmholtzRadialModel(**settings)
    return model if enabled else None


class RadialControl:
    def __init__(self, model):
        self.model = model

    def signature(self):
        return dict(model='helmholtz_z24mm_polynomial_v1', **asdict(self.model))

    @staticmethod
    def geometry(params, target):
        cfg = build_common_config(dict(params, TARGET_SCHEDULE=[(0., np.asarray(target), 1., None)],
                                       INITIAL_ROBOT_POSITIONS=np.zeros((1, 2))))
        return cfg, generate_circular_source_positions(cfg.NUM_SOURCES, cfg.RADIUS)

    def allocate(self, params, target_xy_m, measured_xy_m, applied_angles_deg,
                 reference_inputs, pid_force_n, *, max_angle_step_deg=10.,
                 ratio=1., angle_deg=None, incremental=False, total_hessian=None):
        """Match total force, retaining reference total stiffness unless incremental."""
        started = time.monotonic()
        cfg, sources = self.geometry(params, target_xy_m)
        target, measured = np.asarray(target_xy_m, float), np.asarray(measured_xy_m, float)
        coil, _ = self.model.force_and_hessian(measured, cfg.M_ROBOT_MAGNITUDE)
        _, coil_h = self.model.force_and_hessian(target, cfg.M_ROBOT_MAGNITUDE)
        angles, requested = np.asarray(applied_angles_deg, float), np.asarray(pid_force_n, float)
        n = cfg.NUM_SOURCES
        if (angles.shape != (n,) or requested.shape != (2,)
                or not np.isfinite(angles).all() or not np.isfinite(requested).all()
                or np.any(angles < 0) or np.any(angles > 90)
                or not np.isfinite(max_angle_step_deg) or not 0 < max_angle_step_deg <= 90):
            raise ValueError('Invalid GUI005 radial force allocation request.')
        applied = np.cos(np.deg2rad(angles))
        if reference_inputs is None:
            reference_inputs = (applied if incremental else
                                self.synthesize(params, target, ratio=ratio, angle_deg=angle_deg)['u_values'])
        reference = np.asarray(reference_inputs, float)
        if (reference.shape != (n,) or not np.isfinite(reference).all()
                or np.any(reference < 0) or np.any(reference > 1)):
            raise ValueError('Invalid radial allocation reference inputs.')
        A = build_actuation_matrix(measured, sources, cfg.C_F)
        Href = calculate_potential_hessian(target, sources, cfg.C_F, reference) + coil_h
        Hgoal = Href if total_hessian is None else np.asarray(total_hessian, float)
        columns = [calculate_potential_hessian(target, sources, cfg.C_F, row) for row in np.eye(n)]
        B = np.asarray([[H[0, 0], H[0, 1], H[1, 1]] for H in columns]).T
        hscale = max(float(np.linalg.norm(Hgoal)), float(np.linalg.norm(coil_h)), 1e-30)
        fscale = max(float(np.linalg.norm(A)), float(np.linalg.norm(requested-coil)), 1e-30)
        lower = np.cos(np.deg2rad(np.minimum(90., angles+max_angle_step_deg)))
        upper = np.cos(np.deg2rad(np.maximum(0., angles-max_angle_step_deg)))
        constraints = []
        if not incremental:
            if np.linalg.eigvalsh(Hgoal)[0] <= 0:
                raise ValueError('Radial reference has no positive total restoring stiffness.')
            magnetic_goal = Hgoal-coil_h
            goal = magnetic_goal[[0, 0, 1], [0, 1, 1]] / hscale
            constraints = [LinearConstraint(B/hscale, goal, goal)]
        M, d = A/fscale, (requested-coil)/fscale

        def objective(c):
            return float(np.sum((M@c-d)**2) + 1e-8*np.sum((c-applied)**2)
                         + 1e-9*np.sum((c-reference)**2))

        def gradient(c):
            return 2*M.T@(M@c-d) + 2e-8*(c-applied) + 2e-9*(c-reference)

        result = minimize(objective, np.clip(applied, lower, upper), jac=gradient,
                          method='SLSQP', bounds=list(zip(lower, upper)), constraints=constraints,
                          options={'maxiter': 400, 'ftol': 1e-13})
        c = np.asarray(result.x, float)
        stiffness_position = measured if incremental else target
        _, stiffness_coil_h = self.model.force_and_hessian(stiffness_position, cfg.M_ROBOT_MAGNITUDE)
        H = calculate_potential_hessian(stiffness_position, sources, cfg.C_F, c) + stiffness_coil_h
        if (not result.success or not np.isfinite(c).all()
                or np.any(c < lower-1e-8) or np.any(c > upper+1e-8)
                or np.linalg.eigvalsh(H)[0] <= 0
                or (not incremental and np.linalg.norm(H-Hgoal) > hscale*1e-5)):
            raise ValueError(f'GUI005 radial allocation infeasible or unstable: {result.message}')
        predicted, actual = A@c+coil, A@applied+coil
        return dict(angles_deg=np.rad2deg(np.arccos(np.clip(c, 0, 1))), u_values=c,
                    reference_inputs=reference, requested_force_n=requested,
                    predicted_force_n=predicted, applied_force_n=actual,
                    force_residual_n=float(np.linalg.norm(predicted-requested)),
                    applied_force_residual_n=float(np.linalg.norm(actual-requested)),
                    initial_force_residual_n=float(np.linalg.norm(actual-requested)),
                    current_force_n=actual, force_adjustment_requested_n=requested-actual,
                    delta_inputs=c-applied,
                    magnetic_force_n=A@c, radial_force_n=coil, H=H,
                    stiffness_eigenvalues=np.linalg.eigvalsh(H),
                    allocation_position_m=measured, allocation_iterations=int(result.nit),
                    allocation_mode='gui005_radial_incremental' if incremental else 'gui005_radial_force',
                    force_compensated=True, radial_model=self.signature(),
                    calculation_time_s=time.monotonic()-started)

    def synthesize(self, params, target_xy_m, *, ratio=1., angle_deg=None):
        cfg, sources = self.geometry(params, target_xy_m)
        self.model.force_and_hessian(target_xy_m, cfg.M_ROBOT_MAGNITUDE)
        nominal = calculate_point_control(params, target_xy_m, ratio=ratio, angle_deg=angle_deg)
        reference = np.asarray(nominal['u_values'], float)
        Hgoal = calculate_potential_hessian(np.asarray(target_xy_m), sources, cfg.C_F, reference)
        result = self.allocate(params, target_xy_m, target_xy_m, nominal['angles_deg'], reference,
                               np.zeros(2), max_angle_step_deg=90., total_hessian=Hgoal)
        self.require_balance(result, np.zeros(2))
        return dict(nominal, **result, nominal_magnetic_inputs=reference,
                    desired_pos=np.asarray(target_xy_m, float))

    @staticmethod
    def require_balance(result, requested):
        if result['force_residual_n'] > max(1e-12, np.linalg.norm(requested)*.01):
            raise ValueError('GUI005 radial force/bias is infeasible at the target; no pose sent.')

    def apply_integral(self, params, target, baseline, integral_force_n):
        bias = np.asarray(integral_force_n, float)
        reference = np.cos(np.deg2rad(baseline['angles_deg']))
        result = self.allocate(params, target, target, baseline['angles_deg'], reference,
                               bias, max_angle_step_deg=90.)
        self.require_balance(result, bias)
        return dict(result, desired_pos=np.asarray(target), baseline=baseline,
                    inherited_integral_force_n=bias.copy())

    def anchored(self, params, target, *, anchor_xy_m, anchor_angles_deg,
                 baseline_angles_deg, ratio=1., angle_deg=None):
        # Baselines originate from this commander's compensated calculate path.
        target, anchor = np.asarray(target, float), np.asarray(anchor_xy_m, float)
        angles, baseline = np.asarray(anchor_angles_deg, float), np.asarray(baseline_angles_deg, float)
        cfg, sources = self.geometry(params, target)
        self.model.force_and_hessian(anchor, cfg.M_ROBOT_MAGNITUDE)
        coil, coil_h = self.model.force_and_hessian(target, cfg.M_ROBOT_MAGNITUDE)
        if (angles.shape != (cfg.NUM_SOURCES,) or baseline.shape != angles.shape
                or not np.isfinite([angles, baseline]).all()
                or np.any(angles < 0) or np.any(angles > 90)
                or np.any(baseline < 0) or np.any(baseline > 90)):
            raise ValueError('Invalid GUI005 anchored model angles.')
        nominal = baseline if np.array_equal(target, anchor) else self.synthesize(
            params, target, ratio=ratio, angle_deg=angle_deg)['angles_deg']
        actual = np.cos(np.deg2rad(angles))
        delta = np.cos(np.deg2rad(nominal))-np.cos(np.deg2rad(baseline))
        fraction = 1.
        for value, change in zip(actual, delta):
            if change > 0:
                fraction = min(fraction, (1-value)/change)
            elif change < 0:
                fraction = min(fraction, -value/change)
        fraction = float(np.clip(fraction, 0., 1.))
        controls = np.clip(actual+fraction*delta, 0., 1.)
        H = calculate_potential_hessian(target, sources, cfg.C_F, controls)+coil_h
        if np.linalg.eigvalsh(H)[0] <= 0:
            raise ValueError('GUI005 anchored pose has no positive total restoring stiffness.')
        return dict(angles_deg=np.rad2deg(np.arccos(controls)), u_values=controls,
                    nominal_angles_deg=nominal, anchor_xy_m=anchor, anchor_angles_deg=angles,
                    optimizer_xy_m=target, allocation_fraction=fraction,
                    allocation_mode='gui005_radial_anchored_xy', force_compensated=False,
                    radial_force_n=coil, radial_model=self.signature(),
                    stiffness_eigenvalues=np.linalg.eigvalsh(H))
