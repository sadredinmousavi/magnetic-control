"""Shared control synthesis and field-preparation workflow."""

from dataclasses import dataclass

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
