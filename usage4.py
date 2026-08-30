"""Optimize controls, simulate microrobot/payload dynamics, and animate them."""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from case_loader import build_common_config, case_output_path, get_case_name_from_argv, load_case, require_keys
from control_workflow import run_control_workflow
from functions_pm_microrobots import animate_trajectories, microrobot_payload_dynamics
from functions_utility import SolveIVPProgress, print_optimization_results


REQUIRED_KEYS = [
    "NUM_SOURCES", "RADIUS", "TARGET_SCHEDULE", "SOURCE_MAGNETIZATION",
    "ROBOT_MAGNETIZATION", "L_SOURCE", "L_ROBOT", "GRID_MIN", "GRID_MAX",
    "RESOLUTION", "DENSITY_NDFEB", "FLUID_VISCOSITY", "ALPHA",
    "CAPILLARY_SIN_C", "GAMMA", "INITIAL_ROBOT_POSITIONS", "T_SPAN",
    "T_EVAL_POINTS", "SOLVER_PROGRESS_INTERVAL", "PAYLOAD_RADIUS",
    "PAYLOAD_HEIGHT", "PAYLOAD_DENSITY", "PAYLOAD_DRAG_FACTOR",
    "CONTACT_STIFFNESS", "CONTACT_DAMPING", "PAYLOAD_CAPILLARY_GAIN",
    "PAYLOAD_CAPILLARY_RANGE", "PAYLOAD_INITIAL_POS", "PAYLOAD_INITIAL_VEL",
]


def _initial_state(cfg):
    state = np.zeros(cfg.NUM_ROBOTS * 4 + 4)
    robot_states = state[:cfg.NUM_ROBOTS * 4].reshape(cfg.NUM_ROBOTS, 4)
    robot_states[:, :2] = np.asarray(cfg.INITIAL_ROBOT_POSITIONS)
    state[-4:-2] = cfg.PAYLOAD_INITIAL_POS
    state[-2:] = cfg.PAYLOAD_INITIAL_VEL
    return state


def main(case_name=None):
    case_name = case_name or get_case_name_from_argv()
    params = load_case(case_name)
    require_keys(params, REQUIRED_KEYS, case_name)
    cfg = build_common_config(params)

    print(f"Loaded case: {case_name}")
    workflow = run_control_workflow(cfg, params, report=print_optimization_results)
    progress = SolveIVPProgress(cfg.T_SPAN, min_interval=cfg.SOLVER_PROGRESS_INTERVAL)

    def dynamics(t, state):
        progress.update(t)
        return microrobot_payload_dynamics(
            t, state, workflow.source_positions, workflow.target_controls,
            cfg.M_SOURCE_MAGNITUDE, cfg.M_ROBOT_MAGNITUDE,
            cfg.ROBOT_MASS, cfg.FLUID_DRAG, cfg.ROBOT_RADIUS,
            cfg.CAPILLARY_SIN_C, cfg.GAMMA, cfg.PAYLOAD_RADIUS,
            cfg.PAYLOAD_MASS, cfg.PAYLOAD_DRAG, cfg.CONTACT_STIFFNESS,
            cfg.CONTACT_DAMPING, cfg.PAYLOAD_CAPILLARY_GAIN,
            cfg.PAYLOAD_CAPILLARY_RANGE, cfg.PAYLOAD_CAPILLARY_CUTOFF,
            cfg.USE_OVERDAMPED_DYNAMICS, cfg.DYNAMICS_SPEEDUP,
            cfg.WALL_SEGMENTS, cfg.WALL_STIFFNESS, cfg.WALL_DAMPING,
            cfg.WALL_INTERACTION_RANGE,
        )

    solution = solve_ivp(
        dynamics, cfg.T_SPAN, _initial_state(cfg), t_eval=cfg.T_EVAL,
        method=params.get("SOLVER_METHOD", "RK45"),
        rtol=params.get("SOLVER_RTOL", 1e-5),
        atol=params.get("SOLVER_ATOL", 1e-8),
    )
    progress.finish(solution.message)
    if not solution.success:
        raise RuntimeError(f"Dynamics integration failed: {solution.message}")

    video_filename = Path("outputs") / case_output_path(case_name).with_suffix(".mp4")
    video_filename.parent.mkdir(parents=True, exist_ok=True)
    animate_trajectories(
        solution.t, solution.y, workflow.source_positions, cfg.TARGET_SCHEDULE,
        cfg.GRID_MIN, cfg.GRID_MAX, field_data=workflow.field_data,
        draw_contour=params.get("ANIMATION_DRAW_CONTOUR", True),
        draw_streamlines=params.get("ANIMATION_DRAW_STREAMLINES", False),
        draw_quiver=params.get("ANIMATION_DRAW_QUIVER", False),
        draw_sources=True, draw_all_targets=False, draw_active_target=True,
        draw_target_trajectory=params.get("ANIMATION_DRAW_TARGET_TRAJECTORY", False),
        plot_trajectories=params.get("ANIMATION_DRAW_TRAJECTORIES", False),
        plot_microrobots=True,
        robot_marker_size=params.get("ANIMATION_ROBOT_MARKER_SIZE", 55),
        payload_radius=cfg.PAYLOAD_RADIUS,
        clip_field_to_dish=params.get("PLOT_FIELD_INSIDE_DISH", False),
        dish_center=params.get("DISH_CENTER", (0.0, 0.0)),
        dish_radius=params.get("DISH_RADIUS"),
        dish_outside_fade_alpha=params.get("DISH_OUTSIDE_FADE_ALPHA", 0.86),
        show_magnet_moment_vectors=params.get(
            "SHOW_EXTERNAL_MAGNET_MOMENT_VECTORS", False
        ),
        magnet_moment_arrow_length=params.get("MAGNET_MOMENT_ARROW_LENGTH", 0.035),
        magnet_moment_arrow_color=params.get("MAGNET_MOMENT_ARROW_COLOR", "#d1495b"),
        wall_segments=cfg.WALL_SEGMENTS,
        save_video=params.get("SAVE_VIDEO", True), video_name=video_filename,
        video_dpi=params.get("VIDEO_DPI", 160),
        video_fps=params.get("VIDEO_FPS", 30),
        video_crf=params.get("VIDEO_CRF", 18),
        figure_size=params.get("ANIMATION_FIGURE_SIZE", (8, 8)),
    )
    return solution


if __name__ == "__main__":
    main()
