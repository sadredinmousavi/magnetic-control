import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.constants import mu_0


def select_file_from_dialog(
    initial_dir,
    title,
    filetypes,
    default_path=None,
    cancel_message=None
):
    """Return a selected file path, or a default path when the dialog is unavailable."""
    initial_dir = Path(initial_dir).resolve()

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()

        selected_file = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=filetypes,
        )
        root.destroy()

        if selected_file:
            return Path(selected_file)
    except Exception as exc:
        if default_path is not None:
            print(f"Could not open file dialog ({exc}). Using default: {default_path}")
            return Path(default_path)

        raise RuntimeError(f"Could not open file dialog: {exc}") from exc

    if default_path is not None:
        print(f"{cancel_message or 'No file selected.'} Using default: {default_path}")
        return Path(default_path)

    raise SystemExit(cancel_message or "No file selected.")


def select_case_name_from_dialog(default_case_name):
    case_path = select_file_from_dialog(
        initial_dir=Path.cwd() / "cases",
        title="Select case file",
        filetypes=[
            ("Case files", "case_*.py"),
            ("Python files", "*.py"),
            ("All files", "*.*"),
        ],
        default_path=Path.cwd() / "cases" / f"{default_case_name}.py",
        cancel_message="No case file selected.",
    )

    if case_path.name == "__init__.py":
        raise ValueError("__init__.py is not a runnable case file.")

    return case_path.stem


def get_case_name_from_argv(default_case_name):
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).stem

    return select_case_name_from_dialog(default_case_name)


def load_case(case_name):
    case_name = Path(case_name).stem
    module = importlib.import_module(f"cases.{case_name}")

    if not hasattr(module, "PARAMS"):
        raise ValueError(f"Case module 'cases.{case_name}' must define PARAMS.")

    return module.PARAMS


def require_keys(params, required_keys, case_name):
    missing = [key for key in required_keys if key not in params]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Case '{case_name}' is missing required parameters: {missing_str}"
        )


def unpack_target_schedule_entry(entry):
    """
    Returns schedule fields for supported target entry formats.

    Stable single-equilibrium entry:
        (start_time, target_pos, eig_ratio, eigvec_angle_rad)

    Two-equilibrium entry:
        (start_time, target_pos_1, target_pos_2)
    """
    if len(entry) == 4:
        start_time, target_pos, eig_ratio, eigvec_angle_rad = entry
        return start_time, target_pos, None, eig_ratio, eigvec_angle_rad

    if len(entry) == 3:
        start_time, target_pos_1, target_pos_2 = entry
        return start_time, target_pos_1, target_pos_2, None, None

    raise ValueError(
        "TARGET_SCHEDULE entries must be either "
        "(start_time, target_pos, eig_ratio, eigvec_angle_rad) or "
        "(start_time, target_pos_1, target_pos_2)."
    )


def build_common_config(params):
    num_sources = params["NUM_SOURCES"]
    radius = params["RADIUS"]

    target_schedule = params["TARGET_SCHEDULE"]

    source_magnetization = params["SOURCE_MAGNETIZATION"]
    robot_magnetization = params["ROBOT_MAGNETIZATION"]
    l_source = params["L_SOURCE"]
    l_robot = params["L_ROBOT"]
    m_source_magnitude = (l_source**3) * source_magnetization
    m_robot_magnitude = (l_robot**3) * robot_magnetization
    c_f = (3 * mu_0 / (4 * np.pi)) * m_source_magnitude * m_robot_magnitude
    reference_source_moment = (0.02**3) * 1000e3
    reference_robot_moment = (0.0005**3) * 868e3
    reference_c_f = (3 * mu_0 / (4 * np.pi)) * reference_source_moment * reference_robot_moment
    stiffness_scale = c_f / reference_c_f
    stability_trace_margin = params.get(
        "STABILITY_TRACE_MARGIN",
        1e-6 * stiffness_scale
    )
    stability_det_margin = params.get(
        "STABILITY_DET_MARGIN",
        1e-12 * stiffness_scale**2
    )

    grid_min = params["GRID_MIN"]
    grid_max = params["GRID_MAX"]
    resolution = params["RESOLUTION"]
    initial_robot_positions = params["INITIAL_ROBOT_POSITIONS"]

    density_ndfeb = params.get("DENSITY_NDFEB")
    fluid_viscosity = params.get("FLUID_VISCOSITY")
    alpha = params.get("ALPHA")
    capillary_sin_c = params.get("CAPILLARY_SIN_C")
    gamma = params.get("GAMMA")
    use_overdamped_dynamics = params.get("USE_OVERDAMPED_DYNAMICS", False)
    dynamics_speedup = params.get("DYNAMICS_SPEEDUP", 1.0)
    wall_segments = params.get("WALL_SEGMENTS", [])
    wall_stiffness = params.get("WALL_STIFFNESS", 0.0)
    wall_damping = params.get("WALL_DAMPING", 0.0)
    wall_interaction_range = params.get("WALL_INTERACTION_RANGE", 0.0)

    robot_volume = l_robot**3
    robot_radius = l_robot / 2
    robot_mass = None
    fluid_drag = None
    if density_ndfeb is not None:
        robot_mass = density_ndfeb * robot_volume
    if fluid_viscosity is not None and alpha is not None:
        fluid_drag = fluid_viscosity * 6 * robot_radius * alpha * np.pi

    t_span = params.get("T_SPAN")
    t_eval_points = params.get("T_EVAL_POINTS")
    t_eval = None
    if t_span is not None and t_eval_points is not None:
        t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)
    solver_progress_interval = params.get("SOLVER_PROGRESS_INTERVAL")

    payload_radius = params.get("PAYLOAD_RADIUS")
    payload_height = params.get("PAYLOAD_HEIGHT")
    payload_density = params.get("PAYLOAD_DENSITY")
    payload_drag_factor = params.get("PAYLOAD_DRAG_FACTOR")
    contact_stiffness = params.get("CONTACT_STIFFNESS")
    contact_damping = params.get("CONTACT_DAMPING")
    payload_capillary_gain = params.get("PAYLOAD_CAPILLARY_GAIN")
    payload_capillary_range = params.get("PAYLOAD_CAPILLARY_RANGE")
    payload_initial_pos = params.get("PAYLOAD_INITIAL_POS")
    payload_initial_vel = params.get("PAYLOAD_INITIAL_VEL")

    payload_volume = None
    payload_mass = None
    payload_drag = None
    payload_capillary_cutoff = None
    if payload_radius is not None and payload_height is not None:
        payload_volume = np.pi * payload_radius**2 * payload_height
    if payload_volume is not None and payload_density is not None:
        payload_mass = payload_density * payload_volume
    if fluid_drag is not None and payload_drag_factor is not None:
        payload_drag = fluid_drag * payload_drag_factor
    if payload_capillary_range is not None:
        payload_capillary_cutoff = 3 * payload_capillary_range

    num_robots = len(initial_robot_positions)

    return SimpleNamespace(
        NUM_SOURCES=num_sources,
        RADIUS=radius,
        TARGET_SCHEDULE=target_schedule,
        SOURCE_MAGNETIZATION=source_magnetization,
        ROBOT_MAGNETIZATION=robot_magnetization,
        L_SOURCE=l_source,
        L_ROBOT=l_robot,
        # Legacy aliases for older plotting/debug code that may inspect CFG.
        M_SATURATION=robot_magnetization,
        MAGNETIZATION=robot_magnetization,
        M_SOURCE_MAGNITUDE=m_source_magnitude,
        M_ROBOT_MAGNITUDE=m_robot_magnitude,
        C_F=c_f,
        STABILITY_TRACE_MARGIN=stability_trace_margin,
        STABILITY_DET_MARGIN=stability_det_margin,
        GRID_MIN=grid_min,
        GRID_MAX=grid_max,
        RESOLUTION=resolution,
        INITIAL_ROBOT_POSITIONS=initial_robot_positions,
        DENSITY_NDFEB=density_ndfeb,
        FLUID_VISCOSITY=fluid_viscosity,
        ALPHA=alpha,
        CAPILLARY_SIN_C=capillary_sin_c,
        GAMMA=gamma,
        USE_OVERDAMPED_DYNAMICS=use_overdamped_dynamics,
        DYNAMICS_SPEEDUP=dynamics_speedup,
        WALL_SEGMENTS=wall_segments,
        WALL_STIFFNESS=wall_stiffness,
        WALL_DAMPING=wall_damping,
        WALL_INTERACTION_RANGE=wall_interaction_range,
        ROBOT_VOLUME=robot_volume,
        ROBOT_RADIUS=robot_radius,
        ROBOT_MASS=robot_mass,
        FLUID_DRAG=fluid_drag,
        NUM_ROBOTS=num_robots,
        T_SPAN=t_span,
        T_EVAL_POINTS=t_eval_points,
        T_EVAL=t_eval,
        SOLVER_PROGRESS_INTERVAL=solver_progress_interval,
        PAYLOAD_RADIUS=payload_radius,
        PAYLOAD_HEIGHT=payload_height,
        PAYLOAD_DENSITY=payload_density,
        PAYLOAD_DRAG_FACTOR=payload_drag_factor,
        PAYLOAD_VOLUME=payload_volume,
        PAYLOAD_MASS=payload_mass,
        PAYLOAD_DRAG=payload_drag,
        CONTACT_STIFFNESS=contact_stiffness,
        CONTACT_DAMPING=contact_damping,
        PAYLOAD_CAPILLARY_GAIN=payload_capillary_gain,
        PAYLOAD_CAPILLARY_RANGE=payload_capillary_range,
        PAYLOAD_CAPILLARY_CUTOFF=payload_capillary_cutoff,
        PAYLOAD_INITIAL_POS=payload_initial_pos,
        PAYLOAD_INITIAL_VEL=payload_initial_vel,
    )
