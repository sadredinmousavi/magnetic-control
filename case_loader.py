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

    raise SystemExit(cancel_message or "No file selected.")


def normalize_case_name(case_name):
    case_text = str(case_name).strip()
    case_text = case_text.replace("\\", "/")

    if case_text.endswith(".py"):
        path = Path(case_text)
        parts = path.with_suffix("").parts

        if "cases" in parts:
            case_parts = parts[parts.index("cases") + 1:]
            return ".".join(case_parts)

        if len(parts) >= 2:
            return ".".join(parts[-2:])

        return path.stem

    case_text = case_text.replace("/", ".")

    if "." in case_text:
        return case_text

    return case_text


def case_output_path(case_name):
    return Path(case_output_name(case_name))


def case_output_name(case_name):
    return "_".join(normalize_case_name(case_name).split("."))


def select_case_name_from_dialog():
    case_path = select_file_from_dialog(
        initial_dir=Path.cwd() / "cases",
        title="Select case file",
        filetypes=[
            ("Case files", "*.py"),
            ("Python files", "*.py"),
            ("All files", "*.*"),
        ],
        default_path=None,
        cancel_message="No case file selected.",
    )

    if case_path.name == "__init__.py":
        raise ValueError("__init__.py is not a runnable case file.")

    relative_path = case_path.resolve().relative_to((Path.cwd() / "cases").resolve())
    return normalize_case_name(relative_path)


def get_case_name_from_argv():
    if len(sys.argv) > 1:
        return normalize_case_name(sys.argv[1])

    return select_case_name_from_dialog()


def load_case(case_name):
    case_name = normalize_case_name(case_name)
    module = importlib.import_module(f"cases.{case_name}")

    if not hasattr(module, "PARAMS"):
        raise ValueError(f"Case module 'cases.{case_name}' must define PARAMS.")

    params = dict(module.PARAMS)

    if "." in case_name:
        base_module_name = ".".join(case_name.split(".")[:-1] + ["case"])
        base_module = importlib.import_module(f"cases.{base_module_name}")

        if not hasattr(base_module, "PARAMS"):
            raise ValueError(
                f"Base case module 'cases.{base_module_name}' must define PARAMS."
            )

        merged_params = dict(base_module.PARAMS)
        merged_params.update(params)
        return merged_params

    return params


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

    Four-equilibrium entry:
        (start_time, target_pos_1, target_pos_2, target_pos_3, target_pos_4)
    """
    if len(entry) == 5:
        start_time, target_pos_1, target_pos_2, target_pos_3, target_pos_4 = entry
        additional_positions = [target_pos_2, target_pos_3, target_pos_4]
        return start_time, target_pos_1, additional_positions, None, None

    if len(entry) == 4:
        start_time, target_pos, eig_ratio, eigvec_angle_rad = entry
        return start_time, target_pos, None, eig_ratio, eigvec_angle_rad

    if len(entry) == 3:
        start_time, target_pos_1, target_pos_2 = entry
        return start_time, target_pos_1, target_pos_2, None, None

    raise ValueError(
        "TARGET_SCHEDULE entries must be either "
        "(start_time, target_pos, eig_ratio, eigvec_angle_rad) or "
        "(start_time, target_pos_1, target_pos_2) or "
        "(start_time, target_pos_1, target_pos_2, target_pos_3, target_pos_4)."
    )


def validate_target_schedule(schedule):
    """Validate supported entries and require increasing start times."""
    if not schedule:
        raise ValueError("TARGET_SCHEDULE must contain at least one entry.")

    previous_start = None
    for index, entry in enumerate(schedule):
        start_time, target, additional, ratio, angle = unpack_target_schedule_entry(entry)
        if previous_start is not None and start_time <= previous_start:
            raise ValueError("TARGET_SCHEDULE start times must be strictly increasing.")
        previous_start = start_time

        positions = [target]
        if additional is not None:
            positions.extend(additional if isinstance(additional, list) else [additional])
        for position in positions:
            if np.asarray(position).shape != (2,):
                raise ValueError(
                    f"TARGET_SCHEDULE entry {index} contains a non-2D position."
                )
        if ratio is not None and ratio <= 0:
            raise ValueError(f"TARGET_SCHEDULE entry {index} has a non-positive eig_ratio.")


def build_common_config(params):
    num_sources = params["NUM_SOURCES"]
    radius = params["RADIUS"]

    if not isinstance(num_sources, (int, np.integer)) or num_sources <= 0:
        raise ValueError("NUM_SOURCES must be a positive integer.")
    if radius <= 0:
        raise ValueError("RADIUS must be positive.")

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
    if grid_min >= grid_max:
        raise ValueError("GRID_MIN must be less than GRID_MAX.")
    if not isinstance(resolution, (int, np.integer)) or resolution < 2:
        raise ValueError("RESOLUTION must be an integer of at least 2.")
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
    wall_recovery_depth = params.get("WALL_RECOVERY_DEPTH", 0.0)

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
        if len(t_span) != 2 or t_span[0] >= t_span[1]:
            raise ValueError("T_SPAN must contain increasing start and end times.")
        if not isinstance(t_eval_points, (int, np.integer)) or t_eval_points < 2:
            raise ValueError("T_EVAL_POINTS must be an integer of at least 2.")
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
        WALL_RECOVERY_DEPTH=wall_recovery_depth,
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
