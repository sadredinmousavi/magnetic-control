import importlib
import sys
from types import SimpleNamespace

import numpy as np
from scipy.constants import mu_0


def get_case_name_from_argv(default_case_name):
    if len(sys.argv) > 1:
        return sys.argv[1]
    return default_case_name


def load_case(case_name):
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


def build_common_config(params):
    num_sources = params["NUM_SOURCES"]
    radius = params["RADIUS"]

    target_schedule = params["TARGET_SCHEDULE"]

    m_saturation = params["M_SATURATION"]
    l_source = params["L_SOURCE"]
    l_robot = params["L_ROBOT"]
    magnetization = params["MAGNETIZATION"]
    m_source_magnitude = (l_source**3) * m_saturation
    m_robot_magnitude = (l_robot**3) * m_saturation
    c_f = (3 * mu_0 / (4 * np.pi)) * m_source_magnitude * m_robot_magnitude

    grid_min = params["GRID_MIN"]
    grid_max = params["GRID_MAX"]
    resolution = params["RESOLUTION"]
    initial_robot_positions = params["INITIAL_ROBOT_POSITIONS"]

    density_ndfeb = params.get("DENSITY_NDFEB")
    fluid_viscosity = params.get("FLUID_VISCOSITY")
    alpha = params.get("ALPHA")
    capillary_sin_c = params.get("CAPILLARY_SIN_C")
    gamma = params.get("GAMMA")

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
        M_SATURATION=m_saturation,
        L_SOURCE=l_source,
        L_ROBOT=l_robot,
        MAGNETIZATION=magnetization,
        M_SOURCE_MAGNITUDE=m_source_magnitude,
        M_ROBOT_MAGNITUDE=m_robot_magnitude,
        C_F=c_f,
        GRID_MIN=grid_min,
        GRID_MAX=grid_max,
        RESOLUTION=resolution,
        INITIAL_ROBOT_POSITIONS=initial_robot_positions,
        DENSITY_NDFEB=density_ndfeb,
        FLUID_VISCOSITY=fluid_viscosity,
        ALPHA=alpha,
        CAPILLARY_SIN_C=capillary_sin_c,
        GAMMA=gamma,
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
