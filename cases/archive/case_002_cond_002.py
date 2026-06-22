import numpy as np

# =============================================================================
# NOMENCLATURE & UNITS REFERENCE
# =============================================================================
# u (control_inputs_u)  : Dimensionless [-1, 1]. The COSINE of the magnet angle (u = cos(theta)).
# theta_rad / theta_deg : Radians / Degrees. The actual physical angle of the magnet.
# pos / r / p1 / p2     : Meters (m). Spatial position vectors (x, y, z).
# net_force / F_m       : Newtons (N). The magnetic force exerted on the microrobot.
# C_F                   : N.m^4. Lumped magnetic force constant (permeability & moments).
# mu_0                  : T.m/A (or H/m). Vacuum permeability.
# m_ba / M              : A.m^2. Magnetic dipole moment.
# B                     : Tesla (T). Magnetic flux density (magnetic field).
# U (potential_energy)  : Joules (J). Magnetic potential energy landscape.
# H (Hessian)           : N/m. Spatial derivative of force (magnetic stiffness matrix).
# eigenvalues           : N/m. Eigenvalues of Hessian (negative = stable restoring force).
# =============================================================================

def two_equilibrium_points(center, angle_rad, radius=0.05):
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    center = np.array(center)
    return center + radius * direction, center - radius * direction

SHAPE_DURATION = 30.0

PARAMS = {
    # --- System Geometry ---
    "NUM_SOURCES": 8,
    "RADIUS": 0.25,

    # --- Desired Parameters ---
    "TARGET_SCHEDULE": [
        # (0.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(80.0))),
        (0 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(0.0))),
        (1 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(45.0))),
        (2 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(90.0))),
        (3 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(135.0))),
        (4 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(180.0))),
        (5 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(225.0))),
        (6 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(270.0))),
        (7 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(315.0))),
        (8 * SHAPE_DURATION, *two_equilibrium_points([0.00, 0.00], np.deg2rad(360.0))),
        (9 * SHAPE_DURATION, *two_equilibrium_points([0.001, 0.00], np.deg2rad(80.0))),
    ],
    "TWO_EQUILIBRIUM_SOLVER": "stable",#plain
    "STABILITY_TRACE_MARGIN" : 1e-8,
    "STABILITY_DET_MARGIN" : 1e-14,

    # --- Magnetic Properties ---
    "SOURCE_MAGNETIZATION": 1000e3,
    "ROBOT_MAGNETIZATION": 3.0e4,  # Effective magnetization for PDMS + 20% SPION
    "L_SOURCE": 0.02,
    "L_ROBOT": 0.00025,
    "ROBOT_HEIGHT": 0.00025,

    # --- Observation Space (Grid) ---
    "GRID_MIN": -0.3,
    "GRID_MAX": 0.3,
    "RESOLUTION": 50,

    # --- Microrobot Dynamics Parameters ---
    "DENSITY_NDFEB": 1200,
    "FLUID_VISCOSITY": 0.001,
    "ALPHA": 0.3,
    "CAPILLARY_SIN_C": 0.01,
    "GAMMA": 0.072,

    # --- Initial Robot Positions ---
    # Generate 30 robots in a grid pattern for simplicity
    "INITIAL_ROBOT_POSITIONS": np.array([
        [x, y] for x in np.linspace(-0.05, 0.05, 6)
                for y in np.linspace(-0.05, 0.05, 5)
    ]),  # 6x5=30

    # --- Time / Solver Parameters ---
    "T_SPAN": (0, 13 * SHAPE_DURATION),
    "T_EVAL_POINTS": 300,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "USE_OVERDAMPED_DYNAMICS": True,
    "DYNAMICS_SPEEDUP": 5.0,
    "SOLVER_RTOL": 1e-4,
    "SOLVER_ATOL": 1e-7,

    # --- Payload Parameters ---
    "PAYLOAD_RADIUS": 1e-12,
    "PAYLOAD_HEIGHT": 1.0,
    "PAYLOAD_DENSITY": 1.0,
    "PAYLOAD_DRAG_FACTOR": 0.0,
    "CONTACT_STIFFNESS": 0.0,
    "CONTACT_DAMPING": 0.0,
    "PAYLOAD_CAPILLARY_GAIN": 0.0,
    "PAYLOAD_CAPILLARY_RANGE": 1.0,
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
