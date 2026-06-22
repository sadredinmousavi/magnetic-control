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


PARAMS = {
    # --- System Geometry ---
    "NUM_SOURCES": 8,
    "RADIUS": 0.25,

    # --- Desired Parameters ---
    # The equilibrium center first gathers robots away from the payload, then
    # visits the payload at the origin, transports it, and finally retreats.
    "TARGET_SCHEDULE": [
        (6.0, np.array([-0.105, -0.0375]), 1, np.deg2rad(0.0)),
        (12.0, np.array([-0.07, -0.025]), 1, np.deg2rad(0.0)),
        (18.0, np.array([-0.035, -0.0125]), 1, np.deg2rad(0.0)),
        (24.0, np.array([0.00, 0.00]), 1, np.deg2rad(0.0)),
        (30.0, np.array([0.01, 0.00]), 1, np.deg2rad(0.0)),
        (36.0, np.array([0.01, 0.00]), 1, np.deg2rad(0.0)),
        (42.0, np.array([0.02, 0.005]), 1, np.deg2rad(0.0)),
        (48.0, np.array([0.03, 0.015]), 1, np.deg2rad(0.0)),
        (54.0, np.array([0.04, 0.03]), 1, np.deg2rad(0.0)),
        (60.0, np.array([0.035, 0.045]), 1, np.deg2rad(0.0)),
        (66.0, np.array([0.02, 0.055]), 1, np.deg2rad(0.0)),
        (72.0, np.array([0.00, 0.06]), 1, np.deg2rad(0.0)),
        (78.0, np.array([-0.02, 0.055]), 1, np.deg2rad(0.0)),
        (84.0, np.array([-0.035, 0.045]), 1, np.deg2rad(0.0)),
        (90.0, np.array([-0.045, 0.03]), 1, np.deg2rad(0.0)),
        (96.0, np.array([-0.04, 0.01]), 1, np.deg2rad(0.0)),
        (102.0, np.array([-0.025, -0.005]), 1, np.deg2rad(0.0)),
        (108.0, np.array([0.00, -0.015]), 1, np.deg2rad(0.0)),
        (114.0, np.array([0.025, -0.01]), 1, np.deg2rad(0.0)),
        (120.0, np.array([0.045, 0.00]), 1, np.deg2rad(0.0)),
        (126.0, np.array([0.06, 0.015]), 1, np.deg2rad(0.0)),
        (132.0, np.array([0.07, 0.035]), 1, np.deg2rad(0.0)),
        (138.0, np.array([0.065, 0.055]), 1, np.deg2rad(0.0)),
        (144.0, np.array([0.05, 0.07]), 1, np.deg2rad(0.0)),
        (150.0, np.array([0.03, 0.08]), 1, np.deg2rad(0.0)),
        (156.0, np.array([0.06, 0.06]), 1, np.deg2rad(0.0)),
        (162.0, np.array([-0.10, 0.00]), 1, np.deg2rad(0.0)),
    ],

    # --- Magnetic Properties ---
    "SOURCE_MAGNETIZATION": 1000e3,
    "ROBOT_MAGNETIZATION": 868e3,
    "L_SOURCE": 0.02,
    "L_ROBOT": 0.0005,

    # --- Observation Space (Grid) ---
    "GRID_MIN": -0.3,
    "GRID_MAX": 0.3,
    "RESOLUTION": 50,

    # --- Microrobot Dynamics Parameters ---
    "DENSITY_NDFEB": 7500,
    "FLUID_VISCOSITY": 0.001,
    "ALPHA": 0.3,
    "CAPILLARY_SIN_C": 0.01,
    "GAMMA": 0.072,

    # --- Initial Robot Positions ---
    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),

    # --- Time / Solver Parameters ---
    "T_SPAN": (0, 180.0),
    "T_EVAL_POINTS": 700,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "USE_OVERDAMPED_DYNAMICS": False,
    "DYNAMICS_SPEEDUP": 1.0,
    "SOLVER_RTOL": 1e-4,
    "SOLVER_ATOL": 1e-7,

    # --- Payload Parameters ---
    "PAYLOAD_RADIUS": 0.015,
    "PAYLOAD_HEIGHT": 0.001,
    "PAYLOAD_DENSITY": 50,
    "PAYLOAD_DRAG_FACTOR": 50,
    "CONTACT_STIFFNESS": 2e-5,
    "CONTACT_DAMPING": 5e-5,
    "PAYLOAD_CAPILLARY_GAIN": 5e-7,
    "PAYLOAD_CAPILLARY_RANGE": 0.007,
    "PAYLOAD_INITIAL_POS": np.array([0.0, 0.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
