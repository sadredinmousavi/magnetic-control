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
    # TARGET_SCHEDULE entries:
    # (start_time, target_pos, eig_ratio, eigvec_angle_rad)
    # Write the angle by hand in degrees and convert inline with np.deg2rad(...).
    "TARGET_SCHEDULE": [
        (0.0, np.array([0.01, 0.00]), 1, np.deg2rad(90.0)),
        (3.0, np.array([0.00, 0.00]), 2, np.deg2rad(0.0)),
        (6.0, np.array([0.00, 0.00]), 2, np.deg2rad(45.0)),
        (9.0, np.array([0.00, 0.00]), 2, np.deg2rad(90.0)),
        (12.0, np.array([0.00, 0.00]), 2, np.deg2rad(135.0)),
        (15.0, np.array([0.00, 0.00]), 2, np.deg2rad(180.0)),
        (18.0, np.array([0.00, 0.00]), 2, np.deg2rad(225.0)),
        (21.0, np.array([0.00, 0.00]), 2, np.deg2rad(270.0)),
        (24.0, np.array([0.00, 0.00]), 2, np.deg2rad(315.0)),
        (27.0, np.array([0.00, 0.00]), 2, np.deg2rad(360.0)),
        (30.0, np.array([0.001, 0.00]), 1, np.deg2rad(90.0)),
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
    "CAPILLARY_SIN_C": 0.1,
    "GAMMA": 0.072,

    # --- Initial Robot Positions ---
    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.08, 0.04], [-0.05, 0.06], [-0.02, 0.08],
        [0.02, 0.08], [0.05, 0.06], [0.08, 0.04],
        [0.00, -0.09],
    ]),

    # --- Time / Solver Parameters ---
    "T_SPAN": (0, 33.0),
    "T_EVAL_POINTS": 300,
    "SOLVER_PROGRESS_INTERVAL": 0.5,

    # --- Payload Parameters ---
    "PAYLOAD_RADIUS": 0.015,
    "PAYLOAD_HEIGHT": 0.001,
    "PAYLOAD_DENSITY": 50,
    "PAYLOAD_DRAG_FACTOR": 200,
    "CONTACT_STIFFNESS": 2e-4,
    "CONTACT_DAMPING": 5e-4,
    "PAYLOAD_CAPILLARY_GAIN": 5e-7,
    "PAYLOAD_CAPILLARY_RANGE": 0.007,
    "PAYLOAD_INITIAL_POS": np.array([0.0, 0.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
