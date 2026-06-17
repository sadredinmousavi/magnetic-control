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


PARAMS = {
    # --- System Geometry ---
    "NUM_SOURCES": 8,
    "RADIUS": 0.25,

    # --- Desired Parameters ---
    # TARGET_SCHEDULE entries:
    # Stable one-equilibrium mode:
    #   (start_time, target_pos, eig_ratio, eigvec_angle_rad)
    # Two-equilibrium mode, without eigenvalue/eigenvector constraints:
    #   (start_time, target_pos_1, target_pos_2)
    # Write the angle by hand in degrees and convert inline with np.deg2rad(...).
    "TARGET_SCHEDULE": [
        # (0.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(80.0))),
        (6.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(0.0))),
        (12.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(45.0))),
        (18.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(90.0))),
        (24.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(135.0))),
        (30.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(180.0))),
        (36.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(225.0))),
        (42.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(270.0))),
        (48.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(315.0))),
        (54.0, *two_equilibrium_points([0.00, 0.00], np.deg2rad(360.0))),
        # (60.0, *two_equilibrium_points([0.001, 0.00], np.deg2rad(80.0))),
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
    "T_SPAN": (0, 66.0),
    "T_EVAL_POINTS": 300,
    "SOLVER_PROGRESS_INTERVAL": 0.5,

    # --- Payload Parameters ---
    # Disabled for this case: kept only to satisfy usage4.py's required keys.
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
