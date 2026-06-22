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

SHAPE_DURATION = 40.0

PARAMS = {
    # --- System Geometry ---
    "NUM_SOURCES": 8,
    "RADIUS": 0.25,

    # --- Desired Parameters ---
    # Move through a compact zig-zag channel with several knees. The path stays
    # near the workspace center, away from the PMs at radius 0.25 m.
    "TARGET_SCHEDULE": [
        (0 * SHAPE_DURATION, np.array([-0.115, -0.055]), 1, np.deg2rad(0.0)),
        (1 * SHAPE_DURATION, np.array([-0.075, -0.055]), 1, np.deg2rad(0.0)),
        (2 * SHAPE_DURATION, np.array([-0.075, 0.040]), 1, np.deg2rad(0.0)),
        (3 * SHAPE_DURATION, np.array([-0.020, 0.040]), 1, np.deg2rad(0.0)),
        (4 * SHAPE_DURATION, np.array([-0.020, -0.040]), 1, np.deg2rad(0.0)),
        (5 * SHAPE_DURATION, np.array([0.040, -0.040]), 1, np.deg2rad(0.0)),
        (6 * SHAPE_DURATION, np.array([0.040, 0.050]), 1, np.deg2rad(0.0)),
        (7 * SHAPE_DURATION, np.array([0.085, 0.050]), 1, np.deg2rad(0.0)),
        (8 * SHAPE_DURATION, np.array([0.115, 0.050]), 1, np.deg2rad(0.0)),
    ],

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
    "INITIAL_ROBOT_POSITIONS": np.array([
        [x, y] for x in np.linspace(-0.125, -0.105, 6)
                for y in np.linspace(-0.065, -0.045, 5)
    ]),

    # --- Wall / Corridor Parameters ---
    # Soft wall segments form a compact corridor with open entrance/exit.
    # It is not a hard collision solver: walls repel robots when they approach.
    "WALL_SEGMENTS": [
        # Entrance horizontal section.
        (np.array([-0.115, -0.090]), np.array([-0.055, -0.090])),
        (np.array([-0.115, -0.020]), np.array([-0.055, -0.020])),
        # First vertical climb.
        (np.array([-0.110, -0.055]), np.array([-0.110, 0.040])),
        (np.array([-0.040, -0.055]), np.array([-0.040, 0.040])),
        # Upper horizontal section.
        (np.array([-0.075, 0.005]), np.array([-0.020, 0.005])),
        (np.array([-0.075, 0.075]), np.array([-0.020, 0.075])),
        # Middle vertical descent.
        (np.array([-0.055, 0.040]), np.array([-0.055, -0.040])),
        (np.array([0.015, 0.040]), np.array([0.015, -0.040])),
        # Lower horizontal section.
        (np.array([-0.020, -0.075]), np.array([0.040, -0.075])),
        (np.array([-0.020, -0.005]), np.array([0.040, -0.005])),
        # Final vertical climb.
        (np.array([0.005, -0.040]), np.array([0.005, 0.050])),
        (np.array([0.075, -0.040]), np.array([0.075, 0.050])),
        # Exit horizontal section.
        (np.array([0.040, 0.015]), np.array([0.115, 0.015])),
        (np.array([0.040, 0.085]), np.array([0.115, 0.085])),
    ],
    "WALL_STIFFNESS": 5e-6,
    "WALL_DAMPING": 5e-7,
    "WALL_INTERACTION_RANGE": 0.001,

    # --- Time / Solver Parameters ---
    "T_SPAN": (0, 9 * SHAPE_DURATION),
    "T_EVAL_POINTS": 400,
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
