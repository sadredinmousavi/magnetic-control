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

    # --- Magnetic Properties ---
    "SOURCE_MAGNETIZATION": 1000e3,
    "ROBOT_MAGNETIZATION": 868e3,
    "L_SOURCE": 0.02,
    "L_ROBOT": 0.0005,

    # --- Observation Space (Grid) ---
    "GRID_MIN": -0.3,
    "GRID_MAX": 0.3,
    "RESOLUTION": 50,

    # --- Circular Petri-Dish Workspace ---
    "DISH_CENTER": (0.0, 0.0),
    "DISH_RADIUS": 0.15,
    "PLOT_FIELD_INSIDE_DISH": True,
    "DISH_OUTSIDE_FADE_ALPHA": 0.86,

    # --- External-Magnet Moment Arrows ---
    "SHOW_EXTERNAL_MAGNET_MOMENT_VECTORS": True,
    "MAGNET_MOMENT_ARROW_LENGTH": 0.035,
    "MAGNET_MOMENT_ARROW_COLOR": "#d1495b",

    # --- Microrobot Dynamics Parameters ---
    "DENSITY_NDFEB": 7500,
    "FLUID_VISCOSITY": 0.001,
    "ALPHA": 0.3,
    "CAPILLARY_SIN_C": 0.1,
    "GAMMA": 0.072,

    # --- Animation Parameters ---
    "ANIMATION_ROBOT_MARKER_SIZE": 8,

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
}
