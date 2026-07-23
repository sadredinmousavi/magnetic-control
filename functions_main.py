import numpy as np
from scipy.constants import mu_0  # Magnetic constant (permeability of free space)
from scipy.optimize import minimize


# =============================================================================
# NOMENCLATURE & UNITS REFERENCE
# =============================================================================
#
# --- Control & Actuation Variables ---
# u (control_inputs_u)  : Dimensionless [-1, 1]. The mathematical control input.
#                         CRITICAL: This is the COSINE of the magnet angle (u = cos(θ)), 
#                         NOT the angle itself in radians!
# theta_rad             : Radians (rad). The actual physical angle of the magnet's 
#                         moment relative to the z-axis (calculated via arccos(u)).
# theta_deg             : Degrees (°). The physical angle converted for display/UI.
# 
# --- Geometry & Position Variables ---
# pos / r / p1 / p2     : Meters (m). Spatial position vectors (x, y, z) of the 
#                         microrobot or actuator magnets.
# r_norm (|r|)          : Meters (m). Euclidean distance between magnets.
#
# --- Force & Physics Variables ---
# net_force / F_m       : Newtons (N). The magnetic force exerted on the microrobot.
# C_F                   : N·m^4 (Newtons * meters^4). The lumped magnetic force constant 
#                         combining permeability and magnetic moments.
#                         (Derived from F = C_F * u * (r / |r|^5))
# mu_0                  : T·m/A (Tesla-meters per Ampere) or H/m. Vacuum permeability.
# m_ba / M              : A·m^2 (Ampere-square meters). Magnetic dipole moment.
# B                     : Tesla (T). Magnetic flux density (magnetic field).
# U (potential_energy)  : Joules (J). Magnetic potential energy landscape.
#
# --- Stability & Optimization Variables ---
# H (Hessian)           : N/m (Newtons per meter). The spatial derivative of the 
#                         force (Jacobian of force, Hessian of energy), representing 
#                         the magnetic stiffness matrix.
# eigenvalues           : N/m (Newtons per meter). The eigenvalues of the Hessian matrix. 
#                         Negative eigenvalues indicate stable restoring forces (a well).
# eigenvectors          : Dimensionless (normalized direction vectors). The principal 
#                         directions of the stiffness/instability.
# =============================================================================



# =============================================================================
# CORE PHYSICS MODELS
# =============================================================================

def calculate_field_potential_abbott_2d(source_pos, point_pos, source_moment_vec, target_moment_vec=None):
    """
    Calculates the 2D magnetic field and potential energy using the general dipole model.

    This model is based on the fundamental 3D formulas adapted for a 2D plane.
    It calculates the magnetic field B at a point and the potential energy U of a
    target dipole placed at that point.

    Formulas from Abbott2020.pdf:
    ---------------------------------------------------------------------
    Magnetic Field B(r) = (mu_0 / 4pi) * [ (3r(m_s . r) / |r|^5) - (m_s / |r|^3) ]
      (page 2, eq. 1)

    Potential Energy U = -B . m_t
      (page 3, eq. 5)

    where:
    - m_s = source magnetic moment vector
    - m_t = target (robot) magnetic moment vector
    - r   = vector from source to the point of interest
    ---------------------------------------------------------------------

    Args:
        source_pos (np.ndarray): A 2-element array [x, y] for the source magnet's position.
        point_pos (np.ndarray): A 2-element array [x, y] for the point of interest (e.g., robot pos).
        source_moment_vec (np.ndarray): The 2D magnetic moment vector [mx, my] of the source.
        target_moment_vec (np.ndarray, optional): The 2D magnetic moment vector of the target.
                                                  If None, potential energy will be 0. Defaults to None.

    Returns:
        dict: A dictionary containing {'field': B_vector, 'potential': U_scalar}.
    """
    source_pos = np.asarray(source_pos)
    point_pos = np.asarray(point_pos)
    source_moment_vec = np.asarray(source_moment_vec)

    r_vec = point_pos - source_pos
    r_mag = np.linalg.norm(r_vec)

    if r_mag < 1e-9:
        return {'field': np.array([np.inf, np.inf]), 'potential': np.inf}

    # Calculate magnetic field (B)
    m_s_dot_r = np.dot(source_moment_vec, r_vec)
    term1 = 3 * r_vec * m_s_dot_r / (r_mag**5)
    term2 = source_moment_vec / (r_mag**3)
    field_vec = (mu_0 / (4 * np.pi)) * (term1 - term2)

    # Calculate potential energy (U)
    potential = 0.0
    if target_moment_vec is not None:
        target_moment_vec = np.asarray(target_moment_vec)
        potential = -np.dot(field_vec, target_moment_vec)

    return {'field': field_vec, 'potential': potential}

def calculate_force_yousefi_model(source_pos, robot_pos, control_input_u, m_source, m_robot):
    """
    Calculates the 2D magnetic force based on the simplified planar model from Yousefi et al., 2021.
    This model is computationally efficient and well-suited for control applications.

    Formula from Yousefi2021.pdf (page 4, simplified from eq. 5):
    F_m = C_F * u * r / |r|^5
    """
    source_pos = np.asarray(source_pos)
    robot_pos = np.asarray(robot_pos)

    r_vec = robot_pos - source_pos
    r_mag = np.linalg.norm(r_vec)

    if r_mag < 1e-9:
        return np.zeros(2)

    C_F = (3 * mu_0 * m_source * m_robot) / (4 * np.pi)
    force_vector = C_F * control_input_u * (r_vec / (r_mag**5))

    return force_vector

# --- System Configuration & Simulation ---

def generate_circular_source_positions(num_sources, radius):
    """
    Generates positions for source magnets arranged in a circle.
    """
    positions = np.zeros((num_sources, 2))
    angles = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)
    positions[:, 0] = radius * np.cos(angles)
    positions[:, 1] = radius * np.sin(angles)
    return positions

def calculate_total_force_from_sources(source_positions, control_inputs_u, robot_pos, m_source, m_robot):
    """
    Calculates the total magnetic force on the robot from all source magnets using the Yousefi model.
    """
    total_force = np.zeros(2)
    if len(source_positions) != len(control_inputs_u):
        raise ValueError("Number of source positions must match the number of control inputs.")

    for i, pos in enumerate(source_positions):
        total_force += calculate_force_yousefi_model(pos, robot_pos, control_inputs_u[i], m_source, m_robot)

    return total_force

def calculate_total_field(source_positions, source_moment_vectors, point_pos):
    """
    Calculates the total magnetic field at a point from all source magnets.
    """
    total_field = np.zeros(2)
    if len(source_positions) != len(source_moment_vectors):
        raise ValueError("Number of source positions must match the number of moment vectors.")

    for i, pos in enumerate(source_positions):
        # We only need the field, so we don't pass a target moment
        result = calculate_field_potential_abbott_2d(pos, point_pos, source_moment_vectors[i])
        total_field += result['field']

    return total_field


def calculate_dipole_interaction_force(pos_k, pos_j, m_k, m_j):
    """
    Calculates the repulsive interaction force exerted BY robot k ON robot j.
    Assumes purely vertical (Z-axis) magnetic moments.
    """
    mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability
    
    r_kj = pos_j - pos_k # Vector from k to j
    r_mag = np.linalg.norm(r_kj)
    
    if r_mag < 1e-6: # Prevent division by zero
        return np.zeros(2)
        
    # Simplified purely repulsive force
    force = (3 * mu_0 * m_k * m_j / (4 * np.pi)) * (r_kj / (r_mag**5))
    
    return force


def calculate_capillary_force(pos_j, pos_i, robot_radius, gamma=0.072, meniscus_angle=None, sin_C=None):
    """
    Capillary force exerted BY robot j ON robot i.
    
    Dong formula:
        F_cap_ij = 2*pi*gamma*R^2*sin(C)^2 * r_ij / |r_ij|^2
    
    pos_i, pos_j: 2D positions [x, y] in meters
    robot_radius: R in meters
    gamma: surface tension, water ~0.072 N/m
    meniscus_angle: C in radians, optional
    sin_C: optional direct value of sin(C)
    """
    r_ij = np.asarray(pos_i) - np.asarray(pos_j)
    r_mag = np.linalg.norm(r_ij)
    
    if r_mag < 1e-9:
        return np.zeros(2)
    
    if sin_C is None:
        if meniscus_angle is None:
            raise ValueError("Provide either meniscus_angle or sin_C.")
        sin_C = np.sin(meniscus_angle)
    
    K_cap = 2 * np.pi * gamma * robot_radius**2 * sin_C**2
    return K_cap * r_ij / (r_mag**2)


def calculate_robot_payload_contact_force(
    robot_pos,
    robot_vel,
    payload_pos,
    payload_vel,
    robot_radius,
    payload_radius,
    k_contact,
    c_contact
):
    """
    Contact force exerted BY robot ON payload.
    Equal and opposite force acts on robot.
    """
    r_vec = np.asarray(payload_pos) - np.asarray(robot_pos)
    dist = np.linalg.norm(r_vec)

    if dist < 1e-12:
        return np.zeros(2)

    n = r_vec / dist
    overlap = robot_radius + payload_radius - dist

    if overlap <= 0:
        return np.zeros(2)

    rel_vel = np.asarray(payload_vel) - np.asarray(robot_vel)
    normal_rel_vel = np.dot(rel_vel, n)

    # # Spring-damper normal contact
    # F_mag = k_contact * overlap - c_contact * normal_rel_vel

    # # No adhesive pulling from contact model
    # F_mag = max(F_mag, 0.0)

    closing_speed = max(-normal_rel_vel, 0.0)
    F_mag = k_contact * overlap + c_contact * closing_speed

    return F_mag * n


def calculate_robot_payload_capillary_force(
    robot_pos,
    payload_pos,
    robot_radius,
    payload_radius,
    capillary_gain,
    capillary_range
):
    """
    Approximate attractive capillary force exerted BY robot ON payload.

    This is a phenomenological model, not Dong's exact robot-robot capillary formula.
    It helps nearby robots stick to a floating nonmagnetic payload.
    """
    r_vec = np.asarray(payload_pos) - np.asarray(robot_pos)
    dist = np.linalg.norm(r_vec)

    if dist < 1e-12:
        return np.zeros(2)

    n = r_vec / dist
    gap = dist - (robot_radius + payload_radius)

    if gap < 0:
        gap = 0.0

    # Exponential short-range attraction
    F_mag = capillary_gain * np.exp(-gap / capillary_range)

    return F_mag * n


def calculate_robot_payload_interaction_force(
    robot_pos,
    robot_vel,
    payload_pos,
    payload_vel,
    robot_radius,
    payload_radius,
    k_contact,
    c_contact,
    capillary_gain,
    capillary_range,
    capillary_cutoff,
    adhesion_gap=0.0
):
    """
    Combined robot-payload interaction.

    Returns force exerted BY robot ON payload.

    Model:
    - If robot is outside payload contact distance:
        capillary attraction pulls payload toward robot.
    - If robot overlaps payload:
        soft contact repulsion prevents penetration.
    - Near contact:
        attraction smoothly weakens so capillary/contact do not fight badly.

    adhesion_gap:
        small preferred surface gap. Use 0 for touching.
    """
    robot_pos = np.asarray(robot_pos)
    payload_pos = np.asarray(payload_pos)
    robot_vel = np.asarray(robot_vel)
    payload_vel = np.asarray(payload_vel)

    r_vec = payload_pos - robot_pos
    dist = np.linalg.norm(r_vec)

    if dist < 1e-12:
        return np.zeros(2)

    n = r_vec / dist

    contact_dist = robot_radius + payload_radius
    gap = dist - contact_dist

    rel_vel = payload_vel - robot_vel
    normal_rel_vel = np.dot(rel_vel, n)

    F_total = np.zeros(2)

    # -----------------------------------------------------
    # 1. Capillary attraction: only when not deeply overlapping
    # -----------------------------------------------------
    # Positive direction n means force on payload away from robot.
    # Attraction means payload is pulled toward robot: -n.
    if capillary_gain > 0:
        if gap > capillary_cutoff:
            F_cap_mag = 0.0
        else:
            effective_gap = max(gap - adhesion_gap, 0.0)
            F_cap_mag = capillary_gain * np.exp(-effective_gap / capillary_range)
            
            # If overlapping, fade attraction so it does not fight contact too much.
            if gap < 0:
                penetration = -gap
                fade = np.exp(-penetration / max(robot_radius, 1e-12))
                F_cap_mag *= fade
            
            F_cap_on_payload = -F_cap_mag * n
            F_total += F_cap_on_payload

    # -----------------------------------------------------
    # 2. Contact repulsion: only if overlapping
    # -----------------------------------------------------
    if gap < 0:
        overlap = -gap

        # Repulsion pushes payload away from robot: +n
        # Damping only when bodies are approaching/contact-compressing.
        closing_speed = max(-normal_rel_vel, 0.0)

        F_contact_mag = k_contact * overlap + c_contact * closing_speed
        F_contact_on_payload = F_contact_mag * n

        F_total += F_contact_on_payload

    return F_total








# =============================================================================
# CALCULUS
# =============================================================================

def build_actuation_matrix(target_pos, source_positions, C_F):
    """
    Builds the 2xN actuation matrix A(p) at the target position.
    Based on Yousefi2021 formula: f_m = C_F * u * r / |r|^5
    """
    N = len(source_positions)
    A = np.zeros((2, N))
    
    for i in range(N):
        # r_vec = source_positions[i] - target_pos
        r_vec = target_pos - source_positions[i] 
        distance = np.linalg.norm(r_vec)
        
        # Avoid division by zero if target is exactly on a magnet
        if distance < 1e-6:
            continue
            
        # The contribution of magnet i for u_i = 1
        A[:, i] = C_F * r_vec / (distance**5)
        
    return A


def calculate_potential_hessian(target_pos, source_positions, C_F, u):
    """
    Calculates the Hessian matrix of the Potential Energy 
    (which is exactly the negative of the Force Jacobian).
    A positive definite Hessian indicates a stable equilibrium point (a potential well).
    """
    H = np.zeros((2, 2))
    N = len(source_positions)
    
    for i in range(N):
        r_vec = target_pos - source_positions[i]
        dx, dy = r_vec[0], r_vec[1]
        R_sq = dx**2 + dy**2
        R = np.sqrt(R_sq)
        
        if R < 1e-6:
            continue
            
        R_7 = R**7
        
        # Second derivatives of Potential Energy (negative derivatives of Force)
        hxx = C_F * u[i] * (5 * dx**2 - R_sq) / R_7
        hxy = C_F * u[i] * (5 * dx * dy) / R_7
        hyy = C_F * u[i] * (5 * dy**2 - R_sq) / R_7
        
        H[0, 0] += hxx
        H[0, 1] += hxy
        H[1, 0] += hxy
        H[1, 1] += hyy
        
    return H






# =============================================================================
# OPTIMIZERS
# =============================================================================







def default_target_effort(num_sources):
    """Use average u^2 = 0.25 so effort scales with the number of magnets."""
    return num_sources * 0.25


def find_two_equilibrium_inputs(
    desired_pos_1,
    desired_pos_2,
    source_positions,
    C_F,
    target_effort=None
):
    """
    Finds u that makes two desired positions equilibrium points.

    This enforces zero net force at both positions:
        F(desired_pos_1) = 0
        F(desired_pos_2) = 0

    It does not constrain stability, eigenvalue ratio, or eigenvector angle.
    """
    num_sources = len(source_positions)
    if target_effort is None:
        target_effort = default_target_effort(num_sources)
    desired_pos_1 = np.array(desired_pos_1)
    desired_pos_2 = np.array(desired_pos_2)

    A1 = build_actuation_matrix(desired_pos_1, source_positions, C_F)
    A2 = build_actuation_matrix(desired_pos_2, source_positions, C_F)

    def objective(u):
        return (np.sum(u**2) - target_effort)**2

    def force_constraint(u):
        return np.concatenate((A1 @ u, A2 @ u))

    constraints = [{'type': 'eq', 'fun': force_constraint}]
    bounds = [(0.0, 1.0) for _ in range(num_sources)]
    u0 = np.ones(num_sources) * np.sqrt(target_effort / num_sources)
    u0 = np.clip(u0, 0.1, 0.9)

    result = minimize(objective, u0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        print("Two-equilibrium optimization failed:", result.message)
        return np.zeros(num_sources)


def find_two_equilibrium_with_center_repulsion_inputs(
    desired_pos_1,
    desired_pos_2,
    source_positions,
    C_F,
    target_effort=None,
    center_line_repulsion_margin=1e-7,
    repulsion_weight=1e6,
    stability_weight=1e5,  # Added: Weight to penalize instability at target points
    stability_margin=1e-5  # Added: Minimum curvature required to be considered a stable "bowl"
):
    """
    Finds u for two stable equilibrium points with a soft repelling midpoint, 
    allowing negative inputs.

    Constraints:
        1. F(desired_pos_1) = 0
        2. F(desired_pos_2) = 0
        3. F(center) = 0

    Soft Objective:
        - Minimizes effort.
        - Penalizes positive curvature at the center (encourages saddle/repulsion).
        - Penalizes flat/negative curvature at the desired points (forces stable bowls).
    """
    num_sources = len(source_positions)
    if target_effort is None:
        target_effort = default_target_effort(num_sources) # Assuming defined in your scope
        
    desired_pos_1 = np.array(desired_pos_1)
    desired_pos_2 = np.array(desired_pos_2)
    center_pos = 0.5 * (desired_pos_1 + desired_pos_2)

    line_vec = desired_pos_1 - desired_pos_2
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        raise ValueError("The two equilibrium points must be different.")
    
    line_unit = line_vec / line_len
    
    # Calculate orthogonal vector for 2D stability checks
    if len(line_unit) == 2:
        ortho_unit = np.array([-line_unit[1], line_unit[0]])
    else:
        # Dummy orthogonal vector if you are working in 3D (will need a real cross product later)
        ortho_unit = np.zeros_like(line_unit) 

    # Build actuation matrices
    A1 = build_actuation_matrix(desired_pos_1, source_positions, C_F)
    A2 = build_actuation_matrix(desired_pos_2, source_positions, C_F)
    A_center = build_actuation_matrix(center_pos, source_positions, C_F)

    def center_line_curvature(u):
        H_center = calculate_potential_hessian(center_pos, source_positions, C_F, u)
        return line_unit @ H_center @ line_unit

    def calculate_instability_violation(u, pos):
        """
        Calculates how far the point is from being a stable 'bowl'.
        We check the curvature along the line and orthogonal to it.
        Both should be strictly positive (greater than stability_margin).
        """
        H = calculate_potential_hessian(pos, source_positions, C_F, u)
        
        curv_line = line_unit @ H @ line_unit
        curv_ortho = ortho_unit @ H @ ortho_unit
        
        violation_line = np.maximum(0.0, stability_margin - curv_line)
        violation_ortho = np.maximum(0.0, stability_margin - curv_ortho)
        
        return violation_line**2 + violation_ortho**2

    def objective(u):
        # 1. Effort minimization (works perfectly with negative inputs)
        effort_error = (np.sum(u**2) - target_effort)**2
        
        # 2. Soft Penalty for Curvature at Center (Saddle creation)
        curvature_center = center_line_curvature(u)
        center_violation = np.maximum(0.0, curvature_center + center_line_repulsion_margin)
        center_penalty = repulsion_weight * (center_violation ** 2)
        reward = curvature_center * 1.0 
        
        # 3. Target Points Stability Penalty (Bowl creation)
        instability_1 = calculate_instability_violation(u, desired_pos_1)
        instability_2 = calculate_instability_violation(u, desired_pos_2)
        target_stability_penalty = stability_weight * (instability_1 + instability_2)

        return effort_error + center_penalty + reward + target_stability_penalty

    def force_constraint(u):
        # Ensure zero force at targets and center
        return np.concatenate((A1 @ u, A2 @ u, A_center @ u))

    constraints = [
        {'type': 'eq', 'fun': force_constraint}
    ]
    
    # Allow negative inputs
    bounds = [(-1.0, 1.0) for _ in range(num_sources)]
    
    # Calculate base magnitude for initial guess
    base_u0 = np.sqrt(target_effort / num_sources)
    
    # Randomly assign signs so the solver explores positive and negative space equally
    np.random.seed(42) # Remove or change seed if you want different initializations
    signs = np.random.choice([-1.0, 1.0], size=num_sources)
    u0 = np.ones(num_sources) * base_u0 * signs
    
    # Clip to keep solver slightly away from extreme edges on start
    u0 = np.clip(u0, -0.9, 0.9)

    result = minimize(objective, u0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        print("Two-equilibrium center-repulsion optimization failed:", result.message)
        return np.zeros(num_sources)


def find_two_stable_equilibrium_inputs(
    desired_pos_1,
    desired_pos_2,
    source_positions,
    C_F,
    target_effort=None,
    target_ratio=1.0,
    ratio_weight=10.0,
    trace_margin=1e-8,
    det_margin=1e-14
):
    """
    Finds u for two stable equilibrium points with near-equal eigenvalues.

    Constraints:
        1. F(desired_pos_1) = 0
        2. F(desired_pos_2) = 0
        3. H(desired_pos_1) and H(desired_pos_2) are positive definite

    The objective pulls each Hessian eigenvalue ratio toward 1 using:
        Delta / trace^2 = ((ratio - 1) / (ratio + 1))^2

    This avoids the center-repulsion constraint, which made the endpoints
    unstable in some cases.
    """
    num_sources = len(source_positions)
    if target_effort is None:
        target_effort = default_target_effort(num_sources)
    desired_pos_1 = np.array(desired_pos_1)
    desired_pos_2 = np.array(desired_pos_2)

    A1 = build_actuation_matrix(desired_pos_1, source_positions, C_F)
    A2 = build_actuation_matrix(desired_pos_2, source_positions, C_F)
    force_scale = max(np.max(np.abs(A1)), np.max(np.abs(A2)), 1e-30)
    hessian_cache = {}

    def hessian_metrics(pos, u):
        cache_key = id(pos)
        cached = hessian_cache.get(cache_key)
        if cached is None or not np.array_equal(cached["u"], u):
            H = calculate_potential_hessian(pos, source_positions, C_F, u)
            trace = H[0, 0] + H[1, 1]
            det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
            delta = trace * trace - 4.0 * det
            cached = {
                "u": np.array(u, copy=True),
                "trace": trace,
                "det": det,
                "delta": delta,
            }
            hessian_cache[cache_key] = cached

        return cached["trace"], cached["det"], cached["delta"]

    def anisotropy_penalty(pos, u):
        trace, det, delta = hessian_metrics(pos, u)
        if trace == 0 or det <= 0:
            return np.inf
        
        # General formula for any target ratio r:
        # Penalizes deviation from the expected relationship between trace^2 and det
        r = target_ratio
        expected_ratio = ((r + 1)**2) / r
        actual_ratio = (trace * trace) / det
        
        return (actual_ratio - expected_ratio)**2

    def objective(u):
        effort_error = (np.sum(u**2) - target_effort)**2
        anisotropy_error = (
            anisotropy_penalty(desired_pos_1, u)
            + anisotropy_penalty(desired_pos_2, u)
        )
        return effort_error + ratio_weight * anisotropy_error

    def force_constraint(u):
        return np.concatenate((A1 @ u, A2 @ u)) / force_scale

    def stability_trace_1(u):
        trace, _, _ = hessian_metrics(desired_pos_1, u)
        return (trace / trace_margin) - 1.0

    def stability_trace_2(u):
        trace, _, _ = hessian_metrics(desired_pos_2, u)
        return (trace / trace_margin) - 1.0

    def stability_det_1(u):
        _, det, _ = hessian_metrics(desired_pos_1, u)
        return (det / det_margin) - 1.0

    def stability_det_2(u):
        _, det, _ = hessian_metrics(desired_pos_2, u)
        return (det / det_margin) - 1.0

    constraints = [
        {'type': 'eq', 'fun': force_constraint},
        {'type': 'ineq', 'fun': stability_trace_1},
        {'type': 'ineq', 'fun': stability_trace_2},
        {'type': 'ineq', 'fun': stability_det_1},
        {'type': 'ineq', 'fun': stability_det_2},
    ]
    bounds = [(0, 1.0) for _ in range(num_sources)]

    base = np.ones(num_sources) * np.sqrt(target_effort / num_sources)
    base = np.clip(base, 0.1, 0.9)

    initial_guesses = [
        base,
        np.ones(num_sources) * 0.1,
        np.ones(num_sources) * 0.5,
        np.ones(num_sources) * 0.9,
    ]

    rng = np.random.default_rng(7)
    for _ in range(12):
        initial_guesses.append(rng.uniform(0.05, 0.95, num_sources))

    best_result = None
    for u0 in initial_guesses:
        result = minimize(
            objective,
            u0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500}
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

        if result.success:
            return result.x

    print("Two-stable-equilibrium optimization failed:", best_result.message)
    return best_result.x


def find_stable_equilibrium_inputs(
    desired_pos,
    source_positions,
    C_F,
    target_effort=None,
    ratio=1.0,
    eig_angle_rad=None,
    trace_margin=1e-6,
    det_margin=1e-12,
):
    """
    Finds permanent magnets inputs (u) to create a stable trap at target_pos.
    
    Optimization problem definition
    --------------------------------
    Solve for source inputs u so that the desired point becomes a stable
    equilibrium with prescribed shape/orientation properties.

    Objective:
        Minimize (sum(u^2) - target_effort)^2

    Constraints:
        1. force = 0
        2. trace >= eps
        3. det >= eps
        4. eig_ratio = target_ratio
        5. eig_angle = target_angle   (optional)

    Notes:
    - The Hessian H here is the potential-energy Hessian.
    - Stability requires positive definiteness of H in 2D, enforced through
      positive trace and positive determinant margins.
    - The principal eigenvector angle is computed analytically from the 2x2
      symmetric matrix:
          phi = 0.5 * atan2(2*Hxy, Hxx - Hyy)
    - Because eigenvectors are defined up to sign, the direction is constrained
      modulo pi using cos(2*(phi - phi_target)).
    """
    num_sources = len(source_positions)
    if target_effort is None:
        target_effort = default_target_effort(num_sources)
    desired_pos = np.array(desired_pos)
    
    A = build_actuation_matrix(desired_pos, source_positions, C_F)
    force_scale = max(np.max(np.abs(A)), 1e-30)
    hessian_scale = max(trace_margin, 1e-30)
    hessian_cache = {"u": None, "H": None}

    def hessian_at_input(u):
        if (
            hessian_cache["u"] is None
            or not np.array_equal(hessian_cache["u"], u)
        ):
            hessian_cache["u"] = np.array(u, copy=True)
            hessian_cache["H"] = calculate_potential_hessian(
                desired_pos,
                source_positions,
                C_F,
                u
            )

        return hessian_cache["H"]

    def hessian_trace_det_delta(u):
        J = hessian_at_input(u)
        a = J[0, 0]
        b = J[0, 1]
        c = J[1, 0]
        d = J[1, 1]
        trace = a + d
        det = a * d - b * c
        delta = trace * trace - 4 * det
        return J, trace, det, delta
    
    def objective(u):
        return (np.sum(u**2) - target_effort)**2
        
    
    # ------------------------------------------------------------------
    # Primary equilibrium constraint F_net = 0
    # ------------------------------------------------------------------
    
    def force_constraint(u):
        return (A @ u) / force_scale
        
    # ------------------------------------------------------------------
    # Stability: Trace(J) < 0 and Det(J) > 0 (already in your code)
    # ------------------------------------------------------------------
    
    def stability_trace_constraint(u):
        _, trace, _, _ = hessian_trace_det_delta(u)
        return (trace / trace_margin) - 1.0
        
    def stability_det_constraint(u):
        _, _, det, _ = hessian_trace_det_delta(u)
        return (det / det_margin) - 1.0
        
    # ------------------------------------------------------------------
    # NEW: Analytic eigenvalue ratio constraint
    #
    # For 2x2 matrix J:
    #   trace = J11 + J22
    #   det   = J11*J22 - J12^2
    #   Delta = trace^2 - 4 det
    #
    # Ratio constraint:
    #   (λ1/λ2 = ratio)  →  trace^2 (1 - r)^2  -  Delta (1 + r)^2  = 0
    # ------------------------------------------------------------------
    def eigen_ratio_constraint(u):
        _, trace, _, delta = hessian_trace_det_delta(u)
        
        r = ratio
        ratio_error = trace * trace * (1 - r) ** 2 - delta * (1 + r) ** 2
        return ratio_error / (hessian_scale * hessian_scale)

    def eigen_angle_constraint(u):
        H = hessian_at_input(u)

        hxx = H[0, 0]
        hxy = H[0, 1]
        hyy = H[1, 1]

        # Principal-axis angle of a 2x2 symmetric matrix.
        principal_angle = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)

        # Eigenvector sign ambiguity means phi and phi + pi represent the same
        # axis. Enforce angle equality modulo pi.
        return 1.0 - np.cos(2.0 * (principal_angle - eig_angle_rad))
    
    constraints = [
        {'type': 'eq', 'fun': force_constraint},
        {'type': 'ineq', 'fun': stability_trace_constraint},
        {'type': 'ineq', 'fun': stability_det_constraint},
    ]

    # if not np.isclose(ratio, 1.0):
    #     constraints.append({'type': 'eq', 'fun': eigen_ratio_constraint})
    constraints.append({'type': 'eq', 'fun': eigen_ratio_constraint})

    if eig_angle_rad is not None and not np.isclose(ratio, 1.0):
        constraints.append({'type': 'eq', 'fun': eigen_angle_constraint})
    
    bounds = [(0, 1.0) for _ in range(num_sources)]
    
    base = np.ones(num_sources) * np.sqrt(target_effort / num_sources)
    base = np.clip(base, 0.1, 0.9)

    initial_guesses = [
        base,
        np.ones(num_sources) * 0.1,
        np.ones(num_sources) * 0.5,
        np.ones(num_sources) * 0.9,
    ]

    rng = np.random.default_rng(11)
    for _ in range(8):
        initial_guesses.append(rng.uniform(0.05, 0.95, num_sources))

    best_result = None
    for u0 in initial_guesses:
        result = minimize(
            objective,
            u0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500}
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

        if result.success:
            return result.x

    print("Stable optimization failed:", best_result.message)
    return best_result.x





def find_four_stable_equilibrium_inputs(
    desired_positions,
    source_positions,
    C_F,
    target_effort=None,
    stability_weight=1e5,
    stability_margin=1e-5
):
    """
    Finds signed inputs (-1 to 1) for four equilibrium points.

    The four-point force constraints are often underdetermined but the
    positive-only feasible region can collapse to u=0. Optimizing inside the
    force-zero nullspace avoids the trivial solution while keeping all four
    force constraints exactly satisfied.
    """
    if len(desired_positions) != 4:
        raise ValueError("This function requires exactly 4 desired equilibrium positions.")
        
    num_sources = len(source_positions)
    if target_effort is None:
        target_effort = default_target_effort(num_sources)
        
    desired_positions = [np.array(pos) for pos in desired_positions]
    dim = desired_positions[0].shape[0]
    
    # Generate standard basis vectors dynamically (works for both 2D and 3D)
    # e.g., for 2D: [1, 0] and [0, 1]
    basis_vectors = np.eye(dim)

    # Build actuation matrices for the 4 target positions
    target_matrices = [build_actuation_matrix(pos, source_positions, C_F) for pos in desired_positions]
    force_matrix = np.vstack(target_matrices)
    force_scale = max(np.max(np.abs(force_matrix)), 1e-30)
    force_matrix_scaled = force_matrix / force_scale
    _, singular_values, vt = np.linalg.svd(force_matrix_scaled)
    rank_tolerance = (
        np.max(force_matrix_scaled.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    rank = np.sum(singular_values > rank_tolerance)
    nullspace = vt[rank:].T

    if nullspace.shape[1] == 0:
        raise ValueError("Four-equilibrium targets have no nonzero force-zero nullspace.")

    def calculate_instability_violation(u, pos):
        """Forces positive curvature across all primary axes to guarantee a stable point."""
        H = calculate_potential_hessian(pos, source_positions, C_F, u)
        violation = 0.0
        
        # Check curvature along each dimension axis
        for i in range(dim):
            vec = basis_vectors[i]
            curvature = vec @ H @ vec
            violation += np.maximum(0.0, stability_margin - curvature)**2
            
        return violation

    def objective_from_u(u):
        # 1. Effort Minimization
        effort_error = (np.sum(u**2) - target_effort)**2
            
        # 2. Stability Penalties (Deep bowls for all 4 targets)
        target_stability_penalty = 0.0
        for pos in desired_positions:
            target_stability_penalty += calculate_instability_violation(u, pos)

        return effort_error + (stability_weight * target_stability_penalty)

    def objective(z):
        return objective_from_u(nullspace @ z)

    def lower_bound_constraint(z):
        return (nullspace @ z) + 1.0

    def upper_bound_constraint(z):
        return 1.0 - (nullspace @ z)

    constraints = [
        {'type': 'ineq', 'fun': lower_bound_constraint},
        {'type': 'ineq', 'fun': upper_bound_constraint},
    ]

    rng = np.random.default_rng(42)
    initial_guesses = [np.zeros(nullspace.shape[1])]

    for _ in range(24):
        trial_u = rng.uniform(-0.8, 0.8, num_sources)
        initial_guesses.append(nullspace.T @ trial_u)

    best_result = None
    for z0 in initial_guesses:
        result = minimize(
            objective,
            z0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

        if result.success and np.sum((nullspace @ result.x) ** 2) > 1e-8:
            return nullspace @ result.x

    print("Four-equilibrium optimization failed:", best_result.message)
    return nullspace @ best_result.x
