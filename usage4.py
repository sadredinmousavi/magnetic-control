import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from case_loader import (
    build_common_config,
    case_output_path,
    get_case_name_from_argv,
    load_case,
    require_keys,
    unpack_target_schedule_entry,
)
from functions_main import (
    calculate_potential_hessian,
    calculate_total_force_from_sources,
    find_four_stable_equilibrium_inputs,
    find_two_equilibrium_with_center_repulsion_inputs,
    find_two_stable_equilibrium_inputs,
    find_two_equilibrium_inputs,
    find_stable_equilibrium_inputs,
    generate_circular_source_positions,
)
from functions_utility import (
    SolveIVPProgress,
    compute_grid_fields,
    extract_optimization_info,
    print_optimization_results,
)
from functions_pm_microrobots import (
    animate_trajectories,
    microrobot_payload_dynamics,
)

# =============================================================================
# NOMENCLATURE & UNITS REFERENCE
# =============================================================================
# u (control_inputs_u)  : Dimensionless [-1, 1]. The COSINE of the magnet angle (u = cos(θ)).
# theta_rad / theta_deg : Radians / Degrees. The actual physical angle of the magnet.
# pos / r / p1 / p2     : Meters (m). Spatial position vectors (x, y, z).
# net_force / F_m       : Newtons (N). The magnetic force exerted on the microrobot.
# C_F                   : N·m^4. Lumped magnetic force constant (permeability & moments).
# mu_0                  : T·m/A (or H/m). Vacuum permeability.
# m_ba / M              : A·m^2. Magnetic dipole moment.
# B                     : Tesla (T). Magnetic flux density (magnetic field).
# U (potential_energy)  : Joules (J). Magnetic potential energy landscape.
# H (Hessian)           : N/m. Spatial derivative of force (magnetic stiffness matrix).
# eigenvalues           : N/m. Eigenvalues of Hessian (negative = stable restoring force).
# =============================================================================



# =========================================================================
# 1. SYSTEM PARAMETERS & CONSTANTS
# =========================================================================

DEFAULT_CASE_NAME = "case_payload_baseline"
CASE_NAME = get_case_name_from_argv(DEFAULT_CASE_NAME)
PARAMS = load_case(CASE_NAME)

require_keys(
    PARAMS,
    [
        "NUM_SOURCES",
        "RADIUS",
        "TARGET_SCHEDULE",
        "SOURCE_MAGNETIZATION",
        "ROBOT_MAGNETIZATION",
        "L_SOURCE",
        "L_ROBOT",
        "GRID_MIN",
        "GRID_MAX",
        "RESOLUTION",
        "DENSITY_NDFEB",
        "FLUID_VISCOSITY",
        "ALPHA",
        "CAPILLARY_SIN_C",
        "GAMMA",
        "INITIAL_ROBOT_POSITIONS",
        "T_SPAN",
        "T_EVAL_POINTS",
        "SOLVER_PROGRESS_INTERVAL",
        "PAYLOAD_RADIUS",
        "PAYLOAD_HEIGHT",
        "PAYLOAD_DENSITY",
        "PAYLOAD_DRAG_FACTOR",
        "CONTACT_STIFFNESS",
        "CONTACT_DAMPING",
        "PAYLOAD_CAPILLARY_GAIN",
        "PAYLOAD_CAPILLARY_RANGE",
        "PAYLOAD_INITIAL_POS",
        "PAYLOAD_INITIAL_VEL",
    ],
    CASE_NAME,
)

CFG = build_common_config(PARAMS)




# =========================================================================
# 2. MAIN EXECUTION
# =========================================================================

def main():
    source_positions = generate_circular_source_positions(CFG.NUM_SOURCES, CFG.RADIUS)
    
    print(f"Loaded case: {CASE_NAME}")
    print("Finding optimal control inputs for scheduled targets")
    
    # Optimization
    # control_inputs_u = find_stable_equilibrium_inputs(DESIRED_POS, source_positions, C_F, ratio=EIG_RATIO)
    target_controls = []
    opt_infos = []
    field_data = []
    
    for target_entry in CFG.TARGET_SCHEDULE:
        (
            start_time,
            target_pos,
            second_equilibrium_pos,
            target_eig_ratio,
            target_eig_angle,
        ) = unpack_target_schedule_entry(target_entry)

        print("\n" + "=" * 70)
        additional_equilibrium_positions = []
        if isinstance(second_equilibrium_pos, list):
            additional_equilibrium_positions = second_equilibrium_pos
        elif second_equilibrium_pos is not None:
            additional_equilibrium_positions = [second_equilibrium_pos]

        if not additional_equilibrium_positions:
            print(
                f"Finding control inputs for target {target_pos} "
                f"starting at t={start_time} with eig_ratio={target_eig_ratio} "
                f"and eig_angle={target_eig_angle:.3f} rad"
            )
        else:
            print(
                f"Finding control inputs for {1 + len(additional_equilibrium_positions)} "
                f"equilibrium points starting at t={start_time}"
            )
        print("=" * 70)
        
        if not additional_equilibrium_positions:
            u_target = find_stable_equilibrium_inputs(
                target_pos,
                source_positions,
                CFG.C_F,
                ratio=target_eig_ratio,
                eig_angle_rad=target_eig_angle,
                trace_margin=CFG.STABILITY_TRACE_MARGIN,
                det_margin=CFG.STABILITY_DET_MARGIN
            )
        elif len(additional_equilibrium_positions) == 1:
            second_equilibrium_pos = additional_equilibrium_positions[0]
            two_equilibrium_solver = PARAMS.get("TWO_EQUILIBRIUM_SOLVER", "stable")

            if two_equilibrium_solver == "plain":
                u_target = find_two_equilibrium_inputs(
                    target_pos,
                    second_equilibrium_pos,
                    source_positions,
                    CFG.C_F
                )
            elif two_equilibrium_solver == "center_repulsion":
                u_target = find_two_equilibrium_with_center_repulsion_inputs(
                    target_pos,
                    second_equilibrium_pos,
                    source_positions,
                    CFG.C_F
                )
            elif two_equilibrium_solver == "stable":
                u_target = find_two_stable_equilibrium_inputs(
                    target_pos,
                    second_equilibrium_pos,
                    source_positions,
                    CFG.C_F,
                    trace_margin=CFG.STABILITY_TRACE_MARGIN,
                    det_margin=CFG.STABILITY_DET_MARGIN
                )
            else:
                raise ValueError(
                    "TWO_EQUILIBRIUM_SOLVER must be 'stable', 'plain', "
                    "or 'center_repulsion'."
                )
        elif len(additional_equilibrium_positions) == 3:
            u_target = find_four_stable_equilibrium_inputs(
                [target_pos] + additional_equilibrium_positions,
                source_positions,
                CFG.C_F
            )
        else:
            raise ValueError("Only one, two, or four equilibrium targets are supported.")
        
        target_controls.append(
            (start_time, target_pos, target_eig_ratio, target_eig_angle, u_target)
        )
        
        net_force = calculate_total_force_from_sources(
            source_positions,
            u_target,
            target_pos,
            CFG.M_SOURCE_MAGNITUDE,
            CFG.M_ROBOT_MAGNITUDE
        )

        equilibrium_positions = [target_pos]
        equilibrium_net_forces = [net_force]

        for eq_pos in additional_equilibrium_positions:
            net_force_2 = calculate_total_force_from_sources(
                source_positions,
                u_target,
                eq_pos,
                CFG.M_SOURCE_MAGNITUDE,
                CFG.M_ROBOT_MAGNITUDE
            )
            equilibrium_positions.append(eq_pos)
            equilibrium_net_forces.append(net_force_2)
        
        equilibrium_stability = []
        for eq_pos in equilibrium_positions:
            eq_H = calculate_potential_hessian(
                eq_pos,
                source_positions,
                CFG.C_F,
                u_target
            )
            eq_eigenvalues, eq_eigenvectors = np.linalg.eig(eq_H)
            equilibrium_stability.append({
                "position": eq_pos,
                "H": eq_H,
                "trace": np.trace(eq_H),
                "determinant": np.linalg.det(eq_H),
                "eigenvalues": eq_eigenvalues,
                "eigenvectors": eq_eigenvectors,
            })

        H = equilibrium_stability[0]["H"]
        eigenvalues = equilibrium_stability[0]["eigenvalues"]
        eigenvectors = equilibrium_stability[0]["eigenvectors"]
        
        opt_info = extract_optimization_info(
            control_inputs_u=u_target,
            net_force=net_force,
            H=H,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            desired_pos=target_pos,
            equilibrium_positions=equilibrium_positions,
            equilibrium_net_forces=equilibrium_net_forces,
            equilibrium_stability=equilibrium_stability,
            microrobot_positions=CFG.INITIAL_ROBOT_POSITIONS
        )
        
        opt_infos.append(opt_info)
        
        
        source_moment_vectors = np.zeros_like(source_positions)
        
        for i, pos in enumerate(source_positions):
            radial_unit_vector = pos / np.linalg.norm(pos)
            source_moment_vectors[i] = (
                u_target[i] * CFG.M_SOURCE_MAGNITUDE * radial_unit_vector
            )
        
        X, Y, Fx, Fy, U_pot, Bx, By = compute_grid_fields(
            source_positions,
            u_target,
            source_moment_vectors,
            CFG.GRID_MIN,
            CFG.GRID_MAX,
            CFG.RESOLUTION,
            CFG.M_SOURCE_MAGNITUDE,
            CFG.M_ROBOT_MAGNITUDE
        )
        
        field_data.append({
            "start_time": start_time,
            "target_pos": target_pos,
            "u": u_target,
            "X": X,
            "Y": Y,
            "Fx": Fx,
            "Fy": Fy,
            "U_pot": U_pot,
            "Bx": Bx,
            "By": By,
        })
        
        
        print_optimization_results(opt_info)
    
    
    print("Running dynamics simulation...")
    
    # 1. Set up the initial state:
    # robots:  [x1, y1, vx1, vy1, ..., xN, yN, vxN, vyN]
    # payload: [xp, yp, vxp, vyp]
    initial_state = np.zeros(CFG.NUM_ROBOTS * 4 + 4)

    for i in range(CFG.NUM_ROBOTS):
        initial_state[i * 4] = CFG.INITIAL_ROBOT_POSITIONS[i][0]
        initial_state[i * 4 + 1] = CFG.INITIAL_ROBOT_POSITIONS[i][1]
        # vx, vy remain 0

    payload_idx = CFG.NUM_ROBOTS * 4
    initial_state[payload_idx] = CFG.PAYLOAD_INITIAL_POS[0]
    initial_state[payload_idx + 1] = CFG.PAYLOAD_INITIAL_POS[1]
    initial_state[payload_idx + 2] = CFG.PAYLOAD_INITIAL_VEL[0]
    initial_state[payload_idx + 3] = CFG.PAYLOAD_INITIAL_VEL[1]

    progress = SolveIVPProgress(
        CFG.T_SPAN,
        min_interval=CFG.SOLVER_PROGRESS_INTERVAL
    )

    def dynamics_with_progress(t, y):
        progress.update(t)
        return microrobot_payload_dynamics(
            t,
            y,
            source_positions,
            target_controls,
            CFG.M_SOURCE_MAGNITUDE,
            CFG.M_ROBOT_MAGNITUDE,
            CFG.ROBOT_MASS,
            CFG.FLUID_DRAG,
            CFG.ROBOT_RADIUS,
            CFG.CAPILLARY_SIN_C,
            CFG.GAMMA,
            CFG.PAYLOAD_RADIUS,
            CFG.PAYLOAD_MASS,
            CFG.PAYLOAD_DRAG,
            CFG.CONTACT_STIFFNESS,
            CFG.CONTACT_DAMPING,
            CFG.PAYLOAD_CAPILLARY_GAIN,
            CFG.PAYLOAD_CAPILLARY_RANGE,
            CFG.PAYLOAD_CAPILLARY_CUTOFF,
            CFG.USE_OVERDAMPED_DYNAMICS,
            CFG.DYNAMICS_SPEEDUP,
            CFG.WALL_SEGMENTS,
            CFG.WALL_STIFFNESS,
            CFG.WALL_DAMPING,
            CFG.WALL_INTERACTION_RANGE
        )
    
    # 3. Solve the differential equations
    sol = solve_ivp(
        fun=dynamics_with_progress,
        t_span=CFG.T_SPAN,
        y0=initial_state,
        t_eval=CFG.T_EVAL,
        method=PARAMS.get("SOLVER_METHOD", "RK45"),
        rtol=PARAMS.get("SOLVER_RTOL", 1e-5),
        atol=PARAMS.get("SOLVER_ATOL", 1e-8)
    )

    progress.finish(sol.message)
    
    # Add the trajectory data to opt_info so the plotter can access it
    opt_info['trajectories'] = sol.y
    # =========================================================================


    # --- Plotting ---
    # 1. Plot Mode 1 (Streamlines & Info)
    # plot_mode_1(
    #     X, Y, Fx, Fy, 
    #     source_positions, 
    #     opt_info=opt_info, 
    #     draw_contour=True,          
    #     draw_desired_point=True,
    #     plot_microrobots=True,      
    #     plot_trajectories=True      # <-- ADD THIS FLAG
    # )

    # Show/Save the Animation!
    video_filename = Path("outputs") / case_output_path(CASE_NAME).with_suffix(".mp4")
    video_filename.parent.mkdir(parents=True, exist_ok=True)

    animate_trajectories(
        CFG.T_EVAL,
        sol.y,
        source_positions,
        CFG.TARGET_SCHEDULE,
        CFG.GRID_MIN,
        CFG.GRID_MAX,
        field_data=field_data,
        draw_contour=True,
        draw_streamlines=False,
        draw_quiver=False,
        draw_sources=True,
        draw_all_targets=False,
        draw_active_target=True,
        plot_trajectories=False,
        plot_microrobots=True,
        payload_radius=CFG.PAYLOAD_RADIUS,
        wall_segments=CFG.WALL_SEGMENTS,
        save_video=True,
        video_name=video_filename
    )


if __name__ == '__main__':
    main()
