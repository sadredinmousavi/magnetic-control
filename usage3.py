import numpy as np
import matplotlib.pyplot as plt

from case_loader import (
    build_common_config,
    get_case_name_from_argv,
    load_case,
    require_keys,
)
from functions_main import (
    calculate_potential_hessian,
    calculate_total_force_from_sources,
    find_stable_equilibrium_inputs,
    generate_circular_source_positions,
)
from functions_utility import (
    compute_grid_fields,
    extract_optimization_info,
    plot_mode_1,
    print_optimization_results,
)

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
        "M_SATURATION",
        "L_SOURCE",
        "L_ROBOT",
        "MAGNETIZATION",
        "GRID_MIN",
        "GRID_MAX",
        "RESOLUTION",
        "INITIAL_ROBOT_POSITIONS",
    ],
    CASE_NAME,
)

CFG = build_common_config(PARAMS)
PLOT_MODE_1_DISPLAY_SECONDS = 1.5


# =========================================================================
# 2. MAIN EXECUTION
# =========================================================================

def main():
    source_positions = generate_circular_source_positions(CFG.NUM_SOURCES, CFG.RADIUS)

    print(f"Loaded case: {CASE_NAME}")
    print("Finding optimal control inputs for scheduled targets")

    target_controls = []
    opt_infos = []
    field_data = []

    for start_time, target_pos, target_eig_ratio, target_eig_angle in CFG.TARGET_SCHEDULE:
        print("\n" + "=" * 70)
        print(
            f"Finding control inputs for target {target_pos} "
            f"starting at t={start_time} with eig_ratio={target_eig_ratio} "
            f"and eig_angle={target_eig_angle:.3f} rad"
        )
        print("=" * 70)

        u_target = find_stable_equilibrium_inputs(
            target_pos,
            source_positions,
            CFG.C_F,
            ratio=target_eig_ratio,
            eig_angle_rad=target_eig_angle
        )

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

        H = calculate_potential_hessian(
            target_pos,
            source_positions,
            CFG.C_F,
            u_target
        )

        eigenvalues, eigenvectors = np.linalg.eig(H)

        opt_info = extract_optimization_info(
            control_inputs_u=u_target,
            net_force=net_force,
            H=H,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            desired_pos=target_pos,
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

    for opt_info, field in zip(opt_infos, field_data):
        plot_mode_1(
            field["X"],
            field["Y"],
            field["Fx"],
            field["Fy"],
            source_positions,
            opt_info=opt_info,
            draw_contour=True,
            draw_desired_point=True,
            plot_microrobots=True,
            plot_trajectories=False,
            block=False,
            display_seconds=PLOT_MODE_1_DISPLAY_SECONDS,
            reuse_window=True
        )

    plt.show()


if __name__ == "__main__":
    main()
