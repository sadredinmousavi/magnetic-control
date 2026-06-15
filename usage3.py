import numpy as np
import matplotlib.pyplot as plt

from case_loader import (
    build_common_config,
    get_case_name_from_argv,
    load_case,
    require_keys,
    unpack_target_schedule_entry,
)
from functions_main import (
    calculate_potential_hessian,
    calculate_total_force_from_sources,
    find_two_stable_equilibrium_inputs,
    find_two_equilibrium_inputs,
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
        "SOURCE_MAGNETIZATION",
        "ROBOT_MAGNETIZATION",
        "L_SOURCE",
        "L_ROBOT",
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

    for target_entry in CFG.TARGET_SCHEDULE:
        (
            start_time,
            target_pos,
            second_equilibrium_pos,
            target_eig_ratio,
            target_eig_angle,
        ) = unpack_target_schedule_entry(target_entry)

        print("\n" + "=" * 70)
        if second_equilibrium_pos is None:
            print(
                f"Finding control inputs for target {target_pos} "
                f"starting at t={start_time} with eig_ratio={target_eig_ratio} "
                f"and eig_angle={target_eig_angle:.3f} rad"
            )
        else:
            print(
                f"Finding control inputs for equilibrium points {target_pos} "
                f"and {second_equilibrium_pos} starting at t={start_time}"
            )
        print("=" * 70)

        if second_equilibrium_pos is None:
            u_target = find_stable_equilibrium_inputs(
                target_pos,
                source_positions,
                CFG.C_F,
                ratio=target_eig_ratio,
                eig_angle_rad=target_eig_angle,
                trace_margin=CFG.STABILITY_TRACE_MARGIN,
                det_margin=CFG.STABILITY_DET_MARGIN
            )
        else:
            if PARAMS.get("TWO_EQUILIBRIUM_SOLVER") == "plain":
                u_target = find_two_equilibrium_inputs(
                    target_pos,
                    second_equilibrium_pos,
                    source_positions,
                    CFG.C_F
                )
            else:
                u_target = find_two_stable_equilibrium_inputs(
                    target_pos,
                    second_equilibrium_pos,
                    source_positions,
                    CFG.C_F,
                    trace_margin=CFG.STABILITY_TRACE_MARGIN,
                    det_margin=CFG.STABILITY_DET_MARGIN
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

        equilibrium_positions = [target_pos]
        equilibrium_net_forces = [net_force]

        if second_equilibrium_pos is not None:
            net_force_2 = calculate_total_force_from_sources(
                source_positions,
                u_target,
                second_equilibrium_pos,
                CFG.M_SOURCE_MAGNITUDE,
                CFG.M_ROBOT_MAGNITUDE
            )
            equilibrium_positions.append(second_equilibrium_pos)
            equilibrium_net_forces.append(net_force_2)

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
            equilibrium_positions=equilibrium_positions,
            equilibrium_net_forces=equilibrium_net_forces,
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
