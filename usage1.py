import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from case_loader import (
    build_common_config,
    get_case_name_from_argv,
    load_case,
    require_keys,
    select_file_from_dialog,
)
from functions_main import (
    generate_circular_source_positions,
)
from functions_utility import (
    compute_grid_fields,
    plot_mode_1,
    save_temp_plot,
)


# =========================================================================
# 1. SYSTEM PARAMETERS & CONSTANTS
# =========================================================================

PLOT_MODE_1_DISPLAY_SECONDS = 1.5


def parse_args():
    if len(sys.argv) > 3:
        raise SystemExit(
            "Usage: python usage001.py <case_name> <input_angles_file>\n"
            "Example: python usage001.py case_payload_baseline outputs/case_payload_baseline.txt"
        )

    case_name = get_case_name_from_argv("case_payload_baseline")

    if len(sys.argv) > 2:
        input_filename = Path(sys.argv[2])
    else:
        input_filename = select_file_from_dialog(
            initial_dir=Path.cwd() / "outputs",
            title="Select angle input file",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
            default_path=Path.cwd() / "outputs" / f"{case_name}.txt",
            cancel_message="No angle input file selected.",
        )

    return case_name, input_filename



def load_angle_rows(input_filename):
    angle_rows = []
    wait_values = []
    zero_values = []

    with open(input_filename, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split("|")]

            if len(parts) < 1:
                continue

            angle_text = parts[0]

            if not angle_text.startswith("[") or not angle_text.endswith("]"):
                raise ValueError(
                    f"Invalid angle format at line {line_number}: {line}"
                )

            angles_deg = np.array(
                [float(value.strip()) for value in angle_text[1:-1].split(",")],
                dtype=float,
            )

            wait = float(parts[1]) if len(parts) > 1 else None
            zero = float(parts[2]) if len(parts) > 2 else None

            angle_rows.append(angles_deg)
            wait_values.append(wait)
            zero_values.append(zero)

    return angle_rows, wait_values, zero_values


def angles_deg_to_control_inputs(angles_deg):
    angles_rad = np.radians(angles_deg)
    return np.cos(angles_rad)


def build_plot_opt_info(angles_deg, angles_rad, desired_pos=None):
    if desired_pos is None:
        desired_pos = np.array([0.0, 0.0])

    return {
        "angles_rad": angles_rad,
        "angles_deg": angles_deg,
        "desired_pos": desired_pos,
        "equilibrium_positions": [],
        "eigenvalues": None,
        "eigenvectors": None,
        "microrobot_positions": None,
    }


# =========================================================================
# 2. MAIN EXECUTION
# =========================================================================

def main():
    case_name, input_filename = parse_args()

    params = load_case(case_name)

    require_keys(
        params,
        [
            "NUM_SOURCES",
            "RADIUS",
            "SOURCE_MAGNETIZATION",
            "ROBOT_MAGNETIZATION",
            "L_SOURCE",
            "L_ROBOT",
            "GRID_MIN",
            "GRID_MAX",
            "RESOLUTION",
        ],
        case_name,
    )

    cfg = build_common_config(params)
    source_positions = generate_circular_source_positions(cfg.NUM_SOURCES, cfg.RADIUS)

    angle_rows, wait_values, zero_values = load_angle_rows(input_filename)

    if not angle_rows:
        raise ValueError(f"No angle rows found in {input_filename}")

    print(f"Loaded case: {case_name}")
    print(f"Loaded angle input file: {input_filename}")
    print(f"Found {len(angle_rows)} angle rows")

    for row_index, angles_deg in enumerate(angle_rows, start=1):
        if len(angles_deg) != cfg.NUM_SOURCES:
            raise ValueError(
                f"Angle row {row_index} has {len(angles_deg)} angles, "
                f"but NUM_SOURCES={cfg.NUM_SOURCES}"
            )

        angles_rad = np.radians(angles_deg)
        u_target = angles_deg_to_control_inputs(angles_deg)

        source_moment_vectors = np.zeros_like(source_positions)
        for i, pos in enumerate(source_positions):
            radial_unit_vector = pos / np.linalg.norm(pos)
            source_moment_vectors[i] = (
                u_target[i] * cfg.M_SOURCE_MAGNITUDE * radial_unit_vector
            )

        X, Y, Fx, Fy, U_pot, Bx, By = compute_grid_fields(
            source_positions,
            u_target,
            source_moment_vectors,
            cfg.GRID_MIN,
            cfg.GRID_MAX,
            cfg.RESOLUTION,
            cfg.M_SOURCE_MAGNITUDE,
            cfg.M_ROBOT_MAGNITUDE,
        )

        opt_info = build_plot_opt_info(
            angles_deg=angles_deg,
            angles_rad=angles_rad,
        )

        print("\n" + "=" * 70)
        print(f"Plotting input row {row_index}")
        print(f"wait={wait_values[row_index - 1]}, zero={zero_values[row_index - 1]}")
        print("angles_deg:", np.array2string(angles_deg, precision=2))
        print("control u:", np.array2string(u_target, precision=4))
        print("=" * 70)

        fig = plot_mode_1(
            X,
            Y,
            Fx,
            Fy,
            source_positions,
            opt_info=opt_info,
            draw_contour=True,
            draw_desired_point=False,
            plot_microrobots=False,
            plot_trajectories=False,
            block=False,
            display_seconds=PLOT_MODE_1_DISPLAY_SECONDS,
            reuse_window=True,
        )
        save_temp_plot(fig, row_index, folder_name=case_name)
        # plt.close(fig)

    plt.show()


if __name__ == "__main__":
    main()
