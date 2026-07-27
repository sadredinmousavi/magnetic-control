"""Optimize scheduled targets and write magnet angles to a sequence file."""

from pathlib import Path

from case_loader import build_common_config, case_output_path, get_case_name_from_argv, load_case, require_keys
from control_workflow import run_control_workflow
from functions_utility import print_optimization_results


REQUIRED_KEYS = [
    "NUM_SOURCES", "RADIUS", "TARGET_SCHEDULE", "SOURCE_MAGNETIZATION",
    "ROBOT_MAGNETIZATION", "L_SOURCE", "L_ROBOT", "GRID_MIN", "GRID_MAX",
    "RESOLUTION", "INITIAL_ROBOT_POSITIONS",
]


def main(case_name=None):
    case_name = case_name or get_case_name_from_argv()
    params = load_case(case_name)
    require_keys(params, REQUIRED_KEYS, case_name)
    cfg = build_common_config(params)

    print(f"Loaded case: {case_name}")
    result = run_control_workflow(
        cfg, params, compute_fields=False, report=print_optimization_results
    )

    output_filename = Path("outputs") / case_output_path(case_name).with_suffix(".txt")
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as output:
        output.write("# angles | wait | zero\n")
        for opt_info in result.opt_infos:
            angle_text = "[" + ", ".join(
                f"{angle:.2f}" for angle in opt_info["angles_deg"]
            ) + "]"
            output.write(f"{angle_text} | 2 | 180\n")

    print(f"Wrote {output_filename}")
    return output_filename


if __name__ == "__main__":
    main()
