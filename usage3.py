"""Optimize scheduled targets and render selectable static field plots."""

import matplotlib.pyplot as plt

from case_loader import build_common_config, case_output_path, get_case_name_from_argv, load_case, require_keys
from control_workflow import run_control_workflow
from functions_utility import plot_field, print_optimization_results, save_temp_plot


REQUIRED_KEYS = [
    "NUM_SOURCES", "RADIUS", "TARGET_SCHEDULE", "SOURCE_MAGNETIZATION",
    "ROBOT_MAGNETIZATION", "L_SOURCE", "L_ROBOT", "GRID_MIN", "GRID_MAX",
    "RESOLUTION", "INITIAL_ROBOT_POSITIONS",
]


def main(case_name=None, plot_type=None, save_plots=True):
    case_name = case_name or get_case_name_from_argv()
    params = load_case(case_name)
    require_keys(params, REQUIRED_KEYS, case_name)
    cfg = build_common_config(params)
    plot_type = plot_type or params.get("PLOT_TYPE", "force_info")

    print(f"Loaded case: {case_name}")
    result = run_control_workflow(cfg, params, report=print_optimization_results)

    for index, (opt_info, field) in enumerate(
        zip(result.opt_infos, result.field_data), start=1
    ):
        options = {
            "draw_desired_point": True,
            "clip_field_to_dish": params.get("PLOT_FIELD_INSIDE_DISH", False),
            "dish_center": params.get("DISH_CENTER", (0.0, 0.0)),
            "dish_radius": params.get("DISH_RADIUS"),
            "dish_outside_fade_alpha": params.get("DISH_OUTSIDE_FADE_ALPHA", 0.86),
            "show_magnet_moment_vectors": params.get(
                "SHOW_EXTERNAL_MAGNET_MOMENT_VECTORS", False
            ),
            "magnet_moment_arrow_length": params.get(
                "MAGNET_MOMENT_ARROW_LENGTH", 0.035
            ),
            "magnet_moment_arrow_color": params.get(
                "MAGNET_MOMENT_ARROW_COLOR", "#d1495b"
            ),
        }
        if str(plot_type).lower() in {"1", "force_info"}:
            options.update({
                "draw_contour": True,
                "plot_microrobots": True,
                "plot_trajectories": False,
                "block": False,
                "display_seconds": params.get("PLOT_DISPLAY_SECONDS", 1.5),
                "reuse_window": True,
            })
        fig = plot_field(
            plot_type, field, result.source_positions, opt_info, **options
        )
        if save_plots:
            save_temp_plot(fig, index, folder_name=case_output_path(case_name))

    plt.show()
    return result


if __name__ == "__main__":
    main()
