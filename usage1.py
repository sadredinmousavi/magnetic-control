"""Plot the complete scheduled target trajectory without computing a field."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from case_loader import (
    case_output_name,
    case_output_path,
    get_case_name_from_argv,
    load_case,
    require_keys,
    unpack_target_schedule_entry,
    validate_target_schedule,
)
from functions_main import generate_circular_source_positions


REQUIRED_KEYS = ["NUM_SOURCES", "RADIUS", "TARGET_SCHEDULE", "GRID_MIN", "GRID_MAX"]
PROJECT_DIR = Path(__file__).resolve().parent
ALL_OUTPUT_DIRNAME = "target_trajectories"


def discover_case_conditions(project_dir=PROJECT_DIR):
    """Return every runnable case/condition module name in name order."""
    cases_dir = Path(project_dir) / "cases"
    case_names = []
    for case_dir in sorted(cases_dir.glob("case_*"), key=lambda path: path.name.lower()):
        if not case_dir.is_dir() or not (case_dir / "case.py").is_file():
            continue
        conditions = sorted(
            (path.stem for path in case_dir.glob("cond_*.py") if path.is_file()),
            key=str.lower,
        )
        case_names.extend(f"{case_dir.name}.{condition}" for condition in conditions)
    return case_names


def _target_groups(schedule):
    """Return every scheduled set of one, two, or four target positions."""
    groups = []
    for entry in schedule:
        _, primary, additional, _, _ = unpack_target_schedule_entry(entry)
        positions = [primary]
        if additional is not None:
            positions.extend(additional if isinstance(additional, list) else [additional])
        groups.append(np.asarray(positions, dtype=float))
    return groups


def create_target_trajectory_figure(params):
    """Create a field-free overview of target tracks and workspace geometry."""
    schedule = params["TARGET_SCHEDULE"]
    validate_target_schedule(schedule)
    groups = _target_groups(schedule)

    figure_size = params.get("ANIMATION_FIGURE_SIZE", (8, 8))
    fig, ax = plt.subplots(figsize=figure_size)
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.12, top=0.93)

    ax.set_xlim(params["GRID_MIN"], params["GRID_MAX"])
    ax.set_ylim(params["GRID_MIN"], params["GRID_MAX"])
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(params.get("TARGET_TRAJECTORY_TITLE", "Scheduled Target Trajectory"))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    source_positions = generate_circular_source_positions(
        params["NUM_SOURCES"], params["RADIUS"]
    )
    ax.scatter(
        source_positions[:, 0], source_positions[:, 1],
        c="gray", s=100, marker="s", edgecolors="black",
        label="Source magnets", zorder=8,
    )

    for index, wall in enumerate(params.get("WALL_SEGMENTS", [])):
        start = np.asarray(wall[0], dtype=float)
        end = np.asarray(wall[1], dtype=float)
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color="dimgray", linewidth=4, solid_capstyle="round",
            label="Walls" if index == 0 else None, zorder=6,
        )

    dish_radius = params.get("DISH_RADIUS")
    if dish_radius is not None:
        dish_center = np.asarray(params.get("DISH_CENTER", (0.0, 0.0)), dtype=float)
        ax.add_patch(patches.Circle(
            dish_center, dish_radius, fill=False, edgecolor="black",
            linewidth=2.0, label="Petri dish", zorder=7,
        ))

    max_targets = max(len(group) for group in groups)
    colors = ["#e07a2d"] + [
        plt.cm.tab10(index) for index in range(1, max_targets)
    ]
    for target_index in range(max_targets):
        track = np.asarray([
            group[target_index] for group in groups if target_index < len(group)
        ])
        label = (
            "Target trajectory"
            if target_index == 0
            else f"Target {target_index + 1} trajectory"
        )
        ax.plot(
            track[:, 0], track[:, 1], color=colors[target_index],
            linewidth=1.8, linestyle=(0, (4.0, 2.5)),
            alpha=0.9, label=label, zorder=8,
        )
        ax.scatter(
            track[:, 0], track[:, 1], color=colors[target_index],
            s=24, edgecolors="black", linewidths=0.5, zorder=9,
        )

    primary_track = np.asarray([group[0] for group in groups])
    ax.scatter(
        primary_track[0, 0], primary_track[0, 1],
        c="#2a9d55", s=90, marker="o", edgecolors="black",
        label="Start", zorder=10,
    )
    ax.scatter(
        primary_track[-1, 0], primary_track[-1, 1],
        c="red", s=180, marker="X", edgecolors="black",
        label="End", zorder=10,
    )

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)
    return fig


def save_all_target_trajectories(case_names=None, output_dir=None):
    """Save one target plot for every case/condition without opening plot windows."""
    case_names = list(case_names or discover_case_conditions())
    if not case_names:
        raise FileNotFoundError("No cases/case_*/cond_*.py files were found.")

    output_dir = Path(output_dir or Path("outputs") / ALL_OUTPUT_DIRNAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    print(f"Saving {len(case_names)} target trajectories to {output_dir.resolve()}\n")
    for case_name in case_names:
        params = load_case(case_name)
        require_keys(params, REQUIRED_KEYS, case_name)
        fig = create_target_trajectory_figure(params)
        output_path = output_dir / f"{case_output_name(case_name)}_target_trajectory.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)
        print(f"Saved: {output_path.resolve()}")

    print(f"\nSaved {len(saved_paths)} target trajectory plots.")
    return saved_paths


def main(case_name=None, save_plot=True, show_plot=True):
    case_name = case_name or get_case_name_from_argv()
    if str(case_name).strip().lower() == "all":
        return save_all_target_trajectories()

    params = load_case(case_name)
    require_keys(params, REQUIRED_KEYS, case_name)

    print(f"Loaded case: {case_name}")
    fig = create_target_trajectory_figure(params)

    output_path = Path("outputs") / case_output_path(case_name) / "target_trajectory.png"
    if save_plot:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved target trajectory: {output_path.resolve()}")

    if show_plot:
        plt.show()
    return fig, output_path


if __name__ == "__main__":
    main()
