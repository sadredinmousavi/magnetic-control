"""Extract stationary grid dwells from microrobot tracking data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPEAT_COUNT = 3
ROLLING_WINDOW_FRAMES = 45
STATIONARY_THRESHOLD_PX = 2.5
MIN_DWELL_FRAMES = 30

TIME_COLUMN = "time_s"
X_COLUMN = "robot_center_x_px"
Y_COLUMN = "robot_center_y_px"
ACCEPTED_COLUMN = "accepted"


def choose_grid_size():
    """Ask for the experiment grid before opening the tracking-file dialog."""
    while True:
        answer = input("Grid size (3 for 3x3, 5 for 5x5; Q to quit): ").strip()
        if answer.lower() == "q":
            raise SystemExit(0)
        if answer in ("3", "5"):
            return int(answer)
        print("Please enter 3 or 5.")


def choose_input_file(initial_dir=None):
    """Open a dialog for a GUI004 detection CSV or compatible spreadsheet."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    initial_path = Path(initial_dir or Path.cwd()).resolve()
    if not initial_path.is_dir():
        initial_path = Path.cwd()
    selected = filedialog.askopenfilename(
        title="Select microrobot tracking data",
        initialdir=str(initial_path),
        filetypes=[
            ("Tracking data", "*.csv *.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise SystemExit("No tracking file selected.")
    return Path(selected)


def load_tracking_file(filename):
    path = Path(filename)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    for column in (TIME_COLUMN, X_COLUMN, Y_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"Required column '{column}' not found.")

    if ACCEPTED_COLUMN in frame.columns:
        accepted = frame[ACCEPTED_COLUMN]
        if accepted.dtype == object:
            accepted = accepted.astype(str).str.strip().str.lower().isin(
                ("1", "true", "yes")
            )
        else:
            accepted = accepted == 1
        frame = frame[accepted].copy()

    frame = frame[[TIME_COLUMN, X_COLUMN, Y_COLUMN]].dropna()
    # GUI004 repeats the swarm center on each accepted robot row. Retain one
    # position per video timestamp so rolling windows count actual frames.
    frame = frame.drop_duplicates(subset=[TIME_COLUMN], keep="first")
    return frame.sort_values(TIME_COLUMN).reset_index(drop=True)


def detect_dwell_segments(frame):
    rolling_x = frame[X_COLUMN].rolling(ROLLING_WINDOW_FRAMES, center=True).std()
    rolling_y = frame[Y_COLUMN].rolling(ROLLING_WINDOW_FRAMES, center=True).std()
    stationary = np.hypot(rolling_x, rolling_y) < STATIONARY_THRESHOLD_PX
    indices = np.where(stationary.fillna(False))[0]
    if not len(indices):
        return []

    segments = []
    start = previous = indices[0]
    for current in indices[1:]:
        if current == previous + 1:
            previous = current
            continue
        if previous - start + 1 >= MIN_DWELL_FRAMES:
            segments.append((start, previous))
        start = previous = current
    if previous - start + 1 >= MIN_DWELL_FRAMES:
        segments.append((start, previous))
    return segments


def extract_dwell_statistics(frame, segments):
    rows = []
    for dwell_id, (start, end) in enumerate(segments, start=1):
        section = frame.iloc[start:end + 1]
        start_time = section[TIME_COLUMN].iloc[0]
        end_time = section[TIME_COLUMN].iloc[-1]
        rows.append({
            "Dwell ID": dwell_id,
            "Start (s)": start_time,
            "End (s)": end_time,
            "Duration (s)": end_time - start_time,
            "Median X (px)": section[X_COLUMN].median(),
            "Median Y (px)": section[Y_COLUMN].median(),
            "Mean X (px)": section[X_COLUMN].mean(),
            "Mean Y (px)": section[Y_COLUMN].mean(),
            "Std X (px)": section[X_COLUMN].std(),
            "Std Y (px)": section[Y_COLUMN].std(),
            "Frames": len(section),
        })
    return pd.DataFrame(rows)


def assign_grid_locations(dwells, grid_size):
    point_count = grid_size**2
    if len(dwells) < point_count:
        raise ValueError(
            f"Only {len(dwells)} dwells were detected; {point_count} are needed "
            "to establish the first grid pass."
        )

    result = dwells.copy()
    reference = result.iloc[:point_count][
        ["Median X (px)", "Median Y (px)"]
    ].to_numpy()
    measured = result[["Median X (px)", "Median Y (px)"]].to_numpy()
    distances = np.linalg.norm(
        measured[:, None, :] - reference[None, :, :], axis=2
    )
    result["Grid Point"] = np.argmin(distances, axis=1) + 1
    result["Distance to Reference (px)"] = np.min(distances, axis=1)
    return result


def build_expected_schedule(grid_size, repeat_count=REPEAT_COUNT):
    """Match the alternating forward/reverse paths in Case 100 tests."""
    normal = list(range(1, grid_size**2 + 1))
    rows = []
    schedule_id = 1
    for pass_index in range(1, repeat_count + 1):
        points = normal if pass_index % 2 else normal[::-1]
        for grid_point in points:
            rows.append({
                "Schedule ID": schedule_id,
                "Pass": pass_index,
                "Grid Point": grid_point,
            })
            schedule_id += 1
    return pd.DataFrame(rows)


def align_detected_to_schedule(dwells, schedule):
    observed = dwells["Grid Point"].tolist()
    expected = schedule["Grid Point"].tolist()
    row_count, column_count = len(observed), len(expected)
    costs = np.full((row_count + 1, column_count + 1), np.inf)
    actions = np.empty((row_count + 1, column_count + 1), dtype=object)
    costs[0, 0] = 0

    for column in range(1, column_count + 1):
        costs[0, column] = costs[0, column - 1] + 1
        actions[0, column] = "skip_expected"
    for row in range(1, row_count + 1):
        costs[row, 0] = costs[row - 1, 0] + 2
        actions[row, 0] = "skip_observed"

    choices = ("match", "skip_expected", "skip_observed")
    for row in range(1, row_count + 1):
        for column in range(1, column_count + 1):
            candidates = (
                costs[row - 1, column - 1]
                + (0 if observed[row - 1] == expected[column - 1] else 5),
                costs[row, column - 1] + 1,
                costs[row - 1, column] + 2,
            )
            choice = int(np.argmin(candidates))
            costs[row, column] = candidates[choice]
            actions[row, column] = choices[choice]

    matches = []
    missing_indices = []
    row, column = row_count, column_count
    while row > 0 or column > 0:
        action = actions[row, column]
        if action == "match":
            matches.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif action == "skip_expected":
            missing_indices.append(column - 1)
            column -= 1
        elif action == "skip_observed":
            row -= 1
        else:
            break

    result = dwells.copy()
    result["Pass"] = np.nan
    result["Schedule ID"] = np.nan
    for observed_index, expected_index in reversed(matches):
        if observed[observed_index] == expected[expected_index]:
            result.loc[observed_index, "Pass"] = schedule.loc[expected_index, "Pass"]
            result.loc[observed_index, "Schedule ID"] = schedule.loc[
                expected_index, "Schedule ID"
            ]
    missing = schedule.iloc[list(reversed(missing_indices))].copy()
    return result, missing


def save_results(dwells, output_file):
    dwell_columns = [
        "Dwell ID", "Pass", "Grid Point", "Schedule ID", "Start (s)",
        "End (s)", "Duration (s)", "Median X (px)", "Median Y (px)",
        "Mean X (px)", "Mean Y (px)", "Std X (px)", "Std Y (px)",
        "Frames", "Distance to Reference (px)",
    ]
    dwells[dwell_columns].to_csv(output_file, index=False)


def parse_args():
    if len(sys.argv) > 3:
        raise SystemExit("Usage: python usage5.py [3|5] [tracking_file]")
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else None
    input_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    return grid_size, input_file


def main(grid_size=None, input_filename=None, output_filename=None):
    if grid_size is None:
        grid_size = choose_grid_size()
    if grid_size not in (3, 5):
        raise ValueError("Grid size must be 3 or 5.")
    input_file = Path(input_filename) if input_filename else choose_input_file(
        Path.cwd() / "outputs" / "offline_detection"
    )
    if not input_file.is_file():
        raise FileNotFoundError(f"Tracking file not found: {input_file}")
    output_file = Path(output_filename) if output_filename else input_file.with_name(
        f"{input_file.stem}_dwell_points_{grid_size}x{grid_size}.csv"
    )

    print(f"Grid: {grid_size} x {grid_size}")
    print(f"Loading: {input_file}")
    frame = load_tracking_file(input_file)
    print(f"Accepted unique frames: {len(frame)}")
    segments = detect_dwell_segments(frame)
    print(f"Stationary dwells detected: {len(segments)}")
    dwells = extract_dwell_statistics(frame, segments)
    dwells = assign_grid_locations(dwells, grid_size)
    schedule = build_expected_schedule(grid_size)
    dwells, missing = align_detected_to_schedule(dwells, schedule)
    save_results(dwells, output_file)

    print(f"Expected dwells: {grid_size**2 * REPEAT_COUNT}")
    print(f"Detected dwells: {len(dwells)}")
    print(f"Missing dwells: {len(missing)}")
    if len(missing):
        print("Missing scheduled visits:")
        print(missing.to_string(index=False))
    print(f"Saved: {output_file.resolve()}")
    return output_file


if __name__ == "__main__":
    selected_grid, selected_file = parse_args()
    main(selected_grid, selected_file)
